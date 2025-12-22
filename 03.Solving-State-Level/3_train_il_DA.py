import os
import sys
import argparse
import pathlib
import numpy as np


def pretrain(policy, pretrain_loader):
    policy.pre_train_init()
    i = 0

    # Train one layer at a time
    while True:
        for batch in pretrain_loader:

            batch = batch.to(device)
            constraint_features = batch.constraint_features.unsqueeze(0)
            edge_index = batch.edge_index.unsqueeze(0)
            edge_attr = batch.edge_attr.unsqueeze(0)
            variable_features = batch.variable_features.unsqueeze(0)

            # Train current layer, returns False when layer converges
            if not policy.pre_train(constraint_features, edge_index, edge_attr, variable_features):
                break

        # Move to next layer, returns None when all layers are trained
        if policy.pre_train_next() is None:
            break
        i += 1
    return i


def process(policy, data_loader, top_k=[1, 3, 5, 10], optimizer=None):
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))
    mean_entropy = 0
    n_samples_processed = 0

    # Gradient accumulation configuration
    accumulation_steps = 128
    if optimizer is not None:
        optimizer.zero_grad()

    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(device)
            
            constraint_features = batch.constraint_features.unsqueeze(0)
            edge_index = batch.edge_index.unsqueeze(0)
            edge_attr = batch.edge_attr.unsqueeze(0)
            variable_features = batch.variable_features.unsqueeze(0)

            # Forward pass
            logits = policy(constraint_features, edge_index, edge_attr, variable_features)
            logits = logits.squeeze(0)

            # Extract logits for candidate variables only
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            cross_entropy_loss = F.cross_entropy(logits, batch.candidate_choices, reduction='mean')
            entropy = (-F.softmax(logits, dim=-1)*F.log_softmax(logits, dim=-1)).sum(-1).mean()
            loss = cross_entropy_loss - entropy_bonus*entropy

            # Training step with gradient accumulation
            if optimizer is not None:
                loss = loss / accumulation_steps
                loss.backward()

                # Update weights every accumulation_steps or at end of epoch
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            # Compute top-k accuracy
            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            kacc = []
            for k in top_k:
                # Handle cases where number of candidates < k
                if logits.size()[-1] < k:
                    kacc.append(1.0)
                    continue

                # Get model's top-k predictions
                pred_top_k = logits.topk(k).indices
                pred_top_k_true_scores = true_scores.gather(-1, pred_top_k)

                # Check if best variable is in top-k
                accuracy = (pred_top_k_true_scores == true_bestscore).any(dim=-1).float().mean().item()
                kacc.append(accuracy)

            kacc = np.asarray(kacc)

            # Accumulate metrics
            mean_loss += cross_entropy_loss.item() * batch.num_graphs
            mean_entropy += entropy.item() * batch.num_graphs
            mean_kacc += kacc * batch.num_graphs
            n_samples_processed += batch.num_graphs

    # Average metrics over all samples
    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed
    mean_entropy /= n_samples_processed
    return mean_loss, mean_kacc, mean_entropy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'ufacilities', 'indset', 'mknapsack'],
        default = 'setcover',
        nargs='?',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--wandb',
        help="Use wandb?",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    max_epochs = 1000
    batch_size = 1
    pretrain_batch_size = 1
    valid_batch_size = 1
    lr = 1e-3
    entropy_bonus = 0.0
    top_k = [1, 3, 5, 10]

    # Map problem names to data directories
    problem_folders = {
        'setcover': 'setcover/400r_750c_0.05d',
        'cauctions': 'cauctions/100_500',
        'ufacilities': 'ufacilities/35_35_5',
        'indset': 'indset/500_4',
        'mknapsack': 'mknapsack/100_6',
    }
    problem_folder = problem_folders[args.problem]
    running_dir = f"DA/{args.problem}/{args.seed}"
    os.makedirs(running_dir, exist_ok=True)


    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "cpu"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = f"cuda:0"
    import torch
    import torch.nn.functional as F
    import torch_geometric
    from utilities import log, pad_tensor, GraphDataset, Scheduler
    sys.path.insert(0, os.path.abspath(f'./actor'))
    from actor import Dual_Attention

    embSize = 64
    nConsF = 5
    nEdgeF = 1
    nVarF = 19
    d_model = 64
    d_ff = 128
    d_k = d_v = 64
    n_layers = 4
    n_heads = 2

    policy = Dual_Attention(
        embSize, nConsF, nEdgeF, nVarF,
        n_layers, n_heads, d_model, d_k, d_v, d_ff
    ).to(device)

    # Set random seeds for reproducibility
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    logfile = os.path.join(running_dir, 'il_train_log.txt')
    if os.path.exists(logfile):
        os.remove(logfile)

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"entropy bonus: {entropy_bonus}", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed {args.seed}", logfile)

    if args.wandb:
        import wandb
        wandb.init(project="learn2branch")

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = Scheduler(optimizer, mode='min', patience=10, factor=0.2, verbose=True)

    train_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_folder/'train').glob('sample_*.pkl')]
    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    valid_files = [str(file) for file in (pathlib.Path(f'data/samples')/problem_folder/'valid').glob('sample_*.pkl')]

    pretrain_data = GraphDataset(pretrain_files)
    pretrain_loader = torch_geometric.data.DataLoader(pretrain_data, pretrain_batch_size, shuffle=False)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, valid_batch_size, shuffle=False)

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # Epoch 0: Pre-training only
        if epoch == 0:
            n = pretrain(policy, pretrain_loader)
            log(f"PRETRAINED {n} LAYERS", logfile)

        # Epochs 1+: Supervised training
        else:
            epoch_train_files = rng.choice(train_files, int(np.floor(10000/batch_size))*batch_size, replace=True)
            train_data = GraphDataset(epoch_train_files)
            train_loader = torch_geometric.data.DataLoader(train_data, batch_size, shuffle=True)

            # Training pass
            train_loss, train_kacc, entropy = process(policy, train_loader, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)
            if args.wandb:
                wandb.log({'train_loss':train_loss, 'train_entropy':entropy}, step = epoch)
                wandb.log({f'train_acc@{k}':acc for k, acc in zip(top_k, train_kacc)}, step = epoch)

        # Validation pass
        valid_loss, valid_kacc, entropy = process(policy, valid_loader, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
        if args.wandb:
            wandb.log({'valid_loss':valid_loss, 'valid_entropy':entropy}, step = epoch)
            wandb.log({f'valid_acc@{k}':acc for k, acc in zip(top_k, valid_kacc)}, step = epoch)

        # Learning rate scheduling and early stopping
        scheduler.step(valid_loss)
        if scheduler.num_bad_epochs == 0:
            torch.save(policy.state_dict(), pathlib.Path(running_dir)/'il.pkl')
            log(f"  best model so far", logfile)
        elif scheduler.num_bad_epochs == 10:
            log(f"  10 epochs without improvement, decreasing learning rate", logfile)
        elif scheduler.num_bad_epochs == 20:
            log(f"  20 epochs without improvement, early stopping", logfile)
            break

    # Load best model and evaluate on validation set
    policy.load_state_dict(torch.load(pathlib.Path(running_dir)/'il.pkl'))
    valid_loss, valid_kacc, entropy = process(policy, valid_loader, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
