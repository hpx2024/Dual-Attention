import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np

import torch.nn as nn
from torch_scatter import scatter


class PreNormException(Exception):
    pass


class PreNormLayer(torch.nn.Module):
    def __init__(self, n_units, shift=True, scale=True, name=None):
        super().__init__()
        assert shift or scale
        self.register_buffer('shift', torch.zeros(n_units) if shift else None)
        self.register_buffer('scale', torch.ones(n_units) if scale else None)
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input_):
        if self.waiting_updates:
            self.update_stats(input_)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input_ = input_ + self.shift

        if self.scale is not None:
            input_ = input_ * self.scale

        return input_

    def start_updates(self):
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input_):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input_.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input_.shape[-1]}."

        input_ = input_.reshape(-1, self.n_units)
        sample_avg = input_.mean(dim=0)
        sample_var = (input_ - sample_avg).pow(2).mean(dim=0)
        sample_count = np.prod(input_.size())/self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift = -self.avg

        if self.scale is not None:
            self.var[self.var < 1e-8] = 1
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False
        self.trainable = False
        

class BaseModel(torch.nn.Module):
    """
    Our base model class, which implements pre-training methods.
    """

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pre_train_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def pre_train(self, *args, **kwargs):
        try:
            with torch.no_grad():
                self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True




class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff, d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = self.layer_norm(output + residual)
        return output   


# SelfAttentionKernelBasedLayer
class SelfAttentionKernelBasedLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super(SelfAttentionKernelBasedLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs, self_mask=None):

        residual, batch_size = inputs, inputs.size(0)

        # Project and reshape: (B, S, D) -> (B, S, D_new) -> (B, S, H, W) -> (B, H, S, W)
        Q = self.W_Q(inputs).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(inputs).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(inputs).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)

        # Apply sigmoid as kernel function
        Q = F.sigmoid(Q) 
        K = F.sigmoid(K)

        # Kernel-based linear attention computation
        # Numerator: Q * (K^T * V)
        KV = torch.einsum('bhnk, bhnv -> bhkv', K, V)
        num = torch.einsum('bhnk, bhkv -> bhnv', Q, KV)

        # Denominator: Q * sum(K) for normalization
        k_sum = torch.einsum('bhnk -> bhk', K)
        den = torch.einsum('bhnk, bhk -> bhn', Q, k_sum).unsqueeze(-1) 


        den = den + 1e-8  # Prevent division by zero
        output = (num/den).transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(output)
        output = self.layer_norm(output + residual)
       
        attn = None  # Attention weights not explicitly computed in kernel trick
        return output, attn

# CrossLocalAttentionLayer
class CrossLocalAttentionLayer(nn.Module):

    def __init__(self, n_heads, d_model, d_k, d_v):
        super(CrossLocalAttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_E = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, edge_indices, edge_features, input_Q, input_K, input_V, cross_mask=None):

        residual, batch_size = input_Q, input_Q.size(0)

        # Project to multi-head space: (B, S, D) -> (B, S, H, W) -> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)
        E = self.W_E(edge_features).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)

        # Get edge indices for current batch
        src_nodes, tgt_nodes = edge_indices[0][0], edge_indices[0][1]

        # Select Q, K, V for nodes connected by edges
        Q_edges = Q[:, :, src_nodes, :] 
        K_edges = K[:, :, tgt_nodes, :]
        V_edges = V[:, :, tgt_nodes, :] 

        # Compute attention scores incorporating edge features
        scores = torch.mul(Q_edges, K_edges) / np.sqrt(self.d_k)
        scores = torch.mul(scores, E)
        attn = torch.exp(scores.sum(-1, keepdim=True).clamp(-5, 5))
        msg  = torch.mul(attn, V_edges)

        # Aggregate messages to source nodes using scatter
        index = edge_indices[0][0].view(batch_size, 1, -1)
        index = index.expand(batch_size, self.n_heads, -1)
        output = torch.zeros_like(Q)
        scatter(msg, index, dim=2, out=output, reduce='add')

        # Compute normalization coefficients
        attn_normal_coeff = scores.new_zeros(batch_size, self.n_heads, input_Q.size(1),1)
        scatter(attn, index, dim=2, out=attn_normal_coeff, reduce='add')

        # Normalize results
        output = output / (attn_normal_coeff + 1e-8)

        # Reshape and project back to model dimension
        output = output.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(output)
        output = self.layer_norm(output + residual)

        return output, attn


# Self+Cross_Layer
class Dual_Attention_layer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff):
        super(Dual_Attention_layer, self).__init__()

        # Self-attention for each node type
        self.self_attn_var = SelfAttentionKernelBasedLayer(n_heads, d_model, d_k, d_v)
        self.sef_attn_con = SelfAttentionKernelBasedLayer(n_heads, d_model, d_k, d_v)

        # Cross-attention between node types
        self.cross_attn_var2con = CrossLocalAttentionLayer(n_heads, d_model, d_k, d_v)
        self.cross_attn_con2var = CrossLocalAttentionLayer(n_heads, d_model, d_k, d_v)

        # Feature fusion modules to combine self and cross-attention outputs
        self.fusion_c = nn.Sequential(
            torch.nn.Linear(2*d_model, d_model),
            torch.nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        self.fusion_v = nn.Sequential(
            torch.nn.Linear(2*d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(d_model)
        )

        # Independent feed-forward networks for each stream
        self.ffn_con = PoswiseFeedForwardNet(d_ff, d_model)
        self.ffn_var = PoswiseFeedForwardNet(d_ff, d_model)

    def forward(self, constraint_features, edge_indices, edge_features, variable_features, self_mask_constraint=None, cross_mask_c2v=None, cross_mask_v2c=None):

        # Reverse edge indices for variable -> constraint attention
        reversed_edge_indices = edge_indices[:, [1, 0], :]

        # Variable stream: self-attention + cross-attention from constraints
        self_var, self_attn_var = self.self_attn_var(variable_features)
        cross_var, cross_attn_var = self.cross_attn_var2con(reversed_edge_indices, edge_features,variable_features, constraint_features, constraint_features, cross_mask_v2c)

        # Constraint stream: self-attention + cross-attention from variables
        self_con, sel_attn_con = self.sef_attn_con(constraint_features, self_mask_constraint)
        cross_con, cross_attn_con = self.cross_attn_con2var(edge_indices, edge_features, constraint_features, variable_features, variable_features, cross_mask_c2v)

        # Fuse self and cross-attention outputs
        combined_constraint = self.fusion_c(torch.cat([self_con, cross_con], dim=-1))
        combined_variable = self.fusion_v(torch.cat([self_var, cross_var], dim=-1))

        # Apply feed-forward networks
        output_constraint = self.ffn_con(combined_constraint)
        output_variable = self.ffn_var(combined_variable)

        return output_variable,output_constraint


class Dual_Attention(BaseModel):
    def __init__(self, embSize, nConsF, nEdgeF, nVarF,
                n_layers, n_heads, d_model, d_k, d_v, d_ff
                ):

        super(Dual_Attention, self).__init__()
        self.emb_size = embSize 
        self.cons_nfeats = nConsF
        self.edge_nfeats = nEdgeF
        self.var_nfeats = nVarF

        # Embedding layers: map raw features to embedding space
        self.cons_embedding = torch.nn.Sequential(
            PreNormLayer(self.cons_nfeats),
            torch.nn.Linear(self.cons_nfeats, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
        )

        self.edge_embedding = torch.nn.Sequential(
            PreNormLayer(self.edge_nfeats),
            torch.nn.Linear(self.edge_nfeats, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
        )

        self.var_embedding = torch.nn.Sequential(
            PreNormLayer(self.var_nfeats),
            torch.nn.Linear(self.var_nfeats, self.emb_size), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
        )

        # Stack of Dual-Attention layers
        self.layers = nn.ModuleList([Dual_Attention_layer(n_heads, d_model, d_k, d_v, d_ff) for _ in range(n_layers)])

        # Output module
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(), 
            torch.nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):

        # Embed raw features
        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)
        edge_features = self.edge_embedding(edge_features)

        # Process through Dual-Attention layers
        for layer in self.layers:

            variable_features, constraint_features = layer(
                constraint_features, edge_indices, edge_features, variable_features
            )

        
        # Apply output Module
        output = self.output_module(variable_features).squeeze(-1)

        return output
