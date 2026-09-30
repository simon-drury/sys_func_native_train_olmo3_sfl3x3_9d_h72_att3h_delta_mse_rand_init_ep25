import torch
import torch.nn as nn
import torch.nn.functional as F

class SFLStateAdapter(nn.Module):
    """
    Adapter matrix W_adapt projecting continuous 9D first-difference state deltas
    (Delta z_t in R^9) into hidden dimension d_model = 72.
    
    Subscript coordinates:
    [m_FL,ID, m_FL,IP, m_FL,TX, m_TN,ID, m_TN,IP, m_TN,TX, m_MD,ID, m_MD,IP, m_MD,TX]
    """
    def __init__(self, in_dim=9, hidden_dim=72):
        super().__init__()
        self.w_adapt = nn.Linear(in_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, delta_z):
        return self.layer_norm(self.w_adapt(delta_z))

class SFLMetafunctionalAttention(nn.Module):
    """
    3-Head Metafunctional Attention isolating Ideational, Interpersonal, and Textual subspaces.
    Enforces instant activation clearance boundary post-pass.
    """
    def __init__(self, embed_dim=72, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)

        # Activation Clearance Boundary: Release intermediate tensors
        del q, k, v, scores, attn
        return out

class SFLTrajectoryEngine(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=72):
        super().__init__()
        self.adapter = SFLStateAdapter(in_dim, hidden_dim)
        self.attn = SFLMetafunctionalAttention(hidden_dim, num_heads=3)
        self.head = nn.Linear(hidden_dim, in_dim)

    def forward(self, delta_z):
        h = self.adapter(delta_z)
        h = h + self.attn(h)
        return self.head(h)

def compute_trajectory_loss(delta_z_hat, delta_z):
    return F.mse_loss(delta_z_hat, delta_z)
