import torch
import torch.nn as nn
import torch.nn.functional as F

# Coordinate order: [FL-ID, FL-IP, FL-TX, TN-ID, TN-IP, TN-TX, MD-ID, MD-IP, MD-TX]
# Metafunction columns: ideational {0,3,6}, interpersonal {1,4,7}, textual {2,5,8}
METAFUNCTION_COORDS = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]


class MetafunctionalAttention(nn.Module):
    """Three heads. Head m forms its relations only from metafunction m's coordinates.
    Values are read from the full state. Attention is causal."""

    def __init__(self, hidden=72, in_dim=9):
        super().__init__()
        self.hd = hidden // 3
        self.register_buffer('idx', torch.tensor(METAFUNCTION_COORDS))
        self.q = nn.ModuleList([nn.Linear(3, self.hd) for _ in range(3)])
        self.k = nn.ModuleList([nn.Linear(3, self.hd) for _ in range(3)])
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, h, dz):
        B, T, _ = h.shape
        v = self.v(h).view(B, T, 3, self.hd).transpose(1, 2)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        ctx = []
        for m in range(3):
            sub = dz[..., self.idx[m]]
            s = self.q[m](sub) @ self.k[m](sub).transpose(-2, -1) / self.hd ** 0.5
            s = s.masked_fill(mask, float('-inf'))
            ctx.append(F.softmax(s, -1) @ v[:, m])
        return self.out(torch.cat(ctx, -1))


class SFLModel(nn.Module):
    def __init__(self, in_dim=9, hidden=72):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden))
        self.attn = MetafunctionalAttention(hidden, in_dim)
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.head = nn.Linear(hidden, in_dim)

    def forward(self, dz):
        h = self.project(dz)
        h = h + self.attn(h, dz)
        h = h + self.ff(self.norm(h))
        return self.head(h)


def trajectory_loss(model, z):
    """z: (B, T+1, 9) states. Predict displacement t+1 from displacements up to t."""
    dz = z[:, 1:] - z[:, :-1]
    return F.mse_loss(model(dz[:, :-1]), dz[:, 1:]), dz


if __name__ == '__main__':
    torch.manual_seed(0)
    m = SFLModel()
    z = torch.cumsum(torch.randn(8, 17, 9) * 0.1, 1)
    loss0, dz = trajectory_loss(m, z)
    zero = F.mse_loss(torch.zeros_like(dz[:, 1:]), dz[:, 1:]).item()
    copy = F.mse_loss(dz[:, :-1], dz[:, 1:]).item()
    x = dz[:, :-1].clone(); y0 = m(x); x[:, -1] += 5.0; y1 = m(x)
    causal = torch.allclose(y0[:, :-1], y1[:, :-1], atol=1e-6)
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    for _ in range(300):
        l, _ = trajectory_loss(m, z); opt.zero_grad(); l.backward(); opt.step()
    grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    print(f'params {sum(p.numel() for p in m.parameters())}')
    print(f'baseline zero {zero:.4f} copy {copy:.4f} initial {loss0.item():.4f} final {l.item():.4f}')
    print(f'causal {causal} gradients_finite {grads} overfit {l.item() < 0.05 * loss0.item()}')
