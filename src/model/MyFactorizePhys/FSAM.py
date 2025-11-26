import torch
from torch import nn
from sklearn.decomposition import NMF


class FSAM(nn.Module):
    def __init__(self, ch_input, ch_align, L=1, *a, **k):
        super().__init__()

        # Preprocessing operation (τ×κ×α×β -> τ×κ×α×β)
        self.ξ_pre = nn.Sequential(
            nn.Conv3d(ch_input, ch_align, (1, 1, 1)), 
            nn.ReLU(inplace=True)
        )

        # Non-negative matrix factorization (M×N -> M×L, L×N)
        self.NMF_model = NMF(n_components=L, init='random', random_state=0)

        # Postprocessing operation (τ×κ×α×β -> τ×κ×α×β)
        post_kws = dict(bias=False) #, dtype=torch.float64)
        self.ξ_post = nn.Sequential(
            nn.Conv3d(ch_align, ch_align, (1, 1, 1), **post_kws),
            nn.ReLU(inplace=True),
            # nn.InstanceNorm3d(ch_align), # The authors also don't implement this
            nn.Conv3d(ch_align, ch_input, 1, **post_kws)
        )
    

    @torch.no_grad() # PyTorch doesn't need to concern itself here
    def ϕ(self, V, return_val=None):
        W = self.NMF_model.fit_transform(V.detach().clone().numpy())
        H = self.NMF_model.components_
        if return_val == 'WH': return W, H # Not part of algorithm
        V = V - (V - (W @ H)) # PyTorch has some memory antics I wanna avoid
        return V # Rank-L approximation of V

    
    def forward(self, ε):
        Vst = self.ξ_pre(ε) # Convolution
        n, τ, κ, α, β = Vst.shape
        assert n == 1 # FactorizePhys has a leading 1
        Vst_2D = Vst.view(τ, κ * α * β) # 4D -> 2D
        Vst_hat_2D = self.ϕ(Vst_2D, None) # NMF approximation
        Vst_hat = Vst_hat_2D.view(n, τ, κ, α, β) # 2D -> 4D
        error = torch.dist(Vst, Vst_hat)
        ε_hat = self.ξ_post(Vst_hat) # Convolution
        return ε_hat, error