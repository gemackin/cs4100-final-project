import torch
from torch import nn
from sklearn.decomposition import NMF


class FSAM(nn.Module):
    def __init__(self, ch_input, ch_align, L=1, return_val=None):
        super().__init__()
        self.return_val = return_val # Change this to change output
        kwargs = dict(dtype=torch.float64)

        # Preprocessing operation (τ×κ×α×β -> τ×κ×α×β)
        self.ξ_pre = nn.Sequential(
            nn.Conv3d(ch_input, ch_align, (1, 1, 1), **kwargs), 
            nn.ReLU(inplace=True)
        )

        # Non-negative matrix factorization (M×N -> M×L, L×N)
        self.NMF_model = NMF(n_components=L, init='random', random_state=0)

        # Postprocessing operation (τ×κ×α×β -> τ×κ×α×β)
        self.ξ_post = nn.Sequential(
            nn.Conv3d(ch_align, ch_align, (1, 1, 1), bias=False, **kwargs),
            nn.ReLU(inplace=True),
            # nn.InstanceNorm3d(ch_align), # The authors also don't implement this
            nn.Conv3d(ch_align, ch_input, 1, bias=False, **kwargs)
        )
    

    @torch.no_grad() # PyTorch doesn't need to concern itself here
    def ϕ(self, V):
        W = self.NMF_model.fit_transform(V)
        H = self.NMF_model.components_
        return W, H # Temporal vector, spatial vector

    
    def forward(self, ε):
        Vst = self.ξ_pre(ε) # Convolution
        n, τ, κ, α, β = ε.shape
        Vst_2D = Vst.view(n, τ, κ * α * β) # 4D -> 2D
        W, H = self.ϕ(Vst_2D) # NMF
        if self.return_val == 'W': return W # Not part of algorithm
        elif self.return_val == 'WH': return W, H # Also not part of it
        Vst_hat_2D = W @ H # Approximation
        Vst_hat = Vst_hat_2D.view(n, τ, κ, α, β) # 2D -> 4D
        error = torch.dist(Vst, Vst_hat)
        ε_hat = self.ξ_post(Vst_hat) # Convolution
        return ε_hat, error