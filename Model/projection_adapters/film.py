from torch import nn

from .base import ProjectionAdapter


class FiLMAdapter(ProjectionAdapter):
    """P6 -- FiLM Adapter (FiLM-lite / self-conditioning variant).

    LayerNorm -> gamma, beta = Linear(pooled context), Linear(pooled context)
    x' = (1 + gamma) * x + beta, then projected to 768.

    Two evidence-backed corrections versus a naive "gamma,beta = Linear(x)"
    (see KANIT_projeksiyon_analizi/, P1_P3_P7_PROJECTION_HATA_ANALIZI.md
    -- this was the single most severe bug found in the P1-P7 set):

    1. **Identity-at-init.** FiLM's official code (Perez et al. 2017,
       arXiv:1709.07871; github.com/ethanjperez/film, film_gen.py) uses
       `gamma_option='linear', gamma_baseline=1` so gamma is centered on 1
       at initialization -- FiLM(x) = 1*x + 0 = x at t=0. A plain
       `nn.Linear` (the previous implementation) centers its output near
       0, so `gamma*x` was ~0*x at init: measured on the real mobileclip_s2
       checkpoint (adapter_evidence.json, adapter_specific.P6_film), only
       0.16% of gamma values were anywhere near 1, i.e. the adapter was
       discarding almost all of the visual signal and replacing it with a
       small random constant (beta) before any training happened -- a
       plausible sole cause of "caption independent of the image". Here
       both Linear layers are zero-initialized and the formula uses
       `(1 + gamma)`, so `max|out - Linear(LayerNorm(x))|` at t=0 is
       exactly 0.0 (verified in corrected_design_probe.json).
    2. **Conditioning source.** The original FiLM conditions on a signal
       *external* to the feature map being modulated (in the paper, a
       question embedding; here there is no such second input). Modulating
       every token from *itself* (the previous implementation) is not the
       literature's FiLM. The closest faithful adaptation without a second
       input is to condition each token on a *pooled, image-global*
       summary of the same features (mean over the token/patch dimension),
       which is what this implementation does; this is documented here
       explicitly as a "FiLM-lite" variant rather than silently presented
       as the original mechanism.

    Input:  (B, N, encoder_dim)
    Output: (B, N, 768)
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 768):
        super().__init__(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.to_gamma = nn.Linear(input_dim, input_dim)
        self.to_beta = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, image_features):
        x = self.norm(image_features)
        context = x.mean(dim=1, keepdim=True)
        gamma = self.to_gamma(context)
        beta = self.to_beta(context)
        modulated = (1 + gamma) * x + beta
        return self.out_proj(modulated)
