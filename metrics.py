# metrics.py
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import piq

def _to01(x: torch.Tensor) -> torch.Tensor:
    # expects [-1,1]; maps to [0,1]
    return (x + 1.0) / 2.0

class FIDSSIMPSNR:
    """Track FID(A->B, B->A) + cycle SSIM/PSNR."""
    def __init__(self, device="cuda", use_fid=True):
        self.device = device
        self.use_fid = use_fid
        if use_fid:
            # we'll feed uint8 [0,255], so normalize=False
            self.fid_A2B = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            self.fid_B2A = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        self.reset()

    def reset(self):
        self._nA = 0; self._ssimA = 0.0; self._psnrA = 0.0
        self._nB = 0; self._ssimB = 0.0; self._psnrB = 0.0
        if self.use_fid:
            self.fid_A2B.reset()
            self.fid_B2A.reset()

    @torch.no_grad()
    def update_cycle_A(self, real_A, rec_A):
        a = _to01(real_A).clamp(0, 1)
        b = _to01(rec_A).clamp(0, 1)
        n = a.size(0)
        self._ssimA += piq.ssim(a, b, data_range=1.0).mean().item() * n
        self._psnrA += piq.psnr(a, b, data_range=1.0).mean().item() * n
        self._nA += n

    @torch.no_grad()
    def update_cycle_B(self, real_B, rec_B):
        a = _to01(real_B).clamp(0, 1)
        b = _to01(rec_B).clamp(0, 1)
        n = a.size(0)
        self._ssimB += piq.ssim(a, b, data_range=1.0).mean().item() * n
        self._psnrB += piq.psnr(a, b, data_range=1.0).mean().item() * n
        self._nB += n

    @torch.no_grad()
    def update_fid_A2B(self, real_B, fake_B):
        # torchmetrics FID accepts uint8 [0,255]
        rb = (_to01(real_B).clamp(0, 1) * 255).to(torch.uint8)
        fb = (_to01(fake_B).clamp(0, 1) * 255).to(torch.uint8)
        self.fid_A2B.update(rb, real=True)
        self.fid_A2B.update(fb, real=False)

    @torch.no_grad()
    def update_fid_B2A(self, real_A, fake_A):
        ra = (_to01(real_A).clamp(0, 1) * 255).to(torch.uint8)
        fa = (_to01(fake_A).clamp(0, 1) * 255).to(torch.uint8)
        self.fid_B2A.update(ra, real=True)
        self.fid_B2A.update(fa, real=False)

    @torch.no_grad()
    def compute(self):
        out = {}
        if self._nA:
            out["ssim_cycle_A"] = self._ssimA / self._nA
            out["psnr_cycle_A"] = self._psnrA / self._nA
        if self._nB:
            out["ssim_cycle_B"] = self._ssimB / self._nB
            out["psnr_cycle_B"] = self._psnrB / self._nB
        if self.use_fid:
            out["fid_A2B"] = float(self.fid_A2B.compute().cpu())
            out["fid_B2A"] = float(self.fid_B2A.compute().cpu())
        return out
