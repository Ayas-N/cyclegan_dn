import threading, torch
_state = threading.local()

def set_anchors(xa: torch.Tensor, ya: torch.Tensor):
    _state.xa, _state.ya = xa, ya

def get_anchors(device, batch_size):
    xa = getattr(_state, "xa", None)
    ya = getattr(_state, "ya", None)
    if xa is None or ya is None:
        return (torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.zeros(batch_size, dtype=torch.long, device=device))
    return xa.to(device).long(), ya.to(device).long()
