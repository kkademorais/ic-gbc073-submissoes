import math
import torch


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.tanh(x)          # elemento a elemento, sem parâmetros


# E[f(z)^2] com z ~ N(0,1), estimado uma vez por amostragem (Monte Carlo)
_g = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    desvio = math.sqrt(1.0 / (fan_in * 0.5))
    if camada == n_camadas:
        gain = 0.1  # camada de logits: um pouco menor ajuda o SGD
    else:
        gain = torch.nn.init.calculate_gain("tanh")
    #gain = torch.nn.init.calculate_gain("leaky_relu", 0.01)
    torch.nn.init.orthogonal(W, gain=gain)
    #W.normal_(0.0, desvio)
    b.zero_()
