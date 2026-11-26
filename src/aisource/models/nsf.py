"""Construcción del self.flow normalizador y evaluación de la verosimilitud.

`construir` traduce un diccionario de hiperparámetros a un self.flow de zuko, de
modo que la arquitectura sea un dato de la corrida y no esté escrita en el
código. Arquitecturas disponibles:

    NSF      splines racionales cuadráticas, para el bloque no circular
    NCSF     idem pero circular, para phi (base uniforme en [-pi, pi])
    MAF      autorregresivo afín
    RealNVP  acoplamiento afín (NICE con máscaras aleatorias)

El resto del módulo evalúa log p. La bandera `circular` decide la firma: el
modelo circular es condicional, p(phi | z), y se evalúa como `self.flow(z)`; el
otro es incondicional, p(z), y se evalúa como `self.flow()`. `nll` devuelve la
verosimilitud negativa media ponderada por los pesos del MCPL en el espacio
estandarizado del self.flow (sin offset del cambio de variable).
"""

import math
import time

import numpy as np
import torch
import zuko
from torch.utils.data import DataLoader
from zuko.lazy import Flow, UnconditionalTransform
from zuko.transforms import AffineTransform, SigmoidTransform

from .base import BaseGenerator, Particle


class NSFlow(BaseGenerator):
    """NFs"""

    def __init__(
        self, particle: Particle, device: str, seed: int | None, bounds=None
    ) -> None:
        super().__init__(particle, device, seed)
        self.epochs = None
        self.batch_size = None
        self.lr = None
        self.bounds = [0.0] * (self.ndim - 1), [1.0] * (self.ndim - 1)
        self.flow_z = self.construir(self.ndim - 1, bounds=self.bounds).to(self.device)
        self.flow_phi = self.construir(1, context=self.ndim - 1, bounds=self.bounds).to(
            self.device
        )

    def prep(self, nsf, bounds):
        """Antepone min-max -> logit al NSF para confinar su salida a (a, b) por dimensión.

        La base y el spline viven en R; la logit mapea (a,b)->(0,1)->R al entrenar,
        y la sigmoide confina de vuelta a (a,b) al samplear, así ninguna muestra
        sale del rango. zuko lleva el jacobiano de todo esto en el sel.flog_prob.
        """
        a = torch.as_tensor(bounds[0], dtype=torch.float32)
        b = torch.as_tensor(bounds[1], dtype=torch.float32)
        # (a,b) -> (0,1): s = loc + scale*x
        scale, loc = 1.0 / (b - a), -a / (b - a)
        transforms = [
            UnconditionalTransform(
                AffineTransform, loc, scale, buffer=True
            ),  # (a,b) -> (0,1)
            # logit (0,1) -> R
            UnconditionalTransform(lambda: SigmoidTransform().inv),
            # el NSF en R
            *list(nsf.transform.transforms),
        ]
        return Flow(transform=transforms, base=nsf.base)

    def construir(
        self,
        dim: int,
        arch: str = "NSF",
        context=0,
        layers=3,
        hidden_width=64,
        hidden_depth=2,
        bounds=None,
        bins=6,
    ):
        comun = {
            "context": context,
            "transforms": layers,
            "hidden_features": [hidden_width] * hidden_depth,
            "activation": torch.nn.ReLU,
        }
        if arch == "NSF":
            nsf = zuko.flows.NSF(dim, bins=bins, **comun)
            return nsf if bounds is None else self.prep(nsf, bounds)
        if arch == "NCSF":
            return zuko.flows.NCSF(dim, bins=bins, **comun)
        if arch == "MAF":
            return zuko.flows.MAF(dim, **comun)
        if arch == "RealNVP":
            return zuko.flows.coupling.NICE(dim, randmask=True, **comun)
        raise ValueError(f"err not found: {arch}")

    def n_parametros(
        self,
    ):
        """Cantidad de parámetros entrenables del self.flow."""
        return sum(p.numel() for p in self.flow.parameters() if p.requires_grad)

    def flog_prob(self, flow, z, phi, circular):
        """log p de un batch. El modelo circular es p(phi | z), el otro es p(z)."""
        if circular:
            return flow(z).log_prob(phi)
        return flow().log_prob(z)

    @torch.no_grad()
    def nll(self, flow, particulas, circular):
        """NLL media de una partición, ponderada por los pesos (espacio estandarizado).

        Suma -log p pesado por w y divide por la suma de pesos. Mueve los tensores al
        device del self.flow, así corre en GPU si el self.flow está en GPU.
        """
        flow.eval()
        device = next(flow.parameters()).device
        z = particulas.z.to(device)
        phi = particulas.phi.to(device)
        w = particulas.w.to(device)
        nll_p = float((-self.flog_prob(flow, z, phi, circular) * w).sum())
        return nll_p / float(w.sum())

    def _fit(
        self,
        flow,
        train: np.ndarray,
        validation: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        wgt_dc: float = 0.0,
        seed: int = 0,
        circular: bool = False,
        verbose: bool = True,
        patience: int = 20,
        lr: float = 1e-3,
    ) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        # train = self._check_input_array(train)
        # train = self._to_tensor(train)
        # validation = self._check_input_array(validation)
        # validation = self._to_tensor(validation)
        self.wgt_dc = wgt_dc
        self.seed = seed
        self.circ = circular

        self.device = next(flow.parameters()).device
        train_loader = DataLoader(
            train,
            batch_size=int(batch_size),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=wgt_dc)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best, best_epoch, best_state = math.inf, -1, None
        no_improvement = 0
        history = []
        t0 = time.time()
        for epoca in range(epochs):
            flow.train()
            sum_loss, sum_w = 0.0, 0.0
            for zb, phib, wb in train_loader:
                zb, phib, wb = (
                    zb.to(self.device),
                    phib.to(self.device),
                    wb.to(self.device),
                )
                loss = -(self.flog_prob(flow, zb, phib, circular) * wb).sum() / wb.sum()
                loss.backward()
                opt.step()
                opt.zero_grad()

                sum_loss += float(loss.detach()) * float(wb.sum())
                sum_w += float(wb.sum())
            train_nll = sum_loss / sum_w
            val_nll = self.nll(flow, validation, circular)
            lr = self.lr
            history.append(
                {
                    "epoca": epoca,
                    "train_nll": train_nll,
                    "val_nll": val_nll,
                    "lr": lr,
                    "time": time.time() - t0,
                }
            )
            if verbose:
                print(
                    f"epoch {epoca:3d}  train {train_nll:+.4f}  "
                    f"validation {val_nll:+.4f}  lr {lr:.2e}",
                    flush=True,
                )
            sched.step()
            if val_nll < best - 1e-5:
                best, best_epoch = val_nll, epoca
                best_state = {
                    k: v.detach().cpu().clone() for k, v in flow.state_dict().items()
                }
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(
                        f"early stopping in epoch {epoca} "
                        f"(best: {best_epoch}, validation {best:+.4f})"
                    )
                    break

        print(
            f"\nbest validation NLL = {best:+.6f} (epoch {best_epoch})"
            f"in {time.time() - t0:.0f} s"
        )
        return best_state, history

    def fit(
        self,
        train: np.ndarray,
        validation: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        wgt_dc: float = 0.0,
        seed: int = 0,
        verbose: bool = True,
        patience: int = 20,
        lr: float = 1e-3,
    ) -> None:
        _ = self._fit(
            self.flow_z,
            train,
            validation,
            epochs,
            batch_size,
            wgt_dc,
            seed,
            False,
            verbose,
            patience,
            lr,
        )
        _p = self._fit(
            self.flow_phi,
            train,
            validation,
            epochs,
            batch_size,
            wgt_dc,
            seed,
            True,
            verbose,
            patience,
            lr,
        )

    @torch.no_grad()
    def sample(self, n: int) -> np.ndarray:
        z = self.flow_z().sample((n,))
        p = torch.cat([z, self.flujo_phi(z).sample()], dim=1)
        return p
