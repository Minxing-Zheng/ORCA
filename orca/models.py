"""Stage 1 -- generative model of ``p(Y | X)``.

ORCA only needs a model it can *draw samples from*.  The default one shipped
here is :class:`QuantileNet`: a multi-quantile neural network trained with the
pinball loss, from which we sample by inverting the estimated conditional
quantile function.  :class:`QuantileForest` offers the same interface backed by
a quantile regression forest.

Any object exposing ``fit(X, y)`` and ``sample(X, n_sample, rng)`` can be used
instead -- see ``orca.sampling.generate_samples``.

The network architecture and pinball loss follow the conditional-quantile
network of Sesia & Romano, *Conformal Histogram Regression* (NeurIPS 2021),
https://github.com/msesia/chr -- see NOTICE.md.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

__all__ = ["QuantileNet", "QuantileForest", "default_quantile_grid"]


def default_quantile_grid(step=0.01):
    """Quantile levels ``step, 2*step, ..., 1 - step``."""
    return np.arange(step, 1.0, step)


class _MLP(nn.Module):
    """Feed-forward net emitting one output per quantile level."""

    def __init__(self, num_features, num_quantiles, num_hidden=64, dropout=0.1, no_crossing=True):
        super().__init__()
        self.no_crossing = no_crossing
        self.net = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_quantiles),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.net(x)
        if self.no_crossing:
            # Sorting the outputs rules out quantile crossings.
            out, _ = torch.sort(out, dim=1)
        return out


class _PinballLoss(nn.Module):
    """Average pinball (check) loss across all quantile levels."""

    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer("quantiles", quantiles)

    def forward(self, preds, target):
        errors = target.unsqueeze(1) - preds
        q = self.quantiles.unsqueeze(0)
        return torch.max((q - 1.0) * errors, q * errors).mean()


class QuantileNet:
    """Conditional quantile network usable as a generative sampler.

    Parameters
    ----------
    num_features : int
        Input dimension.
    quantiles : array-like or None
        Quantile levels to fit.  Defaults to ``0.01, ..., 0.99``.
    num_hidden : int
        Width of the two hidden layers.
    dropout : float
        Dropout probability.
    learning_rate, num_epochs, batch_size : float, int, int
        Adam optimiser settings.  ``batch_size >= n_train`` trains full batch.
    no_crossing : bool
        Sort the predicted quantiles so they are monotone in the level.
    random_state : int
        Seed for torch initialisation and batching.
    device : str
        ``"cpu"`` or ``"cuda"``.
    verbose : bool
        Print the training loss every ``log_every`` epochs.
    """

    def __init__(
        self,
        num_features,
        quantiles=None,
        num_hidden=64,
        dropout=0.1,
        learning_rate=5e-3,
        num_epochs=500,
        batch_size=512,
        no_crossing=True,
        random_state=0,
        device="cpu",
        verbose=False,
        log_every=50,
    ):
        self.device = torch.device(device)
        self.random_state = int(random_state)
        torch.manual_seed(self.random_state)

        q = default_quantile_grid() if quantiles is None else np.asarray(quantiles, dtype=float)
        self.quantiles_np = np.sort(q)
        quantiles_t = torch.from_numpy(self.quantiles_np).float().to(self.device)

        self.num_features = int(num_features)
        self.model = _MLP(
            self.num_features,
            len(self.quantiles_np),
            num_hidden=num_hidden,
            dropout=dropout,
            no_crossing=no_crossing,
        ).to(self.device)
        self.loss_func = _PinballLoss(quantiles_t).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.verbose = bool(verbose)
        self.log_every = int(log_every)
        self.loss_history = []

    # -- training ----------------------------------------------------------- #
    def fit(self, X, y):
        """Fit on ``X`` of shape ``(n, d)`` and scalar targets ``y``."""
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was built for {self.num_features}"
            )

        X_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)
        n = X_t.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)

        self.model.train()
        for epoch in range(1, self.num_epochs + 1):
            if self.batch_size >= n:
                batches = [torch.arange(n)]
            else:
                perm = torch.randperm(n, generator=generator)
                batches = perm.split(self.batch_size)

            epoch_loss = 0.0
            for idx in batches:
                idx = idx.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_func(self.model(X_t[idx]), y_t[idx])
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(batches)
            self.loss_history.append(epoch_loss)
            if self.verbose and epoch % self.log_every == 0:
                print(f"  epoch {epoch:4d}/{self.num_epochs} | pinball loss {epoch_loss:.5f}")
        return self

    # -- inference ---------------------------------------------------------- #
    def predict(self, X):
        """Predicted conditional quantiles, shape ``(n, n_quantiles)``."""
        X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(X).to(self.device))
        return out.cpu().numpy()

    def get_quantiles(self):
        return self.quantiles_np

    def _invert_at(self, preds, u):
        """Linearly interpolate the quantile curve ``preds`` at level ``u``."""
        q = self.quantiles_np
        if u <= q[0]:
            return preds[:, 0]
        if u >= q[-1]:
            return preds[:, -1]
        hi = int(np.searchsorted(q, u))
        lo = hi - 1
        w = (u - q[lo]) / (q[hi] - q[lo])
        return (1.0 - w) * preds[:, lo] + w * preds[:, hi]

    def sample(self, X, n_sample, rng=None, shared_quantile_draw=True):
        """Draw ``n_sample`` samples of ``Y`` for every row of ``X``.

        Parameters
        ----------
        shared_quantile_draw : bool
            If ``True`` (the paper's setting) draw one level ``u_k ~ U(0,1)``
            per sample index and evaluate every input at that same level.  Each
            row still receives ``n_sample`` i.i.d. draws from its own conditional
            law; only the coupling *across* rows differs.  If ``False`` every
            (row, sample) pair gets its own independent level.

        Returns
        -------
        np.ndarray of shape ``(n, n_sample, 1)``
        """
        rng = np.random.default_rng() if rng is None else rng
        preds = self.predict(X)  # (n, n_quantiles)
        n = preds.shape[0]
        out = np.empty((n, n_sample), dtype=float)

        if shared_quantile_draw:
            for k in range(n_sample):
                out[:, k] = self._invert_at(preds, float(rng.random()))
        else:
            q = self.quantiles_np
            u = rng.random((n, n_sample))
            hi = np.clip(np.searchsorted(q, u), 1, len(q) - 1)
            lo = hi - 1
            w = (u - q[lo]) / (q[hi] - q[lo])
            rows = np.arange(n)[:, None]
            out = (1.0 - w) * preds[rows, lo] + w * preds[rows, hi]
            out = np.where(u <= q[0], preds[:, [0]], out)
            out = np.where(u >= q[-1], preds[:, [-1]], out)

        return out[:, :, None]


class QuantileForest:
    """Quantile regression forest with the same sampling interface.

    Requires the optional ``quantile-forest`` package
    (``pip install quantile-forest``).
    """

    def __init__(
        self,
        quantiles=None,
        n_estimators=120,
        min_samples_leaf=40,
        n_jobs=1,
        random_state=0,
    ):
        try:
            from quantile_forest import RandomForestQuantileRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "QuantileForest needs the 'quantile-forest' package: pip install quantile-forest"
            ) from exc

        q = default_quantile_grid() if quantiles is None else np.asarray(quantiles, dtype=float)
        self.quantiles_np = np.sort(q)
        self.model = RandomForestQuantileRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(np.asarray(X).reshape(len(X), -1), np.asarray(y).reshape(-1))
        return self

    def predict(self, X):
        X = np.asarray(X).reshape(len(X), -1)
        return np.asarray(self.model.predict(X, quantiles=list(self.quantiles_np)))

    def get_quantiles(self):
        return self.quantiles_np

    def sample(self, X, n_sample, rng=None, shared_quantile_draw=True):
        rng = np.random.default_rng() if rng is None else rng
        X = np.asarray(X).reshape(len(X), -1)
        n = X.shape[0]
        out = np.empty((n, n_sample), dtype=float)
        if shared_quantile_draw:
            for k in range(n_sample):
                out[:, k] = np.asarray(self.model.predict(X, quantiles=float(rng.random()))).ravel()
        else:
            for k in range(n_sample):
                levels = rng.random(n)
                for i in range(n):
                    out[i, k] = float(
                        np.asarray(self.model.predict(X[i : i + 1], quantiles=float(levels[i]))).ravel()[0]
                    )
        return out[:, :, None]
