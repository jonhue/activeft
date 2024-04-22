import torch
import wandb
import numpy as np
from afsl.acquisition_functions.bace import TargetedBaCE, BaCEState

import time


class ITLNoiselessOld(TargetedBaCE):
    r"""
    `ITL` [^3] (*information-based transductive learning*) composes the batch by sequentially selecting the samples with the largest information gain about the prediction targets $\spA$: \\[\begin{align}
        \vx_{i+1} &= \argmax_{\vx \in \spS}\ \I{\vf(\spA)}{y(\vx) \mid \spD_i}.
    \end{align}\\]
    Here, $\spS$ denotes the data set, $f$ is the stochastic process induced by the kernel $k$.[^1]
    We denote (noisy) observations of $\vx_{1:i}$ by $y_{1:i}$ and the first $i$ selected samples by $\spD_i = \\{(\vx_j, y_j)\\}_{j=1}^i$.

    `ITL` can equivalently be interpreted as minimizing the posterior entropy of the prediction targets $\spA$: \\[\begin{align}
        \vx_{i+1} &= \argmin_{\vx \in \spS}\ \H{\vf(\spA) \mid \spD_i, y(\vx)}.
    \end{align}\\]

    .. note::

        The special case where the prediction targets $\spA$ include $\spS$ ($\spS \subseteq \spA$, i.e., the prediction targets include "everything") is [Undirected ITL](undirected_itl).

    `ITL` selects batches via *conditional embeddings*,[^4] leading to diverse batches.

    | Relevance? | Informativeness? | Diversity? | Model Requirement  |
    |------------|------------------|------------|--------------------|
    | ✅          | ✅                | ✅          | embedding / kernel  |

    #### Comparison to VTL

    `ITL` can be expressed as \\[\begin{align}
        \vx_{i+1} &= \argmin_{\vx \in \spS}\ \det{\Var{\vf(\spA) \mid \spD_{i}, y(\vx)}}.
    \end{align}\\]
    That is, `ITL` minimizes the determinant of the posterior covariance matrix of $\vf(\spA)$ whereas [VTL](vtl) minimizes the trace of the posterior covariance matrix of $\vf(\spA)$.
    In practice, this difference amounts to a different "weighting" of the prediction targets in $\spA$.
    While `VTL` attributes equal importance to all prediction targets, [ITL](itl) attributes more importance to the "most uncertain" prediction targets.

    #### Computation

    `ITL` is computed using $\I{\vf(\spA)}{y(\vx) \mid \spD_i} \approx \I{\vy(\spA)}{y(\vx) \mid \spD_i}$ with \\[\begin{align}
        \I{\vy(\spA)}{y(\vx) \mid \spD_i} &= \frac{1}{2} \log\left( \frac{k_i(\vx,\vx) + \sigma^2}{\tilde{k}_i(\vx,\vx) + \sigma^2} \right) \qquad\text{where} \\\\
        \tilde{k}_i(\vx,\vx) &= k_i(\vx,\vx) - \vk_i(\vx,\spA) (\mK_i(\spA,\spA) + \sigma^2 \mI)^{-1} \vk_i(\spA,\vx)
    \end{align}\\] where $\sigma^2$ is the noise variance and $k_i$ denotes the conditional kernel (see afsl.acquisition_functions.bace.BaCE).

    [^1]: A kernel $k$ on domain $\spX$ induces a stochastic process $\\{f(\vx)\\}_{\vx \in \spX}$. See afsl.model.ModelWithKernel.

    [^3]: Hübotter, J., Sukhija, B., Treven, L., As, Y., and Krause, A. Information-based Transductive Active Learning. arXiv preprint, 2024.

    [^4]: see afsl.acquisition_functions.bace.BaCE
    """

    def compute(self, state: BaCEState) -> torch.Tensor:
        start_total = time.time()
        variances = torch.diag(state.covariance_matrix[: state.n, : state.n])

        start = time.time()
        conditional_variances = torch.empty_like(variances)
        unobserved_points = torch.tensor([i for i in torch.arange(state.n) if not ITLNoiselessOld.observed(i, state)], device=ITLNoiselessOld.get_device())
        observed_points = torch.tensor([i for i in torch.arange(state.n) if ITLNoiselessOld.observed(i, state)], device=ITLNoiselessOld.get_device())
        end = time.time()
        print("prefix: " + str(end - start))

        start = time.time()
        for i in unobserved_points:
            conditional_covariance_matrix = state.covariance_matrix.condition_on(
                indices=ITLNoiselessOld.adapted_target_space(state, i),
                target_indices=torch.reshape(i, [1]),
            )[:, :]
            conditional_variances[i] = torch.diag(conditional_covariance_matrix)
        end = time.time()
        print("loop: " + str(end - start))

        
        start = time.time()
        mi = 0.5 * torch.clamp(torch.log(variances / conditional_variances), min=0)
        if observed_points.size(dim = 0) > 0:
            mi.index_fill_(0, observed_points, -float('inf'))
        end = time.time()
        print("mi: " + str(end - start))

        wandb.log(
            {
                "max_mi": torch.max(mi),
                "min_mi": torch.min(mi),
            }
        )

        end_total = time.time()
        print("Total: " + str(end_total - start_total))
        return mi

    @staticmethod
    def adapted_target_space(state: BaCEState, idx) -> torch.Tensor:
        return torch.tensor([i for i in torch.arange(start=state.n, end=state.covariance_matrix.dim) if not ITLNoiselessOld.observed(i, state) and not i == idx], device=ITLNoiselessOld.get_device())

    @staticmethod
    def observed(idx, state: BaCEState):
        return any(ITLNoiselessOld.isClose(state.joint_data[idx], x) for x in state.observed_points)

    @staticmethod
    def isClose(x, y, rel_tol=1e-09, abs_tol=0.0):
        """Checks if two float vectors are almost equal
        Parameters
        ----------
        x : vector, value 1 to check
        y : vector, value 2 to check
        rel_tol : float, optional
            standard value is 0.000001
        abs_tol : float, optional
            standard value is 0.000001
        Returns
        ------
        If the vector x is close to the vector y
        """

        return np.linalg.norm(x - y) <= max(rel_tol * max(np.linalg.norm(x), np.linalg.norm(y)), abs_tol)
    
    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    
    #12.15 180
    #3.26  45
    #0.027 15