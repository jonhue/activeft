from typing import Literal
import torch
from torch import nn
from afsl.embeddings import Embedding
from afsl.model import ClassificationModel
from afsl.utils import get_device, mini_batch_wrapper

ClassificationEmbeddingKind = (
    Literal["cross_entropy_loss"]
    | Literal["summed_cross_entropy_loss"]
    | Literal["norm"]
)
DEFAULT_CLASSIFICATION_EMBEDDING_KIND = "cross_entropy_loss"


class ClassificationEmbedding(Embedding[ClassificationModel]):
    """works only for classification models"""

    kind: ClassificationEmbeddingKind

    def __init__(
        self,
        mini_batch_size: int = 100,
        kind: ClassificationEmbeddingKind = DEFAULT_CLASSIFICATION_EMBEDDING_KIND,
    ):
        super().__init__(mini_batch_size=mini_batch_size)
        self.kind = kind

    def embed(self, model: ClassificationModel, data: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():

            def fn(data):
                data = data.to(get_device(model))
                latent = model.latent(data)  # (N, K)
                outputs = model(data)  # (N, C)
                pred = model.predict(data)  # (N,)
                # losses = nn.CrossEntropyLoss(reduction="none")(outputs, pred)  # (N,)

                K = latent.size(1)
                C = outputs.size(1)

                if self.kind == "cross_entropy_loss":
                    # compute gradient explicitly: eq. (1) of https://arxiv.org/pdf/1906.03671.pdf
                    pred_ = torch.nn.functional.one_hot(pred, K)  # (N, C)
                    A = (outputs - pred_)[:, :, None]  # (N, C, 1)
                    B = latent[:, None, :]  # (N, 1, K)
                    J = torch.matmul(A, B).view(
                        -1, A.shape[1] * B.shape[2]
                    )  # (N, C * K)
                elif self.kind == "summed_cross_entropy_loss":
                    A = (C * outputs - 1)[:, :, None]  # (N, C, 1)
                    B = latent[:, None, :]  # (N, 1, K)
                    J = torch.matmul(A, B).view(
                        -1, A.shape[1] * B.shape[2]
                    )  # (N, C * K)
                elif self.kind == "norm":
                    J = torch.zeros((data.size(0), C * K), dtype=torch.float32)
                    for i in range(data.size(0)):
                        W = model.final_layer.weight  # (C, K)
                        Z = latent[i, :][:, None] @ latent[i, :][None, :]  # (K, K)
                        J[i, :] = (W @ Z).view(-1, J.size(1))
                else:
                    raise NotImplementedError
                return J

            embeddings = mini_batch_wrapper(
                fn=fn,
                data=data,
                batch_size=self.mini_batch_size,
            )
            return embeddings