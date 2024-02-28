import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import afsl
from afsl.acquisition_functions import AcquisitionFunction
from afsl.active_data_loader import ActiveDataLoader
from afsl.utils import get_device
from examples.cifar.data import CollectedData, Dataset
from examples.utils import accuracy


def train(
    model: torch.nn.Module,
    trainloader: DataLoader,
    valset: CollectedData,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs=5,
):
    device = get_device(model)
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        num_batches = 0
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        acc = accuracy(model, valset.inputs, valset.labels)
        wandb.log(
            {"epoch": epoch + 1, "loss": running_loss / num_batches, "accuracy": acc}
        )


def train_loop(
    model: torch.nn.Module,
    labels: torch.Tensor,
    train_inputs: afsl.data.Dataset,
    train_labels: torch.Tensor,
    valset: CollectedData,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    acquisition_function: AcquisitionFunction,
    num_rounds=1_000,
    num_epochs=5,
    query_batch_size=10,
    train_batch_size=64,
    update_target=False,
    reweighting=True,
    reset_parameters=False,
):
    data = Dataset(root="./data")
    wandb.log({"round": 0, "round_accuracy": 0.0})

    data_loader = ActiveDataLoader(
        dataset=train_inputs,
        batch_size=query_batch_size,
        acquisition_function=acquisition_function,
    )

    for i in range(num_rounds):
        batch_indices = data_loader.next(model)
        batch_labels = train_labels[batch_indices]
        batch_mask = (batch_labels[:, None] == labels).any(dim=1)
        batch_inputs = [train_inputs[i] for i in batch_indices[batch_mask]]

        if len(batch_inputs) > 0:
            batch = CollectedData(
                inputs=torch.stack(batch_inputs),
                labels=batch_labels[batch_mask],
            )
            data.add_data(batch.inputs, batch.labels)
            trainloader = DataLoader(data, batch_size=train_batch_size, shuffle=True)

            if update_target and isinstance(
                acquisition_function, afsl.acquisition_functions.Targeted
            ):
                acquisition_function.add_to_target(batch.inputs)

            print("data labels:", torch.unique(torch.tensor(data.targets)))
            if reweighting:
                criterion.weight = (
                    len(data)
                    / torch.bincount(
                        torch.tensor(data.targets), minlength=labels.size(0)
                    )
                ).to(get_device(model))

            if reset_parameters:
                model.reset()
            train(
                model=model,
                trainloader=trainloader,
                valset=valset,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
            )

        acc = accuracy(model, valset.inputs, valset.labels)
        wandb.log(
            {
                "round": i,
                "round_accuracy": acc,
                "data_len": len(data),
                "missing_perc": data.valid_perc(torch.arange(5)),
            }
        )
    wandb.log(
        {
            "label_histogram": wandb.Histogram(
                np_histogram=np.histogram(
                    torch.tensor(data.targets).cpu().numpy(), bins=np.arange(11)
                )
            )
        }
    )