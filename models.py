"""Models."""

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models

TARGET_DISTANCES = [
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            50,
            60,
            70,
            80,
            95,
            110,
            125,
            145,
            165,
        ]

class Net(pl.LightningModule):
    """Neural CNN model class."""

    def __init__(self) -> None:
        super(Net, self).__init__()

        self.model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

        self.change_head_net()

        self.loss_fn = nn.L1Loss()

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.dists = torch.Tensor(TARGET_DISTANCES).to(device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        r = self.model(image).reshape((-1, 17, 3))
        r[:] *= self.dists[:, None]
        return r.reshape((-1, 3 * 17))
    

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images)
        val_loss = self.loss_fn(preds, targets)
        self.log('val_loss', val_loss)
        return val_loss

    def model_parameters(self) -> torch.tensor:
        """Get model parameters."""
        return self.model.parameters()

    def change_head_net(self) -> None:
        """Change the last model classifier step."""
        num_ftrs = self.model.classifier[-1].in_features

        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 51, bias=True),
            # nn.Conv1d(17, 17, 1, bias=True),
            # nn.Conv2d(17, 17, 1, bias=True),
            # nn.Conv3d(17, 17, 1, bias=True),
        )

        self.model.classifier[-1] = head_net


    def compute_metrics(
        self, pred_trajectory: torch.Tensor, target_trajectory: torch.Tensor
    ) -> dict:
        """Compute metric scores.

        Args:
            pred_trajectory (torch.Tensor): predicted path
            target_trajectory (torch.Tensor): true path

        Returns:
            dict: metric scores
        """
        # L1 and L2 distance: matrix of size BSx40x3
        L1_loss = torch.abs(pred_trajectory - target_trajectory)  # noqa: N806
        L2_loss = torch.pow(pred_trajectory - target_trajectory, 2)  # noqa: N806

        # BSx40x3 -> BSx3 average over the predicted points
        L1_loss = L1_loss.mean(axis=1)  # noqa: N806
        L2_loss = L2_loss.mean(axis=1)  # noqa: N806

        # split into losses for each axis and an avg loss across 3 axes
        # All returned tensors have shape (BS)
        return {
            "L1_loss": L1_loss.mean(axis=1),
            "L1_loss_x": L1_loss[:, 0],
            "L1_loss_y": L1_loss[:, 1],
            "L1_loss_z": L1_loss[:, 2],
            "L2_loss": L2_loss.mean(axis=1),
            "L2_loss_x": L2_loss[:, 0],
            "L2_loss_y": L2_loss[:, 1],
            "L2_loss_z": L2_loss[:, 2],
        }
    
    