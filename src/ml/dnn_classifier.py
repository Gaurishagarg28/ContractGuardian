import torch
import torch.nn as nn


class ContractDNN(nn.Module):

    def __init__(
        self,
        input_size=768,
        num_classes=36
    ):

        super().__init__()

        self.network = nn.Sequential(

            # Layer 1
            nn.Linear(
                input_size,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            # Layer 2
            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),


            # Output layer
            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        return self.network(x)