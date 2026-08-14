import torch
import torch.nn as nn


class OptimizedContractDNN(
    nn.Module
):

    def __init__(
        self,
        input_size=768,
        num_classes=36
    ):

        super().__init__()


        self.network = nn.Sequential(

            # -----------------------------------------------
            # Layer 1
            # -----------------------------------------------

            nn.Linear(
                input_size,
                512
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.35
            ),


            # -----------------------------------------------
            # Layer 2
            # -----------------------------------------------

            nn.Linear(
                512,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),


            # -----------------------------------------------
            # Layer 3
            # -----------------------------------------------

            nn.Linear(
                256,
                128
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.GELU(),

            nn.Dropout(
                0.25
            ),


            # -----------------------------------------------
            # Output
            # -----------------------------------------------

            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        return self.network(x)