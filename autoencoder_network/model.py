import torch.nn as nn
import torch


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.height = args.height
        self.width = args.width
        self.dropout = args.dropout
        self.class_num = args.class_num
        self.latent_dim = args.latent_dim
        self.args = args

        self.encoder = nn.Sequential(
            nn.Linear(self.height*self.width, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.latent_dim),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.height*self.width),
            nn.Sigmoid()
        )

    def forward(self, x, debug=False):

        latent_vector = self.encoder(x)
        output = self.decoder(latent_vector)
        output = output.view(self.args.batch_size, -1, self.height, self.width)

        return output, latent_vector

    def resume(self, file, test=False):
        if test and not file:
            self.encoder = nn.Sequential(
                nn.Linear(self.height*self.width, 128),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, self.latent_dim),
                nn.GELU()
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, self.height*self.width),
                nn.Sigmoid()
            )
            return

        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file, map_location='cuda:0')
            print(f"best epoch : {checkpoint['epoch']}")
            checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)
