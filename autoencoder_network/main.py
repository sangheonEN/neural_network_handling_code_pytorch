import dataset_load
import argparse
import torch
import train
from tensorboardX import SummaryWriter


# argparser generation
parser = argparse.ArgumentParser()
parser.add_argument("-max_epoch", default=50, help="sum of epoches")
parser.add_argument("-lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("-weight_decay", default=0.005,
                    type=float, help="Optim weight decay")
parser.add_argument("-momentum", default=0.99,
                    type=float, help="SGD Param")
parser.add_argument("--optim", type=str, default='adam',
                    choices=['adam', 'sgd'])
parser.add_argument("-batch_size", default=32, type=int, help="mini batch")
parser.add_argument("-dropout", default=0.3, type=float, help="dropout rate")
parser.add_argument("-latent_dim", default=5,
                    type=float, help="latent vector dim")
parser.add_argument("-class_num", default=10, type=int,
                    help="total class number")
parser.add_argument("-height", default=28, type=int, help="img_size")
parser.add_argument("-width", default=28, type=int, help="img_size")
parser.add_argument("-save_dir", default="./save_ckpt",
                    type=str, help="save_model_file")
parser.add_argument(
    "-mode_flag",
    default="train",
    type=str,
    help="select the mode: [train], [inference]",
)
args = parser.parse_args()


if __name__ == "__main__":

    # Shape: N, C. Type: Tensor
    train_data, valid_data, test_data = dataset_load.data_load()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensorboard summary
    summary = SummaryWriter()

    trainer = train.Trainer(args, device, summary,
                            train_data, valid_data, test_data)

    trainer.Train_main()
