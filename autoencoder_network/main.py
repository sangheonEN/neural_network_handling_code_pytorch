import dataset_load
import argparse
import torch
import train
import valid


# argparser generation
parser = argparse.ArgumentParser()
parser.add_argument("-max_epoch", default=20, help="sum of epoches")
parser.add_argument("-lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("-weight_decay", default=0.005,
                    type=float, help="Optim weight decay")
parser.add_argument("-momentum", default=0.99,
                    type=float, help="SGD Param")
parser.add_argument("--optim", type=str, default='adam',
                    choices=['adam', 'sgd'])
parser.add_argument("-batch_size", default=3, type=int, help="mini batch")
parser.add_argument("-dropout", default=0.3, type=float, help="dropout rate")
parser.add_argument("-latent_dim", default=5, type=float, help="dropout rate")
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
    x_train, y_train, x_valid, y_valid, x_test, y_test = dataset_load.data_load()

    # x shape: N, H, W. y shape: N, 1
    Custom = dataset_load.Custom_Image_Processing()
    x_train, y_train = Custom.transform(x_train, y_train)
    x_valid, y_valid = Custom.transform(x_valid, y_valid)
    x_test, y_test = Custom.transform(x_test, y_test)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = train.Trainer(args, device, x_train, y_train, x_valid, y_valid)

    trainer.Train_epo()
