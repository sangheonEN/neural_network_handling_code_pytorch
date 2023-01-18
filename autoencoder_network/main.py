import dataset_load
import argparse
import torch
import train
import valid

# argparser generation
parser = argparse.ArgumentParser()
parser.add_argument("-max_epoch", default=20, help="sum of epoches")
parser.add_argument("-lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("-batch_size", default=3, type=int, help="mini batch")
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

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train.trainer()
