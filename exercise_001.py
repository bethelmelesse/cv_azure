import torch
from torch import batch_norm, nn 
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib as plt
import matplotlib.pyplot as plt

print(" ")

device = 'cuda' if torch.cuda.is_available() else "cpu"
print("Using {} device \n".format(device))

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        input_size = 784    # 28 * 28
        hidden_size = [512, 512]
        output_size = 10

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def show_image(data):
    img, label = data
    plt.imshow(img.squeeze(), cmap="gray")          # squeeze() function is used when we want to remove single-dimensional entries from the shape of an array
    plt.axis("off")
    plt.title(label)
    plt.show()


def main():
    training_data_size = len(training_data)
    print(f"Size of training data: {training_data_size}")

    test_data_size = len(test_data)
    print(f"Size of test data: {test_data_size}\n") 

    show_image(training_data[0])
    show_image(test_data[0])

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print(f"Shape of y: {y.shape, y.dtype} \n")
        break
    
    model = NeuralNetwork().to(device)
    print(model)
    
main() 
print(" ")
