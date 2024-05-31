from easyocr import Reader

input_file = "training/IMG_4181/outputIMG_4181.txt"

data = []
with open(input_file, "r") as file:
    data = file.readlines()

# Entferne Newline-Zeichen am Ende jeder Zeile (optional)
data = [line.strip() for line in data]
train_data = []
validation_data = []
test_data = []
# Ausgabe der Liste

listlen = len(data)
train_data_ori = data[:listlen // 3]
validation_data_ori = data[listlen // 3:listlen // 3 * 2]
test_data_ori = data[listlen // 3 * 2:]



print("listenlängen",len(train_data),len(validation_data), len(test_data))
print(train_data_ori)
print(validation_data_ori)
print(test_data_ori)

train_data = []
validation_data = []
test_data = []


for element in train_data_ori:
    print(element)
#    trimmed_string = element.replace('"',"")
    train_data.append(element)

print(train_data[2])

print(train_data)
#print(validation_data[2])
#print(test_data[2])


#listlen=len(data)
#counter=0
#print(listlen)
#for element in data:
 #   if counter < listlen/3:
  #      train_data.append[element]
  #     counter+=1

  #  if counter > listlen/3 and counter < listlen/3*2:
   #     validation_data.append[element]
    #    counter+=1

  #  else:
   #     test_data.append[element]    


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image

# Einfaches CRNN-Modell
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),  # 32x128 -> 32x128
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x128 -> 16x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 16x64 -> 16x64
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x64 -> 8x32
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 8x32 -> 8x32
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1)),  # 8x32 -> 4x31
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 4x31 -> 4x31
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1)),  # 4x31 -> 2x30
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 2x30 -> 2x30
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1)),  # 2x30 -> 1x29
        )
        self.rnn = nn.LSTM(512 * 29, nh, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        for layer in self.cnn:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                print(f"Conv2d output shape: {x.shape}")
            elif isinstance(layer, nn.MaxPool2d):
                print(f"MaxPool2d output shape: {x.shape}")

        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)  # LSTM gibt (output, (h_n, c_n)) zurück, wir brauchen nur output
        x = self.linear(x)
        return x

# Dataset-Klasse
class OCRDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open(img_name).convert('L')
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Trainingseinstellungen
def train(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Hauptskript
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 128)),  # Breite und Höhe anpassen
        transforms.ToTensor(),
    ])

    dataset = OCRDataset(csv_file='training\IMG_4181\outputIMG_4181.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crnn = CRNN(imgH=32, nc=1, nclass=37, nh=256).to(device)  # Beispielwerte
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(crnn.parameters(), lr=0.001)

    train(crnn, dataloader, criterion, optimizer, num_epochs=10)
