import json
import torch
from transforms.augmentations import *
from datasets.cd_dataset import SequenceDataset
from torch.utils.data import DataLoader
from models.networks.Attention_unet import AttentionUnet
from torch import nn, optim
from tqdm import tqdm
import os
from pulse_method.toolbox_pulse import thermograms

operator=thermograms()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Importing data splits
with open('splits/split_50.json','r') as f:
    splits = json.load(f)

train_data=splits['train']
val_data=splits['val']

# Setting up the transforms for the data
transforms=[
    RandomPhaseAwareSpeedChange(),
    RandomBrightnessContrast(),
    PrependFirstFrame(),
    RandomFlip3D(axes=(1,2)),
    RandomElasticTransform(),
    RandomSequenceRotation(),
    RandomCropSequence(),
    AddGaussianNoise3D(),
    NormalizeTo01()
]     

train_transform=Compose3D(transforms)

val_transforms = Compose3D([RandomCropSequence(),
                            NormalizeTo01()])

# Preparation of the datasets
train_dataset=SequenceDataset(train_data,train_transform,operator)
val_dataset=SequenceDataset(val_data,val_transforms,operator)

# Preparation of dataloader
train_loader=DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=4)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=4)

# Defining models, loss and optimizer
model = AttentionUnet(n_classes=1, in_channels=1).to(device)
criterion=nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-4)

# Training loop setup
num_epochs = 500
history = {
    'train_loss': [],
    'val_loss': []
}

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, labels = images.to(device), labels.to(device)

        outputs, _, _, _ = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        

    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs, _, _, _ = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            

    avg_val_loss = val_loss / len(val_loader)
    

    # Logging
    print(f"\nEpoch {epoch + 1}/{num_epochs} ")
    print(f"Train Loss: {avg_train_loss:.4f} ")
    print(f"Val   Loss: {avg_val_loss:.4f}\n")

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)

    # Save model checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    if (epoch+1)%10==0:
        torch.save(model.state_dict(), f'checkpoints/model_unet_epoch_{epoch+1}.pth')

# Save training history

os.makedirs('logs', exist_ok=True)
with open('logs/training_history_attention_unet.json', 'w') as f:
    json.dump(history, f, indent=2)