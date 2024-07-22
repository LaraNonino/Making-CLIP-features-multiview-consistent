import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, ToTensor, Compose
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
from torchvision import transforms
import shutil
import random
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import plotly.express as px
import plotly.io as pio
import pandas as pd


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.5):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim, eps=1e-6)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class BaseDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.image_paths, self.labels = self._load_images_and_labels()

    def _load_images_and_labels(self):
        image_paths = []
        labels = []

        for scene in os.listdir(self.images_folder):
            scene_path = os.path.join(self.images_folder, scene)
            if os.path.isdir(scene_path):
                for obj in os.listdir(scene_path):
                    obj_path = os.path.join(scene_path, obj)
                    if os.path.isdir(obj_path):
                        label = f"{scene}_{obj}"
                        for img_file in os.listdir(obj_path):
                            if img_file.endswith('.jpg'):
                                image_paths.append(os.path.join(obj_path, img_file))
                                labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class TripletDataset(BaseDataset):
    def __init__(self, images_folder, transform=None):
        super(TripletDataset, self).__init__(images_folder, transform)
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

    def __getitem__(self, idx):
        anchor_img_path = self.image_paths[idx]
        anchor_label = self.labels[idx]
        anchor_img = Image.open(anchor_img_path).convert('RGB')
        
        if self.transform:
            anchor_img = self.transform(anchor_img)

        positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img_path = self.image_paths[positive_idx]
        positive_img = Image.open(positive_img_path).convert('RGB')
        if self.transform:
            positive_img = self.transform(positive_img)

        negative_label = random.choice([label for label in self.label_to_indices.keys() if label != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img_path = self.image_paths[negative_idx]
        negative_img = Image.open(negative_img_path).convert('RGB')
        if self.transform:
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img, anchor_label, negative_label

    def __len__(self):
        return len(self.image_paths) 

class TripletEuclidianLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletEuclidianLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)      

class TripletCosineLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletCosineLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)   

def load_checkpoint(projection_head, optimizer, scheduler, weights_dir):
    checkpoints = [os.path.join(weights_dir, f) for f in os.listdir(weights_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0

    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint)
    projection_head.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return epoch

def save_checkpoint(projection_head, optimizer, scheduler, epoch, loss, weights_dir):
    checkpoint = {
        'model_state_dict': projection_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    weights_path = os.path.join(weights_dir, f"epoch_{epoch}.pth")
    torch.save(checkpoint, weights_path)

# Plot the embeddings before the training
def plot_initial_embeddings(model, processor, projection_head, dataloader, device, train_output_dir, label_to_idx, idx_to_label):
    projection_head.eval()
    initial_embeddings = []
    initial_labels = []

    with torch.no_grad():
        for anchor_imgs, _, _, anchor_labels, _ in dataloader:
            anchor_imgs = (anchor_imgs * 255).byte()
            anchor_imgs = anchor_imgs.to(device)
            anchor_inputs = processor(images=anchor_imgs, return_tensors="pt").to(device)
            anchor_outputs = model(**anchor_inputs)
            anchor_embeddings = projection_head(anchor_outputs.pooler_output)
            
            initial_embeddings.append(anchor_embeddings)

            for label in anchor_labels:
                initial_labels.append(label_to_idx[label])

    initial_embeddings = torch.cat(initial_embeddings)
    initial_labels = torch.tensor(initial_labels)

def plot_embeddings(embeddings, labels, epoch, output_dir, idx_to_label):
    pca2 = PCA(n_components=2)
    pca2_result = pca2.fit_transform(embeddings.cpu().detach().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(pca2_result[:, 0], pca2_result[:, 1], c=labels.cpu().numpy(), cmap='viridis', alpha=0.6)
    plt.title(f"PCA of Embeddings at Epoch {epoch}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)

    pca2_dir = os.path.join(output_dir, "PCA_2")
    os.makedirs(pca2_dir, exist_ok=True)
    plt.savefig(os.path.join(pca2_dir, f"epoch_{epoch}.png"))
    plt.close()

    pca3 = PCA(n_components=3)
    pca3_result = pca3.fit_transform(embeddings.cpu().detach().numpy())
    color_sequence = px.colors.qualitative.Set1
    label_names = [idx_to_label[idx] for idx in labels.cpu().numpy()]
    labels_df = pd.DataFrame({'label': label_names})

    num_labels = len(np.unique(labels_df['label']))
    color_sequence = px.colors.sample_colorscale('Viridis', [n / num_labels for n in range(num_labels)])

    # Create the 3D scatter plot
    fig = px.scatter_3d(
        x=pca3_result[:, 0],
        y=pca3_result[:, 1],
        z=pca3_result[:, 2],
        color=labels_df['label'], 
        title=f"PCA of Embeddings at Epoch {epoch}",
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "z": "PCA Component 3"},
        color_discrete_sequence=color_sequence
    )
    fig.update_layout(legend_title_text='Labels')

    pca3_dir = os.path.join(output_dir, "PCA_3")
    os.makedirs(pca3_dir, exist_ok=True)
    pio.write_html(fig, file=os.path.join(pca3_dir, f"epoch_{epoch}_3d.html"))
    fig.write_image(os.path.join(pca3_dir, f"epoch_{epoch}_3d.png"))

def validate(model, processor, projection_head, dataloader, criterion, device, label_to_idx):
    projection_head.eval()
    total_loss = 0
    val_embeddings = []
    val_labels = []
    
    with torch.no_grad():
        for anchor_imgs, positive_imgs, negative_imgs, anchor_labels, _ in dataloader:

            anchor_imgs = (anchor_imgs * 255).byte()
            positive_imgs = (positive_imgs * 255).byte()
            negative_imgs = (negative_imgs * 255).byte()

            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)
            
            anchor_inputs = processor(images=anchor_imgs, return_tensors="pt").to(device)
            positive_inputs = processor(images=positive_imgs, return_tensors="pt").to(device)
            negative_inputs = processor(images=negative_imgs, return_tensors="pt").to(device)
            
            anchor_outputs = model(**anchor_inputs)
            positive_outputs = model(**positive_inputs)
            negative_outputs = model(**negative_inputs)
            
            anchor_embeddings = projection_head(anchor_outputs.pooler_output)
            positive_embeddings = projection_head(positive_outputs.pooler_output)
            negative_embeddings = projection_head(negative_outputs.pooler_output)
            
            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss.item()

            val_embeddings.append(anchor_embeddings)
            for label in anchor_labels:
                val_labels.append(label_to_idx[label])

    avg_loss = total_loss / len(dataloader)
    val_embeddings = torch.cat(val_embeddings)
    val_labels = torch.tensor(val_labels)
    
    return avg_loss, val_embeddings, val_labels

def train(model, processor, projection_head, train_loader, val_loader, optimizer, scheduler, criterion, label_to_idx, idx_to_label, device, plots_dir, weights_dir, num_epochs=10):

    training_plot_dir = os.path.join(plots_dir, "train")
    val_plot_dir = os.path.join(plots_dir, "val")
    os.makedirs(training_plot_dir, exist_ok=True)
    os.makedirs(val_plot_dir, exist_ok=True)

    start_epoch = load_checkpoint(projection_head, optimizer, scheduler, weights_dir)

    if start_epoch == 0:
        plot_initial_embeddings(model, processor, projection_head, train_loader, device, training_plot_dir, label_to_idx, idx_to_label)

    for epoch in range(start_epoch, num_epochs):
        projection_head.train()
        total_loss = 0
        epoch_embeddings = []
        epoch_labels = []

        for anchor_imgs, positive_imgs, negative_imgs, anchor_labels, _ in train_loader:

            anchor_imgs = (anchor_imgs * 255).byte()
            positive_imgs = (positive_imgs * 255).byte()
            negative_imgs = (negative_imgs * 255).byte()

            anchor_imgs = anchor_imgs.to(device)
            positive_imgs = positive_imgs.to(device)
            negative_imgs = negative_imgs.to(device)
            
            anchor_inputs = processor(images=list(anchor_imgs), return_tensors="pt").to(device)
            positive_inputs = processor(images=list(positive_imgs), return_tensors="pt").to(device)
            negative_inputs = processor(images=list(negative_imgs), return_tensors="pt").to(device)

            with torch.no_grad():
                anchor_outputs = model(**anchor_inputs)
                positive_outputs = model(**positive_inputs)
                negative_outputs = model(**negative_inputs)
                
            anchor_embeddings = projection_head(anchor_outputs.pooler_output)
            positive_embeddings = projection_head(positive_outputs.pooler_output)
            negative_embeddings = projection_head(negative_outputs.pooler_output)

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            epoch_embeddings.append(anchor_embeddings)
            for label in anchor_labels:
                epoch_labels.append(label_to_idx[label])
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss, val_embeddings, val_labels = validate(model, processor, projection_head, val_loader, criterion, device, label_to_idx)

        scheduler.step()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        epoch_embeddings = torch.cat(epoch_embeddings)
        epoch_labels = torch.tensor(epoch_labels)  # Convert list of indices to tensor
        
        plot_embeddings(epoch_embeddings, epoch_labels, epoch+1, training_plot_dir, idx_to_label)
        plot_embeddings(val_embeddings, val_labels, epoch+1, val_plot_dir, idx_to_label)

        # Save weights
        save_checkpoint(projection_head, optimizer, scheduler, epoch+1, avg_train_loss, weights_dir)

    logging.info("Training complete.")

# Initialize training parameters
def setup_finetuning(logs_dir, weights_dir, plots_dir, training_data, siglip_directory, projection_head, loss, batch_size, lr, num_epochs=10):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    sys.stdout.flush()

    if not os.path.isdir(training_data):
        print(f"Dataset not found.")
        sys.stdout.flush()
        return

    if isinstance(loss, TripletEuclidianLoss):
        loss_type = "triplet_euclidian_loss"
        train_dataset = TripletDataset(train_images_folder)
        val_dataset = TripletDataset(val_images_folder)
        idx_to_labels = list(set(train_dataset.labels) | set(val_dataset.labels))
        labels_to_idx = {label: idx for idx, label in enumerate(idx_to_labels)}
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    elif isinstance(loss, TripletCosineLoss):
        loss_type = "triplet_cosine_loss"
        train_dataset = TripletDataset(train_images_folder)
        val_dataset = TripletDataset(val_images_folder)
        idx_to_labels = list(set(train_dataset.labels) | set(val_dataset.labels))
        labels_to_idx = {label: idx for idx, label in enumerate(idx_to_labels)}
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        print("Unknown loss type")
        return

    train_images_folder = os.path.join(training_data, "train")
    val_images_folder = os.path.join(training_data, "val")

    transform = Compose([
        Resize((224, 224)),  
        ToTensor()
    ])
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    siglip_vision_model = SiglipVisionModel.from_pretrained(siglip_directory).to(device)
    siglip_processor = AutoProcessor.from_pretrained(siglip_directory)
    projection_head = projection_head.to(device)

    optimizer = optim.Adam(projection_head.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    train(siglip_vision_model, siglip_processor, projection_head, train_loader, val_loader, optimizer, scheduler, loss, labels_to_idx, idx_to_labels, device, plots_dir, weights_dir, num_epochs=num_epochs)


# Define paths for the training outputs and inputs
logs_dir = "/cluster/home/lnonino/Making-CLIP-features-multiview-consistent/logs/logs-0"
weights_dir = "/cluster/home/lnonino/Making-CLIP-features-multiview-consistent/weights/weights-0"
plots_dir = "/cluster/home/lnonino/Making-CLIP-features-multiview-consistent/plots/plots-0"
training_data = "/cluster/project/cvg/data/scannet_multiview/training_dataset_300/project_without_occlusion"
siglip_directory = "/cluster/home/lnonino/Making-CLIP-features-multiview-consistent/siglip-base-patch16-224"

setup_finetuning(
    logs_dir = logs_dir,
    weights_dir = weights_dir,
    plots_dir = plots_dir,
    training_data = training_data,
    siglip_directory = siglip_directory,
    projection_head = ProjectionHead(embedding_dim=768, projection_dim=768),
    loss = TripletCosineLoss(1.0),
    batch_size=512,
    lr=5e-4,
    num_epochs=50
)


