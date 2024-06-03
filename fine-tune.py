import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
from torchvision import transforms
import shutil
import random
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
from torch.optim.lr_scheduler import StepLR
import plotly.express as px
import plotly.io as pio
import pandas as pd



class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.0):
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

class BasicDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.image_paths, self.labels = self._load_images_and_labels()

    def _load_images_and_labels(self):
        image_paths = []
        labels = []
        label_map = {}
        current_label = 0

        for scene in os.listdir(self.images_folder):
            scene_path = os.path.join(self.images_folder, scene)
            if os.path.isdir(scene_path):
                for obj in os.listdir(scene_path):
                    obj_path = os.path.join(scene_path, obj)
                    if os.path.isdir(obj_path):
                        label = f"{scene}_{obj}"
                        if label not in label_map:
                            label_map[label] = current_label
                            current_label += 1
                        for img_file in os.listdir(obj_path):
                            if img_file.endswith('.jpg'):
                                image_paths.append(os.path.join(obj_path, img_file))
                                labels.append(label_map[label])
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

class TripletDataset(BasicDataset):
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

        # Display the images using matplotlib
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(anchor_img)
        axes[0].axis('off')  # Turn off axis labels for the first image
        axes[1].imshow(positive_img)
        axes[1].axis('off')  # Turn off axis labels for the second image
        axes[2].imshow(negative_img)
        axes[2].axis('off')  # Turn off axis labels for the third image
        plt.show()
        
        return anchor_img, positive_img, negative_img, idx, positive_idx, negative_idx

    def __len__(self):
        return len(self.image_paths)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, anchor, positive, negative):
        return self.triplet_loss(anchor, positive, negative)   

def split_dataset(full_dataset_dir, output_dir, val_split=0.2, seed=42, num_images_per_obj=10):
    random.seed(seed)
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    for scene in os.listdir(full_dataset_dir):
        scene_path = os.path.join(full_dataset_dir, scene)
        if os.path.isdir(scene_path):
            os.makedirs(os.path.join(train_dir, scene), exist_ok=True)
            os.makedirs(os.path.join(val_dir, scene), exist_ok=True)
            
            for obj in os.listdir(scene_path):
                obj_path = os.path.join(scene_path, obj)
                if os.path.isdir(obj_path):
                    train_obj_dir = os.path.join(train_dir, scene, obj)
                    val_obj_dir = os.path.join(val_dir, scene, obj)
                    
                    os.makedirs(train_obj_dir, exist_ok=True)
                    os.makedirs(val_obj_dir, exist_ok=True)
                    
                    img_files = [f for f in os.listdir(obj_path) if f.endswith('.jpg')]
                    
                    img_files.sort()
                    
                    if len(img_files) > num_images_per_obj:
                        step = len(img_files) // num_images_per_obj
                        sampled_files = [img_files[i * step] for i in range(num_images_per_obj)]
                    else:
                        sampled_files = img_files
                    
                    random.shuffle(sampled_files)
                    
                    split_idx = int(len(sampled_files) * (1 - val_split))
                    train_files = sampled_files[:split_idx]
                    val_files = sampled_files[split_idx:]

                    for file in train_files:
                        shutil.copy(os.path.join(obj_path, file), train_obj_dir)
                    
                    for file in val_files:
                        shutil.copy(os.path.join(obj_path, file), val_obj_dir)
    
    print(f"Dataset split complete. Training data in: {train_dir}, Validation data in: {val_dir}")

def validate(model, processor, projection_head, dataloader, criterion, device):
    projection_head.eval()
    total_loss = 0
    with torch.no_grad():
        for anchor_imgs, positive_imgs, negative_imgs, _, _, _ in dataloader:
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

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def plot_embeddings(embeddings, labels, epoch, output_dir):
    pca2 = PCA(n_components=2)
    pca2_result = pca2.fit_transform(embeddings.cpu().detach().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(pca2_result[:, 0], pca2_result[:, 1], c=labels.cpu().numpy(), cmap='tab10', alpha=0.6)
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
    labels_df = pd.DataFrame({'label': labels.cpu().numpy()})

    fig = px.scatter_3d(
        x=pca3_result[:, 0],
        y=pca3_result[:, 1],
        z=pca3_result[:, 2],
        color=labels_df['label'].astype(str), 
        title=f"PCA of Embeddings at Epoch {epoch}",
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "z": "PCA Component 3"},
        color_discrete_sequence=color_sequence
    )

    fig.update_layout(legend_title_text='Labels')

    pca3_dir = os.path.join(output_dir, "PCA_3")
    os.makedirs(pca3_dir, exist_ok=True)
    pio.write_html(fig, file=os.path.join(pca3_dir, f"epoch_{epoch}_3d.html"))
    fig.write_image(os.path.join(pca3_dir, f"epoch_{epoch}_3d.png"))

def plot_initial_embeddings(model, processor, projection_head, dataloader, device, output_dir):
    projection_head.eval()
    initial_embeddings = []
    initial_labels = []

    with torch.no_grad():
        for anchor_imgs, _, _, anchor_indices, _, _ in dataloader:
            anchor_imgs = anchor_imgs.to(device)
            anchor_inputs = processor(images=anchor_imgs, return_tensors="pt").to(device)
            anchor_outputs = model(**anchor_inputs)
            anchor_embeddings = projection_head(anchor_outputs.pooler_output)
            
            initial_embeddings.append(anchor_embeddings)
            initial_labels.extend(anchor_indices)

    initial_embeddings = torch.cat(initial_embeddings)
    initial_labels = torch.tensor(initial_labels)
    
    plot_embeddings(initial_embeddings, initial_labels, 0, output_dir)

def train(model, processor, projection_head, train_loader, val_loader, optimizer, scheduler, criterion, device, plots_dir, num_epochs=10):

    plot_initial_embeddings(model, processor, projection_head, train_loader, device, plots_dir)

    for epoch in range(num_epochs):
        projection_head.train()
        total_loss = 0
        epoch_embeddings = []
        epoch_labels = []

        for anchor_imgs, positive_imgs, negative_imgs, anchor_indices, _, _ in train_loader:
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
            epoch_labels.extend(anchor_indices)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate(model, processor, projection_head, val_loader, criterion, device)

        scheduler.step()

        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        epoch_embeddings = torch.cat(epoch_embeddings)
        epoch_labels = torch.tensor(epoch_labels)  # Convert list of indices to tensor
        
        plot_embeddings(epoch_embeddings, epoch_labels, epoch+1, plots_dir)

    logging.info("Training complete.")

def setup_finetuning(data, siglip_version, loss, embedding_dim, projection_dim, batch_size, lr, num_epochs=10):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    original_dataset = os.path.join(os.getcwd(), "outputs/masked_images/", data)
    custom_dataset = os.path.join(os.getcwd(), "custom_dataset/", data)
    if not os.path.isdir(custom_dataset):
        os.makedirs(custom_dataset, exist_ok=True)
        split_dataset(original_dataset, custom_dataset, val_split=0.2, num_images_per_obj=50)

    train_images_folder = os.path.join(custom_dataset, "train")
    val_images_folder = os.path.join(custom_dataset, "val")

    if isinstance(loss, TripletLoss):
        loss_type = "triplet_loss"
        train_dataset = TripletDataset(train_images_folder)
        val_dataset = TripletDataset(val_images_folder)
        train_loader = DataLoader(train_dataset)
        val_loader = DataLoader(val_dataset)
    else:
        print("Unknown loss type")
        return

    plots_dir = os.path.join(os.getcwd(), f"plots/{data}/{loss_type}")
    logs_dir = os.path.join(os.getcwd(), f"logs/{data}/{loss_type}")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "training.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    siglip_vision_model = SiglipVisionModel.from_pretrained(siglip_version).to(device)
    siglip_processor = AutoProcessor.from_pretrained(siglip_version)
    projection_head = ProjectionHead(embedding_dim=embedding_dim, projection_dim=projection_dim).to(device)

    optimizer = optim.Adam(projection_head.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.05)

    train(siglip_vision_model, siglip_processor, projection_head, train_loader, val_loader, optimizer, scheduler, loss, device, plots_dir, num_epochs=num_epochs)
 

setup_finetuning(
    data = "project_without_occlusion",
    siglip_version="google/siglip-base-patch16-224",
    loss = TripletLoss(1.0),
    embedding_dim=768,
    projection_dim=256,
    batch_size=128,
    lr=1e-4,
    num_epochs=10
)
