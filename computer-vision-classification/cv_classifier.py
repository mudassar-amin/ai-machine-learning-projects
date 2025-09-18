# Simple Computer Vision Classification Project
# This single file downloads CIFAR-10 and trains multiple CNN models

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleCV:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    def load_data(self):
        """Download and load CIFAR-10 dataset"""
        print("Downloading CIFAR-10 dataset...")
        
        # Download training data
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True,  # This automatically downloads the dataset
            transform=self.train_transform
        )
        
        # Download test data
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True,  # This automatically downloads the dataset
            transform=self.test_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print("Dataset loaded successfully!")
    
    def create_simple_cnn(self):
        """Create a simple CNN model"""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(128 * 7 * 7, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return SimpleCNN()
    
    def create_resnet(self):
        """Create ResNet50 with transfer learning"""
        model = resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    
    def train_model(self, model, model_name, epochs=10):
        """Train the model"""
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses = []
        train_accuracies = []
        
        print(f"\nTraining {model_name}...")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training loop with progress bar
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{running_loss/(pbar.n+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%")
        
        return model, train_losses, train_accuracies
    
    def test_model(self, model, model_name):
        """Test the model and show results"""
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        
        print(f"\nTesting {model_name}...")
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # Overall accuracy
        overall_acc = 100 * correct / total
        print(f"{model_name} Test Accuracy: {overall_acc:.2f}%")
        
        # Per-class accuracy
        print(f"\nPer-class accuracy for {model_name}:")
        for i in range(10):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"{self.class_names[i]}: {acc:.2f}%")
        
        return overall_acc
    
    def show_sample_predictions(self, model, model_name, num_samples=8):
        """Show sample predictions"""
        model.eval()
        
        # Get a batch of test images
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        
        # Select random samples
        indices = np.random.choice(len(images), num_samples, replace=False)
        sample_images = images[indices]
        sample_labels = labels[indices]
        
        # Make predictions
        with torch.no_grad():
            sample_images = sample_images.to(self.device)
            outputs = model(sample_images)
            _, predicted = torch.max(outputs, 1)
        
        # Plot results
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Convert image for display (denormalize)
            img = sample_images[i].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0)
            
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Labels
            true_label = self.class_names[sample_labels[i]]
            pred_label = self.class_names[predicted[i].cpu()]
            
            color = 'green' if true_label == pred_label else 'red'
            title = f'True: {true_label}\nPred: {pred_label}'
            axes[i].set_title(title, color=color, fontsize=10)
        
        plt.suptitle(f'{model_name} - Sample Predictions', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete training and testing pipeline"""
        print("=" * 60)
        print("COMPUTER VISION CLASSIFICATION PROJECT")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Train and test models
        models = [
            ("Simple CNN", self.create_simple_cnn(), 15),
            ("ResNet50", self.create_resnet(), 10)
        ]
        
        results = {}
        
        for model_name, model, epochs in models:
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            # Train
            start_time = time.time()
            trained_model, losses, accuracies = self.train_model(model, model_name, epochs)
            training_time = time.time() - start_time
            
            # Test
            test_accuracy = self.test_model(trained_model, model_name)
            
            # Show sample predictions
            self.show_sample_predictions(trained_model, model_name)
            
            # Save results
            results[model_name] = {
                'test_accuracy': test_accuracy,
                'training_time': training_time,
                'final_train_loss': losses[-1],
                'final_train_acc': accuracies[-1]
            }
            
            # Save model
            torch.save(trained_model.state_dict(), f'{model_name.lower().replace(" ", "_")}_model.pth')
            print(f"Model saved as {model_name.lower().replace(' ', '_')}_model.pth")
        
        # Compare results
        print(f"\n{'='*20} FINAL RESULTS {'='*20}")
        print(f"{'Model':<15} {'Test Acc':<10} {'Train Time':<12}")
        print("-" * 40)
        for name, result in results.items():
            print(f"{name:<15} {result['test_accuracy']:<10.2f} {result['training_time']:<12.1f}s")
        
        print(f"\nProject completed successfully!")
        print(f"Dataset saved in: ./data/")
        print(f"Models saved as .pth files")


def main():
    # Create and run the project
    cv_project = SimpleCV()
    cv_project.run_complete_pipeline()


if __name__ == "__main__":
    main()