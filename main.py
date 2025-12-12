import os
from data.node import EEGNet
from local.model import GCN
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import random
from data.graph_dataloader import get_dataloaders


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


device = torch.device('cuda:0')

src_path = os.path.join(os.getcwd(), 'data')
train_path = os.path.join(src_path, 'driver_train_dataset.pt')
test_path = os.path.join(src_path, 'driver_test_dataset.pt')


class Trainer(object):
    def __init__(self):
        self.net1 = EEGNet(f1=8, f2=16, d=2,
                           input_time_length=384,
                           embedding_dim=64,
                           dropout_rate=0.5, sampling_rate=128, classes=2).to(device)
        self.net2 = GCN(in_channels=64,
                        out_channels=2,
                        hidden_channels=32,
                        dropout_rate=0.5).to(device)

        self.n_epochs = 100
        self.batch_size = 64
        self.lr = 0.01
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = opt.AdamW(
            list(self.net1.parameters()) + list(self.net2.parameters()),
            lr=self.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.995)

        self.train_dataloader, self.test_dataloader = get_dataloaders(train_path, test_path, 64)

    def node_embed(self, batch):
        # 1. Node feature
        raw_x = batch.x  # (1920, 384)

        s, t = raw_x.shape
        raw_x = raw_x.view(self.batch_size, -1, t)

        node_embeddings = self.net1(raw_x)  # (64, 30, 64)
        embed_dim = node_embeddings.shape[-1]
        node_embeddings = node_embeddings.view(-1, embed_dim)

        return node_embeddings


    def train(self):
        best_train_acc, best_test_acc = 0, 0

        for epoch in range(self.n_epochs):
            self.net1.train()
            self.net2.train()

            total_loss, correct, total_samples = 0, 0, 0

            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                batch = batch.to(device)

                # 1. Node feature Extraction -> Graph Feature Update
                batch.x = self.node_embed(batch) # outputs of EEGNet -> used as node features

                # 2. First Training in Local State (GCN)
                out = self.net2(batch)

                loss = self.criterion(out, batch.y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total_samples += batch.y.size(0)

            self.scheduler.step()

            train_avg_loss = total_loss / len(self.train_dataloader)
            train_accuracy = 100 * correct / total_samples

           # Validation
            self.net1.eval()
            self.net2.eval()

            test_loss, correct, test_samples = 0, 0, 0
            test_preds, test_labels = [], []
            for batch in self.test_dataloader:
                batch = batch.to(device)

                batch.x = self.node_embed(batch)
                out = self.net2(batch)
                loss = self.criterion(out, batch.y)
                test_loss += loss.item()

                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                test_samples += batch.y.size(0)

                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(batch.y.cpu().numpy())

            test_avg_loss = test_loss / len(self.test_dataloader)
            test_accuracy = 100 * correct / test_samples

            if train_accuracy > best_train_acc:
                best_train_acc = train_accuracy

            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy

            print(f"Epoch {epoch + 1:03d} | Train Loss: {train_avg_loss:.4f} | Train ACC: {train_accuracy:.2f}%"
                  f" | Test Loss: {test_avg_loss:.4f} | Test ACC: {test_accuracy:.2f}%")

        print("-" * 60)
        print(f"Training Finished!")
        print(f"Best Train Acc: {best_train_acc:.2f}% | Best Test Acc: {best_test_acc:.2f}%")
        print("-" * 60)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
