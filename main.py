import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data.node import EEGNet
from data.graph_dataloader import get_dataloaders, get_balanced_dataloaders
from global_.model import HierarchicalModel

from sklearn.metrics import roc_auc_score


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


device = torch.device('cuda:0')

src_path = os.path.join(os.getcwd(), 'data')
train_path = os.path.join(src_path, 'mi_train_dataset.pt')
test_path = os.path.join(src_path, 'mi_test_dataset.pt')


class Trainer(object):
    def __init__(self):
        self.n_epochs = 100
        self.batch_size = 32
        self.lr = 0.001

        # self.train_dataloader, self.test_dataloader = get_dataloaders(train_path, test_path, self.batch_size)
        self.train_dataloader, self.test_dataloader = get_balanced_dataloaders(train_path, test_path, self.batch_size)

        sample_batch = next(iter(self.train_dataloader))
        self.n_nodes = int(sample_batch.x.shape[0] / sample_batch.num_graphs)

        self.net1 = EEGNet(f1=8, f2=16, d=2,
                           input_time_length=sample_batch.x.shape[-1],
                           embedding_dim=64,
                           dropout_rate=0.5, sampling_rate=128, classes=2).to(device)
        self.net2 = HierarchicalModel(in_channels=64,
                                      hidden_channels=32,
                                      out_channels=2,
                                      n_nodes=self.n_nodes).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = opt.AdamW(
            list(self.net1.parameters()) + list(self.net2.parameters()),
            lr=self.lr,
            weight_decay=1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.n_epochs,
                                                                    eta_min=1e-5)


    def node_embed(self, batch):
        # 1. Node feature
        raw_x = batch.x  # (1920, 384)

        s, t = raw_x.shape
        raw_x = raw_x.view(self.batch_size, -1, t)

        node_embeddings = self.net1(raw_x)  # (64, 30, 64)
        embed_dim = node_embeddings.shape[-1]
        node_embeddings = node_embeddings.view(-1, embed_dim)

        return node_embeddings

    def plot_results(self, train_acc_list, test_acc_list):
        train_acc, test_acc = np.array(train_acc_list), np.array(test_acc_list)
        epochs = range(1, len(train_acc) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_acc, 'b-', label='Train ACC')
        plt.plot(epochs, test_acc, 'r-', label='Test ACC')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.show()


    def train(self):
        # initial_aux_weight = 0.1  # 초기 보조손실 가중치 비율. Total Loss = Global_CE + gamma * (Local_CE + Meso_CE) + alpha * (Link + Ent)

        best_train_acc, best_test_acc, best_test_auc = 0, 0, 0
        train_acc_list, test_acc_list = [], []

        gamma = 0.5

        for epoch in range(self.n_epochs):
            self.net1.train()
            self.net2.train()

            total_loss, correct, total_samples = 0, 0, 0

            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                batch = batch.to(device)

                # 1. Extract Node Features  -> Update graphs
                batch.x = self.node_embed(batch) # outputs of EEGNet -> used as node features

                # 2. Train Local states + Cluster Meso states
                out_local, out_meso, out_global, link_loss, ent_loss = self.net2(batch)

                ce_loss_global = self.criterion(out_global, batch.y)
                ce_loss_meso = self.criterion(out_meso, batch.y)
                ce_loss_local = self.criterion(out_local, batch.y)

                loss = (1.0 * ce_loss_global) + \
                       (gamma * ce_loss_meso) + \
                       (gamma * ce_loss_local) + \
                       (0.1 * link_loss) + \
                       (0.1 * ent_loss)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = out_global.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total_samples += batch.y.size(0)

            self.scheduler.step()

            train_avg_loss = total_loss / len(self.train_dataloader)
            train_accuracy = 100 * correct / total_samples
            train_acc_list.append(train_accuracy)

           # Validation
            self.net1.eval()
            self.net2.eval()

            test_loss, correct, test_samples = 0, 0, 0
            test_preds, test_labels, test_probes = [], [], []
            for batch in self.test_dataloader:
                batch = batch.to(device)

                batch.x = self.node_embed(batch)
                out_global, out_meso, out_local, link_loss, ent_loss = self.net2(batch)

                ce_loss_global = self.criterion(out_global, batch.y)
                ce_loss_meso = self.criterion(out_meso, batch.y)
                ce_loss_local = self.criterion(out_local, batch.y)

                loss = (1.0 * ce_loss_global) + \
                       (gamma * ce_loss_meso) + \
                       (gamma * ce_loss_local) + \
                       (0.1 * link_loss) + \
                       (0.1 * ent_loss)

                test_loss += loss.item()

                pred = out_global.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                test_samples += batch.y.size(0)

                probs = F.softmax(out_global, dim=1)

                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(batch.y.cpu().numpy())
                test_probes.extend(probs[:, 1].detach().cpu().numpy())

            # Performance 1: ACC
            test_avg_loss = test_loss / len(self.test_dataloader)
            test_accuracy = 100 * correct / test_samples

            test_acc_list.append(test_accuracy)

            # Performance 2: AUC
            try:
                test_auc = roc_auc_score(test_labels, test_probes)
            except ValueError:
                test_auc = 0.0  # 배치 안에 한 클래스만 있는 경우 에러 방지

            # Best Performance
            if train_accuracy > best_train_acc:
                best_train_acc = train_accuracy

            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy

            if test_auc > best_test_auc:
                best_test_auc = test_auc


            print(f"Epoch {epoch + 1:03d} | Train Loss: {train_avg_loss:.4f} | Train ACC: {train_accuracy:.2f}%"
                  f" | Test Loss: {test_avg_loss:.4f} | Test ACC: {test_accuracy:.2f}% | Test AUC: {test_auc:.4f}")

        print("-" * 60)
        print(f"Training Finished!")
        print(f"Best Train Acc: {best_train_acc:.2f}% | Best Test Acc: {best_test_acc:.2f}% | Best Test AUC: {best_test_auc:.4f}")
        print("-" * 60)

        self.plot_results(train_acc_list, test_acc_list)
        

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
