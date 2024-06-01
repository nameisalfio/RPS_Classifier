import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
from os.path import join

class RPSClassifier(nn.Module):
    
    def __init__(self, input_shape=(50, 50, 3), num_classes=3):
        super(RPSClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, test_loader, exp_name='experiment', lr=0.001, epochs=10, m=0.9, logdir='logs', offset=0):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr, momentum=m)  
    
        train_losses = []; train_accuracies = []
        test_losses = []; test_accuracies = []
    
        loss_meter = AverageValueMeter()
        acc_meter = AverageValueMeter()
        writer = SummaryWriter(join(logdir, exp_name))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        loader = {'train' : train_loader, 'test' : test_loader }
    
        global_step = 0
        for e in range(epochs):
            print(f'[Epoch {e+1+offset}/{epochs+offset}]')
            for mode in ['train', 'test']:
                loss_meter.reset()
                acc_meter.reset()
                self.model.train() if mode == 'train' else self.model.eval()
    
                with torch.set_grad_enabled(mode == 'train'):
                    for i, batch in enumerate(loader[mode]):
                        X = batch[0].to(device)
                        y = batch[1].to(device)
                        out = self.model(X)
    
                        n = X.shape[0]
                        global_step += n
                        l = loss_fn(out, y)
    
                        if mode == 'train':
                            l.backward()
                            optimizer.step()
                            optimizer.zero_grad()
    
                        loss_meter.add(l.item(), n)
                        acc = accuracy_score(y.to('cpu'), out.to('cpu').max(1)[1])
                        acc_meter.add(acc, n)
    
                        if mode == 'train':
                            writer.add_scalar('loss/train', loss_meter.value(), global_step=global_step)
                            writer.add_scalar('accuracy/train', acc_meter.value(), global_step=global_step)
    
                writer.add_scalar('loss/'+mode, loss_meter.value(), global_step=global_step)
                writer.add_scalar('accuracy/'+mode, acc_meter.value(), global_step=global_step)
    
                if mode == 'train':
                    train_losses.append(loss_meter.value())
                    train_accuracies.append(acc_meter.value())
                else:
                    test_losses.append(loss_meter.value())
                    test_accuracies.append(acc_meter.value())
                    
            print(f'[Epoch {e+1+offset}/{epochs+offset}] - Train Loss: {train_losses[-1]:.3f}, Train Accuracy: {train_accuracies[-1]:.3f}, Test Loss: {test_losses[-1]:.3f}, Test Accuracy: {test_accuracies[-1]:.3f}\n')
            torch.save(self.model.state_dict(), f'{exp_name}/{exp_name}-{e+1}.pth')

        return train_losses, train_accuracies, test_losses, test_accuracies

    def test_classifier(self, loader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        predictions, labels = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                output = self.model(x)
                preds = output.to('cpu').max(1)[1].numpy()
                labs = y.to('cpu').numpy()
                predictions.extend(list(preds))
                labels.extend(list(labs))
        return np.array(predictions), np.array(labels)

    def perc_error(self, gt, pred):
        return (1 - accuracy_score(gt, pred)) * 100