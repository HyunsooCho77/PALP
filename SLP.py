import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import pickle
from utils import return_filenames, train_converter, test_converter
from torch.optim.lr_scheduler import StepLR
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

class TensorData(Dataset):
  def __init__(self, x_data, y_data):
    self.x_data = torch.FloatTensor(x_data)
    self.y_data = torch.LongTensor(y_data) 
    self.len = self.y_data.shape[0]        

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len


class SLP(nn.Module):
    def __init__(self, rep_dim,num_labels):
        super().__init__()
        self.mlp1 = nn.Linear(rep_dim, num_labels)
        self.do = torch.nn.Dropout(p=0.1, inplace=False)
    def forward(self, x):
        return self.do(self.mlp1(F.relu(x)))


@hydra.main(config_path=".", config_name="model_config.yaml")
def main(args: DictConfig):
    # Load Dataset
    train_file, test_file, _ = return_filenames(args, task_name=args.task_name)

    with open(test_file,'rb') as f:
        test_data = pickle.load(f)
    with open(train_file,'rb') as f:
        train_data = pickle.load(f)

    train_x, train_y =  train_converter(train_data, return_type='list')
    traindata = TensorData(train_x, train_y)
    trainloader = DataLoader(traindata, batch_size= args.nn_batch_size)
    
    test_x1, test_y1 =  test_converter(test_data, return_type='list')
    testdata1 = TensorData(test_x1, test_y1)
    testloader1 = DataLoader(testdata1, batch_size=256)


    # SLP & initialize
    criterion = nn.CrossEntropyLoss()
    rep_dim, num_labels = len(train_x[0]),len(train_data)
    model = SLP(rep_dim,num_labels)
    optimizer = optim.Adam(model.parameters(), lr=args.nn_lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    scheduler = StepLR(optimizer, step_size = 10, gamma= 0.8)
    early_stop_count, best_acc = 0,0


    # Train & evalutate
    torch.manual_seed(42)
    pbar = tqdm(range(1000))
    for epoch in pbar: 
        
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        acc1 = round(calculate_accuracy(model,testloader1), 2)
        
        if acc1>= best_acc:
            best_acc = acc1
            early_stop_count=0
        else:
            early_stop_count+=1
        if early_stop_count >= args.nn_early_stop:
            break
            
        pbar.set_description(f'loss : {running_loss}, acc: {best_acc}, early_stop_count: {early_stop_count}')
        
    print(f'BEST > acc:{best_acc}')
    print('Finished Training')


def calculate_accuracy(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc =  100 * correct / total
    return acc



if __name__ == "__main__":
    main()