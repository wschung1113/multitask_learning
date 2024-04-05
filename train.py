import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split


class MultiTaskModel(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=500, output_dim1=100, output_dim2=7, dropout_p=0.1):
        super(MultiTaskModel, self).__init__()
        
        # 첫 번째 fully connected layer, ReLU activation, dropout
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        
        # 두 번째 fully connected layer (첫 번째 multi-task layer)
        self.fc2_1 = nn.Linear(hidden_dim, output_dim1)
        
        # 세 번째 fully connected layer (두 번째 multi-task layer)
        self.fc2_2 = nn.Linear(hidden_dim, output_dim2)
        
    def forward(self, x):
        # 입력 데이터를 첫 번째 fully connected layer, ReLU activation, dropout을 거쳐 전달
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 두 개의 multi-task layer로 분리
        output1 = self.fc2_1(x)  # 첫 번째 multi-task layer
        output2 = self.fc2_2(x)  # 두 번째 multi-task layer
        
        return output1, output2
    

# 모델 초기화
model = MultiTaskModel()

# CUDA 사용 가능 여부 확인
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# 손실 함수 정의
criterion = nn.BCEWithLogitsLoss()

# 옵티마이저 정의
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 데이터 준비
# .npy 파일 경로
# task 1
ligand_structure_1_file_path = "ligand_structure.npy"
protein_binding_file_path = "protein_binding.npy"

# task 2
ligand_structure_2_file_path = "ligand_structure_2.npy"
drug_category_file_path = "drug_catgory.npy"

# .npy 파일 로드
# 200,000 samples
ligand_structure_1_array = np.load(ligand_structure_1_file_path)
protein_binding_array = np.load(protein_binding_file_path)

# 10,000 samples
ligand_structure_2_array = np.load(ligand_structure_2_file_path)
drug_category_array = np.load(drug_category_file_path)

# extend sample size by repetition for unbiased model training
ligand_structure_2_array = np.repeat(ligand_structure_2_array, repeats=20, axis=0)
drug_category_array = np.repeat(drug_category_array, repeats=20, axis=0)

# define pytorch Dataset and DataLoader
class ProteinBindingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        return torch.tensor(x_sample, dtype=torch.long), torch.tensor(y_sample, dtype=torch.long)

class DrugCategoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        return torch.tensor(x_sample, dtype=torch.long), torch.tensor(y_sample, dtype=torch.long)
    
protein_binding_dataset = ProteinBindingDataset(ligand_structure_1_array, protein_binding_array)
drug_category_dataset = DrugCategoryDataset(ligand_structure_2_array, drug_category_array)

train_size = int(0.8 * len(protein_binding_dataset))
val_size = len(protein_binding_dataset) - train_size

protein_binding_train_dataset, protein_binding_val_dataset = random_split(protein_binding_dataset, [train_size, val_size])
drug_category_train_dataset, drug_category_val_dataset = random_split(drug_category_dataset, [train_size, val_size])

batch_size = 64

protein_binding_train_loader = DataLoader(protein_binding_train_dataset, batch_size=batch_size, shuffle=True)
protein_binding_val_loader = DataLoader(protein_binding_val_dataset, batch_size=batch_size, shuffle=False)

drug_category_train_loader = DataLoader(drug_category_train_dataset, batch_size=batch_size, shuffle=True)
drug_category_val_loader = DataLoader(drug_category_val_dataset, batch_size=batch_size, shuffle=False)


# 모델 학습
batch_iter_n = train_size // batch_size + 1

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    
    for i in range(batch_iter_n):
        ## task 1 training
        inputs_task_1, labels_task_1 = next(iter(protein_binding_train_loader))
        # inputs_task_1, labels_task_1 = inputs_task_1.to(device), labels_task_1.to(device)
        
        # 첫 번째 task에 해당하는 레이어를 unfreeze
        for param in model.fc2_1.parameters():
            param.requires_grad = True
        
        # 두 번째 task에 해당하는 레이어를 freeze
        for param in model.fc2_2.parameters():
            param.requires_grad = False
        
        optimizer.zero_grad()
        outputs1, _ = model(inputs_task_1)  # 첫 번째 task에 대한 출력만 사용
        loss1 = criterion(outputs1, labels_task_1)
        loss1.backward()
        optimizer.step()
        
        ## task 2 training
        inputs_task_2, labels_task_2 = next(iter(drug_category_train_loader))
        # inputs_task_2, labels_task_2 = inputs_task_2.to(device), labels_task_2.to(device)
        
        # 두 번째 task에 해당하는 레이어를 unfreeze
        for param in model.fc2_2.parameters():
            param.requires_grad = True
        
        # 첫 번째 task에 해당하는 레이어를 freeze
        for param in model.fc2_1.parameters():
            param.requires_grad = False
        
        optimizer.zero_grad()
        _, outputs2 = model(inputs_task_2)  # 첫 번째 task에 대한 출력만 사용
        loss2 = criterion(outputs2, labels_task_2)
        loss2.backward()
        optimizer.step()
        
    model.eval()
    
    val_loss_task_1 = 0.0
    val_loss_task_2 = 0.0
    
    with torch.no_grad():
        for inputs, labels in protein_binding_val_loader:
            outputs1, _ = model(inputs)
            val_loss_task_1 += criterion(outputs1.squeeze(), labels.float()).item()
        
        for inputs, labels in drug_category_val_loader:
            _, outputs2 = model(inputs)
            val_loss_task_2 += criterion(outputs2.squeeze(), labels.float()).item() 
            
    val_loss_task_1 /= len(protein_binding_val_loader)
    val_loss_task_2 /= len(drug_category_val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss_task_1:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss_task_2:.4f}')
        