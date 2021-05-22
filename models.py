import torch
import torch.nn as nn

class MLP3(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout_rate = 0.5
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class MLP4(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden2_num = 100
        self.relu = nn.ReLU()
        self.dropout_rate = 0.5
        self.fc2 = nn.Linear(hidden_size, self.hidden2_num)
        self.fc3 = nn.Linear(self.hidden2_num, num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class MLP5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP5, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden2_num = 100
        self.hidden3_num = 200
        self.relu = nn.ReLU()
        self.dropout_rate = 0.65
        self.fc2 = nn.Linear(hidden_size, self.hidden2_num)
        self.fc3 = nn.Linear(self.hidden2_num, self.hidden3_num)
        self.fc4 = nn.Linear(self.hidden3_num, num_classes)
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

