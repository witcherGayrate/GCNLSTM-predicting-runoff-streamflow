import torch
import torch.nn.functional as F
from torch_geometric_temporal import GCLSTM


class GCN_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GCN_LSTM, self).__init__()
        self.recurrent = GCLSTM(input_size, hidden_size,K=1)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 4, output_size))
 
    def forward(self, x, edge_index, edge_weight):
        x=x.to(torch.float32)
        x, _ = self.recurrent(x, edge_index, edge_weight)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.MLP(x)
        return x.sequeeze(-1)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features,hidden_size,output_size):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(node_features, hidden_size,K=1) #k relay on the link depth of nodes
        #self.lstm = torch.nn.LSTM(hidden_size,hidden_size)
        #self.linear = torch.nn.Linear(hidden_size, 1)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 2, hidden_size//2 ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.01),
            #torch.nn.Linear(hidden_size // 4, hidden_size//4)
            #torch.nn.ReLU(inplace=True)
            torch.nn.Linear(hidden_size//2,output_size)
            )

    def forward(self, x, edge_index, edge_weight, h, c):
        x = x.to(torch.float32)
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        #h_0,c_0 = self.lstm(h_0)
        h = F.relu(h_0)
        #h = self.linear(h)
        h = self.MLP(h)
        return h.squeeze(-1), h_0.squeeze(-1), c_0.squeeze(-1)

class LSTM(torch.nn.Module):
    '''
    docstring for Lstm
    :param input_size:一日数据大小 = sequence * channels
    :param hidden_size:隐藏层神经元
    :param hidden_dim: 隐藏层个数
    :param output_size: 输出大小
    '''
    def __init__(self, input_size,hidden_size,hidden_dim,output_size):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size = input_size,hidden_size = hidden_size,num_layers=hidden_dim,batch_first=True,bidirectional=False)
        #self.attention = torch.nn.TransformerEncoderLayer(hidden_size,nhead=8,batch_first=True)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),  #if bidirectional specify as ture  the out_size will multiple 2
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 2, hidden_size//2 ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.01),
            #torch.nn.Linear(hidden_size // 4, hidden_size//4)
            #torch.nn.ReLU(inplace=True)
            torch.nn.Linear(hidden_size//2,output_size)
            ) 

    def forward(self,x):
        self.lstm.flatten_parameters()
        x = x.to(torch.float32)
        output,(w,y) = self.lstm(x)
        #output = F.relu(output)
        #output = self.attention(output)
        output = F.relu(output)
        output = self.MLP(output)
        return output

class GRU(torch.nn.Module):
    '''
    docstring for Lstm
    :param input_size:一日数据大小 = sequence * channels
    :param hidden_size:隐藏层神经元
    :param hidden_dim: 隐藏层个数
    :param output_size: 输出大小
    '''
    def __init__(self, input_size,hidden_size,output_size=1):
        super(GRU,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.GRU = torch.nn.GRU(input_size = input_size,hidden_size = hidden_size,batch_first=True)
        #self.dense = torch.nn.Linear(in_features=hidden_size,out_features=output_size) 
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 2, hidden_size//2 ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.01),
            torch.nn.Linear(hidden_size//2,output_size)
            )

    def forward(self,x):
        self.GRU.flatten_parameters()
        x = x.to(torch.float32)
        output,h_n = self.GRU(x)
        output = F.relu(output)
        output = self.MLP(output)
        return output

class RNN(torch.nn.Module):
    """docstring for RNN"""
    def __init__(self, input_size,hidden_size,num_layers=1,batch_first=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_size,hidden_size//2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 2, hidden_size//2 ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.01),
            torch.nn.Linear(hidden_size//2,1) #output_size=1
            )
    def forward(self,x):
        self.rnn.flatten_parameters()
        x = x.to(torch.float32)
        output,h_n = self.rnn(x)
        output = F.relu(output)
        output = self.MLP(output)
        return output


class MLP(torch.nn.Module):
    """docstring for MLP"""
    def __init__(self, input_size,hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size,hidden_size//2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 2, hidden_size//2 ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.01),
            torch.nn.Linear(hidden_size//2,1) #output_size=1
            )
    def forward(self,x):
        x = x.to(torch.float32)
        output = self.mlp(x)
        return output


class biLSTM(torch.nn.Module):
    '''
    docstring for Lstm
    :param input_size:一日数据大小 = sequence * channels
    :param hidden_size:隐藏层神经元
    :param hidden_dim: 隐藏层个数
    :param output_size: 输出大小
    '''
    def __init__(self, input_size,hidden_size,hidden_dim,output_size):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size = input_size,hidden_size = hidden_size,num_layers=hidden_dim,batch_first=True,bidirectional=True)
        #self.attention = torch.nn.TransformerEncoderLayer(hidden_size,nhead=8,batch_first=True)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, hidden_size // 2),  #if bidirectional specify as ture  the out_size will multiple 2
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size // 2, hidden_size//2 ),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.01),
            #torch.nn.Linear(hidden_size // 4, hidden_size//4)
            #torch.nn.ReLU(inplace=True)
            torch.nn.Linear(hidden_size//2,output_size)
            ) 

    def forward(self,x):
        self.lstm.flatten_parameters()
        x = x.to(torch.float32)
        output,(w,y) = self.lstm(x)
        #output = F.relu(output)
        #output = self.attention(output)
        output = F.relu(output)
        output = self.MLP(output)
        return output