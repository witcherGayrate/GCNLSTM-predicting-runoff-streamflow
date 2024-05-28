import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import RecurrentGCN,GCN_LSTM,LSTM
from torch_geometric_temporal import StaticGraphTemporalSignal
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dataset(normalized_data = False,data_seq=3,pred_len = 1,offset = 1,train_test_ratio = 0.8):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    if type(normalized_data) is bool:
        normalized_data = np.load('../bin/value_matrix.npy') #(nodes,batch,channels)
        train_len = int(normalized_data.shape[1]*train_test_ratio)
        for i in range(0,train_len-data_seq-pred_len,offset):
            input_seq = normalized_data[:,i:i+data_seq,:]
            X_train.append(input_seq)
            target_lable = normalized_data[:,i+data_seq+pred_len-1,-1] #runoff is label
            Y_train.append(target_lable)
        test_len = normalized_data.shape[1]-train_len
        for j in range(0,test_len-data_seq-pred_len,offset):
            input_seq = normalized_data[:,train_len+j:train_len+j+data_seq,:]
            X_test.append(input_seq)
            target_lable = normalized_data[:,train_len+j+data_seq+pred_len-1,-1]#-1 is runoff
            Y_test.append(target_lable)
    else:
        train_len = int(normalized_data.shape[1]*train_test_ratio)
        for i in range(0,train_len-data_seq-pred_len,offset):
            input_seq = normalized_data[:,i:i+data_seq,:]
            X_train.append(input_seq)
            target_lable = normalized_data[:,i+data_seq+pred_len-1,-1] #runoff is label
            Y_train.append(target_lable)
        test_len = normalized_data.shape[1]-train_len
        for j in range(0,test_len-data_seq-pred_len,offset):
            input_seq = normalized_data[:,train_len+j:train_len+j+data_seq,:]
            X_test.append(input_seq)
            target_lable = normalized_data[:,train_len+j+data_seq+pred_len-1,-1]#-1 is runoff
            Y_test.append(target_lable)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    print("------------Completed splitting the data and stored it-----------")
    return X_train,Y_train,X_test,Y_test
def MaxMin_Normalized(value_matrix,min_max_values): #not suitable to feature ,which has constant value in node .such as longitude. The max is the same as min of these features.
    # Normalize value_matrix using min_max_values
    # The time range of the feature remains unchanged, and the maximum and minimum values remain unchanged.
    for i in range(value_matrix.shape[0]):
        for j in range(value_matrix.shape[-1]):
            value_matrix[i, :, j] = (value_matrix[i, :, j] - min_max_values[i, j, 0]) / (min_max_values[i, j, 1] - min_max_values[i, j, 0])
    print(f'value_matirx_normalized.shape: {value_matrix.shape}')
    return value_matrix
def Anti_Normalized(label,min_max_values):
    for node in range(label.shape[-1]):
        label[:,node] = label[:,node]*(min_max_values[node,-1,1]-min_max_values[node,-1,0])+min_max_values[node,-1,0]
    print(f'anti_normalized_label.shape: {label.shape}\n')
    return label

#---------data preprogress-------
value_matrix = np.load('../bin/value_matrix.npy')[:,:,3:] #remove constant value :longitude,latitude,altitude
#data scaler
min_max_values = np.load('../bin/min_max_values.npy')

#normalized dynamic features
v_m_normalized = MaxMin_Normalized(value_matrix,min_max_values)
#split train data and test data
X_train,Y_train,X_test,Y_test = create_dataset(v_m_normalized)


print(f"X_train.shape: {X_train.shape}") #(batchsize,nodes,sequence,channels)
print(f"Y_train.shape: {Y_train.shape}")#(batchsize,nodes,channel)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2]*X_train.shape[3])#(batch,nodes,seq*channel)
Y_train = Y_train.reshape(Y_train.shape[0],Y_train.shape[1],1) #(batch,node,channel)

X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]*X_test.shape[3])#(batch,nodes,seq*channel)
Y_test = Y_test.reshape(Y_test.shape[0],Y_test.shape[1],1) #(batch,node,channel)

#load edges_index and edges_weight
edge_index = np.load("../bin/edge_index.npy")

edge_weight = np.load("../bin/edge_weight.npy")


'''PyG Temporal中的类似DataLoader的数据处理器：
    edge_index：是邻接矩阵；
    edge_weight：是每个链路的权重；
    features：是输入历史节点特征矩阵；
    targets：是输入预测节点特征举证ground-truth；
'''
train_dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=X_train, targets=Y_train)
test_dataset = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight, features=X_test, targets=Y_test)



print('----------------Created the dataloader of Train Data-------------')


# Define the model (GCN_LSTM) and other hyperparameters

input_size = X_train.shape[2] #input_size = input_dim*input_seq
hidden_size = 128
hidden_dim = 1
out_size = 1

model = RecurrentGCN(input_size,hidden_size,out_size)#.to(device)
#model = Lstm(input_size,hidden_size,hidden_dim,out_size)
num_epochs = 120
# Loss function (Mean Squared Error)
criterion = nn.MSELoss()
params = list(model.parameters())
# Optimizer (e.g., Adam)
optimizer = optim.Adam(params, lr=0.01)

model.train()
cost_list = []
# Training loop
for epoch in range(num_epochs): #迭代次数
    cost = 0.0
    h,c=None,None
    for batch,snapshot in enumerate(train_dataset):  # Iterate over your training data #训练批次
        # Forward pass
        #snapshot = snapshot.to(device) 
        y_hat,h,c= model(snapshot.x,snapshot.edge_index,snapshot.edge_weight,h,c)
        #y_hat= model(snapshot.x)
        # Calculate the loss
        cost += criterion(y_hat.squeeze(-1),snapshot.y.squeeze(-1))
    cost = cost/(batch+1) #一次epoch后样本平均loss
    cost_list.append(cost.item())
    # Backpropagation
    cost.backward()
    # Update weights
    optimizer.step()
    # Zero the parameter gradients
    optimizer.zero_grad()   
    print("[Epoch {}]:loss {}".format(epoch, cost*1000))   
cost = cost.item()

print("Finished training model")
#torch.save(model.state_dict(),'./GCN_LSTM_params.pth')
print("Train MSE: {:.4f}".format(cost))

plt.plot(cost_list)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Average of Training cost for nodes")
plt.show()


model.eval() #滚动预测输出作为新输入,预测时长从1到365天
cost = 0
test_real = []
test_pre = []
cha = []
for time, snapshot in enumerate(test_dataset):
    #snapshot = snapshot.to(device)
    y_hat,h,c = model(snapshot.x, snapshot.edge_index, snapshot.edge_weight,h,c)
    #y_hat= model(snapshot.x)
    test_pre.append(y_hat.detach().cpu().numpy())
    test_real.append(snapshot.y.detach().cpu().numpy())
    cost = cost + torch.mean((y_hat.squeeze(-1) - snapshot.y.squeeze(-1)) ** 2)
    cha_node = y_hat - snapshot.y
    cha.append(cha_node.detach().cpu().numpy())


cost = cost / (time + 1)
cost = cost.item()
print("Test MSE: {:.4f}".format(cost))
#list to numpy.array
test_real = np.array(test_real)
test_real = test_real.reshape(-1,X_train.shape[1]) #(batch_size,nodes)
print(f'Test_real.shape: {np.array(test_real).shape}')
test_real = Anti_Normalized(test_real,min_max_values)
#np.save('../bin/UsaSample/three_basin/test_real_3_1_0.9',test_real)

test_pre = np.array(test_pre)
test_pre = test_pre.reshape(-1, X_train.shape[1])
print(f'Test_pre.shape: {np.array(test_pre).shape}')
test_pre = Anti_Normalized(test_pre,min_max_values)
#np.save('../bin/UsaSample/three_basin/test_pre_3_1_0.9',test_pre)
R2 = r2_score(test_real.flatten(),test_pre.flatten())
print(f'R2 of test dataset with no offset: {R2}')
MAE = mean_absolute_error(test_real.flatten(),test_pre.flatten())
print(f'Mean absolute error of no offset: {MAE}')
RMSE = mean_squared_error(test_real.flatten(),test_pre.flatten())**0.5
print(f'Root mean squared error of no offset: {RMSE}')

plt.figure(1)
for i in range(test_real.shape[1]):
    plt.subplot(4, 7, 1+i) # 7* 4 = nodes
    plt.plot(test_real[:, i], label='real data')
    plt.plot(test_pre[:, i], label='pre data')
    plt.xlabel("Time")
    plt.ylabel("Normalized Runoff")
    plt.suptitle("Prediction Against Truth")
    plt.legend()
plt.show()

