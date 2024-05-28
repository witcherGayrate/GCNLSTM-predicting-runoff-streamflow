# GCNLSTM-predicting-runoff-streamflow
基于pytorch搭建的卷积长短时间记忆网络（**GCN-LSTM**）、（双向）长短时间记忆网络(**biLSTM/LSTM**)、门控循环单元(**GRU**)、循环神经网络(**RNN**)、多层感知机(**MLP**)

模型与数据均部署在CPU上，如需部署在GPU上只需改动少量代码

输入特征：历史径流、降雨、温度、水汽压、日照时长、雪水当量、短波辐射通量

输出特征：未来n天径流值

水文站数据属于美国的某片流域，28个水文站点

注意：应用不同的神经网络需要更改网络的输入大小input_size以及模型的输入参数model(arg1,arg2,arg3..)

其中，图卷积神经网络的权重矩阵edge_weight可以自定义，以及邻接矩阵edge_index根据水文站连接情况自定义

