import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import math
import os
import torch.nn.functional as F

# embedding_dimension =1
# max_sequence_length = 1024
# PE = torch.zeros(max_sequence_length, embedding_dimension)
# position = torch.arange(0, max_sequence_length).unsqueeze(1)  # shape(max_sequence_len, 1) 转为二维，方便后面直接相乘
# position = position.float()
# buff = torch.pow(1 / 10000, 2 * torch.arange(0, embedding_dimension / 2)/ embedding_dimension)  # embedding_dimension/2
# PE[:, ::2] = torch.sin(position * buff)
# PE[:, 1::2] = torch.cos(position * buff)
# plt.figure()
# plt.plot(PE.numpy())
# plt.show()

'''
input: shape(batch_size, max_sequence_length)
output: shape(batch_size, max_sequence_length, embedding_dimension)
'''

# max_sequence_length
# encode_length
# decode_length
'''上述三个参数对应最大序列长度  encoder的positional encoding的长度 decoder的position的长度'''

# def positional_encoding(x, max_sequence_length):
#     '''
#     :param x:
#     :param max_sequence_length:
#     :return:
#     位置编码的格式为
#     PE(pos, 2i) = sin(pos/10000^(2i/d_model))
#     PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#     d_model = embedding_dimension =1
#     PE的维度为【seq最大长度， 编码维数】
#     '''
#     PE = torch.zeros(max_sequence_length, embedding_dimension)
#     position = torch.arange(0, max_sequence_length).unsqueeze(1) #shape(max_sequence_len, 1) 转为二维，方便后面直接相乘
#     position = position.float()
#     buff = torch.pow(1 / 10000, 2*torch.arange(0, embedding_dimension/2)/embedding_dimension)  # embedding_dimension/2
#     PE[:, ::2] = torch.sin(position * buff)
#     PE[:, 1::2] = torch.cos(position * buff)
#     return PE
#矩阵乘积也就是不带bias的nn.Linear
# nn.Linear(embedding_dimension, embedding_dimension, bias=False)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    '''
    self-attention
    '''

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, q, k, v):
        '''
        q, k, v: shape(batch_size, n_heads, sequence_length, embedding_dimension)
        attr_mask: shape(batch_size, n_heads, sequence_length, sequence_length)
        '''
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(1))
        # k为四维张量，不能用转置 k.transpose(-1, -2)
        score = torch.softmax(score, dim=-1)
        return torch.matmul(score, v)

    def padding_mask(k, q):
        '''
        用于attention计算softmax前，对pad位置进行mask，使得softmax时该位置=0
        k, q: shape(batch_size, sequence_lenth)
        '''
        batch_size, seq_k = k.size()
        batch_size, seq_q = q.size()
        # P = 0
        mask = k.data.eq(0).unsqueeze(1)  # shape(batch_size, 1, sequence_length)
        return mask.expand(batch_size, seq_k, seq_q)

class Multi_head_attention(nn.Module):
    def __init__(self, embedding_dimension,n_heads):
        super(Multi_head_attention, self).__init__()
        self.n_heads = n_heads
        self.embedding_dimension = embedding_dimension
        self.w_q = nn.Linear(embedding_dimension, embedding_dimension*n_heads, bias=False)
        self.w_k = nn.Linear(embedding_dimension, embedding_dimension*n_heads, bias=False)
        self.w_v = nn.Linear(embedding_dimension, embedding_dimension*n_heads, bias=False)
        self.fc = nn.Linear(embedding_dimension*n_heads, embedding_dimension, bias=False)
        self.LayerNorm = nn.LayerNorm(self.embedding_dimension)

    def forward(self, attr_q, attr_k, attr_v):
        '''
        attr_q, attr_k, attr_v: shape(batch_size, sequence_length, embedding_dim)
        attr_mask: shape(batch_size, sequence_length, sequence_length)

        q, k, v: shape(batch_size, n_heads, sequence_length, embedding_dim)
        attr_mask expend : shape(shape(batch_size, n_heads, seq_len, seq_len)

        context : shape(batch_size, n_heads, sequence_length, embedding_dim)
        context reshape: shape(batch_size, sequence_length, n_heads*embedding_dim)
        context fc: shape(batch_size, sequence_length, embedding_dim)

                ## https://zhuanlan.zhihu.com/p/130883313
                MultiHead(Q, K, V) = Concat(head_1, ..., head-h)
                    where head_i = Attention(QW_qi, KW_ki, VW_vi)

        '''
        batch_size = attr_q.shape[0]

        attr_q = attr_q.to(device)
        attr_k = attr_k.to(device)
        attr_v = attr_v.to(device)
        q = self.w_q(attr_q).view(attr_q.size(0), -1, self.n_heads, self.embedding_dimension).transpose(1, 2)
        k = self.w_k(attr_k).view(attr_k.size(0), -1, self.n_heads, self.embedding_dimension).transpose(1, 2)
        v = self.w_v(attr_v).view(attr_v.size(0), -1, self.n_heads, self.embedding_dimension).transpose(1, 2)

        context = Attention()(q, k, v)

        #3.3.3 节输出为以上部分context

        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads*self.embedding_dimension)
        context = self.fc(context)
        out = self.LayerNorm(context + attr_q)


        # return nn.LayerNorm(self.embedding_dimension)(context + attr_q) #残差+layernorm
        return out #残差+layernorm


class feedforward(nn.Module):
    def __init__(self,embedding_dimension, feature_dimension):
        '''
        构造两层linear，先升维，再降回原来维度，达到特征提取的效果
        '''
        super(feedforward, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.fc = nn.Sequential(
            nn.Linear(embedding_dimension, feature_dimension, bias=False),
            nn.ReLU(),
            nn.Linear(feature_dimension, embedding_dimension, bias=False)
        )
        self.LayerNorm = nn.LayerNorm(self.embedding_dimension)
    def forward(self, x):
        output = self.fc(x)
        output = self.LayerNorm(output+x)
        # return nn.LayerNorm(self.embedding_dimension)(output+x)  #残差+layernorm
        return output


class Encode_layer(nn.Module):
    def __init__(self,embedding_dimension):
        super(Encode_layer, self).__init__()
        self.attention = Multi_head_attention(embedding_dimension,n_heads=1)
        self.fc = feedforward(embedding_dimension, feature_dimension=1)
    def forward(self, encode_inputs, mask):
        encode_output = self.attention(encode_inputs,encode_inputs, encode_inputs, mask)
        encode_output = self.fc(encode_output)
        return encode_output


class Encode(nn.Module):
    '''
    1. embedding/ postion encoding
    2. multi-head attention add+layernorm
    3. feed-forward add+layernorm
    '''

    def __init__(self, embedding_dimension, encode_length, n_layers):
        super(Encode, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.encode_length = encode_length
        # self.embedding = torch.reshape([encode_length, embedding_dimension]) #已经是响应 无需编码#

        self.layers = nn.ModuleList([Encode_layer(embedding_dimension) for _ in range(n_layers)])

    def forward(self, encode_inputs):
        # embedding/ postion encoding / pad mask
        # encode_inputs = torch.LongTensor(encode_inputs.numpy())
        encode_inputs = torch.FloatTensor(encode_inputs.cpu().numpy())
        x = encode_inputs.reshape([encode_inputs.shape[0],self.encode_length, self.embedding_dimension])
        # 近似代替编码 将seq转化为【batch_size, seq_length, embedding_dim】

        # x = encode_inputs
        pe = self.positional_encoding(x, self.encode_length)
        pad_mask = self.padding_mask(encode_inputs, encode_inputs)
        pe = torch.reshape(pe,[pe.shape[0],pe.shape[1],1])
        x = x + pe

        # multi-head attention add+layernorm
        # feed-forward add+layernorm

        for layer in self.layers:
            x = layer(x, pad_mask)
        return x  # encode 输出

    def positional_encoding(self, x, max_sequence_length):
        PEBatch = torch.zeros(x.shape[0], max_sequence_length)
        PE = torch.zeros(max_sequence_length, self.embedding_dimension)
        position = torch.arange(0, max_sequence_length).unsqueeze(1)  # shape(max_sequence_len, 1) 转为二维，方便后面直接相乘
        position = position.float()
        buff = torch.pow(1 / 10000, 2 * torch.arange(0,
                                                     self.embedding_dimension / 2) / self.embedding_dimension)  # embedding_dimension/2
        PE[:, ::2] = torch.sin(position * buff)
        PE[:, 1::2] = torch.cos(position * buff)


        for i in range(0, x.shape[0]):
            PE = PE.reshape([max_sequence_length])
            PEBatch[i,:] = PE

        return PEBatch

    def padding_mask(self, k, q):
        '''
        用于attention计算softmax前，对pad位置进行mask，使得softmax时该位置=0
        k, q: shape(batch_size, sequence_lenth)
        '''
        batch_size, seq_k = k.size()
        batch_size, seq_q = q.size()
        # <pad> = 0
        mask = k.data.eq(0).unsqueeze(1)  # shape(batch_size, 1, sequence_length)
        return mask.expand(batch_size, seq_k, seq_q)



def subsequence_mask(v):
    '''
    用于对输入句子进行mask，屏蔽未来时刻的单词内容
    采用上移1的上三角矩阵
    '''
    batch_size, seq_v = v.size()
    mask = np.triu(np.ones((batch_size, seq_v, seq_v)), k=1)
    return torch.from_numpy(mask).byte()

class Decode_layer(nn.Module):
    def __init__(self,embedding_dimension):
        super(Decode_layer, self).__init__()
        self.attention1 = Multi_head_attention(embedding_dimension, n_heads=1)
        self.attention2 = Multi_head_attention(embedding_dimension, n_heads=1)
        self.fc = feedforward(embedding_dimension, feature_dimension=1)
    def forward(self, encode_output, decode_inputs, dec_mask, dec_enc_mask):
        decode_output = self.attention1(decode_inputs,decode_inputs, decode_inputs, dec_mask)
        decode_output = self.attention2(decode_output, encode_output, encode_output, dec_enc_mask)
        decode_output = self.fc(decode_output)
        return decode_output


class Decode(nn.Module):
    def __init__(self, embedding_dimension, decode_length, n_layers):
        super(Decode, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.decode_length =decode_length
        # self.embedding = nn.Embedding(len(vocab_decode), embedding_dimension)
        self.layers = nn.ModuleList([Decode_layer(embedding_dimension) for _ in range(n_layers)])

    def forward(self, encode_inputs, decode_inputs, encode_outputs):
        # embedding/ postion encoding / pad mask
        decode_inputs = torch.FloatTensor(decode_inputs.cpu().numpy())
        # x = self.embedding(decode_inputs)
        x = decode_inputs.reshape([decode_inputs.shape[0],self.decode_length, self.embedding_dimension])
        # x = decode_inputs
        pe = self.positional_encoding(x, self.decode_length)
        pad_mask = self.padding_mask(decode_inputs, decode_inputs)
        subsequence_mask = self.subsequence_mask(decode_inputs)
        dec_mask = torch.gt((pad_mask + subsequence_mask), 0)
        dec_enc_mask = self.padding_mask(decode_inputs, encode_inputs)
        pe = torch.reshape(pe,[pe.shape[0],pe.shape[1],1])

        x = x + pe

        # multi-head attention add+layernorm
        # feed-forward add+layernorm
        for layer in self.layers:
            x = layer(encode_outputs, x, dec_mask, dec_enc_mask)
        return x  # encode 输出

    def positional_encoding(self, x, max_sequence_length):
        PEBatch = torch.zeros(x.shape[0], max_sequence_length)
        PE = torch.zeros(max_sequence_length, self.embedding_dimension)
        position = torch.arange(0, max_sequence_length).unsqueeze(1)  # shape(max_sequence_len, 1) 转为二维，方便后面直接相乘
        position = position.float()
        buff = torch.pow(1 / 10000, 2 * torch.arange(0,
                                                     self.embedding_dimension / 2) / self.embedding_dimension)  # embedding_dimension/2
        PE[:, ::2] = torch.sin(position * buff)
        PE[:, 1::2] = torch.cos(position * buff)

        for i in range(0, x.shape[0]):
            PE = PE.reshape([max_sequence_length])
            PEBatch[i,:] = PE

        return PEBatch
        # return PE


    def padding_mask(self, q, k):
        '''
        用于attention计算softmax前，对pad位置进行mask，使得softmax时该位置=0
        k, q: shape(batch_size, sequence_lenth)
        '''
        batch_size, seq_k = k.size()
        batch_size, seq_q = q.size()
        # <pad> = 0
        mask = k.data.eq(0).unsqueeze(1)  # shape(batch_size, 1, sequence_length)
        return mask.expand(batch_size, seq_q, seq_k)


    def subsequence_mask(self, v):
        '''
        用于对输入句子进行mask，屏蔽未来时刻的单词内容
        采用上移1的上三角矩阵
        '''
        batch_size, seq_v = v.size()
        mask = np.triu(np.ones((batch_size, seq_v, seq_v)), k=1)
        return torch.from_numpy(mask).byte()


class Transformer(nn.Module):
    def __init__(self, embedding_dimension, encode_length, decode_length):
        super(Transformer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.encode_length = encode_length
        self.decode_length = decode_length
        self.encode = Encode(embedding_dimension=embedding_dimension, encode_length=encode_length, n_layers=1)
        self.decode = Decode(embedding_dimension=embedding_dimension,decode_length=decode_length, n_layers=1)
        # self.linear = nn.Linear(embedding_dimension, len(vocab_decode))
        self.linear = nn.Linear(embedding_dimension, 1)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encode(enc_inputs)
        dec_outputs = self.decode(enc_inputs, dec_inputs, enc_outputs)
        output = self.linear(dec_outputs)
        return output

class Unet(nn.Module):
    def __init__(self, LayerNumber, NumberofFeatureChannel, Fs, T):
        super(Unet,self).__init__()
        # print('unet')
        # nlayers = LayerNumber
        # nefilters=NumberofFeatureChannel  ### 每次迭代时特征增加数量###
        self.num_layers = LayerNumber
        self.nefilters = NumberofFeatureChannel
        self.Fs = Fs
        self.T = T
        filter_size = 21
        merge_filter_size = 21
        self.encoder = nn.ModuleList()  ### 定义一个空的modulelist命名为encoder###
        self.decoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dbatch = nn.ModuleList()
        echannelin = [1] + [(i + 1) * self.nefilters for i in range(self.num_layers-1)]
        echannelout = [(i + 1) * self.nefilters for i in range(self.num_layers)]
        dchannelout = echannelout[::-1]
        dchannelin = [dchannelout[0]*2]+[(i) * self.nefilters + (i - 1) * self.nefilters for i in range(self.num_layers,1,-1)]

        for i in range(self.num_layers):
            self.encoder.append(nn.Conv1d(echannelin[i],echannelout[i],filter_size,padding=filter_size//2))
            self.decoder.append(nn.Conv1d(dchannelin[i],dchannelout[i],merge_filter_size,padding=merge_filter_size//2))
            self.ebatch.append(nn.BatchNorm1d(echannelout[i]))  #  moduleList 的append对象是添加一个层到module中
            self.dbatch.append(nn.BatchNorm1d(dchannelout[i]))

        self.middle = nn.Sequential(
            nn.Conv1d(echannelout[-1],echannelout[-1],filter_size,padding=filter_size//2), # //双斜杠取整
            nn.BatchNorm1d(echannelout[-1]),
            # nn.LeakyReLU(0.1)
            nn.Tanh()
        )

        #################################################################################################
        ## attention 的ht generation
        seq_len = echannelout[-1]; ## encoder output feature dimension
        embedding_dimension = int ((self.T*self.Fs)/(self.nefilters**self.num_layers))  ## time dimension feature

        convFeatureDim = echannelout[-1] ## 更像是 embedding dimension
        timeFeature_Len = int ((self.T*self.Fs)/(self.nefilters**self.num_layers))  ## time dimension feature 更像是seq length

        self.self_attention = Multi_head_attention(timeFeature_Len, n_heads=6)
        ##################################################################################################

        #################################################################################################
        ## LSTM 的ht  generation
        LSTMFeature = int((Fs * T) / 2 ** LayerNumber)
        self.middleLSTM = nn.LSTM(input_size = LSTMFeature,hidden_size=LSTMFeature,batch_first=True)

        ######################################################################################################

        self.out = nn.Sequential(
            nn.Conv1d(self.nefilters+2, self.nefilters+2, filter_size,padding=filter_size//2),
            # nn.Tanh()
            nn.LeakyReLU(0.1)
        )
        self.inputFeature1 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=7, padding=7 // 2),
            nn.LeakyReLU(0.1)
        )
        self.inputFeature2 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=3 // 2),
            nn.LeakyReLU(0.1)
        )

        self.smooth = nn.Sequential(
            nn.Conv1d(self.nefilters+1, 1, kernel_size=filter_size, padding=filter_size//2,),
            nn.LeakyReLU(0.1)
        )
    def forward(self,x):
        encoder = list()
        input = x     #x.shape[batch_size, 1-channel, 10240-T*Fs-->feature]

        for i in range(self.num_layers):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            x = F.leaky_relu(x,0.1)
            encoder.append(x)
            x = x[:,:,::2]

        ##############################################################
        ## attention 中间层的ht generation

        #last layer encoder size[batch_size, last layer encoder out size(echannelout[-1]), featureNum( T*Fs/(self.nefilters^self.num_layers) )]

        # x = x.transpose(2, 1)
        x = self.self_attention(x, x, x)
        # x= x.transpose(2,1)


        ########################################################

        ########################################################
        ## LSMT 中间层的ht generation

        # h0=torch.full(
        #     [1, x.shape[0], int((self.Fs * self.T) / 2 ** self.num_layers)], 0);
        # c0=torch.full([1, x.shape[0], int((self.Fs * self.T) / 2 ** self.num_layers)], 0)
        # h0=h0.to(device);c0 =c0.to(device)
        # self.middleLSTM.flatten_parameters()
        # x,(h0, c0) = self.middleLSTM(x,(h0, c0))

        #######################################################################


        for i in range(self.num_layers):
            # x = F.upsample(x,scale_factor=2,mode='linear')
            x = F.interpolate(x, scale_factor = 2, mode='linear')
            x = torch.cat([x,encoder[self.num_layers - i - 1]],dim=1)  ##特征合并过程中维数不对####
            x = self.decoder[i](x)
            x = self.dbatch[i](x)
            x = F.leaky_relu(x,0.1)

        # inputfeature1 = self.inputFeature1(input)
        # inputfeature2 = self.inputFeature2(input)
        # x = torch.cat([x, inputfeature1, inputfeature2], dim=1)
        # x = self.out(x)

        x = torch.cat([x, input],dim=1)
        x = self.smooth(x)

        return x