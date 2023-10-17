import os, csv, sys, json, time, random, pickle, argparse, scipy
import config_file
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
import scipy.sparse as sp
import torch
import clip
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import torch.nn.init as init
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch_geometric.data import Data
from models.layers_vol1 import StructuralAttentionLayer, TemporalAttentionLayer
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

import networkx as nx
import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')


parser = argparse.ArgumentParser()
parser.description = "ini"
parser.add_argument("-t", "--task", type=str, default="weibo")
parser.add_argument("-g", "--gpu_id", type=str, default="1")
parser.add_argument("-c", "--config_name", type=str, default="single3.json")
parser.add_argument("-T", "--thread_name", type=str, default="Thread-1")
parser.add_argument("-d", "--description", type=str, default="exp_description")
args = parser.parse_args()


def process_config(config):
    for k,v in config.items():
        config[k] = v[0]
    return config

class DySAT(nn.Module):
    def __init__(self,config,num_features,time_steps):
        super().__init__()
        self.config = config
        self.num_features = num_features
        self.time_steps = time_steps

        self.structural_head_config = [16,8,8]
        self.structural_layer_config = [512]
        self.temporal_head_config = [16]
        self.temporal_layer_config = [512]

        self.spatial_drop = config['spatial_drop'] #0.1
        self.temporal_drop = config['temporal_drop'] #0.5
        self.residual = config['residual'] #True
        self.num_time_steps = config['time_steps_weibo'] #6

        self.structural_attn, self.temporal_attn = self.build_model()

    def build_model(self):
        input_dim = self.num_features
        # print(self.num_features) #8650

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.residual)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]
        
        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        return structural_attention_layers, temporal_attention_layers
    
    def forward(self, graphs):

        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps):
            # print(graphs[t])
            structural_out.append(self.structural_attn(graphs[t]))
        structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = structural_outputs[-1].shape[0]
        out_dim = structural_outputs[-1].shape[-1]
        structural_outputs_padded = []
        for out in structural_outputs:
            zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, zero_padding), dim=0)
            structural_outputs_padded.append(padded)
        structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        
        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_padded)
        # print(temporal_out.shape)

        # print(self.)

        return temporal_out

class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att


    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output

class NerualNetwork(nn.Module):
    def __init__(self):
        super(NerualNetwork,self).__init__()
        self.init_clip_max_norm = None
        self.device = device
        self.best_acc = 0
        

    def mdegcn(self,x_tid,y,loss,i,total):
        self.optimizer.zero_grad()
        logit_origin = self.forward(x_tid)
        loss_classofication = loss(logit_origin, y)
        # loss_mse = nn.MSELoss()
        # loss_dis = loss_mse(dist[0],dist[1])

        loss_defense =  loss_classofication

        loss_defense.backward()

        self.optimizer.step()
        corrects = (torch.max(logit_origin, 1)[1].view(y.size()).data == y.data).sum()
        accuracy = 100 * corrects / len(y)
        print('Batch[{}/{}] - loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,loss_defense.item(), accuracy, corrects,y.size(0)))

    def fit(self,X_train_tid,y_train,X_dev_tid,y_dev):
        if torch.cuda.is_available():
            self.cuda()
        batch_size = self.config['batch_size']
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-3, weight_decay=config['weight_decay'])

        X_train_tid = torch.LongTensor(X_train_tid)
        # X_train = torch.LongTensor(X_train)
        y_train = torch.LongTensor(y_train)
        # print(X_train.shape,X_train) #X_train.shape==1412
        # print(y_train.shape,y_train)

        dataset = TensorDataset(X_train_tid, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        loss = nn.CrossEntropyLoss()
        params = [(name, param) for name, param in self.named_parameters()]
        print('Number of parameters:',len(params)) 

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch + 1, "/", self.config['epochs'])
            self.train()
            avg_loss = 0
            avg_acc = 0
            for i, data in enumerate(dataloader):
                total = len(dataloader) #1412/64=22.0625,total=23
                batch_x_tid, batch_y = (item.cuda(device=self.device) for item in data)
                # for name, param in self.named_parameters():
                #     print(name, param)
                # print(batch_x_tid,batch_x_text,batch_y)
                # print(batch_x_tid.shape,batch_x_text.shape,batch_y.shape) #[64],[64,768],[64]
                # sys.exit()
                self.mdegcn(batch_x_tid, batch_y, loss, i, total)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)
            self.evaluate(X_dev_tid, y_dev)

    def evaluate(self,X_dev_tid,y_dev):
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        y_pred = []
        X_dev_tid = torch.LongTensor(X_dev_tid).cuda()

       

        dataset = TensorDataset(X_dev_tid)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid = data[0]
                logits = self.forward(batch_x_tid)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        acc = accuracy_score(y_dev, y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y_dev, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("saved model at ",self.config['save_path'])

    def predict(self,config,X_test_tid):
        model = MDEGCN(config)
        model.load_state_dict(torch.load(self.config['save_path']))
        model.to(device)
        model.eval()
        
        y_pred = []
        X_test_tid = torch.LongTensor(X_test_tid).cuda()
        #X_test = torch.LongTensor(X_test).cuda()
       

        dataset = TensorDataset(X_test_tid)
        dataloader = DataLoader(dataset, batch_size=50)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_x_tid = data[0]
                # print(X_test_tid[i])
                logits = model(batch_x_tid)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred


class MDEGCN(NerualNetwork):
    def __init__(self,config):
        super(MDEGCN,self).__init__()
        self.config = config
        self.newid2imgnum = config['newid2imgnum']
        self.newid2text = config['newid2text']
        dropout_rate = config['dropout']
        self.graphs = config['graphs']
        self.adjs = config['adjs']
        self.feats = config['feats']
        self.feats = [torch.tensor(feat.toarray()).float() for feat in self.feats]
        self.pyg_graphs = self._build_pyg_graphs()
        self.pyg_graphs = [g.to(device) for g in self.pyg_graphs]

        self.attention = TransformerBlock(input_size=512, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1)
        self.dysat = DySAT(config,config['feats'][0].shape[1],config['time_steps_weibo']).to(device) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_t = nn.Linear(512, 300)
        self.fc_v = nn.Linear(512, 300)
        self.fc_g = nn.Linear(512, 300)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, config['num_classes'])

        self.init_weight()


    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc_t.weight)
        init.xavier_normal_(self.fc_v.weight)
        init.xavier_normal_(self.fc_g.weight)


    def _build_pyg_graphs(self):
        pyg_graphs = []
        for feat, adj in zip(self.feats, self.adjs):
            x = feat
            # expand_rows = self.num_nodes - x.size(0)
            # zero_padding = torch.zeros(expand_rows, x.size(1))
            # x = torch.cat((x, zero_padding), dim=0).T
            edge_index, edge_weight = torch_geometric.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        # print("I'm here!!!")
        # print(pyg_graphs)
        return pyg_graphs

    def clip_model(self,image):
        self.path = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_images/weibo_images_all/'
        textual_features = []
        visual_features = []
        for newid in image.cpu().numpy():
            imgnum = self.newid2imgnum[newid]
            textnum = self.newid2text[newid]
            imgpath = self.path + imgnum + '.jpg'
            image = preprocess(Image.open(imgpath)).unsqueeze(0).to(device)
            # image = Image.open(imgpath)
            with torch.no_grad():
                textual_feature = clip_model.encode_text(clip.tokenize(textnum).to(device)) # 将文本进行编码
                image_feature = clip_model.encode_image(image) # 将图片进行编码
                textual_feature /= textual_feature.norm(dim=-1, keepdim=True)   
                image_feature /= image_feature.norm(dim=-1, keepdim=True) 
                textual_features.append(textual_feature)
                visual_features.append(image_feature) # 将图片进行编码

        textual_features = torch.stack(textual_features)
        textual_features = textual_features.squeeze(1).cuda()
        visual_features = torch.stack(visual_features)
        visual_features = visual_features.squeeze(1).cuda()
        return textual_features,visual_features

    def forward(self,X_train_tid):
        textual_feature, visual_feature = self.clip_model(X_train_tid)
        all_graph_feature = self.dysat(self.pyg_graphs)
        graph_feature = torch.sum(all_graph_feature, dim=1) / all_graph_feature.size(1)
        graph_feature = graph_feature[X_train_tid]

        textual_feature = textual_feature.to(torch.float32)
        visual_feature = visual_feature.to(torch.float32)

        bsz = textual_feature.size()[0]
        
        self_att_t = self.attention(textual_feature.view(bsz, -1, 512), textual_feature.view(bsz, -1, 512), \
                                      textual_feature.view(bsz, -1, 512))
        self_att_v = self.attention(visual_feature.view(bsz, -1, 512), visual_feature.view(bsz, -1, 512), \
                                        visual_feature.view(bsz, -1, 512))
        self_att_g = self.attention(graph_feature.view(bsz, -1, 512), graph_feature.view(bsz, -1, 512), \
                                        graph_feature.view(bsz, -1, 512))
        
        co_att_tg = self.attention(self_att_g,self_att_t,self_att_t).view(bsz, 512)
        co_att_gt = self.attention(self_att_t,self_att_g,self_att_g).view(bsz, 512)
        co_att_vg = self.attention(self_att_g,self_att_v,self_att_v).view(bsz, 512)
        co_att_gv = self.attention(self_att_v,self_att_g,self_att_g).view(bsz, 512)

        # align_t = self.align_t(textual_feature)
        # # align_v = self.align_v(co_att_gt)
        # align_g = self.align_g(graph_feature)
        # dist = [align_t,align_g]

        att_feature = torch.cat((co_att_tg,co_att_gt,co_att_vg,co_att_gv), dim=1)
        a1 = self.relu(self.fc1(att_feature))
        a2 = self.relu(self.fc2(a1))
        a3 = self.fc3(a2)
        logit_origin = self.fc4(a3)

        return logit_origin


def load_data():
    pre = os.path.dirname(os.getcwd()) + '/dataset/weibo/weibo_files/'
    [X_train_tid, X_train, y_train] = pickle.load(open(pre + "/train.pkl", 'rb'))
    [X_dev_tid, X_dev, y_dev] = pickle.load(open(pre + "/dev.pkl", 'rb'))
    [X_test_tid, X_test, y_test] = pickle.load(open(pre + "/test.pkl", 'rb'))



    config['node_embedding'] = pickle.load(open(pre + "/node_embedding.pkl", 'rb'))[0]
    print(config['node_embedding'].shape)

    with open(pre+"/graphs_new.pkl", "rb") as f:
        graphs = pickle.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    adjs = [nx.adjacency_matrix(g) for g in graphs]

    #节点特征
    config['feats'] = [graph.graph['feature'] for graph in graphs]
    #print(config['feats'][0]) #768
    config['graphs'] = graphs
    config['adjs'] = adjs

    with open(pre+ '/node_id_new.json', 'r') as f:
        newid2mid = json.load(f)
        newid2mid = dict(zip(newid2mid.values(), newid2mid.keys()))
    mid2num = {}
    for file in os.listdir(os.path.dirname(os.getcwd())+'/dataset/weibo/weibocontentwithimage/original-microblog/'):
        mid2num[file.split('_')[-2]] = file.split('_')[0]
    newid2num = {}
    for id in X_train_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_dev_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    for id in X_test_tid:
        newid2num[id] = mid2num[newid2mid[id]]
    config['newid2imgnum'] = newid2num

    newid2text = {}
    for i in range(len(X_train)):
        newid2text[X_train_tid[i]] = X_train[i]
    for i in range(len(X_dev)):
        newid2text[X_dev_tid[i]] = X_dev[i]
    for i in range(len(X_test)):
        newid2text[X_test_tid[i]] = X_test[i]
    config['newid2text'] = newid2text

    return X_train_tid, y_train, \
           X_dev_tid, y_dev, \
           X_test_tid, y_test



def train_and_test(model):
    model_suffix = model.__name__.lower().strip("text")
    res_dir = 'exp_result'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.task)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = os.path.join(res_dir, args.description)
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)
    #
    dir_path = os.path.join('exp_result', args.task, args.description)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    X_train_tid, y_train, \
    X_dev_tid, y_dev, \
    X_test_tid, y_test= load_data()

    # X_train = torch.stack([torch.Tensor(ele) for ele in X_train]).squeeze()
    # X_dev = torch.stack([torch.Tensor(ele) for ele in X_dev]).squeeze()
    # X_test = torch.stack([torch.Tensor(ele) for ele in X_test]).squeeze()

    
    print('MDEGCN Instantiating')
    nn = model(config)

    print('MDEGCN Training')
    nn.fit(X_train_tid, y_train,X_dev_tid, y_dev)
    print('MDEGCN Testing')
    y_pred = nn.predict(config,X_test_tid)
    # print(y_pred)
    # print(y_test)
    # print(X_test_tid)
    num = []
    # words = []
    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            num.append(X_test_tid[i])
    print(num)

    res = classification_report(y_test, y_pred, target_names=config['target_names'], digits=3, output_dict=True)
    for k, v in res.items():
        print(k, v)
    print("result:{:.4f}".format(res['accuracy']))
    res2={}
    res_final = {}
    res_final.update(res)
    res_final.update(res2)
    print(res)
    return res

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子


config = process_config(config_file.config)
seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
model = MDEGCN
train_and_test(model)





print('Runing Time: ', time.time()-start_time)
