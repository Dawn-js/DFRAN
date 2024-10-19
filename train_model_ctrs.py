import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision import models
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from avmodel_AF import AVmodel
from opts import AverageMeter
from Dataset_lvxing import get_dataset as MVSA_S
from transformers import BertTokenizer, BertModel

# 加载BERT模型和分词器
model_name = '/data_home/home/sunbo1/Datasets/pretrained_berts/bert_cn'  # 可以根据需要更改为其他BERT模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model_bert = BertModel.from_pretrained(model_name)

class ModelArgs:
    def __init__(self, num_heads, layers, attn_mask, output_dim, attn_dropout, relu_dropout, res_dropout, out_dropout, embed_dropout):
        self.num_heads = num_heads
        self.layers = layers
        self.attn_mask = attn_mask
        self.output_dim = output_dim
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

args_mvsa_single = ModelArgs(num_heads=8, layers=6, attn_mask=True, output_dim=3, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, out_dropout=0.1, embed_dropout=0.15)
args_mvsa_multi = ModelArgs(num_heads=8, layers=6, attn_mask=True, output_dim=3, attn_dropout=0.2, relu_dropout=0.2, res_dropout=0.2, out_dropout=0.2, embed_dropout=0.2)
args_ctrs = ModelArgs(num_heads=8, layers=6, attn_mask=True, output_dim=3, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, out_dropout=0.1, embed_dropout=0.1)

# 假设你在某个地方确定了当前使用的数据集
current_dataset = '11'

if current_dataset == 'args_mvsa_single':
    model_args = args_mvsa_single
elif current_dataset == 'args_mvsa_multi':
    model_args = args_mvsa_multi
else:
    model_args = args_ctrs

device = torch.device("cuda")

# 设置随机种子
seed = 11111
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

# 调用 MultimodalDataLoader 获取数据加载器
batch_size = 16
train_data, test_data = MVSA_S(batch_size=batch_size)
dataset = "ctrs"

if dataset == "mvsa":
    model_lr = 5e-5
    flow_lr = 5e-4
elif dataset == "ctrs":
    model_lr = 1e-5
    flow_lr = 5e-4
elif dataset == "mvsa_multe":
    model_lr = 2e-4
    flow_lr = 5e-4 

epoch = 10

# vit_model = models.vit_b_16(pretrained=True)  # 使用 torchvision 中的 ViT

# CLIP pre-trained model's path
clip_pth = 'ViT-B/16'
if clip_pth in ['ViT-B/16', 'ViT-B/32']:
    dim = 512
elif clip_pth in ['ViT-L/14']:
    dim = 768
#    pad_size = 77
model, _ = clip.load(clip_pth)
clip_model = model.to(device)

fc = nn.Linear(768, 512)
net = AVmodel(model_args).cuda()

XE_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=model_lr, weight_decay=1e-5)  # best 1e-5
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.86)

best_acc1 = -1
best_F1 = -1

def train_(epoch, total_epoch, l1, l2, l3, l4):
    net.train()
    tasks_top1 = AverageMeter()  # 存储output的准确率 
    tasks_losses = AverageMeter()

    for batch_idx, data in enumerate(tqdm(train_data)):
        image, text, emo_label = data[0], data[1], data[2]

        image, emo_label = image.to(device), emo_label.to(device)


        clip_model.eval()
        # 将文本编码为BERT输入格式
        inputs = tokenizer(text, max_length=197, padding='max_length', truncation=True, return_tensors='pt')

        # 使用BERT提取文本特征
        with torch.no_grad():
            outputs = model_bert(**inputs)

        # 提取最后一层隐藏状态
        last_hidden_states = outputs.last_hidden_state
        text_f = fc(torch.as_tensor(last_hidden_states, dtype=torch.float32)).to(device)
        image_f = clip_model.get_image_feature(image)
        image_f = torch.as_tensor(image_f, dtype=torch.float32)

        output, output_T_aux, output_I_aux = net(text_f, image_f, training=True)
        output_all = (output + output_T_aux + output_I_aux)  /3
        emo_res = output_all.max(1)[1]  # emo 预测结果
        cor = emo_res.eq(emo_label).sum().item()

        loss1 = XE_loss(output, emo_label.long())
        loss2 = XE_loss(output_T_aux, emo_label.long())
        loss3 = XE_loss(output_I_aux, emo_label.long())

        loss = l1 * loss1 + l2 * loss2 + l3 * loss3 
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tasks_top1.update(cor * 100 / (emo_label.size(0) + 0.0), emo_label.size(0))
        tasks_losses.update(loss.item(), emo_label.size(0))

    print("Epoch [{}/{}], Loss Avg: {:.4f}, Acc Avg: {:.4f}".format(epoch + 1, total_epoch, tasks_losses.avg, tasks_top1.avg))

def test_(epoch, total_epoch, l1, l2, l3, l4):
    global best_acc1
    global best_F1
    net.eval()
    tasks_top1 = AverageMeter()  # 存储output的准确率
    tasks_losses = AverageMeter()

    f1_macro_values = []
    f1_micro_values = []
    f1_weighted_values = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_data)):
            image,  text, emo_label = data[0], data[1], data[2]

            image,  emo_label = image.to(device), emo_label.to(device)

            clip_model.eval()

            # 将文本编码为BERT输入格式
            inputs = tokenizer(text, max_length=197, padding='max_length', truncation=True, return_tensors='pt')

            # 使用BERT提取文本特征
            with torch.no_grad():
                outputs = model_bert(**inputs)

            # 提取最后一层隐藏状态
            last_hidden_states = outputs.last_hidden_state
            text_f = fc(torch.as_tensor(last_hidden_states, dtype=torch.float32)).to(device)
            image_f = clip_model.get_image_feature(image)
            image_f = torch.as_tensor(image_f, dtype=torch.float32)

                    
            output, output_T_aux, output_I_aux = net(text_f, image_f)
            output_all = (output + output_T_aux + output_I_aux)/3
            emo_res = output_all.max(1)[1]  # emo 预测结果
            cor = emo_res.eq(emo_label).sum().item()

            loss1 = XE_loss(output, emo_label.long())
            loss2 = XE_loss(output_T_aux, emo_label.long())
            loss3 = XE_loss(output_I_aux, emo_label.long())

            loss = l1 * loss1 + l2 * loss2 + l3 * loss3

            tasks_top1.update(cor * 100 / (emo_label.size(0) + 0.0), emo_label.size(0))
            tasks_losses.update(loss.item(), emo_label.size(0))
            # 在计算F1值之前将预测结果和标签移动到CPU上
            emo_res_cpu = emo_res.cpu().numpy()
            emo_label_cpu = emo_label.cpu().numpy()

            # 计算 f1_score
            F1_macro = f1_score(emo_label_cpu, emo_res_cpu, average='macro')
            F1_micro = f1_score(emo_label_cpu, emo_res_cpu, average='micro')
            F1_weighted = f1_score(emo_label_cpu, emo_res_cpu, average='weighted')

            f1_macro_values.append(F1_macro)
            f1_micro_values.append(F1_micro)
            f1_weighted_values.append(F1_weighted)

            overall_f1_macro = sum(f1_macro_values) / len(f1_macro_values)
            overall_f1_micro = sum(f1_micro_values) / len(f1_micro_values)
            overall_f1_weighted = sum(f1_weighted_values) / len(f1_weighted_values)

        print("Epoch [{}/{}], Loss Avg: {:.4f}, Acc Avg: {:.4f}, F1 (Macro): {:.4f}, F1 (Micro): {:.4f}, F1 (Weighted): {:.4f}".format(
                epoch + 1, total_epoch, tasks_losses.avg, tasks_top1.avg, overall_f1_macro, overall_f1_micro, overall_f1_weighted))


    acc = tasks_top1.avg  # 当前的准确率
    
    if acc > best_acc1:  # 如果当前准确率超过最好准确率
        best_acc1 = acc  # 更新最好准确率
        best_F1 = overall_f1_micro
        print("当前最好的准确率为：{:.4f},F1值为:{:.4f}".format(best_acc1, best_F1))
    else:
        print("当前最好的准确率为：{:.4f},F1值为:{:.4f}".format(best_acc1, best_F1))


    return best_acc1, tasks_losses

for epoch_ in range(epoch):
    l1, l2, l3, l4 = 0.3, 0.3, 0.3, 0.1
    train_(epoch_, epoch, l1, l2, l3, l4)
    test_acc, current_loss = test_(epoch_, epoch, l1, l2, l3, l4)
    print("第%d个epoch的学习率:%f" % (epoch_ + 1, optimizer.param_groups[0]['lr']))
    # lr_scheduler.step()