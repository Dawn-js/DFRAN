import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score
from avmodel_AF import AVmodel, Flow
from opts import AverageMeter
from dataset import MultimodalDataLoader

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
args_mvsa_multi = ModelArgs(num_heads=8, layers=6, attn_mask=True, output_dim=3, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, out_dropout=0.1, embed_dropout=0.1)
args_ctrs = ModelArgs(num_heads=8, layers=6, attn_mask=True, output_dim=3, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, out_dropout=0.1, embed_dropout=0)

# 假设你在某个地方确定了当前使用的数据集
current_dataset = 'args_mvsa_single'

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

dataset = "mvsa"  # mvsa, ctros, mvsa_multe
# 数据文件路径
train_data_path = f"/data_home/home/sunbo1/A-test/Datasets/{dataset}/train_data.pt"
val_data_path = f"/data_home/home/sunbo1/A-test/Datasets/{dataset}/val_data.pt"
test_data_path = f"/data_home/home/sunbo1/A-test/Datasets/{dataset}/test_data.pt"

# 调用 MultimodalDataLoader 获取数据加载器
batch_size = 16
data_loaders = MultimodalDataLoader(train_data_path, val_data_path, test_data_path, batch_size=batch_size)

# 获取训练、验证、测试数据加载器
train_loader, _, test_loader = data_loaders.get_loaders()

if dataset == "mvsa":
    model_lr = 2e-4
    flow_lr = 5e-4
elif dataset == "ctrs":
    model_lr = 1e-5
    flow_lr = 5e-4
elif dataset == "mvsa_multe":
    model_lr = 2e-4
    flow_lr = 5e-4 

epoch = 10

fc = nn.Linear(768, 512)
net = AVmodel(model_args).cuda()
FlowNet = Flow().cuda()

XE_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=model_lr, weight_decay=1e-5)  # best 1e-5
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.86)
optimizer_1 = optim.Adam(FlowNet.parameters(), lr=flow_lr)

best_acc1 = -1
best_F1 = -1

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_(epoch, total_epoch, l1, l2, l3, l4):
    net.train()
    FlowNet.train()
    tasks_top1 = AverageMeter()  # 存储output的准确率 
    tasks_losses = AverageMeter()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        image, text, emo_label = data[0], data[1], data[2]
        image, text, emo_label = image.to(device), (fc(text)).to(device), emo_label.to(device)

        _, _, loss_rec_t, loss_rec_i = FlowNet(image, text)

        output, output_T_aux, output_I_aux = net(text, image, training=True)
        output_all = (output + output_T_aux + output_I_aux)  /3
        emo_res = output_all.max(1)[1]  # emo 预测结果
        cor = emo_res.eq(emo_label).sum().item()

        loss1 = XE_loss(output, emo_label.long())
        loss2 = XE_loss(output_T_aux, emo_label.long())
        loss3 = XE_loss(output_I_aux, emo_label.long())

        loss = l1 * loss1 + l2 * loss2 + l3 * loss3 + l4 * (loss_rec_t + loss_rec_i)
        # loss = l1 * loss1 + l2 * loss2 + l3 * loss3
        # Backward and optimize
        optimizer.zero_grad()
        optimizer_1.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_1.step()

        tasks_top1.update(cor * 100 / (emo_label.size(0) + 0.0), emo_label.size(0))
        tasks_losses.update(loss.item(), emo_label.size(0))

    print("Epoch [{}/{}], Loss Avg: {:.4f}, Acc Avg: {:.4f}".format(epoch + 1, total_epoch, tasks_losses.avg, tasks_top1.avg))

def test_(epoch, total_epoch, l1, l2, l3, l4):
    global best_acc1
    global best_F1
    net.eval()
    # FlowNet.eval()
    tasks_top1 = AverageMeter()  # 存储output的准确率
    tasks_losses = AverageMeter()
    all_preds = []
    all_labels = []
    f1_macro_values = []
    f1_micro_values = []
    f1_weighted_values = []
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            image, text, emo_label = data[0], data[1], data[2]
            image, text, emo_label = image.to(device), (fc(text)).to(device), emo_label.to(device)

            image , _, _, _ = FlowNet(image, text)
                    
            output, output_T_aux, output_I_aux = net(text, image)
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

            all_preds.extend(emo_res_cpu)
            all_labels.extend(emo_label_cpu)

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
    lr_scheduler.step()
