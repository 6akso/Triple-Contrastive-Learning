import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, base_model, num_classes, method):
        super().__init__()
        self.base_model = base_model #base_model应该是一个包含Transformer编码器的预训练模型，例如BERT、RoBERTa等。
        self.num_classes = num_classes
        self.method = method
        self.linear = nn.Linear(base_model.config.hidden_size, num_classes)
#这段代码定义了一个线性层，用于对输入特征进行分类。线性层是一种全连接层，将输入特征映射到输出特征的过程可以表示为y = xW + b，其中x是输入特征，W是要学习的权重矩阵，b是偏置向量，y是输出特征。
#在这里，输入特征是预训练模型输出的每个序列的第一个隐藏状态，即一个形状为[batch_size, hidden_size]的张量。base_model.config.hidden_size表示预训练模型的隐藏状态的维度，num_classes表示分类的类别数。线性层将输入特征映射到一个num_classes维的向量空间，然后使用softmax函数将这个向量空间中的值归一化，得到输入序列属于各个类别的概率。
#具体来说，这段代码定义了一个nn.Linear对象，它的输入维度为base_model.config.hidden_size，输出维度为num_classes。在模型的前向传播过程中，输入特征会先经过一层Dropout操作，然后再传递给线性层进行计算。线性层会将输入特征映射到一个num_classes维的向量空间，然后使用softmax函数将这个向量空间中的值归一化，得到输入序列属于各个类别的概率。
        self.dropout = nn.Dropout(0.5)
        for param in base_model.parameters():
            param.requires_grad_(True) #这段代码将BERT模型中所有的参数都设置为可训练的（requires_grad=True），以便在模型的训练过程中更新这些参数。

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
#在Python中，使用 ** 表示将一个字典中的键值对展开为多个关键字参数传递给函数。在这段代码中， ** inputs的作用是将一个包含输入张量的字典inputs中的键值对，展开为多个关键字参数，传递给self.base_model(**inputs)函数。
#具体来说，inputs字典中包含了输入文本向量、注意力掩码、文本段落标记等信息。当将其传递给self.base_model(**inputs)函数时， ** inputs会将这些键值对展开为多个关键字参数，分别传递给self.base_model函数，例如：
# input_ids = inputs['input_ids']
# attention_mask = inputs['attention_mask']
# token_type_ids = inputs['token_type_ids']
# 这些关键字参数分别对应BERT模型的输入张量，包括文本序列的ID索引、注意力掩码、文本段落标记等信息。通过这种方式，可以方便地将多个输入张量传递给BERT模型，并进行相应的处理
        hiddens = raw_outputs.last_hidden_state
# bert模型的输出可以包括四个：
# last_hidden_state：torch.FloatTensor类型的，最后一个隐藏层的序列的输出。大小是(batch_size, sequence_length, hidden_size) sequence_length是我们截取的句子的长度，hidden_size是768.
# pooler_output： torch.FloatTensor类型的，[CLS]的这个token的输出，输出的大小是(batch_size, hidden_size)
# hidden_states ：tuple(torch.FloatTensor)这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
# attentions：这也是输出的一个可选项，如果输出，需要指定config.output_attentions=True,它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值
        cls_feats = hiddens[:, 0, :]
#这里的[:, 0, :]是Python中的扩展切片语法（Extended Slicing Syntax），它可以用于多维数组或列表的切片操作。
# 在Python中，如果要对一个一维数组或列表进行切片操作，可以使用slice(start, stop, step)函数，例如a[start:stop:step]。而对于多维数组或列表，扩展切片语法则提供了更灵活的切片操作方式。它可以通过使用Ellipsis（...）、整数索引、切片、省略号等符号，对多维数组或列表进行切片操作。
# 具体来说，[:, 0, :]是一个三元组，它表示对一个三维数组或张量进行切片操作。其中，第一个冒号（:）表示对第一维（即batch_size）进行全局切片，即选取所有批次的数据；第二个0表示对第二维（即sequence_length）选择索引为0的位置，即选取所有序列的第一个位置（即CLS标记）；第三个冒号（:）表示对第三维（即hidden_size）进行全局切片，即选取所有隐藏层的输出。
        if self.method in ['ce', 'scl']:
            label_feats = None
            predicts = self.linear(self.dropout(cls_feats))
        else:
            label_feats = hiddens[:, 1:self.num_classes+1, :]
            predicts = torch.einsum('bd,bcd->bc', cls_feats, label_feats)
#torch.einsum('bd,bcd->bc', cls_feats, label_feats)表示对CLS标记特征和标签特征进行点积，返回一个形状为(batch_size, num_classes)的二维张量
        outputs = {
            'predicts': predicts,
            'cls_feats': cls_feats,
            'label_feats': label_feats
        }
        return outputs
