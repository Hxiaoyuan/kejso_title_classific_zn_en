from abc import ABC

from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kejso_title_classification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p, label_ignore_idx=0,
                 head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        # self.linear_2 = nn.Linear(hidden_size, 1)
        self.classification = nn.Linear(hidden_size, n_labels)

        self.label_ignore_idx = label_ignore_idx

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)

        self.device = device

        # 初始化分类器
        self.classification.weight.data.normal_(mean=0.0, std=head_init_range)

    def forward(self, inputs_ids, labels, valid_mask):
        transformer_out, _ = self.model(inputs_ids, features_only=True)

        out_1 = F.relu(self.linear_1(transformer_out))

        out_1 = torch.mean(out_1, dim=1)  # 句向量

        out_1 = self.dropout(out_1)

        logits = self.classification(out_1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            loss = loss_fct(logits.view(-1, self.n_labels), labels.view(-1))
            return loss
        else:
            return logits

    def encode_word(self, s):
        # 待改为新的方法可以  训练
        # tensor_cn_ids = self.xlmr.encode(title_cn)
        tensor_ids = self.xlmr.encode(s)
        if self.is_chinese(s):
            return tensor_ids.cpu().numpy().tolist()[2:-1]
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]
    # 中英文的判断
    def is_chinese(self, s):
        for word in s:
            if '\u4e00' <= word <= '\u9fff':
                return True
        return False
