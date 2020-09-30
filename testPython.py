import torch

from seqeval.metrics import f1_score, classification_report

# a = torch.tensor([[1, 2, 3], [45, 6.1, 6]], dtype=torch.long)
#
# v = torch.argmax(a, dim=1)
#
# print(v)

# y_true = [['O'], ['A'], ['B'], ['C'], ['A']]
# y_pred = [['O'], ['B'], ['B'], ['A'], ['A']]

y_true = [['A'], ['B'], ['C'], ['A']]
y_pred = [['B'], ['B'], ['C'], ['A']]
report = classification_report(y_true, y_pred, digits=4)
print(report)
