import torch
from sklearn.metrics import f1_score as f1


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def f1_score(output, target, threshold=0.5):
    with torch.no_grad():
        target = target.int()
        prob = torch.sigmoid(output)

        return f1(target.cpu(), prob.cpu() > threshold, average='macro')
