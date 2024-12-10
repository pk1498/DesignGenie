import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_outputs, teacher_outputs, labels):
        # Hard loss (classification loss)
        hard_loss = F.cross_entropy(student_outputs, labels)

        # Soft loss (distillation loss)
        student_softmax = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_softmax = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_loss = self.kl_div(student_softmax, teacher_softmax) * (self.temperature ** 2)

        # Combine losses
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
