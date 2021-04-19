import torch
from torch import nn
from losses.utils import indices_to_dense_vector, sigmoid_cross_entropy_with_logits


class IouLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0):

        super(SigmoidFocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction
        self._loss_weight = loss_weight

    def forward(self, prediction_tensor, target_tensor, weights=None, class_indices=None ):
        """Compute loss function.

        Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices. If provided, computes loss only for the specified class indices.

        Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes] representing the value of the loss function.
        """
        weights = weights.unsqueeze(2)
        if class_indices is not None:
            weights *= (indices_to_dense_vector(class_indices, prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor))

        per_entry_cross_ent = sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor)
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = (target_tensor * prediction_probabilities) + ((1 - target_tensor) * (1 - prediction_probabilities))
        modulating_factor = 1.0

        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)

        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha)

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)

        return focal_cross_entropy_loss * weights