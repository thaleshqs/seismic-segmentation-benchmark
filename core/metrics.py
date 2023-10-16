import torch


class RunningMetrics(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = torch.zeros((n_classes, n_classes))
    

    def __compute_matrix(self, pred_label, true_label, n_classes):
        mask = (true_label >= 0) & (true_label < n_classes)
        hist = torch.bincount(
            n_classes*true_label[mask].to(int) + pred_label[mask], minlength=n_classes*n_classes
        ).reshape(n_classes, n_classes)
        
        return hist.cpu()
    

    def update(self, images, targets):
        pred_labels = images.detach().max(dim=1)[1]
        true_labels = targets.detach()

        for p_label, t_label in zip(pred_labels, true_labels):
            self.confusion_matrix += self.__compute_matrix(p_label.flatten(), t_label.flatten(), self.n_classes)


    def get_scores(self):
        '''
        Computes and returns the following metrics:

            - Pixel Accuracy
            - Class Accuracy
            - Mean Class Accuracy
            - Mean Intersection Over Union (mIoU)
            - Frequency Weighted IoU
            - Confusion Matrix
        '''

        hist = self.confusion_matrix

        pixel_accuracy = torch.diag(hist).sum() / hist.sum()

        class_accuracy = torch.diag(hist) / hist.sum(dim=1)
        mean_class_accuracy = torch.nanmean(class_accuracy)

        iou = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
        mean_iou = torch.nanmean(iou)

        frequency = hist.sum(dim=1) / hist.sum() # fraction of the pixels that come from each class
        frequency_weighted_iou = (frequency[frequency > 0] * iou[frequency > 0]).sum()

        return {
            'pixel_accuracy'        : pixel_accuracy.item(),
            'class_accuracy'        : class_accuracy.tolist(),
            'mean_class_accuracy'   : mean_class_accuracy.item(),
            'mean_iou'              : mean_iou.item(),
            'frequency_weighted_iou': frequency_weighted_iou.item(),
            'confusion_matrix'      : self.confusion_matrix.tolist()
        }
    

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))


class EarlyStopper:

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            
            if self.counter >= self.patience:
                return True
            
        return False

