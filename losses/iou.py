import torch
from tqdm import tqdm


def mIoU(dataloader_val, network: torch.nn.Module, device, num_classes):
    with torch.no_grad():
        network.eval()
        network = network.to(device)
        total_gt = torch.zeros(num_classes - 1, dtype=torch.int).to(device)
        correct = torch.zeros(num_classes - 1, dtype=torch.int).to(device)
        total_pred = torch.zeros(num_classes - 1, dtype=torch.int).to(device)
        for images, labels in tqdm(dataloader_val, desc='Calculate mIoU'):
            images = images.to(device)
            labels = labels.to(device)
            prediction = network(images)
            prediction_index = torch.argmax(prediction, dim=1)
            # shift labels by 1 to use 0 as an incorrect flag
            labels += 1
            prediction_index += 1
            correct_mask = torch.where(labels == prediction_index, 1, 0)
            correct_labels = correct_mask * labels
            correct += torch.bincount(correct_labels.view(-1), minlength=num_classes)[1:]
            total_pred += torch.bincount(prediction_index.view(-1), minlength=num_classes)[1:]
            total_gt += torch.bincount(labels.view(-1), minlength=num_classes)[1:num_classes]
        return torch.mean(correct / (total_gt + total_pred - correct))
