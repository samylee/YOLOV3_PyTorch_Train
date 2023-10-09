import torch
from torch.utils.data import DataLoader

from data.dataset import VOCDataset
from models.YOLOV3 import YOLOV3
from utils.loss import YOLOV3Loss
from utils import transforms
from utils.utils import load_darknet_pretrain_weights


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    base_lr = 1e-4
    weight_decay = 0

    lr_steps = [200, 250]
    num_epochs = 280

    batch_size = 32
    B, C = 3, 20
    anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]

    pretrain = 'pretrain/reference.weights'
    train_label_list = 'data/voc0712/train.txt'

    print_freq = 5
    save_freq = 5

    # def model
    yolov3 = YOLOV3(B=B, C=C)
    load_darknet_pretrain_weights(yolov3, pretrain)
    yolov3.to(device)

    # def loss
    criterion = YOLOV3Loss(B, C, anchors=anchors, device=device)

    # def optimizer
    optimizer = torch.optim.Adam(yolov3.parameters(), lr=base_lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps)

    # def dataset
    transforms = transforms.Compose([
        transforms.RandomCrop(jitter=0.3),
        transforms.RandomFlip(prob=0.5),
        transforms.RandomHue(prob=0.5),
        transforms.RandomSaturation(prob=0.5),
        transforms.RandomBrightness(prob=0.5)
    ])
    train_dataset = VOCDataset(train_label_list, transform=transforms, is_train=True, B=B, C=C)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)

    print('Number of training images: ', len(train_dataset))

    # train
    for epoch in range(num_epochs):
        yolov3.train()
        total_loss = 0.0
        for i, (images, targets) in enumerate(train_loader):
            current_lr = get_lr(optimizer)

            images = images.to(device)
            targets = targets.to(device)

            preds = yolov3(images)
            loss = criterion(preds, targets, images.size(2))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print current loss.
            if i % print_freq == 0:
                print('Epoch [%d/%d], Iter [%d/%d], LR: %.6f, Size: %d, Loss: %.4f, Average Loss: %.4f'
                      % (epoch, num_epochs, i, len(train_loader), current_lr, images.shape[2], loss.item(), total_loss / (i+1)))

        lr_scheduler.step()
        if epoch % save_freq == 0:
            torch.save(yolov3.state_dict(), 'weights/yolov3_' + str(epoch) + '.pth')

    torch.save(yolov3.state_dict(), 'weights/yolov3_final.pth')