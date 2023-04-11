import os, sys
sys.path.append("/home/cosmo/Desktop/cse404/")

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from boundingBoxDataset import boundingBoxDataset
from classificationDataset import classificationDataset
from common.utils import *
from common.YTCelebrityDatasetFirstFrame import YTCelebrityDatasetFirstFrame
from common.process import get_bounding_box, estimate_iou, draw_square_by_label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor(pretrained=False):
    feature_extractor = models.alexnet() # weights=models.AlexNet_Weights.DEFAULT)
    feature_extractor.classifier = nn.Sequential(*list(feature_extractor.classifier.children())[:3])

    if pretrained:
        if not os.path.exists('RCNN/checkpoints/feature_extractor.pth'):
            raise FileNotFoundError("No feature extractor checkpoint found.")
        feature_extractor = load_model(feature_extractor, 'RCNN/checkpoints/feature_extractor.pth')
    return feature_extractor

def get_regressor(pretrained=False):
    regressor = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4, bias=True)
    )

    if pretrained:
        if not os.path.exists('RCNN/checkpoints/regression.pth'):
            raise FileNotFoundError("No regressor checkpoint found.")
        regressor = load_model(regressor, 'RCNN/checkpoints/regression.pth')
    return regressor

def get_classifier(pretrained=False):
    classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 2, bias=True)
    )

    if pretrained:
        if not os.path.exists('RCNN/checkpoints/classification.pth'):
            raise FileNotFoundError("No classifier checkpoint found.")
        classifier = load_model(classifier, 'RCNN/checkpoints/classification.pth')
    return classifier

def get_model(model_mode="regression", pretrained=False):
    model_dict = {"regression": get_regressor, "classification": get_classifier}
    model = model_dict[model_mode](pretrained=pretrained)
    return model

def get_data_loaders(json_path, model_mode="regression"):
    dataset_dict = {"regression": boundingBoxDataset, "classification": classificationDataset}

    dataset = dataset_dict[model_mode](json_path)

    generator = torch.Generator().manual_seed(41)
    training, validation, testing = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2], generator=generator)

    train_loader = DataLoader(training, batch_size=8, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(testing, batch_size=8, shuffle=True, num_workers=2)

    return train_loader, validation_loader, test_loader

def get_criteria(model_mode="regression"):
    criteria_dict = {"regression": nn.MSELoss, "classification": nn.CrossEntropyLoss}
    criterion = criteria_dict[model_mode]()
    return criterion

def get_optimizer(model, optim_mode="SGD", **kwargs):
    optimizer_dict = {"SGD": optim.SGD, "Adam": optim.Adam}
    optimizer = optimizer_dict[optim_mode](model.parameters(), **kwargs)
    return optimizer

def train(dataloader, feature_extractor, model, criterion, optimizer,lr_scheduler):
    feature_extractor.train(), model.train()

    train_loss = 0.0

    for image, label in dataloader:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        
        feature = feature_extractor(image)
        output = model(feature).softmax(dim=1)

        loss = criterion(output, label)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item() * image.size(0)

    train_loss /= dataloader.dataset.__len__()
    return train_loss
        

def evaluate(dataloader, feature_extractor, model, criterion):
    feature_extractor.eval(), model.eval()

    eval_loss = 0.0
    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)

            feature = feature_extractor(image)
            output = model(feature).softmax(dim=1)
            pred = output.argmax(dim=1)

            loss = criterion(output, label)
            eval_loss += loss.item() * image.size(0)

    eval_loss /= dataloader.dataset.__len__()
    return eval_loss

def param_init(model):
    for m in model._modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity=nn.Relu)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def trainer(mode='train', model_mode="regression", optim_mode="SGD", **kwargs):
    if mode not in ['train', 'validate', 'test']:
        raise ValueError(f"Selected mode {mode} not in 'train', 'validate', or 'test'")
    if model_mode not in ['regression', 'classification']:
        raise ValueError(f"Selected model mode {model_mode} not in 'regression' or 'classification'")

    train_loader, validation_loader, test_loader = get_data_loaders(json_path, model_mode=model_mode)
    feature_extractor = get_feature_extractor()
    model = get_model(model_mode=model_mode)
    criterion = get_criteria(model_mode=model_mode)
    optimizer = get_optimizer(model, optim_mode, **kwargs)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    if mode == 'train':
        model.apply(param_init)
    
    feature_extractor.to(device), model.to(device)

    if mode == 'test':
        eval_loss = evaluate(test_loader, feature_extractor, model, criterion)
        print(f"Test loss: {eval_loss:.5f} ")
        return

    train_loss_track = []
    eval_loss_track = []

    pbar = tqdm(total=epoch, desc=f'Training {model_mode}', position=0)
    for e in range(epoch):
        train_loss = train(train_loader, feature_extractor, model, criterion, optimizer, lr_scheduler)
        eval_loss = 0
        if mode == 'validate':
            eval_loss = evaluate(validation_loader, feature_extractor, model, criterion)
        
        train_loss_track.append(train_loss)
        eval_loss_track.append(eval_loss)
        pbar.update(1)
        pbar.write(f"Epoch: {e}  Train loss: {train_loss:.5f} Validation loss: {eval_loss:.5f}")
        pbar.refresh()
    pbar.close()

    if not os.path.exists('RCNN/checkpoints'):
        os.makedirs('RCNN/checkpoints')
    save_model(model, f'RCNN/checkpoints/{model_mode}.pth')
    save_model(feature_extractor, f'RCNN/checkpoints/feature_extractor.pth')

    if not os.path.exists('RCNN/results'):
        os.makedirs('RCNN/results')
    
    fig, axs = plt.subplots(1)
    fig.suptitle(f'{model_mode}')
    axs.set_title('Loss')
    axs.plot(train_loss_track, label="Train")
    if mode == 'validate':
        axs.plot(eval_loss_track, label="Validation")
    axs.legend()
    plt.savefig(f'RCNN/results/{model_mode}.png')

def inference_single_image(image, feature_extractor, classification_model, regression_model):
    score, bounding_label = [], []

    image = image.permute(1,2,0).numpy()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])

    for rect in tqdm(selective_search(image), desc="Classifing regions", leave=None):
        region = get_region(image, rect)
        region = np.uint8(region * 255)
        region = transform(region).to(device)
        feature = feature_extractor(region[None, :])
        classification_output = classification_model(feature)
        probility = torch.softmax(classification_output, dim=1)
        if probility[0, 1] > 0.5:
            score.append(probility[0, 1].item())
            bounding_label.append(rect)
    
    keep = nms(bounding_label, score)

    new_score, new_bounding_label = [], []
    for i in keep:
        new_score.append(score[i])
        new_bounding_label.append(bounding_label[i])
    score, bounding_label = new_score, new_bounding_label

    ## for box in tqdm(bounding_label, desc="Regressing bounding boxes", leave=None):
    ##     region = get_region(image, box)
    ##     region = np.uint8(region * 255)
    ##     region = transform(region).to(device)
    ##     feature = feature_extractor(region[None, :])
    ##     diff = regression_model(feature)
    ##     for i in range(4):
    ##         box[i] += diff[0, i].item()

    region = get_region(image, bounding_label[0])
    region = np.uint8(region * 255)
    region = transform(region).to(device)
    feature = feature_extractor(region[None, :])
    diff = regression_model(feature)
    for i in range(4):
        bounding_label[0][i] += diff[0, i].item()

    predict = bounding_label[0]
    return predict


def inference(dataloader, pretrained = True):
    feature_extractor = get_feature_extractor(pretrained=pretrained)
    regression_model = get_model(model_mode="regression", pretrained=pretrained)
    classification_model = get_model(model_mode="classification", pretrained=pretrained)
    
    feature_extractor.to(device), regression_model.to(device), classification_model.to(device)
    feature_extractor.eval(), regression_model.eval(), classification_model.eval()

    iou = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Classifing images"):
            for image, label in zip(images, labels):
                predict = inference_single_image(image, feature_extractor, classification_model, regression_model)
                iou_score = estimate_iou(get_bounding_box(*label), get_bounding_box(*predict))
                iou += iou_score
             
        iou /= dataloader.dataset.__len__()
        print(f"Average IOU: {iou:.5f}")

    return iou

json_path = "/home/cosmo/Desktop/cse404/data.json"
epoch = 20

if __name__ == '__main__':
    # trainer(mode='validate', model_mode="regression", optim_mode="SGD", lr=1e-3, momentum=0.1)
    # trainer(mode='validate', model_mode="classification", optim_mode="Adam", lr=1e-3)
    dataset = YTCelebrityDatasetFirstFrame(json_path)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    # inference(dataloader, pretrained=True)

    # for i in [7, 503, 504, 505]:
    #     image = dataset[i][0]
        
    #     feature_extractor = get_feature_extractor(pretrained=True).to(device)
    #     regression_model = get_model(model_mode="regression", pretrained=True).to(device)
    #     classification_model = get_model(model_mode="classification", pretrained=True).to(device)

    #     predict = inference_single_image(image, feature_extractor, classification_model, regression_model)
    #     image = image.permute(1,2,0).numpy()
    #     image = np.uint8(image * 255)

    #     image = draw_square_by_label(image, *predict)
    #     cv2.imshow("image", image)
    #     cv2.waitKey(0)
        