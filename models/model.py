import matplotlib.pyplot as plt
from torch.nn import functional as F
import torchmetrics
import torch
from torchvision import transforms
import pytorch_lightning as pl
from enum import Enum
import torch.nn as nn
from land_ml.models.resnet50_fpn import Net as resnet
from torchvision.models import vgg16, vgg19
from land_ml.models.VGGs import VGG
# from land_ml.models.models_ribal import vgg19
from torchvision.models.vgg import VGG19_Weights, VGG16_Weights
#from land_ml.models.models_ribal import PyTorchVGG16 as Net

# pip install "ray[tune]"
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# config = {
#  "lr": tune.loguniform(1e-4, 1e-1),
#  "batch_size": tune.choice([32, 64, 128])
# }
def modify_model_cam(model):
    for param in model.parameters():
        param.requires_grad = False

    #modify the last two convolutions
    model.features[-5] = nn.Conv2d(512,512,3, padding=1)
    model.features[-3] = nn.Conv2d(512,2,3, padding=1)

    #remove fully connected layer and replace it with AdaptiveAvePooling
    model.classifier = nn.Sequential(
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.LogSoftmax()
                                    )

def get_model(model_name, pretrained, output_class, dropout):
        if model_name == "ResNet":
            return resnet(output_class, pretrained)
        elif model_name == "ResNetK3":
            return resnet(output_class, pretrained, reduced_k_size=True)
        else: #VGG
            if model_name == "VGG16":
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
                # vgg = VGG(16)
            elif model_name == "VGG19":
                vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1, dropout=dropout)
                # vgg = VGG(19)
                # features = vgg.features
                # print(type(features))
                # print(features)
                # modify_model_cam(vgg)
            # Modify the classifier for single-class classification
            num_features = vgg.classifier[-1].in_features
            vgg.classifier[-1] = nn.Linear(num_features, output_class)
            return vgg
        
class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class Model(pl.LightningModule):  # ResnetFPN
    def __init__(
        self,
        model_type,
        focus_loss,
        test_threshold,
        batch_transform,
        lr,
        num_epochs,
        steps_per_epoch,
        pretrained,
        output_class=1,
        weight_decay=0,
        dropout = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()  # usefull for loading from checkpoint, the init params will be preserved
        self.model_name = model_type
        self.model = get_model(model_type, pretrained, output_class, dropout)
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        if focus_loss:
            self.loss = WeightedFocalLoss()
        else:
            self.loss = F.binary_cross_entropy_with_logits

        # train metrics
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_f1_score = torchmetrics.classification.BinaryF1Score()
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")

        # val metrics
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_f1_score = torchmetrics.classification.BinaryF1Score()
        self.val_auroc = torchmetrics.AUROC(task="binary")

        # test metrics
        # test_threshold = 0.44
        self.test_accuracy = torchmetrics.classification.BinaryAccuracy(threshold=test_threshold)
        self.test_precision = torchmetrics.Precision(task="binary", threshold=test_threshold)
        self.test_recall = torchmetrics.Recall(task="binary", threshold=test_threshold)
        self.test_f1_score = torchmetrics.classification.BinaryF1Score(threshold=test_threshold)
        self.test_pr_curve = torchmetrics.PrecisionRecallCurve(task="binary")
        self.test_auroc = torchmetrics.AUROC(task="binary")

        # TODO: what is this?
        if batch_transform:
            self.__batch_transform = transforms.RandomChoice(
                transforms=[
                    transforms.Resize((100, 100)),
                    transforms.Resize((50, 50)),
                    transforms.Resize((1000, 1000)),
                ]
            )
        else:
            self.__batch_transform = lambda x: x

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # print(f"training step {batch_idx}")
        x, y = batch
        y_pred_logits = self.model(self.__batch_transform(x))
        loss = self.loss(y_pred_logits, y.unsqueeze(dim=1).float())
        y_pred = torch.sigmoid(y_pred_logits)
        outputs = {"loss": loss, "preds": y_pred, "target": y}

        # update metrics
        self.train_accuracy.update(outputs["preds"], outputs["target"].unsqueeze(dim=1).float())
        self.train_precision.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.train_recall.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.train_f1_score.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        return outputs

    # def training_step_end(self, outputs):
    #     # update and log
    #     # Metric auto accumalate ref. https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
    #     # print(f"train_preds: {outputs['preds'].squeeze()}, targets: {outputs['target']}")
    #     # print("training step end")
    #     # print(outputs['preds'])
    #     # print(outputs["target"])
    #     # print(outputs['preds'].device)
    #     self._log(outputs=outputs, stage="train", epoch_step="step")
    #     self.train_accuracy.reset()
    #     self.train_precision.reset()
    #     self.train_recall.reset()
    #     self.train_f1_score.reset()

    def training_epoch_end(self, outputs) -> None:
        self._log(outputs=outputs, stage="train", epoch_step="epoch")
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1_score.reset()

    def evaluate(self, batch):
        x, y = batch
        y_pred_logits = self.model(x)
        loss = self.loss(y_pred_logits, y.unsqueeze(dim=1).float())
        y_pred = torch.sigmoid(y_pred_logits)
        return {"loss": loss, "preds": y_pred, "target": y}

    def validation_step(self, batch, batch_idx):
        outputs = self.evaluate(batch=batch)
        self.val_accuracy.update(outputs["preds"], outputs["target"].unsqueeze(dim=1).float())
        # print(outputs["preds"].squeeze())
        # print(outputs["target"])
        # print(self.val_f1_score(outputs["preds"].squeeze(), outputs["target"]))
        self.val_f1_score.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.val_precision.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.val_recall.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.val_auroc.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        return outputs

    # def validation_step_end(self, outputs):
    # print("valid")
    # print(outputs['preds'])
    # print(outputs["target"])
    # print(outputs['preds'].device)
    # self._log(outputs=outputs, stage="val", epoch_step="step")
    # self.val_accuracy.reset()
    # self.val_f1_score.reset()
    # self.val_precision.reset()
    # self.val_recall.reset()

    def validation_epoch_end(self, outputs) -> None:
        self._log(outputs=outputs, stage="val", epoch_step="epoch")
        self.val_accuracy.reset()
        self.val_f1_score.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        outputs = self.evaluate(batch=batch)
        self.test_accuracy.update(outputs["preds"], outputs["target"].unsqueeze(dim=1).float())
        self.test_f1_score.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.test_precision.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.test_recall.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.test_pr_curve.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        self.test_auroc.update(outputs["preds"], outputs["target"].unsqueeze(dim=1))
        return outputs

    def test_epoch_end(self, outputs) -> None:
        self._log(outputs=outputs, stage="test", epoch_step="epoch")
        precision, recall, _ = self.test_pr_curve.compute()

        fig, ax = plt.subplots()
        ax.plot(recall.cpu(), precision.cpu())
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        plt.show()

        confmat = torchmetrics.ConfusionMatrix(task="binary", num_classes=2, threshold=0.44).cpu()
        for it in outputs:
            confmat.update(it["preds"].cpu(), it["target"].cpu().unsqueeze(dim=1))

        self.test_accuracy.reset()
        self.test_f1_score.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_auroc.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    # def forward(self, x):
    #     return torch.sigmoid(self(x))

    def _log(self, outputs, stage, epoch_step):
        if epoch_step == "epoch":
            all_loss = sum([o["loss"] for o in outputs]) / float(len(outputs))
        else:
            all_loss = outputs["loss"]

        # calculate
        acc = getattr(self, f"{stage}_accuracy").compute()
        precision = getattr(self, f"{stage}_precision").compute()
        recall = getattr(self, f"{stage}_recall").compute()
        f1_score = getattr(self, f"{stage}_f1_score").compute()

        # log
        self.log(f"{stage}_loss_{epoch_step}", all_loss, sync_dist=True)
        self.log(f"{stage}_accuracy_{epoch_step}", acc, sync_dist=True)
        self.log(f"{stage}_precision_{epoch_step}", precision, sync_dist=True)
        self.log(f"{stage}_recall_{epoch_step}", recall, sync_dist=True)
        self.log(f"{stage}_f1_{epoch_step}", f1_score, sync_dist=True)

        # val metrics only
        if stage in ("val", "test"):
            auroc = getattr(self, f"{stage}_auroc").compute()
            self.log(f"{stage}_auroc_{epoch_step}", auroc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config["LR"], max_lr=config["LR"]*10.0, cycle_momentum=False) # Error with pytorch 1.13 when autolog, can't pikle
        # TODO: fix total_steps: You must either provide a value for total_steps or provide a value for both
        # epochs and steps_per_epoch.
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr,
        #     total_steps=self.steps_per_epoch*self.num_epochs,
        #     epochs=self.num_epochs,
        #     verbose=True,
        # )
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}

    def get_model_name(self):
        return self.model_name