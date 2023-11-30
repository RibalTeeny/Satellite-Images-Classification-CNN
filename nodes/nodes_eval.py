from pytorch_lightning import Trainer
import mlflow
import logging
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
# from torchvision.models import VGG
from land_ml.utils.image_processor import ImageProcessor
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

log = logging.getLogger(__name__)

# YAML Keys
TRAINER = "evaluate_trainer"
MLFLOW_URL = "mlflow_url"
EXPERIMENT = "experiment_name"


def get_model(run_id: str, params: dict):
    run_id = "373ab074ec8145df9a7c49a489c2ac0d" # ResNet
    # run_id = "a160cb800d62473188f94e5aa3175a3b" # VGG19
    mlflow.set_tracking_uri(params[MLFLOW_URL])
    mlflow.set_experiment(experiment_name=params[EXPERIMENT])
    log.debug("Loading best model from run: %s" %run_id)
    model = mlflow.pytorch.load_model("runs:/%s/best-model" %run_id)
    print(type(model))
    return model

def evaluate_model(test_loader, model, params: dict):
    test_trainer = Trainer(**params[TRAINER])
    result = test_trainer.test(model, test_loader)

    return result[0]

def get_filenames(directory):
    paths = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            paths.append(os.path.join(path, name))
    return paths

def show_data(data):
    print(data)

def clear_reporting():
    import shutil
    folder = '/flat6/Incubator/workspaces/ribal/land-ml/data/08_reporting'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path) and file_path.endswith(".png"):
                os.unlink(file_path)
            # elif os.path.isdir(file_path):
            #     shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def CAM(run_id, params):
    clear_reporting()
    run_id="373ab074ec8145df9a7c49a489c2ac0d"
    CATS = ["suspicious_site"]
    path = "runs:/%s/best-model" %run_id
    #STATE_DICT_PATH = "mlflow-artifacts:/1/28ae54e7eb5a4643aa602303b0c94ad4/artifacts/best-model"
    model = 'land_ml.models.resnet50_fpn'
    # model = 'land_ml.models.Model'
    ip = ImageProcessor(CATS, path, model=model, mlflow_url=params[MLFLOW_URL], mlflow_exp_name=params[EXPERIMENT])
    dir = "/flat6/Incubator/workspaces/ribal/land-ml/data/05_model_input/binary_classification/test/"
    negatives = get_filenames(dir+"0/")
    positives = get_filenames(dir+"1/")
    fn = 0
    fp = 0
    for neg in negatives:
        # img_path = "0/53_2022-01-01_2022-12-30/visual_53_2022-01-01_2022-12-30.png"
        ip.set_gpu(0)
        iw = ip.execute_cams_pred(neg)
        print(iw.classification_scores, iw.predicted_categories)
        if iw.classification_scores[0] > 0.44:
            fp+=1
            iw.show_global_cams("FP"+neg.split("/")[-1])
    for pos in positives:
        # img_path = "0/53_2022-01-01_2022-12-30/visual_53_2022-01-01_2022-12-30.png"
        ip.set_gpu(0)
        iw = ip.execute_cams_pred(pos)
        print(iw.classification_scores, iw.predicted_categories)
        if iw.classification_scores[0] < 0.44 :
            fn+=1
            iw.show_global_cams("FN"+pos.split("/")[-1])
    print("len of positives:", len(positives))
    print("false positives:", fp)
    print("len of negatives:", len(negatives))
    print("false negatives:", fn)

def cam(test_loader, model):
    clear_reporting()
    save_dir = '/flat6/Incubator/workspaces/ribal/land-ml/data/08_reporting/'
    eval_model = model.eval()
    
    cam_extractor = SmoothGradCAMpp(eval_model.model, "smooth1")
    for i, (img, label) in enumerate(test_loader):
        out = eval_model(img)
        label = label.item()
        pred = 0 if out < 0.44 else 1
        activation_map = cam_extractor(class_idx=0)
        img = img[0]
        print(activation_map)
        save_img(activation_map, img, save_dir+"%d_pred%d_00%d.png"%(label, pred, i))

def cam2(test_loader, model):
    dir = "/flat6/Incubator/workspaces/ribal/land-ml/data/05_model_input/np_new/test"
    clear_reporting()

    eval_model = model.eval()
    cam_extractor = SmoothGradCAMpp(eval_model)

    save_dir = '/flat6/Incubator/workspaces/ribal/land-ml/data/08_reporting/'
    
    negatives = get_filenames(dir+"/0")
    positives = get_filenames(dir+"/1")
    print(negatives)
    fn = 0
    fp = 0
    for neg in negatives:
        out, img = get_output(eval_model, neg)
        print(out.data[0])
        score = out.data[0]
        if score > 0.5:
            fp+=1
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            save_img(activation_map, img, save_dir+"FP"+neg.split("/")[-1])
    for pos in positives:
        out, img = get_output(eval_model, pos)
        print(out.data[0])
        score = out.data[0]
        if score < 0.5 :
            fn+=1
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            save_img(activation_map, img, save_dir+"FP"+neg.split("/")[-1])
    print("len of positives:", len(positives))
    print("false positives:", fp)
    print("len of negatives:", len(negatives))
    print("false negatives:", fn)
    

    return activation_map, img

def read_file(path):
    with open(path, "rb") as f:
        # Convert array to Image
        arr = np.load(f, allow_pickle=True) # data is BGR
    arr = arr[...,::-1].copy() # RGB
    arr = np.moveaxis(arr, 0, -1) # (3, 151, 150) -> (151, 150, 3)
    arr = arr.astype("float32")
    assert arr.shape[2]==3, "Image %s does not contain 3 channels" %path
    return arr

def get_output(eval_model, img_path):
    # Get your input
    img = read_file(img_path)
    # Preprocess it for your chosen model
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Preprocess your data and feed it to the model
    out = eval_model(input_tensor.unsqueeze(0))

    return out, img

def save_img(activation_map, img, fname):

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    plt.figimage(result); plt.axis('off'); plt.tight_layout(); plt.savefig(fname)

# def cam(model):
#     #freeze layers
#     for param in model.parameters():
#         param.requires_grad = False

#     #I got better results when I changed the last two convolutions
#     #which is why you see features[-5]
#     model.features[-5] = nn.Conv2d(512,512,3, padding=1)
#     model.features[-3] = nn.Conv2d(512,2,3, padding=1)

#     #remove fully connected layer and replace it with AdaptiveAvePooling
#     model.classifier = nn.Sequential(
#                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(),
#                                 nn.LogSoftmax()
#                                 )

# def get_params_weights(model):
    
#     params = list(model.parameters())
#     # weight = np.squeeze(params[-1].data.numpy())

#     return params

# def return_CAM(feature_conv, weight, class_idx):
#     # generate the class -activation maps upsample to 256x256
#     size_upsample = (256, 256)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     for idx in class_idx:
#         print(idx)
#         print(weight)
#         beforeDot =  feature_conv.reshape((nc, h*w))
#         cam = np.matmul(weight[idx], beforeDot)
#         cam = cam.reshape(h, w)
#         cam = cam - np.min(cam)
#         cam_img = cam / np.max(cam)
#         cam_img = np.uint8(255 * cam_img)
#         output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam

# def viz(test_loader, model, params):
#     resnet50 = model.model
#     mod = nn.Sequential(*list(resnet50.children())[:-1])
#     normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225]
#     )
#     preprocess = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     normalize
#     ])

#     predicted_labels=[]
    
#     # for i, info in enumerate(test_loader.dataset.imgs[1]):
#     info = test_loader.dataset.imgs[1]
#     fname = info[0]
#     img_pil = Image.open(fname)
#     img_tensor = preprocess(img_pil)
#     img_variable = Variable(img_tensor.unsqueeze(0))

#     logit = resnet50(img_variable)

    
#     h_x = F.softmax(logit, dim=1).data.squeeze()

#     probs, idx = h_x.sort(0, True)
#     probs = probs.detach().numpy()
#     idx = idx.numpy()

#     print("idx", idx)
    

#     # predicted_labels.append(idx[0])
#     predicted =  test_loader.dataset.classes[idx]
    
#     print("Target: " + fname + " | Predicted: " +  predicted)

#     weight = np.squeeze(params[-1].data.numpy())

#     print(type(mod))
#     features_blobs = mod(img_variable)
#     print("B")
#     features_blobs1 = features_blobs.cpu().detach().numpy()
#     CAMs = return_CAM(features_blobs1, weight, [idx])

#     readImg = fname
#     img = cv2.imread(readImg)
#     height, width, _ = img.shape
#     heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
#     result = heatmap * 0.5 + img * 0.5

#     cv2.imwrite("image_1", result)