import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from fcos import FCOSDetector
from VOC_dataset import VOCDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root_dir = './test_images/'
image_list = [os.path.join(root_dir, fn) for fn in os.listdir(root_dir) if fn.endswith('.jpg')]

ckpt = torch.load("./models/voc2012_512x800_epoch100_loss0.6055.pth",map_location=torch.device('cpu'))
model=FCOSDetector(mode="inference")
model.load_state_dict(ckpt)
model.to(device)
model.eval()

resize = transforms.Resize((512))
to_tensor = transforms.ToTensor()
to_normalize = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))

def run_demo(filename, image, model):
    tensor_img = to_normalize(to_tensor(resize(image)))
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(device)

    image = image.resize((tensor_img.shape[3], tensor_img.shape[2]))
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    start_t=time.time()
    with torch.no_grad():
        scores, classes, boxes = model(tensor_img)
    end_t=time.time()
    cost_t=1000*(end_t-start_t)
    print("===>success processing img, cost time %.2f ms"%cost_t)

    boxes=boxes[0].cpu().numpy().tolist()
    classes=classes[0].cpu().numpy().tolist()
    scores=scores[0].cpu().numpy().tolist()

    for i, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        cv2.rectangle(cv_img,pt1,pt2, (0,255,0), 2)
        cv2.putText(cv_img,"%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]+5),int(box[1])+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)

    cv2.imwrite("./out_images/"+filename,cv_img)
    # cv2.imshow('', cv_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

for f_path in image_list:
    filename = f_path.split('/')[-1]
    image = Image.open(f_path).convert('RGB')
    run_demo(filename, image, model)