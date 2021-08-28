
import cv2
from fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from VOC_dataset import VOCDataset
import time

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.float32)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output

if __name__=="__main__":
    # class Config():
    #     #backbone
    #     pretrained=False
    #     freeze_stage_1=True
    #     freeze_bn=True

    #     #fpn
    #     fpn_out_channels=256
    #     use_p5=True
        
    #     #head
    #     class_num=80
    #     use_GN_head=True
    #     prior=0.01
    #     add_centerness=True
    #     cnt_on_reg=False

    #     #training
    #     strides=[8,16,32,64,128]
    #     limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #     #inference
    #     score_threshold=0.5
    #     nms_iou_threshold=0.2
    #     max_detection_boxes_num=150

    # model=FCOSDetector(mode="inference",config=Config)
    model=FCOSDetector(mode="inference")
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model.load_state_dict(torch.load("./models/voc2012_512x800_epoch100_loss0.6055.pth",map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model=model.cuda().eval()
    print("===>success loading model")

    import os
    root="./test_images/"
    names=os.listdir(root)
    for name in names:
        img_bgr=cv2.imread(root+name)
        img_pad=preprocess_img(img_bgr,[512,800])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        # img=img_pad.copy()
        img_t=torch.from_numpy(img).float().permute(2,0,1)
        # img1= transforms.Normalize([102.9801, 115.9465, 122.7717],[1.,1.,1.])(img_t)
        img1=transforms.ToTensor()(img1)
        img1= transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225),inplace=True)(img1)
        img1=img1.cuda()
        
        start_t=time.time()
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        print(out)
        scores,classes,boxes=out
        
        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            img_pad=cv2.rectangle(img_pad,pt1,pt2,(0,255,0))
            img_pad=cv2.putText(img_pad,"%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,[0,200,20],2)
        cv2.imwrite("./out_images/"+name,img_pad)