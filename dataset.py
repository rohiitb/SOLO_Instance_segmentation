import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms 
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor


class Datasetbuilder(Dataset):
    def __init__(self, path):
        # Load everything from the path
        self.images = h5py.File(path["images"],'r')['data']
        self.masks = h5py.File(path["masks"],'r')['data']
        self.bboxes = np.load(path["bboxes"], allow_pickle=True)
        self.labels = np.load(path["labels"], allow_pickle=True)
        
        self.corresponding_masks = []
        # Aligning masks with labels
        count = 0
        for i, label in enumerate(self.labels):
            n = label.shape[0] 
            temp = []
            for j in range(n):
                temp.append(self.masks[count])
                count += 1
            self.corresponding_masks.append(temp)

        # Applying rescaling and mean, std, padding
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((800, 1066)), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.resize = torchvision.transforms.Resize((800, 1066))
        self.pad = torch.nn.ZeroPad2d((11,11,0,0))
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):                                  # Returns single data image
        one_label = self.labels[index]
        one_image = self.images[index]
        one_bbox = self.bboxes[index]
        one_mask = self.corresponding_masks[index]

        one_label = torch.tensor(one_label)
        one_image, one_mask, one_bbox = self.pre_process(one_image, one_mask, one_bbox)

        return one_image, one_label, one_mask, one_bbox

    def pre_process(self, img, msks, box):
        img_normalized = img / 255.                     # Normalized between 0 & 1
        img_normalized = torch.tensor(img_normalized, dtype=torch.float)   # Converted to tensor
        img_scaled = self.transform(img_normalized)    # Rescaled to (800, 1066) and adjusted for given mean and std

        img_final = self.pad(img_scaled)              # Padded with zeros to get the shape as (800, 1088)

        msk_final = torch.zeros((len(msks), 800, 1088))           #Initializing mask tensor

        for i, msk in enumerate(msks):
            msk = msk/1.                                  # Converting it to uint8
            msk = torch.tensor(msk, dtype=torch.float).view(1,300,400)         # Converting it to tensor
            msk_scaled = self.pad(self.resize(msk).view(800,1066))            # Padding and resizing
            msk_scaled[msk_scaled < 0.5] = 0
            msk_scaled[msk_scaled > 0.5] = 1
            msk_final[i] = msk_scaled

        box = torch.tensor(box, dtype=torch.float)
        box_final = torch.zeros_like(box)
        box_final[:,0] = box[:,0] * 800/300                  # Scaling x
        box_final[:,2] = box[:,2] * 800/300                  # Scaling x
        box_final[:, 1] = box[:,1] * (1066/400) + 11                # Scaling y
        box_final[:, 3] = box[:,3] * (1066/400) + 11                # Scaling y

        return img_final, msk_final, box_final

def custom_collate_fn(batch):

        images = []
        labels = []
        masks = []
        bboxes = []

        for i in batch:
            images.append(i[0])
            labels.append(i[1])
            masks.append(i[2])
            bboxes.append(i[3])

        return torch.stack(images, dim=0), labels, masks, bboxes

def visualize_raw_processor(img, label, mask, bbox, alpha=0.5):
    processed_mask = mask.clone().detach().squeeze().bool()
    img = img.clone().detach()[:, :, 11:-11]

    inv_transform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[ 0., 0., 0. ], std=[1/0.229, 1/0.224, 1/0.255]), torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
    pad = torch.nn.ZeroPad2d((11,11,0,0))
    processed_img = pad(inv_transform(img))

    processed_img = processed_img.numpy()

    processed_img = np.clip(processed_img, 0, 1)
    processed_img = torch.from_numpy((processed_img * 255.).astype(np.uint8))

    img_to_draw = processed_img.detach().clone()

    if processed_mask.ndim == 2:
        processed_mask = processed_mask[None, :, :]
    for mask, box, lab in zip(processed_mask, bbox, label):
        if lab.item() == 1 :            # vehicle
            colored = 'blue'
            box = box.numpy().astype(np.uint8)
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
        if lab.item() == 2 :            # person
            colored = 'green'
            box = box.numpy().astype(np.uint8)
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
        if lab.item() == 3 :            # animal
            colored = 'red'
            box = box.numpy().astype(np.uint8)
            color = torch.tensor(ImageColor.getrgb(colored), dtype=torch.uint8)
            img_to_draw[:, mask] = color[:, None]
 
    out = (processed_img * (1 - alpha) + img_to_draw * alpha).to(torch.uint8)
    out = draw_bounding_boxes(out, bbox, colors='red', width=2)
    final_img = out.numpy().transpose(1,2,0)
    return final_img


if __name__ == "__main__":
    path = {}
    path["images"] = "./data/hw3_mycocodata_img_comp_zlib.h5" 
    path["labels"] = "./data/hw3_mycocodata_labels_comp_zlib.npy"
    path["bboxes"] = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
    path["masks"] = "./data/hw3_mycocodata_mask_comp_zlib.h5"

    solo = Datasetbuilder(path)

    trainsize = 2600
    testsize = len(solo) - 2600

    torch.random.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(solo, [trainsize, testsize])

    trainloader = DataLoader(train_dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    testloader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    fig, ax = plt.subplots(5, figsize=(10, 10))
    count = 0
    temp_flag = True

    for i, batch_set in enumerate(trainloader):
        img_set = batch_set[0]
        lab_set = batch_set[1]
        mask_set = batch_set[2]
        bbox_set = batch_set[3]
        for single_img, single_lab, single_mask, single_bbox in zip(img_set, lab_set, mask_set, bbox_set):
            final_img = visualize_raw_processor(single_img, single_lab, single_mask, single_bbox)
            print("count 1: ", count)
            ax[count].imshow(final_img)
            count += 1
            print("count 2: ", count)
            if count == 5:
                print("Inside")
                temp_flag = False
                break
        if temp_flag == False:
            break


    
