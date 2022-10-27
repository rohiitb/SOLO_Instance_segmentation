
from model import *

# Credit to SOLO Author's code
# This function does NMS on the heat map (category_prediction), grid-level
# Input:
#     heat: (batch_size, C-1, S, S)
# Output:
#     (batch_size, C-1, S, S)
def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

# This function performs Matrix NMS
# Input:
#     sorted_masks: (n_active, image_h/4, image_w/4)
#     sorted_scores: (n_active,)
# Output:
#     decay_scores: (n_active,)
def MatrixNMS(sorted_masks, sorted_scores, method='gauss', gauss_sigma=0.5):
    n = len(sorted_scores)
    sorted_masks = sorted_masks.reshape(n, -1)
    intersection = torch.mm(sorted_masks, sorted_masks.T)
    areas = sorted_masks.sum(dim=1).expand(n, n)
    union = areas + areas.T - intersection
    ious = (intersection / union).triu(diagonal=1)

    ious_cmax = ious.max(0)[0].expand(n, n).T
    if method == 'gauss':
        decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
    else:
        decay = (1 - ious) / (1 - ious_cmax)
    decay = decay.min(dim=0)[0]
    return sorted_scores * decay


def visualize_nms_image(img_orig, ff_msks, classes,thresh=0.2, transp=0.2):
  manipulated_mask = torch.where(ff_msks>thresh, 1, 0)
  classes = torch.tensor([0,2,2,1,0])
  edit_im = img_orig.detach().cpu().numpy().copy()

  for x in range(len(manipulated_mask[0])):
    negative = np.where(manipulated_mask[0][x]==1, 0, 1)
    positive = np.where(manipulated_mask[0][x]==1, 255., 0)
    edit_im = img_orig.detach().cpu().numpy().copy()
    c = classes[x]
    edit_im[c] = edit_im[c]*negative + positive*(1-transp) + edit_im[c]*transp
    im = edit_im.squeeze().transpose(1,2,0)
    im = np.clip(im, 0, 1)
  plt.figure(figsize = (20,20))
  plt.imshow(im)

if __name__ == "__main__":
    model.eval()
    model.to(device)
    count = 0
    temp_flag = True
    for i, batch_set in enumerate(testloader):

        img_set  = batch_set[0]
        lab_set  = batch_set[1]
        mask_set = batch_set[2]
        bbox_set = batch_set[3]
        cat_tar, msk_tar, act_msk = model.generate_targets(bbox_set, lab_set, mask_set)

        cat_pred, msk_pred = model.forward(img_set, eval=True)

        for i in range(batch_size):
            img_raw = img_set[i].squeeze(0)
            nms_ip = [points_nms(cat_pred[j][i].unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1) for j in range(5)]
            cat_ip = [each_cat_level.squeeze(0) for each_cat_level in nms_ip]
            msk_ip = [msk_pred[j][i] for j in range(5)]
            fin_msk, fin_cls = model.post_process(cat_ip, msk_ip)
            visualize_nms_image(img_raw, fin_msk, fin_cls,thresh=0.2, transp=0.2)
            count += 1
            if count == 6:
                temp_flag = False
                break
        if temp_flag == False:
            break  