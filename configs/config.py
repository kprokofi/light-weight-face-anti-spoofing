dataset = 'celeba-spoof'

datasets = dict(LCCFASD_root='/home/prokofiev/pytorch/LCC_FASD', Celeba_root='/home/prokofiev/pytorch/antispoofing/CelebA_Spoof')

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

optimizer = dict(lr=0.005, momentum=0.9, weight_decay=5e-4)

schedular = dict(milestones=[20,40], gamma=0.2)

data = dict(cuda=True, batch_size=256, data_loader_workers=4, sampler = None, pin_memory=True, data_root='/home/prokofiev/pytorch/antispoofing/CelebA_Spoof')

resize = dict(height=128, width=128)

checkpoint = dict(snapshot_name="MobileNet3_52.pth.tar", 
                  experiment_path='/home/prokofiev/pytorch/antispoofing/log_tensorboard/MobileNet_Celeba_52')

loss = dict(loss_type='amsoftmax', 
            amsoftmax=dict(m=0.5, s=2, margin_type='cos', label_smooth=False, smoothing=0.1, ratio=[1,1]),
            soft_triple=dict(cN=2, K=10, s=5, tau=.2, m=0.35))

epochs = dict(start_epoch=0, max_epoch=80)

model= dict(model_type='Mobilenet3', model_size = 'large', use_amsoftmax=True, pretrained=True, embeding_dim=128)

aug = dict(type_aug=None, alpha=0.5, beta=0.5, cutmix_prob=0.5)

curves = dict(det_curve='det_curve_1.png', roc_curve='roc_curve_1.png')

dropout = dict(prob_dropout=0.2, classifier=0.35, type='gaussian', mu=0.5, sigma=0.3)

data_parallel = dict(use_parallel=False, parallel_params=dict(device_ids=[0,1], output_device=0))

evaulation = True

RSC = dict(use_rsc=True, p=0.333, b=0.333)

test_dataset = dict(type='LCC_FASD')