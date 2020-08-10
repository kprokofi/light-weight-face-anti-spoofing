dataset = 'celeba-spoof'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

optimizer = dict(lr=0.01, momentum=0.9, weight_decay=5e-4)
schedular = dict(milestones=[25,150], gamma=0.1)
data = dict(cuda=True, batch_size=100, data_loader_workers=2, pin_memory=True, data_root='/home/prokofiev/pytorch/antispoofing/CelebA_Spoof')

resize = dict(height=224, width=224)
checkpoint = dict(snapshot_name="MobileNet3_22_1.pth.tar", 
                  experiment_path='/home/prokofiev/pytorch/antispoofing/log_tensorboard/MobileNet_Celeba_22_1')
loss = dict(loss_type='amsoftmax')
epochs = dict(start_epoch=0, max_epoch=50)
amsoftmax = dict(m=0.5, s=5, margin_type='arc', label_smooth=False, smoothing=0.01)
model= dict(model_type='Mobilenet3', use_amsoftmax=True, pretrained=True)
aug = dict(type_aug=None, alpha=0.5)