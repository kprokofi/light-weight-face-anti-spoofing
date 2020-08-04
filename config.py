
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

optimizer = dict(lr=0.05, momentum=0.9, weight_decay=5e-4)
schedular = dict(milestones=[100,150], gamma=0.1)
data = {
  "cuda": True,
  "batch_size": 100,
  "data_loader_workers": 2,
  "pin_memory": True,
  "data_root": '/home/prokofiev/pytorch/LCC_FASD'
}

resize = dict(height=224, width=224)
checkpoint = dict(snapshot_name="my_best_modelMobileNet2_1.5.pth.tar", 
                  experiment_path='/home/prokofiev/pytorch/antispoofing/log_tensorboard/MobileNet_LCFAD_7')
loss = dict(amsoftmax = 'amsoftmax', crossentropy='crossentropy')
epochs = dict(start_epoch=0, max_epoch=200)
