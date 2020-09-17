'''MIT License

Copyright (C) 2019-2020 Intel Corporation
 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import mixup_target, AverageMeter, load_checkpoint, save_checkpoint, precision, cutmix, make_dataset
import os
from eval_protocol import evaulate

class Trainer:
    def __init__(self, model, criterion, optimizer, args, config, train_loader, val_loader, test_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_step, self.val_step = 0, 0
        self.best_accuracy, self.current_accuracy, self.current_auc, self.current_eer, self.best_acer = 0, 0, 0, float('inf'), float('inf')
        self.multitask = self.config['multi_task_learning']
        self.augmentation = self.config['aug']['type_aug']
        self.experiment_path = self.config['checkpoint']['experiment_path']
        self.snapshot_name = self.config['checkpoint']['snapshot_name']
        self.path_to_checkpoint = os.path.join(self.experiment_path, self.snapshot_name) 
        self.data_parallel = self.config['data_parallel']['use_parallel']
        self.cuda = self.config['data']['cuda']
        self.writer = SummaryWriter(self.experiment_path)

    def train(self, epoch: int):
        ''' method to train your model for epoch '''
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to train mode and train one epoch
        self.model.train()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for i, (input, target) in loop:
            if i == 10:
                break
            if self.cuda:
                input = input.cuda(device=self.args.GPU)
                target = target.cuda(device=self.args.GPU)
            # compute output and loss
            if self.augmentation:
                if self.augmentation == 'mixup':
                    aug_output = mixup_target(input, target, self.config, self.args)
                else:
                    assert self.augmentation == 'cutmix'
                    aug_output = cutmix(input, target, self.config, self.args)
                input, target_a, target_b, lam = aug_output
                tuple_target = (target_a, target_b, lam)
                if self.multitask:
                    hot_target = lam*F.one_hot(target_a[:,0], 2) + (1-lam)*F.one_hot(target_b[:,0], 2)
                else:
                    hot_target = lam*F.one_hot(target_a, 2) + (1-lam)*F.one_hot(target_b, 2)
                output = self.make_output(input, hot_target)
                if self.multitask:
                    loss = self.multi_task_criterion(output, tuple_target)  
                else:
                    loss = self.mixup_criterion(self.criterion, output, target_a, target_b, lam, 2)
            else:
                new_target = F.one_hot(target[:,0], num_classes=2) if self.multitask else F.one_hot(target, num_classes=2)
                output = self.make_output(input, new_target)
                loss = self.multi_task_criterion(output, target) if self.multitask else self.criterion(output, new_target)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure accuracy and record loss
            s = self.config['loss']['amsoftmax']['s']
            acc = precision(output[0], target[:,0].reshape(-1), s) if self.multitask else precision(output, target, s)
            losses.update(loss.item(), input.size(0))
            accuracy.update(acc, input.size(0))

            # write to writer for tensorboard
            self.writer.add_scalar('Train/loss', loss, global_step=self.train_step)
            self.writer.add_scalar('Train/accuracy',  accuracy.avg, global_step=self.train_step)
            self.train_step += 1

            # update progress bar
            max_epochs = self.config['epochs']['max_epoch']
            loop.set_description(f'Epoch [{epoch}/{max_epochs}]')
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg, lr=self.optimizer.param_groups[0]['lr'])
        return losses.avg, accuracy.avg

    def validate(self):
        ''' method to validate model on current epoch '''
        # meters
        losses = AverageMeter()
        accuracy = AverageMeter()
        # switch to evaluation mode and inference the model
        self.model.eval()
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False)
        criterion = self.criterion[0] if self.multitask else self.criterion
        for i, (input, target) in loop:
            if i == 10:
                break
            if self.cuda:
                input = input.cuda(device=self.args.GPU)
                target = target.cuda(device=self.args.GPU)
            if len(target.shape) > 1:
                target = target[:, 0].reshape(-1)
            # computing output and loss
            with torch.no_grad():
                features = self.model(input)
                if self.data_parallel:
                    model1 = self.model.module
                else:
                    model1 = self.model
                
                output = model1.make_logits(features)
                if type(output) == tuple:
                    output = output[0]

                new_target = F.one_hot(target, num_classes=2)  
                loss = criterion(output, new_target)

            # measure accuracy and record loss
            acc = precision(output, target, s=self.config['loss']['amsoftmax']['s'])
            losses.update(loss.item(), input.size(0))
            accuracy.update(acc, input.size(0))

            # update progress bar
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg, acc=acc, avr_acc=accuracy.avg)

        print(f'val accuracy on epoch: {round(accuracy.avg, 3)}, loss on epoch:{round(losses.avg, 3)}')
        # write val in writer
        self.writer.add_scalar('Val/loss', losses.avg, global_step=self.val_step)
        self.writer.add_scalar('Val/accuracy',  accuracy.avg, global_step=self.val_step)
        self.val_step += 1

        return accuracy.avg

    def eval(self, epoch: int, epoch_accuracy: float):
        # evaluate on last 10 epoch and remember best accuracy, AUC, EER, ACER and then save checkpoint
        if (epoch == 0 or epoch >=60) and (epoch_accuracy > self.current_accuracy) and (self.args.save_checkpoint):
            AUC, EER, _ , apcer, bpcer, acer = evaulate(self.model, self.val_loader, self.config, self.args, compute_accuracy=False)
            print(f'__VAL__: epoch: {epoch}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')
            if acer < self.best_acer:
                self.best_acer = acer
                checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                save_checkpoint(checkpoint, f'{self.path_to_checkpoint}')
                self.current_accuracy = epoch_accuracy
                self.current_eer = EER
                self.current_auc = AUC
                AUC, EER, accur, apcer, bpcer, acer, _, _ = evaulate(self.model, self.test_loader, self.config, self.args, compute_accuracy=True) 
                print(f'__TEST__: epoch: {epoch}  accur: {round(np.mean(accur),3)}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')

        # evaluate on val every 10 epoch except last one and save checkpoint
        if (epoch%10 == 0) and (epoch not in (self.config['epochs']['max_epoch'] - 1, 0, 60)) and (self.args.save_checkpoint):
            # printing results
            AUC, EER, accur, apcer, bpcer, acer, _, _ = evaulate(self.model, self.test_loader, self.config, self.args, compute_accuracy=True) 
            print(f'epoch: {epoch}  accur: {round(np.mean(accur),3)}   AUC: {AUC}   EER: {EER}   APCER: {apcer}   BPCER: {bpcer}   ACER: {acer}')
            checkpoint = {'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch':epoch}
            save_checkpoint(checkpoint, f'{self.path_to_checkpoint}')

    def make_output(self, input: torch.tensor, target: torch.tensor):
        ''' target - one hot for main task
        return output 
        If use rsc compute output applying rsc method'''
        assert target.shape[1] == 2
        if self.config['RSC']['use_rsc']:
            # making features before avg pooling
            features = self.model(input)
            if self.data_parallel:
                model1 = self.model.module
            else:
                model1 = self.model
            # do everything after convolutions layers, strating with avg pooling
            all_tasks_output = model1.make_logits(features)
            logits = all_tasks_output[0] if self.multitask else all_tasks_output
            if type(logits) == tuple:
                logits = logits[0]
            # take a derivative, make tensor, shape as features, but gradients insted features
            if self.augmentation:
                fold_target = target.argmax(dim=1)
                target = F.one_hot(fold_target, num_classes=target.shape[1]) 
            target_logits = torch.sum(logits*target, dim=1)
            gradients = torch.autograd.grad(target_logits, features, grad_outputs=torch.ones_like(target_logits), create_graph=True)[0]
            # get value of 1-p quatile
            quantile = torch.tensor(np.quantile(a=gradients.data.cpu().numpy(), q=1-self.config['RSC']['p'], axis=(1,2,3)), device=input.device)
            quantile = quantile.reshape(input.size(0),1,1,1)
            # create mask
            mask = gradients < quantile
            
            # element wise product of features and mask, correction for expectition value 
            new_features = (features*mask)/(1-self.config['RSC']['p'])
            # compute new logits
            new_logits = model1.spoof_task(new_features)
            if type(new_logits) == tuple:
                new_logits = new_logits[0]
            # compute this operation batch wise
            random_uniform = torch.rand(size=(input.size(0), 1), device=input.device)
            random_mask = random_uniform <= self.config['RSC']['b']
            output = torch.where(random_mask, new_logits, logits)
            if self.config['loss']['loss_type'] == 'soft_triple':
                output = (output, all_tasks_output[0][1]) if self.multitask else (output, all_tasks_output[1])
            output = (output, *all_tasks_output[1:])
            return output
        else:
            assert self.config['RSC']['use_rsc'] == False
            features = self.model(input)
            if self.data_parallel:
                model1 = self.model.module
            else:
                model1 = self.model
            output = model1.make_logits(features)
            return output

    def multi_task_criterion(self, output: tuple, target: torch.tensor, C: float=1., Cs: float=0.1, Ci: float=0.01, Cf: float=1.):
        ''' output -> tuple of given losses
        target -> torch tensor of a shape [batch*num_tasks]
        return loss function '''
        SM, CE, BCE = self.criterion
        if self.augmentation:
            target_a, target_b, lam = target
            spoof_loss = self.mixup_criterion(SM, output[0], target_a[:,0], target_b[:,0], lam, 2)
            spoof_type_loss = self.mixup_criterion(CE, output[1], y_a=target_a[:,1], 
                                                                y_b=target_b[:,1],
                                                                lam=lam, num_classes=11)
            lightning_loss = self.mixup_criterion(CE, output[2], y_a=target_a[:,2], 
                                                                y_b=target_b[:,2],
                                                                lam=lam, num_classes=5)
            real_atr_loss = lam*BCE(output[3], target_a[:,3:].type(torch.float32)) + (1-lam)*BCE(output[3], target_b[:,3:].type(torch.float32))

        else:
            # spoof loss, take derivitive
            spoof_target = F.one_hot(target[:,0], num_classes=2)
            spoof_type_target = F.one_hot(target[:,1], num_classes=11)
            lightning_target = F.one_hot(target[:,2], num_classes=5)     
            # compute losses
            spoof_loss = SM(output[0], spoof_target)
            spoof_type_loss =  CE(output[1], spoof_type_target)
            lightning_loss =  CE(output[2], lightning_target)

            # filter output for real images and compute third loss
            mask = target[:,0] == 0
            filtered_output = output[3][mask] 
            filtered_target = target[:,3:][mask].type(torch.float32)
            real_atr_loss = BCE(filtered_output, filtered_target)
            
        # taking derivitives
        self.optimizer.zero_grad()
        spoof_loss.backward(retain_graph=True)

        self.optimizer.zero_grad()
        spoof_type_loss.backward(retain_graph=True)

        self.optimizer.zero_grad()
        lightning_loss.backward(retain_graph=True)

        self.optimizer.zero_grad()
        real_atr_loss.backward(retain_graph=True)
        # combine losses
        loss = C*spoof_loss + Cs*spoof_type_loss + Ci*lightning_loss + Cf*real_atr_loss
        return loss

    @staticmethod
    def mixup_criterion(criterion, pred, y_a, y_b, lam, num_classes):
        ''' y_a and y_b considered to be folded target labels. 
        All losses waits to get one_hot target as an input except the BCELoss '''
        ya_hot = F.one_hot(y_a, num_classes=num_classes)
        yb_hot = F.one_hot(y_b, num_classes=num_classes)
        mixed_target = lam * ya_hot  + (1 - lam) * yb_hot
        return criterion(pred, mixed_target)

    def test(self, transform, file_name, flag=None):
        if flag:
            self.config['test_dataset']['type'] = 'celeba-spoof'

        print('_____________EVAULATION_____________')
        # load snapshot
        epoch_of_checkpoint = load_checkpoint(self.path_to_checkpoint, self.model, map_location=torch.device(f'cuda:{self.args.GPU}'), optimizer=None, strict=True)
        # making dataset
        test_dataset = make_dataset(self.config, val_transform=transform, mode='eval')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.config['data']['batch_size'],
                                                    shuffle=True, pin_memory=self.config['data']['pin_memory'],
                                                    num_workers=self.config['data']['data_loader_workers'])
        # printing results
        AUC, EER, accur, apcer, bpcer, acer, _, _ = evaulate(self.model, test_loader, self.config, self.args, compute_accuracy=True)
        results = f'''accuracy on test data = {round(np.mean(accur),3)}\t\
            AUC = {round(AUC,3)}\t\
                    EER = {round(EER*100,2)}\t\
                        apcer = {round(apcer*100,2)}\t\
                            bpcer = {round(bpcer*100,2)}\t\
                                acer = {round(acer*100,2)}\t\
                                    checkpoint made on {epoch_of_checkpoint} epoch'''    
    
        with open(os.path.join(self.experiment_path, file_name), 'w') as f:
            f.write(results)
    
    def get_exp_info(self):
        exp_num = self.config['exp_num']
        print(f'_______INIT EXPERIMENT {exp_num}______')
        train_dataset, test_dataset = self.config['dataset'], self.config['test_dataset']['type']
        print(f'training on {train_dataset}, testing on {test_dataset}')
        print('\n\nSNAPSHOT')
        for key, item in self.config['checkpoint'].items():
                print(f'{key} --> {item}')
        print('\n\nMODEL')
        for key, item in self.config['model'].items():
                print(f'{key} --> {item}')
        loss_type = self.config['loss']['loss_type']
        print(f'\n\nLOSS TYPE : {loss_type.upper()}')
        for key, item in self.config['loss'][f'{loss_type}'].items():
            print(f'{key} --> {item}')
        print('\n\nDROPOUT PARAMS')
        for key, item in self.config['dropout'].items():
                print(f'{key} --> {item}')
        print('\n\nOPTIMAIZER')
        for key, item in self.config['optimizer'].items():
                print(f'{key} --> {item}')
        print('\n\nADDITIONAL USING PARAMETRS')
        if self.augmentation:
            type_aug = self.config['aug']['type_aug']
            print(f'\nAUG TYPE = {type_aug} USING')
            for key, item in self.config['aug'].items():
                print(f'{key} --> {item}')
        if self.config['RSC']['use_rsc']:
            print(f'RSC USING')
            for key, item in self.config['RSC'].items():
                print(f'{key} --> {item}') 
        if self.data_parallel:
            ids = self.config['data_parallel']['parallel_params']['device_ids']
            print(f'USING DATA PATALLEL ON {ids[0]} and {ids[1]} GPU')
        if self.config['data']['sampler']:
            print('USING SAMPLER')
        if self.config['loss']['amsoftmax']['ratio'] != [1,1]:
            print('USING ADAPTIVE LOSS')
        if self.config['multi_task_learning']:
            print('multi_task_learning using'.upper())
        theta = self.config['conv_cd']['theta']
        if theta > 0:
            print(f'CDC method: {theta}')