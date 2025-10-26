import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import monai
from dataloader_multi_label import artData
import cv2
import os
import numpy as np
import time
from model import myNetBaseUNetAttCtv, myNetSwinUNETRAttCtv
import SimpleITK as sitk
import surface_distance as surfdist
from monai.losses.dice import DiceLoss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Train():
    def __init__(self, mode='train'):
        self.device = 'cuda:0'
        self.label = 'ctv'
        self.use_ram = True
        self.input_labels = ['C1', 'G1'] # ['C1', 'G1', 'G2']
        self.adapt = 'none'

        netName = 'your_network_name'
        self.checkpointPath = '/path_to_log/'+netName+'-Log/checkpoint'
        if not os.path.exists(self.checkpointPath):
            os.makedirs(self.checkpointPath)
        self.trainOutputPath = '/path_to_log/'+netName+'-Log/output'
        if not os.path.exists(self.trainOutputPath):
            os.makedirs(self.trainOutputPath)
        self.testOutputPath = '/path_to_log/'+netName+'-Log/testOutput'
        if not os.path.exists(self.testOutputPath):
            os.makedirs(self.testOutputPath)
        self.log_path = '/path_to_log/'+netName+'-Log/log'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.netName = netName
        print(self.netName)
        
        if mode == 'train':
            self.writer = SummaryWriter(log_dir=self.log_path)
            self.train_loader = DataLoader(artData('train', label=self.label), batch_size=2, shuffle=True, pin_memory=True, num_workers=16)

        self.test_loader = DataLoader(artData('test', label=self.label), batch_size=1, pin_memory=True)
        self.epochs = 200
        self.netInit()
        self.dice_loss = DiceLoss()


    def netInit(self, mode='train'):
        out_channels = 1
        in_channels = 2
        num_encoders = len(self.input_labels)
        if 'SwinUNETR' in self.netName:
            pretrained_path = "/path_to_log/your_SwinUNETR_network_name/checkpoint/epoch_499.pth"
            self.net = myNetSwinUNETRAttCtv(in_channels=in_channels, out_channels=out_channels, num_encoders=num_encoders, ada_type=self.adapt)
            state_dict = torch.load(pretrained_path)
            self.net.load_state_dict(state_dict, strict=False)
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=0.001)
        else:
            pretrained_path = "/path_to_log/your_UNet_network_name/checkpoint/epoch_499.pth"
            self.net = myNetBaseUNetAttCtv(in_channels=in_channels, out_channels=out_channels, num_encoders=num_encoders, ada_type=self.adapt)
            state_dict = torch.load(pretrained_path)
            self.net.load_state_dict(state_dict, strict=False)
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999)
        self.net.to(self.device)
    

    def saveModel(self, epoch):
        self.net.to('cpu')
        torch.save(obj=self.net.state_dict(), f=os.path.join(self.checkpointPath, 'epoch_'+str(epoch)+'.pth'))
        self.net.to(self.device)
    

    def loadModel(self, epoch):
        self.net.load_state_dict(torch.load(os.path.join(self.checkpointPath, 'epoch_'+str(epoch)+'.pth')), strict=False)


    def train(self):
        scaler = GradScaler()
        def norm(x):
            x = (x - x.min())/(x.max() - x.min())*255
            x = x.astype(np.uint8)
            return x

        for epoch in range(self.epochs):
            dice_mean = []
            self.net.train()
            for i, data in enumerate(self.train_loader):
                ct1 = data['ct1'].to(self.device)
                ct2 = data['ct2'].to(self.device)
                gt1 = data['gt1'].to(self.device)
                gt2 = data['gt2'].to(self.device)
                data_type = data['data_type']

                gtv1, ctv1 = gt1[:,:1], gt1[:,1:]
                gtv2, ctv2 = gt2[:,:1], gt2[:,1:]
                input_list = [ctv1, gtv1, gtv2]
                gt1 = torch.cat(input_list[:len(self.input_labels)], dim=1)
                gt2 = ctv2#torch.cat([ctv2, gtv2], dim=1)
                if 'SwinUNETR' in self.netName and 'ada' not in self.netName:
                    with autocast(dtype=torch.bfloat16):
                        pred = self.net(ct1, ct2, gt1)
                        loss = self.dice_loss(pred, gt2)
                    
                    dice_mean.append(loss.item())
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    pred = self.net(ct1, ct2, gt1)
                    loss = self.dice_loss(pred, gt2)
                    
                    dice_mean.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                print('epoch:', epoch, 
                      'iter:', i, 
                      'dice:', loss.item(), 
                      'time:', time.strftime('%Y-%m-%d %H:%M:%S'))
            self.scheduler.step()
            
            pngSave = []
            s = torch.argmax(gt2[0,0,:,:,:].sum(dim=(0,1))).item()
            for t in range(gt2.shape[1]):
                ct1_png = norm(ct1[0,0,:,:,s].clone().detach().cpu().float().numpy())
                gtp1_png = norm(gt1[0,t,:,:,:].sum(1).clone().detach().cpu().float().numpy())
                gt1_png = norm(gt1[0,t,:,:,s].clone().detach().cpu().float().numpy())
                ct2_png = norm(ct2[0,0,:,:,s].clone().detach().cpu().float().numpy())
                gtp2_png = norm(gt2[0,t,:,:,:].sum(1).clone().detach().cpu().float().numpy())
                gt2_png = norm(gt2[0,t,:,:,s].clone().detach().cpu().float().numpy())
                pred_png = norm(pred[0,t,:,:,s].clone().detach().cpu().float().numpy())
                predp_png = norm(pred[0,t,:,:,:].sum(1).clone().detach().cpu().float().numpy())
                pngSave.append(np.concatenate((ct1_png, gtp1_png, gt1_png*ct1_png*0.8+ct1_png*0.2, ct2_png, gtp2_png, gt2_png*ct2_png*0.8+ct2_png*0.2, pred_png, predp_png, gtp1_png/2+gtp2_png/2), axis=1))
            
            pngSave = np.concatenate(pngSave)
            cv2.imwrite(os.path.join(self.trainOutputPath, 'epoch_'+str(epoch)+'.png'), pngSave)
            if (epoch+1) % 50 == 0:
                self.saveModel(epoch)
            
            self.writer.add_scalar('DICE Loss/TRAIN', np.array(dice_mean).mean(), epoch)
        self.writer.close()
    

    def test(self, epoch, load_model=True, save_nii=False):
        def norm(x):
            x = (x - x.min())/(x.max() - x.min())*255
            x = x.astype(np.uint8)
            return x
        
        def write_nii(img, save_file_name):
            img = sitk.GetImageFromArray(img)
            img.SetSpacing((1.2,1.2,3))
            sitk.WriteImage(img, save_file_name)
        
        if load_model:
            self.loadModel(epoch)

        print('---------- testing ----------')
        metric = {'dsc':[], 'iou':[], 'acc':[], 'rvd':[], 'asd':[], 'hsd':[]}
        total_time = 0
        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                ct1 = data['ct1'].to(self.device)
                ct2 = data['ct2'].to(self.device)
                gt1 = data['gt1'].to(self.device)
                gt2 = data['gt2'].to(self.device)
                data_type = data['data_type']
                print(data['f'])

                gtv1, ctv1 = gt1[:,:1], gt1[:,1:]
                gtv2, ctv2 = gt2[:,:1], gt2[:,1:]
                input_list = [ctv1, gtv1, gtv2]
                gt1 = torch.cat(input_list[:len(self.input_labels)], dim=1)
                gt2 = ctv2

                st = time.time()
                pred = self.net(ct1, ct2, gt1)
                ed = time.time()
                total_time += (ed-st)*1000
                if save_nii:
                    pred_mask = torch.zeros_like(pred)
                    pred_mask[pred>=0.5] = 1
                    pred_mask[pred<0.5] = 0
                    dsc = (200*(pred_mask*gt2).sum()/(pred_mask.sum()+gt2.sum())).item()
                    iou = (100*(pred_mask*gt2).sum()/((pred_mask+gt2)>0).sum()).item()
                    acc = (100*(pred_mask == gt2).sum()/np.prod(gt2.shape)).item()
                    rvd = 100*abs(pred_mask.sum()/gt2.sum()-1).item()
                    surface_distances = surfdist.compute_surface_distances(gt2[0,0].cpu().numpy().astype(bool), pred_mask[0,0].cpu().numpy().astype(bool), spacing_mm=(1.2, 1.2, 3.0))
                    hd_dist = surfdist.compute_robust_hausdorff(surface_distances, 95)
                    surf_dist = surfdist.compute_average_surface_distance(surface_distances)
                    avg_surf_dist = (surf_dist[0]+surf_dist[1])/2
                    print(i, dsc/(200-dsc)*100)
                else:
                    dsc = (1-self.dice_loss(pred, gt2).item())*100
                    iou, acc, rvd, avg_surf_dist, hd_dist = 0, 0, 0, 0, 0
                
                print(i, dsc, iou, acc, rvd, avg_surf_dist, hd_dist)
                metric['dsc'].append(dsc)
                metric['iou'].append(iou)
                metric['acc'].append(acc)
                metric['rvd'].append(rvd)
                metric['asd'].append(avg_surf_dist)
                metric['hsd'].append(hd_dist)

                if save_nii:
                    ct1_nii = ct1[0,0].clone().detach().cpu().float().numpy()
                    ct2_nii = ct2[0,0].clone().detach().cpu().float().numpy()
                    gt1_nii = gt1[0,0].clone().detach().cpu().float().numpy()
                    gt2_nii = gt2[0].clone().detach().cpu().float().numpy()
                    pred_nii = pred[0].clone().detach().cpu().float().numpy()

                    write_nii(np.concatenate((gt2_nii, pred_nii)).transpose((3,2,1,0)), os.path.join(self.testOutputPath, 'gt2_pred_'+str(i)+'.nii.gz'))
                    write_nii(gt1_nii.transpose((2,1,0)), os.path.join(self.testOutputPath, 'gt1_'+str(i)+'.nii.gz'))
                    write_nii(ct2_nii.transpose((2,1,0)), os.path.join(self.testOutputPath, 'ct2_'+str(i)+'.nii.gz'))
                    write_nii(ct1_nii.transpose((2,1,0)), os.path.join(self.testOutputPath, 'ct1_'+str(i)+'.nii.gz'))

                s = torch.argmax(gt2[0,0,:,:,:].sum(dim=(0,1))).item()
                ct1 = norm(ct1[0,0,:,:,s].clone().detach().cpu().float().numpy())
                gtp1 = norm(gt1[0,0,:,:,:].sum(1).clone().detach().cpu().float().numpy())
                gt1 = norm(gt1[0,0,:,:,s].clone().detach().cpu().float().numpy())
                ct2 = norm(ct2[0,0,:,:,s].clone().detach().cpu().float().numpy())
                gtp2 = norm(gt2[0,0,:,:,:].sum(1).clone().detach().cpu().float().numpy())
                gt2 = norm(gt2[0,0,:,:,s].clone().detach().cpu().float().numpy())
                predp = norm(pred[0,0,:,:,:].sum(1).clone().detach().cpu().float().numpy())
                pred = norm(pred[0,0,:,:,s].clone().detach().cpu().float().numpy())

                pngSave = np.concatenate((ct1, gtp1, gt1*ct1*0.8+ct1*0.2, ct2, gtp2, gt2*ct2*0.8+ct2*0.2, pred, predp, gtp1/2+gtp2/2), axis=1)
                cv2.imwrite(os.path.join(self.testOutputPath, 'num_'+str(i)+'.png'), pngSave)
        
        for k in metric:
            metric[k] = np.array(metric[k])
        
        np.savez(os.path.join(self.testOutputPath, 'result.npz'), dsc=metric['dsc'], iou=metric['iou'], acc=metric['acc'], rvd=metric['rvd'], asd=metric['asd'], hsd=metric['hsd'])
        print(self.netName, '%.2f ± %.2f & %.2f ± %.2f & %.2f ± %.2f & %.2f ± %.2f & %.2f ± %.2f & %.2f ± %.2f'% (
              metric['dsc'].mean(), metric['dsc'].std(),
              metric['iou'].mean(), metric['iou'].std(), 
              metric['acc'].mean(), metric['acc'].std(), 
              metric['rvd'].mean(), metric['rvd'].std(), 
              metric['asd'].mean(), metric['asd'].std(), 
              metric['hsd'].mean(), metric['hsd'].std(),
              ), ', %.2f ms per case.' % (total_time/len(self.test_loader)))
        
        return metric['dsc'].mean()


if __name__ == '__main__':
    trainer = Train()
    trainer.train()
    trainer = Train('test')
    trainer.test(199, True, True)
