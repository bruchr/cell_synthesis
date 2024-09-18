import itertools
import os
from pathlib import Path
import time

import hjson
import torch
from torchsummary import summary
from tqdm import tqdm

from models.losses import gan_loss
from models.networks import get_scheduler, define_resnet, define_discriminator
from models.networks_3d import define_resnet as define_resnet_3d, define_discriminator as define_discriminator_3d
from utils.sliding_window import generateSlices_half_cut
from utils.image_pool import ImagePool
from utils.utils import tensor2np, inference_save_images
from utils.visualization import Visualizer

class CycleGANModel():
    """[summary]

    Returns:
        [type] -- [description]
    """
    def __init__(self, opt):
        """[summary]

        Arguments:
            opt {[type]} -- [description]
        """
        self.opt = opt
        if torch.cuda.is_available():
            self.opt["device"] = "cuda"
        else:
             self.opt["device"] = "cpu"
             print('\n### No Cuda detected. CPU is used! ###\n')
        self.device = torch.device(self.opt["device"])
        self.img_dim = opt['img_dim']
        assert self.img_dim == 2 or self.img_dim == 3, 'img_dim must be either 2 or 3, but is {}'.format(self.img_dim)
        # Initialization of the different networks
        # For inference only one generator model is needed
        if self.opt["mode"] == "train" or self.opt["direction_inference"] == 'AtoB':
            self.netG_A = define_resnet(opt, "A") if self.img_dim==2 else define_resnet_3d(opt, "A")
        if self.opt["mode"] == "train" or self.opt["direction_inference"] == 'BtoA':
            self.netG_B = define_resnet(opt, "B") if self.img_dim==2 else define_resnet_3d(opt, "B")
        self.optimizers = []
        # define losses
        self.loss = {"G": 0, "G_A": 0, "G_B": 0, "D_A": 0, "D_B": 0,
                     "Cycle_A": 0, "Cycle_B": 0, "Idt_A": 0, "Idt_B": 0,
                     "G_Seg": 0, "D_Seg": 0,
                     "D_A_Acc": 0, "D_B_Acc": 0, "D_Seg_Acc": 0}
        self.epoch_loss = self.loss.copy()
        # define images images will be added, the first time used
        self.images = {}
        self.paths = {}
        self.path_output = Path(opt["output_folder"])

        if opt["mode"] == "train":
            discriminator = define_discriminator(self.opt) if self.img_dim==2 else define_discriminator_3d(self.opt)
            self.netD_A = discriminator(self.opt, "B")  # To distinguish between real_B and fake_B (Corresponds to netG_A)
            self.netD_B = discriminator(self.opt, "A")  # To distinguish between real_A and fake_A (Corresponds to netG_B)
            self.netD_Seg = discriminator(self.opt, "Seg")  # To distinguish between real_B_seg and fake_B_seg (Corresponds to netG_A)
            self.fake_A_pool = ImagePool(opt)
            self.fake_B_img_pool = ImagePool(opt)
            self.fake_B_seg_pool = ImagePool(opt)
            # define loss functions
            self.criterionGAN = gan_loss
            self.criterionCycle = torch.nn.L1Loss()   # L1Loss()
            self.criterionIdt = torch.nn.L1Loss()     # L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), opt["learning_rate"], betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), opt["learning_rate"], betas=(0.5, 0.999))
            self.optimizer_D_Seg = torch.optim.Adam(itertools.chain(self.netD_Seg.parameters()), opt["learning_rate"], betas=(0.5, 0.999))
            # initialize schedulers
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_Seg)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            # initialize tensorboard writer and define loss groups
            self.visualizer = Visualizer(opt)
            self.visualizer.define_loss_group("Discriminators", ["D_A", "D_B", "D_Seg","D_A_Acc", "D_B_Acc", "D_Seg_Acc"])
            self.visualizer.define_loss_group("Generators", ["G", "G_A", "G_B", "Cycle_A", "Cycle_B", "Idt_A", "Idt_B", "G_Seg"])
            

    def set_input(self, input):
        """ Get input data from dataloader and process it"""
        self.images["real_A"] = input["A"].to(self.device)
        self.images["real_B_img"] = input["B"].to(self.device)
        self.images["real_B_seg"] = input["Seg"].to(self.device)
        self.paths["paths_A"] = input["A_path"]
        self.paths["paths_B"] = input["B_path"]

    def forward(self):
        """" Run forward pass"""
        # self.images["fake_B"] = self.netG_A(self.images["real_A"])
        tmp = self.netG_A(self.images["real_A"])
        self.images["fake_B_img"] = tmp[:, :-1]
        self.images["fake_B_seg"] = tmp[:, -1:]
        self.images["rec_A"] = self.netG_B(self.images["fake_B_img"])
        self.images["fake_A"] = self.netG_B(self.images["real_B_img"])
        # self.images["rec_B"] = self.netG_A(self.images["fake_A"])
        tmp = self.netG_A(self.images["fake_A"])
        self.images["rec_B_img"] = tmp[:, :-1]
        self.images["rec_B_seg"] = tmp[:, -1:]

    def set_requires_grad(self, nets, requires_grad=False):
        """ Set requires_grad=False for all networks given"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def backward_G(self):
        """Calculate loss for the generators"""
        lambda_idt = self.opt["lambda_identity"]
        lambda_cycle_A = self.opt["lambda_cycle_A"]
        lambda_cycle_B = self.opt["lambda_cycle_B"]
        # Identity loss
        if 0: # lambda_idt > 0: # Nicht implementiert für Segmentation
            # G_A should be identity if real_B is fed: ||G_A(B) - B ||
            self.images["idt_A"] = self.netG_A(self.images["real_B"])
            self.loss["Idt_A"] = self.criterionIdt(self.images["idt_A"], self.images["real_B"]) * lambda_cycle_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A ||
            self.images["idt_B"] = self.netG_B(self.images["real_A"])
            self.loss["Idt_B"] = self.criterionIdt(self.images["idt_B"], self.images["real_A"]) * lambda_cycle_A * lambda_idt
        else:
            self.loss["Idt_A"] = torch.tensor(0)
            self.loss["Idt_B"] = torch.tensor(0)

        # GAN loss D_A(G_A(A)) and l D_B(G_B(B))
        self.loss["G_A"] = self.criterionGAN(self.netD_A(self.images["fake_B_img"]), True, self.device)
        self.loss["G_B"] = self.criterionGAN(self.netD_B(self.images["fake_A"]), True, self.device)
        self.loss["G_Seg"] = self.criterionGAN(self.netD_Seg(self.images["fake_B_seg"]), True, self.device)
        # Cycle loss ||G_B(G_A(A))|| and ||G_A(G_B(B))||
        self.loss["Cycle_A"] = self.criterionCycle(self.images["rec_A"], self.images["real_A"]) * lambda_cycle_A
        self.loss["Cycle_B"] = self.criterionCycle(self.images["rec_B_img"], self.images["real_B_img"]) * lambda_cycle_B
        # combine losses and calculate gradients
        self.loss["G"] = self.loss["G_A"] + self.loss["G_B"] + self.loss["G_Seg"] + self.loss["Cycle_A"] + self.loss["Cycle_B"] + self.loss["Idt_A"] + self.loss["Idt_B"]
        self.loss["G"].backward()

    def backward_D(self, netD, real, fake):
        """ Calculate GAN loss for discriminators"""
        # Loss of real image
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True, self.device)
        #  Loss for fake image
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False, self.device)
        # Combine loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()

        with torch.no_grad():
            mean_axes = (2,3,4) if self.img_dim == 3 else (2,3)
            true_pos = torch.sum(torch.mean(pred_real.cpu(), axis=mean_axes) >= 0.5)
            true_neg = torch.sum(torch.mean(pred_fake.cpu(), axis=mean_axes) < 0.5)
            acc_d = torch.div((true_pos+true_neg).float(), (real.shape[0]+fake.shape[0]))

        return loss_D, acc_d

    def optimize_parameters(self, epoch):
        """ Calculate losses, gradients and update network weights"""
        self.forward()  # compute fake images and reconstruction images (Cycle)
        # Optimize generators
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_Seg], False)
        self.optimizer_G.zero_grad()  # zero the gradients
        self.backward_G()
        self.optimizer_G.step()
        # Optimize discriminators
        fake_A = self.fake_A_pool.query(self.images["fake_A"])
        fake_B_img = self.fake_B_img_pool.query(self.images["fake_B_img"])
        fake_B_seg = self.fake_B_seg_pool.query(self.images["fake_B_seg"])
        if self.opt['train_d_every_x']==0 or epoch%self.opt['train_d_every_x']==0:
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_Seg], True)
            self.optimizer_D.zero_grad()
            self.optimizer_D_Seg.zero_grad()
            self.loss["D_A"], self.loss['D_A_Acc'] = self.backward_D(self.netD_A, self.images["real_B_img"], fake_B_img)
            self.loss["D_B"], self.loss['D_B_Acc'] = self.backward_D(self.netD_B, self.images["real_A"], fake_A)
            self.loss["D_Seg"], self.loss['D_Seg_Acc'] = self.backward_D(self.netD_Seg, self.images["real_B_seg"], fake_B_seg)
            self.optimizer_D.step()
            self.optimizer_D_Seg.step()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def save_results(self, epoch_nr=None):
        # Save opt to .json file and models to .pth file
        folder = Path.joinpath(self.path_output, self.opt["experiment"])
        if epoch_nr is not None:
            folder = Path.joinpath(folder, 'epoch_{}'.format(epoch_nr))
        folder.mkdir(parents=True, exist_ok=True)
        self.save_hyperparameters(folder, epoch_nr)
        self.save_models(folder)

    def save_hyperparameters(self, folder, epoch_nr=None):
        path = folder.joinpath("settings.hjson")
        if epoch_nr is not None:
            opt_save = self.opt.copy()
            opt_save['experiment'] = 'epoch_{}'.format(epoch_nr)
            opt_save['output_folder'] = str(Path.joinpath(self.path_output, self.opt["experiment"]).as_posix())
        else:
            opt_save = self.opt
        with open(path, 'w', encoding='utf-8') as outfile:
            hjson.dump(opt_save, outfile, ensure_ascii=False, indent=2)

    def save_models(self, folder):
        # Save the trained model parameters (netG_A, netG_B, netD_A, netD_B)
        path = folder.joinpath("models.pth")
        if self.device == torch.device("cuda"):  # only get state_dict from .module for DataParallel() networks
            torch.save({
                        'netG_A_state_dict': self.netG_A.module.state_dict(),
                        'netG_B_state_dict': self.netG_B.module.state_dict(),
                        'netD_A_state_dict': self.netD_A.module.state_dict(),
                        'netD_B_state_dict': self.netD_B.module.state_dict(),
                        'netD_Seg_state_dict': self.netD_Seg.module.state_dict(),
                        'optimizer_G': self.optimizer_G.state_dict(),
                        'optimizer_D': self.optimizer_D.state_dict(),
                        'optimizer_D_Seg': self.optimizer_D_Seg.state_dict(),
                        'scheduler_0': self.schedulers[0].state_dict(),
                        'scheduler_1': self.schedulers[1].state_dict(),
                        'scheduler_2': self.schedulers[2].state_dict(),
                        }, path)
        else:
            torch.save({
                        'netG_A_state_dict': self.netG_A.state_dict(),
                        'netG_B_state_dict': self.netG_B.state_dict(),
                        'netD_A_state_dict': self.netD_A.state_dict(),
                        'netD_B_state_dict': self.netD_B.state_dict(),
                        'netD_Seg_state_dict': self.netD_Seg.state_dict(),
                        'optimizer_G': self.optimizer_G.state_dict(),
                        'optimizer_D': self.optimizer_D.state_dict(),
                        'optimizer_D_Seg': self.optimizer_D_Seg.state_dict(),
                        'scheduler_0': self.schedulers[0].state_dict(),
                        'scheduler_1': self.schedulers[1].state_dict(),
                        'scheduler_2': self.schedulers[2].state_dict(),
                        }, path)
        print("saving model")

    def load_models(self):
        if self.opt["resume_training"] == 1 and self.opt["mode"] == "train" and not self.opt["no_timestamp"]:
            folder = Path.joinpath(self.path_output, self.opt["experiment"][:-13])
        else:
            folder = Path.joinpath(self.path_output, self.opt["experiment"])
        path = folder.joinpath("models.pth")
        checkpoint = torch.load(path, map_location=self.device)
        if self.device == torch.device("cuda"):
            if self.opt["mode"] == "train" or self.opt["direction_inference"] == 'AtoB':
                self.netG_A.module.load_state_dict(checkpoint['netG_A_state_dict'])
            if self.opt["mode"] == "train" or self.opt["direction_inference"] == 'BtoA':
                self.netG_B.module.load_state_dict(checkpoint['netG_B_state_dict'])
            
            if self.opt["mode"] == "train":
                self.netD_A.module.load_state_dict(checkpoint['netD_A_state_dict'])
                self.netD_B.module.load_state_dict(checkpoint['netD_B_state_dict'])
                self.netD_Seg.module.load_state_dict(checkpoint['netD_Seg_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                self.optimizer_D_Seg.load_state_dict(checkpoint['optimizer_D_Seg'])
                self.schedulers[0].load_state_dict(checkpoint['scheduler_0'])
                self.schedulers[1].load_state_dict(checkpoint['scheduler_1'])
                self.schedulers[2].load_state_dict(checkpoint['scheduler_2'])
        else:
            if self.opt["mode"] == "train" or self.opt["direction_inference"] == 'AtoB':
                self.netG_A.load_state_dict(checkpoint['netG_A_state_dict'])
            if self.opt["mode"] == "train" or self.opt["direction_inference"] == 'BtoA':
                self.netG_B.load_state_dict(checkpoint['netG_B_state_dict'])
            
            if self.opt["mode"] == "train":
                self.netD_A.load_state_dict(checkpoint['netD_A_state_dict'])
                self.netD_B.load_state_dict(checkpoint['netD_B_state_dict'])
                self.netD_Seg.load_state_dict(checkpoint['netD_Seg_state_dict'])
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
                self.optimizer_D_Seg.load_state_dict(checkpoint['optimizer_D_Seg'])
                self.schedulers[0].load_state_dict(checkpoint['scheduler_0'])
                self.schedulers[1].load_state_dict(checkpoint['scheduler_1'])
                self.schedulers[2].load_state_dict(checkpoint['scheduler_2'])

    def show_summary(self, dataloader):
        images = next(iter(dataloader))
        print("Printing Generator Network A:")
        summary(self.netG_A, images["A"][0].size(), batch_size=images["A"].size()[0])
        images = next(iter(dataloader))
        print("Printing Generator Network B:")
        summary(self.netG_B, images["B"][0].size(), batch_size=images["B"].size()[0])  
        images = next(iter(dataloader))
        print("Printing Discriminator Network A:")
        summary(self.netD_A, images["B"][0].size(), batch_size=images["B"].size()[0])    
        print("Printing Discriminator Network B:")
        summary(self.netD_A, images["A"][0].size(), batch_size=images["A"].size()[0])
        print("Printing Discriminator Network Seg:")
        summary(self.netD_Seg, images["Seg"][0].size(), batch_size=images["Seg"].size()[0])  

    def train(self, dataloader):
        if self.opt["resume_training"] == 1:
            self.load_models()

        self.show_summary(dataloader)

        save_counter = 0
        zero_loss_counter_da = 0
        zero_loss_counter_db = 0
        time_last_save = time.time()
        for epoch in range(self.opt["epoch_count"], self.opt["learning_rate_fix"] + self.opt["learning_rate_decay"]):  # Epoch loop
            self.opt["epoch_count"] = epoch + 1
            epoch_start_time = time.time()
            # Reset epoch loss for new epoch
            for key in self.epoch_loss.keys():
                self.epoch_loss[key] = 0
            for i, data in enumerate(dataloader):  # Batch loop
                #  save_batch(data, "img")
                self.set_input(data)
                self.optimize_parameters(epoch)
                # Add loss for all batches multiplied by its batch size
                for key in self.epoch_loss.keys():
                    self.epoch_loss[key] += data["A"].shape[0] * self.loss[key].detach()
            self.update_learning_rate()  # Update learning rate after epoch

            # Mean for all losses
            for key in self.epoch_loss.keys():
                self.epoch_loss[key] = self.epoch_loss[key] / len(dataloader.dataset)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch: {self.opt['epoch_count']}")
            print(f"Epoch duration: {epoch_duration}")
            print(f"learning rate 0: {self.optimizers[0].param_groups[0]['lr']}")
            print(f"learning rate 1: {self.optimizers[1].param_groups[0]['lr']}")
            print(f"learning rate 2: {self.optimizers[2].param_groups[0]['lr']}")
            self.visualizer.write_losses(self.epoch_loss, self.opt["epoch_count"])
            self.visualizer.write_images(self.images, self.opt["epoch_count"])

            # Detect convergernce failure
            if self.loss["D_A"] < 10**(-3):
                zero_loss_counter_da +=1
            else:
                zero_loss_counter_da = 0
            if self.loss["D_B"] < 10**(-3):
                zero_loss_counter_db +=1
            else:
                zero_loss_counter_db = 0

            if max(zero_loss_counter_da, zero_loss_counter_db) >= 10:
                print('Zero Loss detected in Discriminator.\nStopping training...')
                break
            if torch.isnan(self.loss["D_A"]) or torch.isnan(self.loss["D_B"]):
                print('NaN Loss detected in Discriminator.\nStopping training...')
                break

            save_counter += 1
            if save_counter == self.opt["save_model_every_x_epochs"]:
                self.save_results(self.opt["epoch_count"])
                save_counter = 0
                time_last_save = time.time()
            elif time.time() - time_last_save >= 3600:
                self.save_results()
                time_last_save = time.time()

            if self.opt["time_limit"] != 0 and time.time()-self.opt["time_start"] > self.opt["time_limit"]:
                self.save_results()
                break

        self.save_results(self.opt["epoch_count"])

    def inference(self, dataloader):
        print("inference mode")

        a2b = self.opt["direction_inference"] == 'AtoB' # Bool for direction
        dom = "A" if a2b else "B"
        dom_path = "A_path" if a2b else "B_path"
        out_nc = self.opt["B_nc"]+1 if a2b else self.opt["A_nc"]

        if 'inf_patch_size' in self.opt.keys() and self.opt['inf_patch_size'] is not None:
            squeeze_ind = torch.where(torch.tensor(self.opt['inf_patch_size'])==1)[0]+2
            squeeze_ind = squeeze_ind.item() if len(squeeze_ind) == 1 else None

        if os.path.basename(self.opt["dataset_folder"]) == '':
            dataset_last_folder = os.path.basename(os.path.dirname(self.opt["dataset_folder"]))
        else:
            dataset_last_folder = os.path.basename(self.opt["dataset_folder"])

        self.load_models()
        net = self.netG_A if a2b else self.netG_B
        net.eval()

        for i, data in enumerate(dataloader):
            # Hier muss das Bild aufgeteilt werden...
            print(f"Processing batch: {i+1}/{len(dataloader)}")
            # Set input
            images = data[dom].to(self.device)
            filenames = data[dom_path]

            if 'inf_patch_size' not in self.opt.keys() or self.opt['inf_patch_size'] is None:
                with torch.no_grad():
                    fakes = net(images)
            else:
                with torch.no_grad():
                    out_shape = (images.shape[0], out_nc, *images.shape[2:])
                    fakes = torch.zeros(out_shape)
                    slices_in, slices_out, slices_res = generateSlices_half_cut(images.shape[2:], self.opt['inf_patch_size'], self.opt['inf_patch_overlap'])
                    for sl_in, sl_out, sl_res in tqdm(zip(slices_in, slices_out, slices_res), total=len(slices_in)):
                        sl_in = (slice(None), slice(None),) + sl_in
                        sl_out = (slice(None), slice(None),) + sl_out
                        sl_res = (slice(None), slice(None),) + sl_res
                        
                        if squeeze_ind is None:
                            fakes[sl_out] = net(images[sl_in])[sl_res]
                        else:
                            fakes[sl_out] = net(images[sl_in].squeeze(dim=squeeze_ind)).unsqueeze(dim=squeeze_ind)[sl_res]
                
            folder = Path.joinpath(self.path_output, self.opt["experiment"], "results_"+ dataset_last_folder, self.opt["direction_inference"])
            fakes_np = tensor2np(fakes, imtype="uint16")
            inference_save_images(fakes_np, folder, filenames, self.img_dim)