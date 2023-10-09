import torch.nn as nn
import torch
import torch.nn.functional as F

from torchvision.utils import make_grid
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

from typing import Tuple
import random
import os

from data_utils import collect_only_files, collect_only_dirs, update_dict, calculate_mean_for_list_of_dicts
from demo_utils import normalize_image

def replace_model_last_linear_layer(model, new_out_dim):
        last_linear_name = ""
        last_linear_layer = None
        for name, layer in model.named_modules():
            if layer.__class__.__name__ == "Linear":
                last_linear_name = name
                last_linear_layer = layer

        new_classification_head = nn.Linear(last_linear_layer.in_features, new_out_dim)

        attrs = last_linear_name.split(".")
        last_obj = model

        for attr in attrs[:-1]:
            last_obj = getattr(last_obj, attr)

        setattr(last_obj, attrs[-1], new_classification_head)

        last_linear_obj = getattr(last_obj, attrs[-1])

        return model, last_linear_obj

def calculate_intermediate_hws(img_hw, generator_dims):
    input_hw = torch.tensor(list(img_hw))[None, :] # (100, 100)
    layer_nr = torch.arange(len(generator_dims)).flip(0)[:, None] # (3, 2, 1, 0)
    generator_intermediate_hws = (input_hw / (2**layer_nr)).ceil().type(torch.int) # ((12, 12), (25, 25), (50, 50), (100, 100,)
    print(f"generator_intermediate_hws: {generator_intermediate_hws}")
    gen_input = torch.zeros((generator_dims[0], generator_intermediate_hws[0, 0], generator_intermediate_hws[0, 1]))
    dec_out_shape = gen_input.reshape(-1).shape[0]

    return gen_input.shape, dec_out_shape, generator_intermediate_hws


def get_real_version(log_folder, experiment_name):
    paths, names = collect_only_dirs(os.path.join(log_folder, experiment_name))
    current_version = sorted([int(name.split("_")[1]) for name in names if name.split("_")[0]=="version"])[-1]
    print(f"CURRENT VERSION: {current_version}")
    return current_version

class IdentityBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_size: Tuple[int, int]):

        super().__init__()
        self.out_size = out_size
        self.conv = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):

        x = F.interpolate(x, size=self.out_size)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
    
class GeneratorHead(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding='same'
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding='same'
        )
        self.final_act = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.final_act(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.seq(x)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()
        self.dsampler = nn.AvgPool2d(2, stride=2)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dsampler(x)

        return x

class TwoPicsGenerator(nn.Module):
    def __init__(self, emb_size, img_hw: Tuple[int, int]=[100,200], generator_dims=[2, 2, 1, 1]):
        super().__init__()

        self.generator_dims = generator_dims

        G_inp_shape, G_inp_flat, G_hws = calculate_intermediate_hws(img_hw, generator_dims)

        #### DENSE DECODER ####
        self.decoder = nn.Sequential(
            DenseBlock(emb_size, G_inp_flat),
        )

        #### GENERATOR ####
        generator_modules = []
        for i in range(len(generator_dims) - 1):
            generator_modules.append(
                GeneratorBlock(
                    in_channels=generator_dims[i],
                    out_channels=generator_dims[i + 1],
                    out_size=tuple(G_hws[i + 1]),
                )
            )
        generator_modules.append(GeneratorHead(generator_dims[-1], 1))
        self.generator = nn.Sequential(*generator_modules)

    def decode(self, z):
        return self.decoder(z)
    
    def generate(self, z):
        out = z.reshape(-1, *self.gen_input.shape)
        return self.generator(out)
    
    def forward(self, x):
        """ 
        this is a train loop forward pass 
        """
        z = self.decode(x)
        return self.generate(z)


class LitTwoPicsGenerator(pl.LightningModule):
    def __init__(self, model, loss_fx, log_folder="lightning_logs", experiment_name="experiment", lr=1e-2, l2reg=0):
        super().__init__()    

        self.automatic_optimization = False
        self.enable_scheduler = False

        self.model = model
        self.loss_fx = loss_fx

        self.lr = lr
        self.l2reg = l2reg

        self.log_folder = log_folder
        self.experiment_name = experiment_name     

        self.T_losses_list = []

        self.save_hyperparameters(ignore=['model', 'loss_fx', 'enable_scheduler', 'lr', 'l2reg'])

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):

        opts = []
        scheds = []

        opts += [torch.optim.Adam(self.model.decoder.parameters(), lr=self.lr, weight_decay=self.l2reg)]
        opts += [torch.optim.Adam(self.model.generator.parameters(), lr=self.lr, weight_decay=self.l2reg)]

        if self.enable_scheduler:
            for opt in opts:
                scheds += [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10,T_mult=2, eta_min=self.lr / 1000000)]

        return opts, scheds
    
    def _step_schedulers(self):
        if self.enable_scheduler:
            highest_lr = 0
            for sched in self.lr_schedulers():
                if sched.get_last_lr()[0] > highest_lr:
                    highest_lr = sched.get_last_lr()[0]
                sched.step()
        else: highest_lr = self.lr
        return highest_lr
    
    def _shared_eval(self, batch):
        img, x = batch
        img_hat = self.forward(x)

        loss = self.loss_fx(img_hat, img)
        loss_dict = {"loss": loss}
        
        tensors_dict = {"x": x, 
                   "img": img, 
                   "img_hat": img_hat}

        return loss, loss_dict, tensors_dict
    
    def on_fit_start(self):
        self.real_version = get_real_version(self.log_folder, self.experiment_name)
        
    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        for opt in opts: opt.zero_grad()
        
        loss, loss_dict, tensors_dict = self._shared_eval(batch)

        self.manual_backward(loss)
        for opt in opts: opt.step()

        self.T_losses_list.append(loss_dict)
  
        self.log_dict(loss_dict)

        s = random.randint(0, tensors_dict["img"].shape[0]-1)
        self.log_tb_recon_images((tensors_dict["x"][s], tensors_dict["img"][s], tensors_dict["img_hat"][s]), batch_idx)

    def on_train_epoch_end(self):

        lr_to_log = self._step_schedulers()

        avg_loss = calculate_mean_for_list_of_dicts(self.T_losses_list)

        self.T_losses_list.clear()

        tb_log = {"LR": lr_to_log}
        tb_log = update_dict(tb_log, avg_loss, "avg")
        self.log_dict(tb_log)

    def log_tb_recon_images(self, viz_batch, batch_idx) -> None:

        x, img, img_hat = viz_batch
        grid = make_grid([img, img_hat])

        self.logger.experiment.add_image(f"alpha reconstructions/v{self.real_version}_e{self.current_epoch}_b{batch_idx}_x{x.item():6.2f}", grid, 0)

class Reshaper(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x):
        return x.reshape(-1, *self.output_shape)


def extract_normalize_transform(t_compose):
    out = lambda x: x
    for t in t_compose.transforms:
        if t.__class__.__name__ == "Normalize":
            return lambda x: TF.normalize(x, t.mean, t.std)

class VanillaGAN(nn.Module):
    def __init__(self, emb_size, dataloader_t, img_hw: Tuple[int, int]=[100,200], generator_dims=[64, 16, 4, 1]):
        super().__init__()

        self.generator_dims = generator_dims
        self.emb_size = emb_size

        self.transforms = extract_normalize_transform(dataloader_t)
        
        G_inp_shape, G_inp_flat, G_hws = calculate_intermediate_hws(img_hw, generator_dims)

        #### GENERATOR ####
        generator_modules = [
            DenseBlock(emb_size, G_inp_flat),
            Reshaper(G_inp_shape)]
        
        for i in range(len(generator_dims) - 1):
            generator_modules.append(
                GeneratorBlock(
                    in_channels=generator_dims[i],
                    out_channels=generator_dims[i + 1],
                    out_size=tuple(G_hws[i + 1]),
                )
            )
        generator_modules.append(GeneratorHead(generator_dims[-1], 1))
        self.generator = nn.Sequential(*generator_modules)

        generator_dims.reverse()

        #### DISCRIMINATOR ####
        discriminator_modules = []

        for i in range(len(generator_dims) - 1):
            discriminator_modules.append(
                DiscriminatorBlock(
                    in_channels=generator_dims[i],
                    out_channels=generator_dims[i + 1],
                )
            )

        discriminator_modules.append(nn.Flatten())
        discriminator_modules.append(DenseBlock(G_inp_flat, 2))
        discriminator_modules.append(nn.Softmax(dim=-1))
        self.discriminator = nn.Sequential(*discriminator_modules)


    def generate(self, z):
        x = self.generator(z)
        return self.transforms(x)
    
    def discriminate(self, x):
        return self.discriminator(x)[:, 1]

    

class LitGAN(pl.LightningModule):
    def __init__(self, model, loss_G_fx, loss_D_fx, log_folder="lightning_logs", experiment_name="experiment", lr=1e-2, l2reg=0):
        super().__init__()    

        self.log_folder = log_folder
        self.experiment_name = experiment_name     

        self.automatic_optimization = False
        self.enable_scheduler = False

        self.train_G = False
        self.train_D = False
        self.switch_period = 20 # how many batches/steps 
        self.switch_period_incremental = 1.05
        self.last_step_of_previous_cycle = 0

        self.model = model
        self.loss_G_fx = loss_G_fx
        self.loss_D_fx = loss_D_fx
        

        self.lr = lr
        self.l2reg = l2reg

        self.T_losses_list = []

        self.save_hyperparameters(ignore=['model', 'loss_G_fx', 'loss_D_fx', 'enable_scheduler', 'lr', 'l2reg'])

    def G(self, z):
        return self.model.generate(z)
    
    def D(self, x):
        return self.model.discriminate(x)

    def configure_optimizers(self):

        opts = []
        scheds = []

        opts += [torch.optim.RMSprop(self.model.generator.parameters(), lr=self.lr, weight_decay=self.l2reg)]
        opts += [torch.optim.RMSprop(self.model.discriminator.parameters(), lr=self.lr, weight_decay=self.l2reg)]

        if self.enable_scheduler:
            for opt in opts:
                scheds += [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10,T_mult=2, eta_min=self.lr / 1000000)]

        return opts, scheds
    
    def _step_schedulers(self):
        if self.enable_scheduler:
            highest_lr = 0
            for sched in self.lr_schedulers():
                if sched.get_last_lr()[0] > highest_lr:
                    highest_lr = sched.get_last_lr()[0]
                sched.step()
        else: highest_lr = self.lr
        return highest_lr
    
    def _shared_eval(self, batch):

        loss_D, loss_G = torch.tensor(0.), torch.tensor(0.)

        img_real, _ = batch
        
        B, C, H, W = img_real.shape
        z = torch.randn(B, self.model.emb_size).to(img_real.device)

        img_fake = self.G(z)
        probs_fake = self.D(img_fake)

        probs_real = self.D(img_real)
        loss_D = self.loss_D_fx(probs_fake, probs_real)

        loss_G = self.loss_G_fx(probs_fake)

        loss = loss_G + loss_D

        loss_dict = {"loss": loss, "loss_G": loss_G, "loss_D": loss_D}
        
        tensors_dict = {"z": z, 
                   "img_real": img_real, 
                   "img_fake": img_fake}

        return loss, loss_dict, tensors_dict
    
    def on_fit_start(self):
        self.real_version = get_real_version(self.log_folder, self.experiment_name)

    def D_G_switch(self):
        current_step_in_period = self.global_step - self.last_step_of_previous_cycle
        new_state_of_G = current_step_in_period > (self.switch_period / 2)

        if current_step_in_period==self.switch_period:
            self.last_step_of_previous_cycle = self.global_step
            self.switch_period = int(self.switch_period * self.switch_period_incremental)
        
        self.train_G = new_state_of_G
        self.train_D = not self.train_G

    def training_step(self, batch, batch_idx):
        self.D_G_switch()

        opts = self.optimizers()
        for opt in opts: opt.zero_grad()
        
        loss, loss_dict, tensors_dict = self._shared_eval(batch)

        self.manual_backward(loss)
        if self.train_G: opts[0].step()
        if self.train_D: opts[1].step()

        self.T_losses_list.append(loss_dict)
  
        self.log_dict(loss_dict)

        if (self.global_step % 100) == 0:
            s = random.randint(0, tensors_dict["img_fake"].shape[0]-1)
            self.log_tb_recon_images((tensors_dict["z"][s], tensors_dict["img_fake"][s]), batch_idx)

    def on_train_epoch_end(self):

        lr_to_log = self._step_schedulers()

        avg_loss = calculate_mean_for_list_of_dicts(self.T_losses_list)

        self.T_losses_list.clear()

        tb_log = {"LR": lr_to_log}
        tb_log = update_dict(tb_log, avg_loss, "avg")
        self.log_dict(tb_log)

    def log_tb_recon_images(self, viz_batch, batch_idx) -> None:

        # normalize image!!!
        z, img_fake = viz_batch
        z = z.detach().cpu().numpy().tolist()
        z = f"[{z[0]:4.2f}; {z[1]:4.2f}; {z[2]:4.2f}]"

        self.logger.experiment.add_image(f"G_out/v{self.real_version}_e{self.current_epoch}_b{batch_idx}_z{z}", normalize_image(img_fake.detach().cpu().numpy())/255, 0)
