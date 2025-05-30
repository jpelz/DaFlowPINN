import torch
from .architectures import Generator, Discriminator
from .data import PatchDataset_SingleFile2 as PatchDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
#import os

def label_smoothing(real_labels, smoothing=0.2):
    return real_labels - torch.abs(torch.randn_like(real_labels))*smoothing

def progressbar(progress, length=20):
    progress = int(progress * length / 100)
    bar = '[' + '=' * progress
    bar += '>' if progress < length else ''
    bar += ' ' * (length - progress - 1) + ']'
    return bar

class Generator_Trainer:
    def __init__(self, name, n_train, n_val, n_residual_blocks=3, conditional=False):

        self.name = name
        self.conditional = conditional
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(n_residual_blocks=n_residual_blocks).to(self.device)

        if conditional:
            cond_vec_train = f'/scratch/jpelz/srgan/TrainingData_DA01/full/training/cond_vec_n{n_train}.0.npy'
            cond_vec_val = f'/scratch/jpelz/srgan/TrainingData_DA01/full/validation/cond_vec_n{n_val}.0.npy'
        else:
            cond_vec_train = None
            cond_vec_val = None

        self.training_data = PatchDataset(lr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/training/y_n{n_train}.0_LR.npy",
                                          hr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/training/y_n{n_train}.0_HR.npy", device=self.device, batch_size=512, shuffle=True,
                                          condition_vector=cond_vec_train)
        
        self.training_loader = DataLoader(self.training_data, batch_size=None, shuffle=False)# prefetch_factor=2, num_workers=2)

        self.validation_data = PatchDataset(lr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/validation/y_n{n_val}.0_LR.npy",
                                            hr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/validation/y_n{n_val}.0_HR.npy", device=self.device, batch_size=512, shuffle=True,
                                            condition_vector=cond_vec_val)

        self.validation_loader = DataLoader(self.validation_data, batch_size=None, shuffle=False)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)


        self.loss_func = torch.nn.functional.mse_loss

        self.epoch = 0
        self.iteration = 0

    def train_step(self, data):
        if self.conditional:
            low_resolution, high_resolution, _ = data
        else:
            low_resolution, high_resolution = data
        
        low_resolution, high_resolution = low_resolution, high_resolution

        self.generator_optimizer.zero_grad()

        super_resolution = self.generator(low_resolution)
        
        mse_loss = self.loss_func(super_resolution, high_resolution)
        

        mse_loss.backward()

        self.generator_optimizer.step()

        return mse_loss.detach()
    
    def validation_step(self, data):
        with torch.no_grad():
            if self.conditional:
                low_resolution, high_resolution, _ = data
            else:
                low_resolution, high_resolution = data

            low_resolution, high_resolution = low_resolution, high_resolution

            super_resolution = self.generator(low_resolution)
            mse_loss = self.loss_func(super_resolution, high_resolution)

            return mse_loss.detach()
        

    def train(self, epochs=1000):

        train_summary_writer = SummaryWriter(f'logs/{self.name}/train/{time.strftime("%Y-%m-%d_%H-%M-%S")}')
        valid_summary_writer = SummaryWriter(f'logs/{self.name}/valid/{time.strftime("%Y-%m-%d_%H-%M-%S")}')
        


        for epoch in range(epochs):
            self.epoch = epoch

            print(f'\n---- Epoch {epoch} ----')

            for i, data in enumerate(self.training_loader):
                self.iteration += 1
                mse_loss = self.train_step(data)
                progress = (i + 1) / len(self.training_loader) * 100
            
                print(f'\r{progressbar(progress)} {progress:.2f}% - MSE Loss: {mse_loss:.4e}', end='')
                train_summary_writer.add_scalar('MSE Loss', mse_loss, self.iteration)

            print('\n', end='')

            MSE_valid_epoch = []

            for  i, data in enumerate(self.validation_loader):
                mse_loss = self.validation_step(data)
                MSE_valid_epoch.append(mse_loss)

            MSE_valid_epoch = torch.mean(torch.stack(MSE_valid_epoch))

            print(f'Validation - MSE Loss: {MSE_valid_epoch:.4e}')
            valid_summary_writer.add_scalar('MSE Loss', MSE_valid_epoch, epoch)

            if epoch % 100 == 0:
                torch.save(self.generator.state_dict(), f'checkpoints/generator_{self.name}_{epoch}.pt')





class SRGAN_Trainer:
    def __init__(self, name, n_train, n_val, n_residual_blocks=3, conditional=False):
        #torch.autograd.set_detect_anomaly(True)

        self.name = name
        self.conditional = conditional

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(n_residual_blocks=n_residual_blocks).to(self.device)
        self.discriminator = Discriminator(conditional=conditional).to(self.device)

        if conditional:
            cond_vec_train = f'/scratch/jpelz/srgan/TrainingData_DA01/full/training/cond_vec_n{n_train}.0.npy'
            cond_vec_val = f'/scratch/jpelz/srgan/TrainingData_DA01/full/validation/cond_vec_n{n_val}.0.npy'
        else:
            cond_vec_train = None
            cond_vec_val = None

        self.training_data = PatchDataset(
            lr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/training/y_n{n_train}.0_LR.npy",
            hr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/training/y_n{n_train}.0_HR.npy",
            device=self.device, batch_size=512, shuffle=True,
            condition_vector=cond_vec_train
        )
        self.training_loader = DataLoader(self.training_data, batch_size=None, shuffle=False)

        self.validation_data = PatchDataset(
            lr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/validation/y_n{n_val}.0_LR.npy",
            hr_file=f"/scratch/jpelz/srgan/TrainingData_DA01/full/validation/y_n{n_val}.0_HR.npy",
            device=self.device, batch_size=512, shuffle=True,
            condition_vector=cond_vec_val
        )
        self.validation_loader = DataLoader(self.validation_data, batch_size=None, shuffle=False)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.adversarial_ratio = 1e-3

        self.loss_func = torch.nn.functional.mse_loss
        self.cross_entropy = torch.nn.functional.binary_cross_entropy

        self.epoch = 0
        self.iteration = 0

    def train_step(self, data):
        if self.conditional:
            low_resolution, high_resolution, cond_vec = data
            low_resolution, high_resolution, cond_vec = low_resolution, high_resolution, cond_vec
        else:
            low_resolution, high_resolution = data
            low_resolution, high_resolution = low_resolution, high_resolution
            cond_vec = None

        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        super_resolution = self.generator(low_resolution)

        mse_loss = self.loss_func(super_resolution, torch.clamp(high_resolution, 0, 1))

        self.discriminator.eval()
        sr_feedback = self.discriminator(super_resolution, condition=cond_vec)

        self.discriminator.train()
        if self.iteration % 2 == 0:
            fake_output = self.discriminator(super_resolution, condition=cond_vec)
            disc_loss = self.cross_entropy(fake_output, torch.clamp(torch.zeros_like(fake_output), 0, 1))
        else:
            real_output = self.discriminator(high_resolution, condition=cond_vec)
            soft_real_labels = label_smoothing(torch.ones_like(real_output))
            disc_loss = self.cross_entropy(real_output, torch.clamp(soft_real_labels, 0, 1))

        adv_loss_g = self.cross_entropy(sr_feedback, torch.clamp(torch.ones_like(sr_feedback), 0, 1))
        gen_loss = mse_loss + adv_loss_g * self.adversarial_ratio

        gen_loss.backward(retain_graph=True)
        disc_loss.backward()

        self.generator_optimizer.step()
        self.discriminator_optimizer.step()

        return mse_loss.detach(), adv_loss_g.detach(), disc_loss.detach()

    def validation_step(self, data):
        with torch.no_grad():
            if self.conditional:
                low_resolution, high_resolution, cond_vec = data
                low_resolution, high_resolution, cond_vec = low_resolution, high_resolution, cond_vec
            else:
                low_resolution, high_resolution = data
                low_resolution, high_resolution = low_resolution, high_resolution
                cond_vec = None

            super_resolution = self.generator(low_resolution)
            mse_loss = self.loss_func(super_resolution, torch.clamp(high_resolution, 0, 1))

            fake_output = self.discriminator(super_resolution, condition=cond_vec)
            real_output = self.discriminator(high_resolution, condition=cond_vec)
            disc_loss_fake = self.cross_entropy(fake_output, torch.clamp(torch.zeros_like(fake_output), 0, 1))
            disc_loss_real = self.cross_entropy(real_output, torch.clamp(torch.ones_like(real_output), 0, 1))
            disc_loss = 0.5 * (disc_loss_fake + disc_loss_real)

            adv_loss_g = self.cross_entropy(fake_output, torch.clamp(torch.ones_like(fake_output), 0, 1))

            return mse_loss.detach(), adv_loss_g.detach(), disc_loss.detach()

    def train(self, epochs=1000):

        train_summary_writer = SummaryWriter(f'logs/{self.name}/train/{time.strftime("%Y-%m-%d_%H-%M-%S")}')
        valid_summary_writer = SummaryWriter(f'logs/{self.name}/valid/{time.strftime("%Y-%m-%d_%H-%M-%S")}')

        for epoch in range(epochs):
            self.epoch = epoch

            print(f'\n---- Epoch {epoch} ----')

            for i, data in enumerate(self.training_loader):
                self.iteration += 1
                mse_loss, adv_loss_g, disc_loss = self.train_step(data)
                progress = (i + 1) / len(self.training_loader) * 100

                print(f'\r{progressbar(progress)} {progress:.2f}% - MSE Loss: {mse_loss:.4e}, Adv Loss G: {adv_loss_g:.4e}, Disc Loss: {disc_loss:.4e}', end='')
                train_summary_writer.add_scalar('MSE Loss', mse_loss, self.iteration)
                train_summary_writer.add_scalar('Adv Loss Generator', adv_loss_g, self.iteration)
                train_summary_writer.add_scalar('Adv Loss Discriminator', disc_loss, self.iteration)

            print('\n', end='')

            MSE_valid_epoch, adv_loss_g_valid_epoch, disc_loss_valid_epoch = [], [], []

            for i, data in enumerate(self.validation_loader):
                mse_loss, adv_loss_g, disc_loss = self.validation_step(data)
                MSE_valid_epoch.append(mse_loss)
                adv_loss_g_valid_epoch.append(adv_loss_g)
                disc_loss_valid_epoch.append(disc_loss)

            MSE_valid_epoch = torch.mean(torch.stack(MSE_valid_epoch))
            adv_loss_g_valid_epoch = torch.mean(torch.stack(adv_loss_g_valid_epoch))
            disc_loss_valid_epoch = torch.mean(torch.stack(disc_loss_valid_epoch))

            print(f'Validation - MSE Loss: {MSE_valid_epoch:.4e}, Adv Loss G: {adv_loss_g_valid_epoch:.4e}, Disc Loss: {disc_loss_valid_epoch:.4e}')
            valid_summary_writer.add_scalar('MSE Loss', MSE_valid_epoch, epoch)
            valid_summary_writer.add_scalar('Adv Loss Generator', adv_loss_g_valid_epoch, epoch)
            valid_summary_writer.add_scalar('Adv Loss Discriminator', disc_loss_valid_epoch, epoch)

            if epoch % 100 == 0:
                torch.save(self.generator.state_dict(), f'checkpoints/generator_{self.name}_{epoch}.pt')
                torch.save(self.discriminator.state_dict(), f'checkpoints/discriminator_{self.name}_{epoch}.pt')
                
class SRGAN_Trainer_old:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.training_data = PatchDataset(data_directory='/content/drive/MyDrive/MA/_SuperResolution', device=self.device, batch_size=16, shuffle=True)
        self.training_loader = DataLoader(self.training_data, batch_size=1, shuffle=False)

        self.validation_data = PatchDataset(data_directory='/content/drive/MyDrive/MA/_SuperResolution', device=self.device, batch_size=16, shuffle=True)
        self.validation_loader = DataLoader(self.validation_data, batch_size=1, shuffle=False)

        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.adversarial_ratio = 1e-3

        self.loss_func = torch.nn.functional.mse_loss
        self.cross_entropy = torch.nn.functional.binary_cross_entropy

        self.epoch = 0
        self.iteration = 0

    def train_step(self, data):
        low_resolution, high_resolution = data
        low_resolution, high_resolution = low_resolution[0], high_resolution[0]

        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

        
        super_resolution = self.generator(low_resolution)
        
        mse_loss = self.loss_func(super_resolution, high_resolution)

        self.discriminator.eval()
        sr_feedback = self.discriminator(super_resolution)

        self.discriminator.train()
        if self.iteration % 2 == 0:
            fake_output = self.discriminator(super_resolution)
            disc_loss = self.cross_entropy(fake_output, torch.zeros_like(fake_output), )
        else:
            real_output = self.discriminator(high_resolution)
            soft_real_labels = label_smoothing(torch.ones_like(real_output))
            disc_loss = self.cross_entropy(real_output, soft_real_labels)

        adv_loss_g = self.cross_entropy(sr_feedback, torch.ones_like(sr_feedback))
        gen_loss = mse_loss + adv_loss_g * self.adversarial_ratio

        gen_loss.backward(retain_graph=True)
        disc_loss.backward()

        self.generator_optimizer.step()
        self.discriminator_optimizer.step()

        return mse_loss.detach(), adv_loss_g.detach(), disc_loss.detach()
    
    def validation_step(self, data):
        with torch.no_grad():
            low_resolution, high_resolution = data
            low_resolution, high_resolution = low_resolution[0], high_resolution[0]

            super_resolution = self.generator(low_resolution)
            mse_loss = self.loss_func(super_resolution, high_resolution)

            fake_output = self.discriminator(super_resolution)
            real_output = self.discriminator(high_resolution)
            disc_loss_fake = self.cross_entropy(fake_output, torch.zeros_like(fake_output))
            disc_loss_real = self.cross_entropy(real_output, torch.ones_like(real_output),)
            disc_loss = 0.5 * (disc_loss_fake + disc_loss_real)

            adv_loss_g = self.cross_entropy(fake_output, torch.ones_like(fake_output))

            return mse_loss.detach(), adv_loss_g.detach(), disc_loss.detach()
        

    def train(self, epochs=100):

        train_summary_writer = SummaryWriter('logs/train')
        valid_summary_writer = SummaryWriter('logs/valid')
        


        for epoch in range(epochs):
            self.epoch = epoch

            print(f'\n---- Epoch {epoch} ----')

            for i, data in enumerate(self.training_loader):
                self.iteration += 1
                mse_loss, adv_loss_g, disc_loss = self.train_step(data)
                progress = (i + 1) / len(self.training_loader) * 100
            
                print(f'\r{progressbar(progress)} {progress:.2f}% - MSE Loss: {mse_loss:.4e}, Adv Loss G: {adv_loss_g:.4e}, Disc Loss: {disc_loss:.4e}', end='')
                train_summary_writer.add_scalar('MSE Loss', mse_loss, self.iteration)
                train_summary_writer.add_scalar('Adv Loss Generator', adv_loss_g, self.iteration)
                train_summary_writer.add_scalar('Adv Loss Discriminator', disc_loss, self.iteration)

            print('\n', end='')

            MSE_valid_epoch, adv_loss_g_valid_epoch, disc_loss_valid_epoch = [], [], []

            for  i, data in enumerate(self.validation_loader):
                mse_loss, adv_loss_g, disc_loss = self.validation_step(data)
                MSE_valid_epoch.append(mse_loss)
                adv_loss_g_valid_epoch.append(adv_loss_g)
                disc_loss_valid_epoch.append(disc_loss)

            MSE_valid_epoch = torch.mean(torch.stack(MSE_valid_epoch))
            adv_loss_g_valid_epoch = torch.mean(torch.stack(adv_loss_g_valid_epoch))
            disc_loss_valid_epoch = torch.mean(torch.stack(disc_loss_valid_epoch))

            print(f'Validation - MSE Loss: {MSE_valid_epoch:.4e}, Adv Loss G: {adv_loss_g_valid_epoch:.4e}, Disc Loss: {disc_loss_valid_epoch:.4e}')
            valid_summary_writer.add_scalar('MSE Loss', MSE_valid_epoch, epoch)
            valid_summary_writer.add_scalar('Adv Loss Generator', adv_loss_g_valid_epoch, epoch)
            valid_summary_writer.add_scalar('Adv Loss Discriminator', disc_loss_valid_epoch, epoch)

            if epoch % 1000 == 0:
                torch.save(self.generator.state_dict(), f'checkpoints/generator_{epoch}.pt')
                torch.save(self.discriminator.state_dict(), f'checkpoints/discriminator_{epoch}.pt')