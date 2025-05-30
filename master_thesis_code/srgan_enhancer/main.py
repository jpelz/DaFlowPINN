import sys

sys.path.append("/scratch/jpelz/srgan")

from SRGAN.training import SRGAN_Trainer, Generator_Trainer

#trainer = SRGAN_Trainer()

name = sys.argv[1]
# n_train = sys.argv[2]
# n_val = sys.argv[3]

#trainer = Generator_Trainer(name=name, n_train=40000, n_val=10000, n_residual_blocks=3, conditional=True)
trainer = SRGAN_Trainer(name=name, n_train=40000, n_val=10000, n_residual_blocks=3, conditional=True)
trainer.train()

