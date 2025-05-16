from torch.optim.lr_scheduler import LRScheduler

class ReduceLROnPlateau_custom(LRScheduler):
    def __init__(self, optimizer, epoch_window: int = 100, factor: float = 0.1, threshold: float = 1e-4, cooldown: int = 10):
        self.epoch_window = epoch_window
        self.factor = factor
        self.optimizer = optimizer
        self.last_epoch = 0
        self.threshold = threshold
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.last_n_losses = []
        self.mean_loss = float('inf')

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, loss):
        loss = float(loss)
        self.last_epoch += 1
        if len(self.last_n_losses) < self.epoch_window:
            self.last_n_losses.append(loss)
        else:
            self.last_n_losses.pop(0)
            self.last_n_losses.append(loss)

        if self.last_epoch > self.epoch_window:

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return

            self.mean_loss = sum(self.last_n_losses) / len(self.last_n_losses)
            max_loss = max(self.last_n_losses)
            min_loss = min(self.last_n_losses)

            if abs(max_loss - min_loss)/self.mean_loss < self.threshold:
                self._reduce_lr()
                self.cooldown_counter = self.cooldown

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.factor
            param_group['lr'] = new_lr
            self._last_lr[i] = new_lr


class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer, epochs = 20):
        self.optimizer = optimizer
        self.epochs = epochs
        self.aim_lr = [group["lr"] for group in self.optimizer.param_groups]
        self.init_lr = 0
        self.new_lr = self.init_lr
        self._last_lr = self.init_lr
        self.last_epoch = 0
  
    def step(self, *args, **kwargs):
        if self.last_epoch < self.epochs-1:
          self.last_epoch += 1
          
          for i, group in enumerate(self.optimizer.param_groups):
            self.new_lr += (self.aim_lr[i]-self.init_lr) / (self.epochs)
            group['lr'] = self.new_lr 

          self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


          


    