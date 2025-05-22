from torch.optim.lr_scheduler import LRScheduler
from typing import List

class ReduceLROnPlateau_custom(LRScheduler):
    """
    Custom learning rate scheduler that reduces the learning rate when a metric has stopped improving.

    Attributes:
        optimizer: torch.optim.Optimizer
            Wrapped optimizer whose learning rate will be scheduled.
        epoch_window: int
            Number of epochs to consider for loss stagnation.
        factor: float
            Factor by which the learning rate will be reduced.
        threshold: float
            Threshold for detecting loss stagnation.
        cooldown: int
            Number of epochs to wait before resuming normal operation after lr has been reduced.
        cooldown_counter: int
            Counter for cooldown epochs.
        last_n_losses: List[float]
            List of recent loss values.
        mean_loss: float
            Mean of the recent loss values.
        last_epoch: int
            Index of the last epoch.

    Methods:
        step(loss: float) -> None:
            Update the learning rate if loss stagnates.
    """
    def __init__(
        self,
        optimizer,
        epoch_window: int = 100,
        factor: float = 0.1,
        threshold: float = 1e-4,
        cooldown: int = 10
    ):
        """
        Args:
            optimizer: Wrapped optimizer.
            epoch_window (int): Number of epochs to consider for loss stagnation.
            factor (float): Factor by which the learning rate will be reduced.
            threshold (float): Threshold for detecting loss stagnation.
            cooldown (int): Number of epochs to wait before resuming normal operation after lr has been reduced.
        """
        self.epoch_window: int = epoch_window
        self.factor: float = factor
        self.optimizer = optimizer
        self.last_epoch: int = 0
        self.threshold: float = threshold
        self.cooldown: int = cooldown
        self.cooldown_counter: int = 0
        self.last_n_losses: List[float] = []
        self.mean_loss: float = float('inf')

        self._last_lr: List[float] = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, loss: float) -> None:
        """
        Update the learning rate if loss stagnates.

        Args:
            loss (float): Current loss value.
        """
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

            # Check if loss has stagnated
            if abs(max_loss - min_loss) / self.mean_loss < self.threshold:
                self.__reduce_lr()
                self.cooldown_counter = self.cooldown

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def __reduce_lr(self) -> None:
        """
        Reduce the learning rate for all parameter groups.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr: float = float(param_group['lr'])
            new_lr: float = old_lr * self.factor
            param_group['lr'] = new_lr
            self._last_lr[i] = new_lr


class WarmupScheduler(LRScheduler):
    """
    Learning rate scheduler with a warmup phase.
    Gradually increases the learning rate from an initial value (default 0) to the target learning rate(s)
    over a specified number of epochs.
    Args:
        optimizer: torch.optim.Optimizer
            Wrapped optimizer whose learning rate will be scheduled.
        epochs: int, optional
            Number of epochs over which to linearly increase the learning rate to the target value(s).
            Default is 20.
    Attributes:
        optimizer: torch.optim.Optimizer
            The optimizer being scheduled.
        epochs: int
            Number of warmup epochs.
        aim_lr: list[float]
            Target learning rates for each parameter group.
        init_lr: float
            Initial learning rate (default 0).
        new_lr: float
            Current learning rate during warmup.
        last_epoch: int
            Index of the last epoch.
    Methods:
        step(*args, **kwargs):
            Updates the learning rate for each parameter group according to the warmup schedule.
    """
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


          


    