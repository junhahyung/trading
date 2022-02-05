import os
import tqdm
import wandb
import torch

class Trainer():
    def __init__(self, args, model, optimizer, dataloader_train, dataloader_test, loss_fn, device):
        self.args = args
        self.model = model.to(device)
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.n_epochs = args.training.n_epochs

        self.output_dir = args.training.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"output dir: {self.output_dir}")


    def train(self):
        wandb.init(entity='junha', project="trading")
        global_step = 0

        for epoch in range(self.n_epochs):
            print(f'start epoch [{epoch}]')
            for step, batch in enumerate(tqdm.tqdm(self.dataloader_train)):
                global_step += step
                self.train_step(batch, global_step, epoch)




    def train_step(self, batch, step, epoch):
        # x: (bs, seq_length, dim)
        # y: (bs, ntarget, target_dim)
        # anchor: (bs, 2, target_dim) first dim: prev_mean, second dim: prev

        x, y, _, anchor = batch
        x = x.to(self.device)
        y = y.to(self.device)
        anchor = anchor.to(self.device)
        prev = anchor[:,1]
        # set to the same dimension
        prev = prev.unsqueeze(1).repeat(1, y.shape[1], 1) # (bs, ntarget, target_dim)

        self.model.train()

        self.optimizer.zero_grad()
        pred = self.model(x, prev)
        loss = self.loss_fn(pred, y.view(pred.shape[0], -1))
        loss.backward()
        self.optimizer.step()


        if step % self.args.training.save_freq == 0:
            save_dict = {}
            save_dict["model"] = self.model.state_dict()
            save_dict["step"] = step
            save_name = os.path.join(self.output_dir, f"{epoch}_{step}.pth")
            torch.save(save_dict, save_name)
            print(f"saved model {save_name}")

        if step % self.args.training.val_freq == 0:
            print('[start validation]')
            val_loss_sum = 0
            prev_mean_loss_sum = 0
            prev_loss_sum = 0
            count = 0
            for val_batch in tqdm.tqdm(self.dataloader_test):
                _vl, bs = self.validate_step(val_batch)
                # val loss
                val_loss_sum += _vl[0]*bs
                # prev mean loss
                prev_mean_loss_sum += _vl[1]*bs
                # prev loss
                prev_loss_sum += _vl[2]*bs
                count += bs
            val_loss = val_loss_sum / count
            prev_mean_loss = prev_mean_loss_sum / count
            prev_loss = prev_loss_sum / count
            wandb.log({"train_loss": loss, "val_loss": val_loss, "prev_mean_loss": prev_mean_loss, "prev_loss": prev_loss, "epoch": epoch}, step=step)
        else:
            wandb.log({"train_loss": loss, "epoch": epoch}, step=step)


    def validate_step(self, batch):
        # anchor: (bs, 2, target_dim) first dim: prev_mean, second dim: prev
        x, y, _, anchor = batch
        x = x.to(self.device)
        y = y.to(self.device)

        anchor = anchor.to(self.device)
        prev_mean = anchor[:,0]
        prev = anchor[:,1]
        # set to the same dimension
        prev_mean = prev_mean.unsqueeze(1).repeat(1, y.shape[1], 1)
        prev = prev.unsqueeze(1).repeat(1, y.shape[1], 1) # (bs, ntarget, target_dim)
        

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x, prev)
            bs = pred.shape[0]
            val_loss = self.loss_fn(pred, y.view(bs, -1))
            prev_mean_loss = self.loss_fn(prev_mean.view(bs, -1), y.view(bs, -1))
            prev_loss = self.loss_fn(prev.view(bs, -1), y.view(bs, -1))

        bs = y.shape[0]
        val_loss = [val_loss, prev_mean_loss, prev_loss]

        return val_loss,bs






