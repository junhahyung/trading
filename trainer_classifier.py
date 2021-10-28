import os
import tqdm
import wandb
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

        x, _, y, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        # set to the same dimension

        self.model.train()

        self.optimizer.zero_grad()
        pred = self.model(x) # (bs, ntarget*target_equity*3)
        pred = pred.view(-1, 3) # (-1, 3)
        loss = self.loss_fn(pred, y.view(-1))
        loss.backward()
        self.optimizer.step()

        _, max_ind = torch.max(pred, -1) # (-1)
        acc = self.calc_acc(max_ind, y.view(-1))


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
            val_acc_sum = 0
            count = 0
            true_list = []
            pred_list = []

            for val_batch in tqdm.tqdm(self.dataloader_test):
                _vl, bs, p, t = self.validate_step(val_batch)
                # val loss
                val_loss_sum += _vl[0]*bs
                # val acc
                val_acc_sum += _vl[1]*bs
                count += bs

                pred_list += p.cpu().tolist()
                true_list += t.cpu().tolist()

            val_loss = val_loss_sum / count
            val_acc = val_acc_sum / count
            #confusion = get_confusion(pred_list, true_list)
            wandb.log({"train_loss": loss, "train_acc": acc, "val_loss": val_loss, "val_acc": val_acc, "epoch": epoch, "conf_mat": wandb.plot.confusion_matrix(y_true=true_list, preds=pred_list, class_names=["0", "pos", "neg"])}, step=step)
            print(step)
            print(self.get_confusion(pred_list, true_list))
        else:
            wandb.log({"train_loss": loss, "train_acc": acc, "epoch": epoch}, step=step)


    def validate_step(self, batch):
        x, _, y, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        bs = x.shape[0]

        self.model.eval()
        with torch.no_grad():
            pred = self.model(x) # (bs, ntarget*target_equity*3)
            pred = pred.view(-1, 3) # (-1, 3)
            _, max_ind = torch.max(pred, -1) # (-1)
            val_loss = self.loss_fn(pred, y.view(-1))
            acc = self.calc_acc(max_ind, y.view(-1))

        val_loss = [val_loss, acc]

        return val_loss, bs, max_ind, y.view(-1)


    @staticmethod
    def get_confusion(pred_list, target_list):
        return confusion_matrix(target_list, pred_list)


    @staticmethod
    def calc_acc(pred, target):
        # (bs)

        assert len(pred.shape) == 1
        assert len(target.shape) == 1

        return torch.sum(pred==target) / pred.shape[0]






