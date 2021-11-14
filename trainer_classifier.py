import os
import tqdm
import wandb
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# trainer class
class Trainer():
    def __init__(self, args, models, optimizers, dataloader_train, dataloader_test, loss_fn, device):
        self.args = args
        self.models = []
        self.optimizers = []
        for model in models:
            self.models.append(model.to(device))

        for optimizer in optimizers:
            self.optimizers.append(optimizer)

        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.device = device
        self.loss_fn = loss_fn.to(device)
        self.n_epochs = args.training.n_epochs
        self.nway = args.training.nway
        if self.nway == 3:
            self.class_names = ["0", "pos", "neg"] 
        elif self.nway == 2:
            self.class_names = ["neg", "pos"] 


        self.best_acc = 0
        self.softmax = nn.Softmax(dim=1)

        self.output_dir = os.path.join(args.training.output_dir, args.name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"output dir: {self.output_dir}")


    # training function
    def train(self):
        wandb.init(entity='junha', project="trading", name=self.args.name)
        global_step = 0

        for epoch in range(self.n_epochs):
            print(f'start epoch [{epoch}]')
            for step, batch in enumerate(tqdm.tqdm(self.dataloader_train)):
                global_step += step
                # train step
                self.train_step(batch, global_step, epoch)

        # write summary file
        fn = os.path.join(self.output_dir, 'summary.txt')
        with open(fn, 'w') as fp:
            fp.write(str(self.args))
            fp.write('\n==================\n')
            fp.write(f'best val acc: {self.best_acc}\n')
            fp.write(f'best confusion: {self.best_confusion}')


    # save model checkpoint
    def save_model(self, name, epoch, step):
        save_dict = {}
        state_dicts = []
        for model in self.models:
            state_dicts.append(model.state_dict())
        save_dict["model"] = state_dicts
        save_dict["step"] = step
        save_name = os.path.join(self.output_dir, f"{name}.pth")
        torch.save(save_dict, save_name)
        print(f"saved model {save_name}")


    # train step function
    def train_step(self, batch, step, epoch):
        # x: (bs, seq_length, dim)
        # y: (bs, ntarget, target_dim)

        # prepare data pair
        x, _, y, _, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # calculate and update each model
        preds = []
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
            pred = model(x) # (bs, ntarget*target_equity*3 or (bs, ntarget*target_equity*2)
            pred = pred.view(-1, self.nway) # (-1, 3) or (-1, 2)
            preds.append(pred)
            loss = self.loss_fn(pred, y.view(-1))
            loss.backward()
            optimizer.step()

        # calculate acc
        pred = torch.stack(preds).mean(0)
        _, max_ind = torch.max(pred, -1) # (-1)
        acc = self.calc_acc(max_ind, y.view(-1))


        # validation step
        if step % self.args.training.val_freq == 0:
            print('[start validation]')
            val_loss_sum = 0
            val_acc_sum = 0
            val_thres_acc_sum = 0
            count = 0 # sum of batch sizes
            t_count = 0 # number of datapoints above thres
            a_count = 0 # number of datapoints
            true_list = []
            pred_list = []

            for val_batch in tqdm.tqdm(self.dataloader_test):
                _vl, bs, p, t = self.validate_step(val_batch)
                bs, t_bs, a_bs = bs
                # val loss
                val_loss_sum += _vl[0]*bs
                # val acc
                val_acc_sum += _vl[1]*bs
                # val thres acc
                val_thres_acc_sum += _vl[2]*t_bs
                count += bs
                t_count += t_bs
                a_count += a_bs

                pred_list += p.cpu().tolist()
                true_list += t.cpu().tolist()

            val_loss = val_loss_sum / count
            val_acc = val_acc_sum / count
            if t_count == 0:
                val_thres_acc = 0
            else:
                val_thres_acc = val_thres_acc_sum / t_count
            confusion = self.get_confusion(pred_list, true_list)
            wandb.log({"train_loss": loss, "train_acc": acc, "val_loss": val_loss, "val_acc": val_acc, "val_thres_acc": val_thres_acc, "above_thres_prob": t_count/float(a_count), "epoch": epoch, "conf_mat": wandb.plot.confusion_matrix(y_true=true_list, preds=pred_list, class_names=self.class_names)}, step=step)
            if self.best_acc < val_acc:
                self.best_acc = val_acc
                self.best_confusion = confusion
                print(f'current best acc: {self.best_acc}')
                print(f'current best confusion: {self.best_confusion}')
                self.save_model('best_model', epoch, step)

        else:
            wandb.log({"train_loss": loss, "train_acc": acc, "epoch": epoch}, step=step)


    def validate_step(self, batch):
        x, _, y, _, _ = batch
        x = x.to(self.device)
        y = y.to(self.device)
        bs = x.shape[0]

        for model in self.models:
            model.eval()

        preds = []
        with torch.no_grad():
            for model in self.models:
                pred = model(x) # (bs, ntarget*target_equity*3) or (bs, ntarget*target_equity*2)
                pred = pred.view(-1, self.nway) # (-1, 3) or (-1, 2)
                preds.append(pred)

            pred = torch.stack(preds).mean(0)
            a_bs = pred.shape[0]
            _, max_ind = torch.max(pred, -1) # (-1)
            val_loss = self.loss_fn(pred, y.view(-1))
            acc = self.calc_acc(max_ind, y.view(-1))

            pred = self.softmax(pred)
            trunc_pred = torch.where(pred > 0.9, pred, torch.zeros_like(pred).to(self.device))
            logit_sum = torch.sum(trunc_pred, -1)
            nonzero_ind = torch.nonzero(logit_sum)
            trunc_bs = len(nonzero_ind)
            #print(nonzero_ind)
            #print(trunc_bs)
            trunc_max_ind  = max_ind[nonzero_ind]
            trunc_y = y.view(-1)[nonzero_ind]
            if trunc_bs != 0:
                thres_acc = self.calc_acc(trunc_max_ind.view(-1), trunc_y.view(-1))
            else:
                thres_acc = 0
            #np.where(pred > 0.8)
            #np.where(pred > 0.9)

        val_loss = [val_loss, acc, thres_acc]
        bs = [bs, trunc_bs, a_bs]

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






