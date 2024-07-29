import os
import torch
from tqdm import tqdm
import mlflow
import optuna


class SNGPUtils:
    def activate_uncert_head(self, deactivate_backbone=True):
        """
        Activates the einet output layer for second stage training and inference.
        """
        return

    def deactivate_uncert_head(self):
        """
        Deactivates the einet output layer for first stage training.
        """
        return

    def start_train(
        self,
        train_dl,
        valid_dl,
        device,
        num_epochs,
        learning_rate,
        early_stop=10,
        checkpoint_dir=None,
        trial=None,
        **kwargs,
    ):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        val_increase = 0
        lowest_val_loss = torch.inf
        epoch = 0
        t = tqdm(range(num_epochs))
        for epoch in t:
            t.set_description(f"Epoch {epoch}")
            loss = 0.0
            self.sngp.reset_precision_matrix()
            for data, target in train_dl:
                optimizer.zero_grad()
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss_v = torch.nn.CrossEntropyLoss()(output, target)
                loss += loss_v.item()
                loss_v.backward()
                optimizer.step()
            t.set_postfix(dict(train_loss=loss / len(train_dl.dataset)))
            val_loss = 0.0
            with torch.no_grad():
                for data, target in valid_dl:
                    optimizer.zero_grad()
                    target = target.type(torch.LongTensor)
                    data, target = data.to(device), target.to(device)
                    output = self(data)
                    loss_v = torch.nn.CrossEntropyLoss()(output, target)
                    val_loss += loss_v.item()
            t.set_postfix(
                dict(
                    train_loss=loss / len(train_dl.dataset),
                    val_loss=val_loss / len(valid_dl.dataset),
                )
            )
            mlflow.log_metric(
                key="train_loss", value=loss / len(train_dl.dataset), step=epoch
            )
            mlflow.log_metric(
                key="val_loss", value=val_loss / len(valid_dl.dataset), step=epoch
            )
            # early stopping via optuna
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    mlflow.set_tag("pruned", f"{epoch}")
                    raise optuna.TrialPruned()
            if val_loss <= lowest_val_loss:
                lowest_val_loss = val_loss
                val_increase = 0
                if checkpoint_dir is not None:
                    self.save(checkpoint_dir + "checkpoint.pt")
            else:
                # early stopping when val increases
                val_increase += 1
                if val_increase >= early_stop:
                    print(
                        f"Stopping early, val loss increased for the last {early_stop} epochs."
                    )
                    break
        if checkpoint_dir is not None:
            # load best
            self.load(checkpoint_dir + "checkpoint.pt")
        return lowest_val_loss

    def save(self, path):
        # create path if not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, backbone_only=False):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)
