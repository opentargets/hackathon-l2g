import torch
import torch.nn.functional as F
import mlflow
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        """
        patience: epochs to wait before stopping
        min_delta: minimal improvement to reset counter
        mode: 'min' (e.g., val_loss) or 'max' (e.g., val_acc)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.best_state = None
        self.should_stop = False

    def step(self, current_value, model):
        if self.best_value is None:
            self.best_value = current_value
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        improved = (
            current_value < self.best_value - self.min_delta
            if self.mode == "min"
            else current_value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = current_value
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader=None, device="cuda", run_name=None, early_stopping=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_name = run_name
        self.early_stopping = early_stopping

    def train(self, num_epochs):
        with mlflow.start_run(run_name=self.run_name):
            mlflow.log_params({
                "num_epochs": num_epochs,
                "optimizer": self.optimizer.__class__.__name__,
                "lr": self.optimizer.param_groups[0]["lr"],
                "model_params": sum(p.numel() for p in self.model.parameters())
            })

            for epoch in range(1, num_epochs + 1):
                train_loss = self._train_epoch()
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                print(f"Epoch {epoch:03d} | train_loss: {train_loss:.4f}", end="")

                if self.val_loader:
                    val_loss, val_acc, val_fdr = self._validate_epoch()
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_acc", val_acc, step=epoch)
                    mlflow.log_metric("val_fdr", val_fdr, step=epoch)
                    print(f" | val_loss: {val_loss:.4f} | val_acc: {val_acc:.3f} | val_FDR: {val_fdr:.3f}")

                    if self.early_stopping and self.early_stopping.step(val_loss, self.model):
                        print(f"⏹️  Early stopping triggered at epoch {epoch}.")
                        self.early_stopping.restore_best_weights(self.model)
                        break
                else:
                    print()

    def _train_epoch(self):
        self.model.train()
        total_loss, total_count = 0.0, 0

        for x, y, mask in self.train_loader:
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            logits, _ = self.model(x, padding_mask=mask)
            loss = F.cross_entropy(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)

        return total_loss / total_count

    def _validate_epoch(self):
        self.model.eval()
        total_loss, total_correct, total_count = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y, mask in self.val_loader:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                logits, _ = self.model(x, padding_mask=mask)
                loss = F.cross_entropy(logits, y)

                total_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=-1)

                total_correct += (preds == y).sum().item()
                total_count += x.size(0)

                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        # compute FDR
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        fp = ((all_preds != all_labels)).sum().item()
        tp_plus_fp = len(all_preds)
        fdr = fp / tp_plus_fp if tp_plus_fp > 0 else 0.0

        val_loss = total_loss / total_count
        val_acc = total_correct / total_count
        return val_loss, val_acc, fdr