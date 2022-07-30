import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from TS_Dataset import *
# Get cpu or gpu device for training.
from model import NeuralNetwork_pl

t_min = 0
t_max = 10000




def timepoint(t, divisor = 10):
    t = t / divisor
    return np.sqrt(t) * np.sin(t) / 3 + np.sin(t * 2)

df = pd.DataFrame({"t":np.arange(t_min, t_max), "value":timepoint(np.arange(t_min, t_max))})
dataset = TS_Dataset(df, x_cols=["t", "value"], y_col= "value", sequence_length=59)


hparams = dict(train_dataset=dataset, input_features=len(dataset.x_cols), output_features=1, n_hidden_layers=5, dropout=0.3, lr=0.01, batch_size=819)
model = NeuralNetwork_pl(**hparams)


checkpoint_callback = ModelCheckpoint(dirpath="logs/", save_top_k=2, monitor="val_loss")
trainer = pl.Trainer(gradient_clip_val = 0.0,
                     log_every_n_steps=1,
                     min_epochs=10, max_epochs=1000,
                     auto_lr_find=True,
                     auto_scale_batch_size=False,
                     track_grad_norm=2,
                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=30, min_delta=0.01), checkpoint_callback],
                     terminate_on_nan=False
                     )


# trainer.tune(model)
print(model.lr)
trainer.fit(model, val_dataloaders=model.train_dataloader())

model = model.load_from_checkpoint(checkpoint_path= checkpoint_callback.best_model_path, **hparams)
trainer.validate(model, model.train_dataloader())
device = "cpu"
N_steps_for_plot = 300
X,y = df.loc[:N_steps_for_plot, dataset.x_cols].values, df.loc[1:(N_steps_for_plot+1), dataset.y_col].values.reshape((-1, 1))
X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()
X, y = X.to(device), y.to(device)
y_pred =model.forward(X)[0]
plt.plot(y_pred.reshape(-1).detach().numpy(), label = "pred")
plt.plot(y.reshape(-1), label = "real")
plt.plot(X[:,1].reshape(-1), label = "x")
plt.legend()
