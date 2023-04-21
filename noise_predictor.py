from utils.noise_utils import (
    get_sounds,
    get_notes,
    sound_note_contains,
    train_test_split,
    get_data_loader,
)
from utils.model_utils import train_model
from torch import device as torch_device, cuda
from utils.model_arquitecture import NoiseModel

sounds, names = get_sounds(
    records_dir="data/tensors/resized_records_5sec.pt",
    names_dir="data/tensors/records_names_5sec.xz",
)

sound_notes = get_notes(
    csv_dir="data/physionet.org/files/ephnogram/1.0.0/ECGPCGSpreadsheet.csv",
    id_col="Record Name",
    notes_col="PCG Notes",
    names=names,
)
labels = sound_note_contains(sound_notes, contains="Good")

X_train, y_train, X_test, y_test = train_test_split(sounds, labels)
print(X_train.shape, y_train.shape)

train = get_data_loader(X=X_train, y=y_train, batch_size=32, shuffle=True)
test = get_data_loader(X=X_test, y=y_test, batch_size=128, shuffle=True)

noise_model = NoiseModel(input_size=X_train.shape[1], num_classes=1)
device = torch_device("cuda" if cuda.is_available() else "cpu")
noise_model = noise_model.to(device)

from lightning.pytorch import Trainer

trainer = Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=noise_model, train_dataloaders=train)

# train_model(
#     model=noise_model,
#     train_loader=train,
#     test_loader=test,
#     device=device,
#     num_epochs=3,
#     lr=0.001,
# )
