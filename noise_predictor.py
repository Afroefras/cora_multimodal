from utils.noise_utils import (
    get_sounds,
    get_notes,
    sound_note_contains,
    get_data_loaders,
)
from lightning.pytorch import Trainer
from torch import device as torch_device, cuda
from utils.model_arquitecture import NoiseClassifier

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
print(f"Sounds: {sounds.shape}")
print(f"Labels: {labels.shape}\n")

train, test = get_data_loaders(sounds, labels, batch_size=256, shuffle=True)

noise_model = NoiseClassifier()
device = torch_device("cuda" if cuda.is_available() else "cpu")
noise_model = noise_model.to(device)


trainer = Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=noise_model, train_dataloaders=train)
