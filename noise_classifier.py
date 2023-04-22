from utils.noise_utils import (
    get_sounds,
    get_notes,
    sound_note_contains,
    get_data_loaders,
)
from lightning.pytorch import Trainer
from torch import device as torch_device, cuda
from utils.noise_arquitecture import NoiseClassifier

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

train, test = get_data_loaders(
    sounds,
    labels,
    batch_size=256,
    num_workers=1,
)

noise_classifier = NoiseClassifier()
device = torch_device("cuda" if cuda.is_available() else "cpu")
noise_classifier = noise_classifier.to(device)


trainer = Trainer(limit_train_batches=100, max_epochs=1)

if __name__=="__main__":
    trainer.fit(model=noise_classifier, train_dataloaders=train, val_dataloaders=test)
