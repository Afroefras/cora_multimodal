# Autoencoder multimodal con datos cardiacos

wget -r -N -c -np https://physionet.org/files/ephnogram/1.0.0/

ed = ExtractData()
IMPORT_DIR = "data/physionet.org/files/ephnogram/1.0.0/WFDB"
ed.extract_n_export(import_dir=IMPORT_DIR, export_dir="data", verbose=True)

#1: ECGPCG0026 imported
#2: ECGPCG0005 imported
...

Min shape: (240000, 2)
#1: torch.Size([1, 14400000, 5]) --> torch.Size([60, 240000, 2])
#2: torch.Size([1, 240000, 5]) --> torch.Size([1, 240000, 2])
...

Tensor 'records.pt' is now saved at:
data/tensors

Citations:
Kazemnejad, A., Gordany, P., & Sameni, R. (2021). EPHNOGRAM: A Simultaneous Electrocardiogram and Phonocardiogram Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/tjtq-5911.
