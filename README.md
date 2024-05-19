- Conda env oluştur:
conda create --name cv_project python=3.11

- env aktifleştir:
conda activate env_name

- Requirements.txt'deki paketleri indir:
pip install -r requirements.txt

train.ipynb'deki aşağıdaki satırları yorum satırına al, bu satırlar datanın daha küçük bir kısmı ile çalışmak için eklendi:
train_dataset = torch.utils.data.Subset(train_dataset, range(20))
valid_dataset = torch.utils.data.Subset(valid_dataset, range(20))

