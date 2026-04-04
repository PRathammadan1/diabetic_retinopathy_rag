import os #isaka liya import filekohandel karna kaliya 
import zipfile #zipko extract karna ka liya 
import pandas as pd #  pandas mana csv ko read karna ka liya 
from PIL import Image  #python image library then we can used for to import image ko open , resize karna ka 
from sklearn.model_selection import train_test_split # bhai muja na 80/20 ratio ma dataset ko dividekarna tha isliye ya impor kiya (genarally ya sab ka sath use hota hia )
import torch #ya library pytorch aur gpu dono ko load karta hai
from torch.utils.data import Dataset, DataLoader #(Dtaset ka kam apni custom datset bana ka liya use hota hai ,Datal
from torchvision import transforms #image ko transform / preprocess karne ke liye hi use hota hai.
from sklearn.utils.class_weight import compute_class_weight
import numpy as np;


# STEP 1: Extract dataset 

zip_path = "dataset.zip"  # Path to your zip file
extract_path = "C:\\Users\\user\\Desktop\\diabetic model\\diabetic_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Zip extracted successfully!")

# STEP 2: Read CSV

csv_path = os.path.join(extract_path, "train.csv")
df = pd.read_csv(csv_path)
print("First 5 rows of CSV:")
print(df.head())


# STEP 3: STEP 3: 80/10/10 Train/Validation/Test Split

# Step 1: Train (80%) + Temp (20%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,              # 20% side me
    stratify=df['diagnosis'],
    random_state=42
)

# Step 2: Validation (10%) + Test (10%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,              # 20% ka half = 10% test
    stratify=temp_df['diagnosis'],
    random_state=42
)

print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")


train_labels = train_df['diagnosis'].values
classes = np.unique(train_labels)

weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
class_weights = torch.tensor(weights, dtype=torch.float)
print("Class weights:", class_weights)
# ---------------------------
# STEP 4: Define Custom Dataset
# ---------------------------
class AutoImageDataset(Dataset):
    def __init__(self, dataframe, img_folder, transform=None, rare_transform=None, rare_classes=[3,4], img_col='id_code', label_col='diagnosis'):
        self.img_folder = os.path.abspath(img_folder)
        self.transform = transform
        self.rare_transform = rare_transform
        self.rare_classes = rare_classes
        self.data = dataframe.copy()
        self.img_col = img_col
        self.label_col = label_col

        # Collect all images from folder and subfolders
        possible_ext = ['.png', '.jpg', '.jpeg']
        all_files = []
        for root, _, files in os.walk(self.img_folder):  #os.walk i sused for to read all the images from all 
            for f in files:
                if f.lower().endswith(tuple(possible_ext)):
                    all_files.append(os.path.join(root, f))

        # Map filename (without extension) to full path
        file_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in all_files}

        # Keep only rows with matching images
        valid_rows = []
        for _, row in self.data.iterrows():
            key = str(row[img_col])
            if key in file_dict:
                row['filename'] = file_dict[key] #here we get a image name 
                valid_rows.append(row)

        self.data = pd.DataFrame(valid_rows)
        if transform is not None:
            print(f"Dataset size: {len(self.data)}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filename']
        label = self.data.iloc[idx][self.label_col]
        image = Image.open(img_path).convert("RGB")
        if self.rare_transform is not None and label in self.rare_classes:
                image = self.rare_transform(image)
        elif self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# ---------------------------
# STEP 5: Define Transforms 🔥 TRAIN TRANSFORM (AUGMENTATION + NORMALIZATION)
# ---------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),              # Roughly same size
    transforms.RandomHorizontalFlip(p=0.5),      # Safe flip
    transforms.RandomRotation(15),               # ±15° rotation
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2, 
        saturation=0.1
    ),                                           # Moderate brightness/contrast
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])   # Standard ImageNet normalization
])


# ❄️ TEST / VALID TRANSFORM (NO AUGMENTATION)
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# STEP 5b: Rare Class Augmentation
# ---------------------------
rare_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),          # extra flip
    transforms.RandomRotation(20),                 # more rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1), # stronger jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------
# STEP 6: DataLoader Function
# ---------------------------
def get_loader(dataframe, img_folder, batch_size=32, train=True):
    transform = train_transform if train else val_transform
    dataset = AutoImageDataset(dataframe, img_folder, transform=transform , rare_transform=rare_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train,num_workers=0,pin_memory=True)
    return loader

# ---------------------------
# STEP 7: Create Loaders
# ---------------------------
img_folder = os.path.join(extract_path, "gaussian_filtered_images")
test_loader = get_loader(test_df, img_folder, batch_size=64, train=False)
train_loader = get_loader(train_df, img_folder, batch_size=64, train=True )#AUGMENTATION ON (train=True)
val_loader   = get_loader(val_df, img_folder, batch_size=64, train=False) #ARGUMENTATION OFF (TRAIN=fALSE

# # ---------------------------
# # STEP 8: Test Loader
# # ---------------------------
for images, labels in train_loader:
    print("Batch images shape:", images.shape)
    print("Batch labels:", labels)
    break


