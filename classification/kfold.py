
from torchvision import datasets, transforms

from models import run_gridsearch_kfold

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

run_gridsearch_kfold(
    data_dir=r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single_cropped",  
    transform=transform,
    architectures=["resnet18", "vgg16", "custom_cnn"],
    lrs=[1e-3, 1e-4],
    batch_sizes=[16, 32],
    num_epochs_list=[20], 
    k_folds=5,
    output_csv="wyniki_gridsearch.csv"
)
