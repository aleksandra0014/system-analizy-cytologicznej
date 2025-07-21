from dataset import train_loader, test_loader, val_loader, class_counts
from models import CytologyClassifier
from xai import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
classifier = CytologyClassifier(num_classes=3, architecture='custom_cnn', class_counts=class_counts)
classifier.train(train_loader, val_loader=val_loader, num_epochs=50, save_best_path='mine_cnn_0.001.32.50_v3.pth')


# 🔍 Ewaluacja
# classifier.load('mine_cnn_0.001.32.50_v2.pth')
# # Wydrukuj wszystkie warstwy i ich indeksy
# for i, layer in enumerate(classifier.model.features):
#     print(f"{i}: {layer}")

# acc = classifier.evaluate(test_loader)
# print(f"Test Accuracy: {acc:.2f}%")
# for i in [r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\HSIL\6b_2.bmp"]:
#     print(classifier.predict(i))
# classifier.model.eval()
# target_layer = classifier.model.features[23]

# xai = XAIExplainer(classifier.model, transform, class_names=['HSIL', 'LSIL', 'NSIL'], target_layer=target_layer)
# xai.gradcam(R'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\HSIL\4c_1.bmp')

# classifier.load(r'C:\Users\aleks\OneDrive\Documents\inzynierka\resnet_cytologia50.pth')
# # Wydrukuj wszystkie warstwy i ich indeksy
# # for i, layer in enumerate(classifier.model.features):
# #     print(f"{i}: {layer}")

# acc = classifier.evaluate(test_loader)
# print(f"Test Accuracy: {acc:.2f}%")
# for i in [r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\HSIL\6b_2.bmp"]:
#     print(classifier.predict(i))

# xai = XAIExplainer(classifier.model, transform, class_names=['HSIL', 'LSIL', 'NSIL'])
# xai.gradcam(R'C:\Users\aleks\OneDrive\Documents\inzynierka\data\data_single\NSIL\10b_1.bmp')
