import torch
from models.classifier.model import create_model
from models.classifier.dataset import create_dataloader
from collections import defaultdict

device = 'mps'
model = create_model('dog', pretrained=False)
ckpt = torch.load('models/classifier/checkpoints/dog_best.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
model.eval()

val_loader = create_dataloader(
    data_paths=['eye_data/VL/개/안구/일반'],
    animal_type='dog',
    batch_size=32,
    img_size=224,
    is_training=False,
    num_workers=0,
    use_sampler=False
)

diseases = model.get_disease_names()
correct = defaultdict(int)
total = defaultdict(int)
tp = defaultdict(int)
fn = defaultdict(int)

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        for disease in diseases:
            dlabels = labels[disease].to(device)
            valid = dlabels >= 0
            if valid.sum() == 0:
                continue
            preds = torch.argmax(outputs[disease][valid], dim=1)
            gt = dlabels[valid]
            correct[disease] += (preds == gt).sum().item()
            total[disease] += valid.sum().item()
            pos = gt > 0
            if pos.sum() > 0:
                tp[disease] += ((preds == gt) & pos).sum().item()
                fn[disease] += ((preds != gt) & pos).sum().item()

print()
print('질환별 성능:')
for d in diseases:
    acc = correct[d] / total[d] * 100 if total[d] > 0 else 0
    recall = tp[d] / (tp[d] + fn[d]) * 100 if (tp[d] + fn[d]) > 0 else 0
    print(f'{d:15s} Acc: {acc:.1f}%  Recall: {recall:.1f}%')
