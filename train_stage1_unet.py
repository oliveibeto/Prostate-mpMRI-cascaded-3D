import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# MONAI imports for 3D U-Net, loss functions, and evaluation metrics
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from dataset_prostate158 import Prostate158Dataset


def train_model():
    print("=" * 60)
    print("INITIALIZING STAGE 1: 3D U-NET 25-EPOCH LOCAL PRODUCTION TRAINING")
    print("=" * 60)

    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training on device: {device}")

    # 2. Dataset and Strict 70/15/15 Splitting Setup
    dataset_dir = Path(r"C:\ProstateData\prostate158_train\train")  # set to your Prostate158 train folder

    full_dataset = Prostate158Dataset(
        dataset_dir=dataset_dir,
        target_spacing=(1.0, 1.0, 1.0),
        target_shape=(32, 128, 128),
        cache_data=False
    )

    # Strict 70% Training, 15% Validation, 15% Testing Split
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Captures rounding remainder

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f" Dataset Splits -> Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_subset)}")

    train_loader = DataLoader(train_subset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=2, shuffle=False, num_workers=0)

    # 3. Instantiate 3D U-Net Architecture
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # 4. Loss Function, Optimizer, Scheduler, and Metrics
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Cosine Annealing LR Scheduler configured for 25 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    post_pred = AsDiscrete(argmax=True, to_onehot=3, dim=1)
    post_label = AsDiscrete(to_onehot=3, dim=1)

    # 5. Production Training & Validation Loop (25 Epochs)
    num_epochs = 25
    best_val_dice = 0.0
    print(f"\nStarting 25-epoch local training loop (clean epoch-by-epoch logging)...\n")

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                preds_encoded = post_pred(outputs)
                labels_encoded = post_label(labels)
                dice_metric(y_pred=preds_encoded, y=labels_encoded)
                batch_dice = dice_metric.aggregate()[0].item()
                dice_metric.reset()
                epoch_dice += batch_dice

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)

        # --- VALIDATION PHASE (Unseen 15% Validation Subset) ---
        model.eval()
        val_dice_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for val_batch in val_loader:
                val_images = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device).long()

                val_outputs = model(val_images)
                preds_encoded = post_pred(val_outputs)
                labels_encoded = post_label(val_labels)

                dice_metric(y_pred=preds_encoded, y=labels_encoded)
                batch_val_dice = dice_metric.aggregate()[0].item()
                dice_metric.reset()

                val_dice_sum += batch_val_dice
                val_steps += 1

        avg_val_dice = val_dice_sum / max(1, val_steps)

        # Step learning rate scheduler at the end of each epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Clean single-line output per epoch
        print(
            f"Epoch [{epoch + 1}/25] | Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {current_lr:.6f}")

        # Save model weights if a new peak Validation Dice score is achieved
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), "prostate_unet_stage1_best.pth")
            print(f"    New best model weights saved! (Peak Val Dice: {best_val_dice:.4f})")

    # Save final weights at completion
    torch.save(model.state_dict(), "prostate_unet_stage1_final.pth")
    print(f"\n LOCAL PRODUCTION TRAINING COMPLETED! Best Val Dice: {best_val_dice:.4f}")
    print(" Weights saved to 'prostate_unet_stage1_best.pth' and 'prostate_unet_stage1_final.pth'!")

    # 6. Final Evaluation on Held-Out Test Set (15%)
    print("\n" + "=" * 60)
    print("EVALUATING BEST MODEL ON HELD-OUT TEST SET (15%)")
    print("=" * 60)

    model.load_state_dict(torch.load("prostate_unet_stage1_best.pth"))
    model.eval()

    test_dice_sum = 0.0
    test_steps = 0

    with torch.no_grad():
        for test_batch in test_loader:
            test_images = test_batch["image"].to(device)
            test_labels = test_batch["label"].to(device).long()

            test_outputs = model(test_images)
            preds_encoded = post_pred(test_outputs)
            labels_encoded = post_label(test_labels)

            dice_metric(y_pred=preds_encoded, y=labels_encoded)
            batch_test_dice = dice_metric.aggregate()[0].item()
            dice_metric.reset()

            test_dice_sum += batch_test_dice
            test_steps += 1

    final_test_dice = test_dice_sum / max(1, test_steps)
    print(f" Final Test Set Dice Score: {final_test_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_model() 