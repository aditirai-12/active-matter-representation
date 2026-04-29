# scripts/check_collapse.py
encoder.load_state_dict(torch.load("checkpoints/.../ConvEncoder_3.pth"))
encoder.eval()

stds = []
for batch in val_loader:
    with torch.no_grad():
        embed = encoder(batch['context'].cuda())
        stds.append(embed.float().std(dim=0).mean().item())

print(f"Mean embedding std: {np.mean(stds):.4f}")