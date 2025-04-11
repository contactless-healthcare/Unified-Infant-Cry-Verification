# src/training.py
# Standard library imports
import os
import copy

# Third-party imports
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import numpy as np
# Local imports
from online_triplet_loss.losses import batch_hard_triplet_loss

def get_embeddings(wavs, lens, encoder, model, args):
    """
    Compute embeddings for the batch.
    """
    feats = encoder.mods.mean_var_norm(encoder.mods.compute_features(wavs), lens)
    embeddings = model(feats)
    return torch.nn.functional.normalize(embeddings.squeeze(), p=2.0, dim=-1)

def compute_loss(dataloader, model, encoder, args, dataset_len, optimizer=None):
    """
    Compute the total loss for a given dataloader.
    Optionally applies backpropagation if optimizer is provided.
    """
    total_loss = 0.0
    if optimizer:
        model.train()  # Set model to training mode
    else:
        model.eval()
    
    for wavs, lens, batch_labels in tqdm(dataloader):
        wavs, lens, batch_labels = wavs.to(args.device), torch.Tensor(lens).to(args.device), torch.Tensor(batch_labels).to(args.device)
        embeddings = get_embeddings(wavs, lens, encoder, model, args)
        loss = batch_hard_triplet_loss(batch_labels, embeddings, margin=args.margin_triplet)
        total_loss += loss.item() * wavs.size(0)

        if optimizer:  # If optimizer is provided, perform backward and optimization step
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
    
    return total_loss / dataset_len

def save_best_model(epoch_loss_val, model, model_saved, epoch, args):
    """
    Save model if validation loss is improved.
    """
    if len(model_saved) < args.best_model_num or any(epoch_loss_val < v for v in model_saved.values()):
        print('Saving model...')
        if len(model_saved) >= args.best_model_num:
            max_loss_ckpt = max(model_saved, key=model_saved.get)
            os.remove(f"{args.save_path}/{max_loss_ckpt}")
            del model_saved[max_loss_ckpt]
        checkpoint_name = f'checkpoint_epoch_{epoch + 1}_valloss_{epoch_loss_val:.4f}.ckpt'
        model_saved[checkpoint_name] = epoch_loss_val
        torch.save(model.state_dict(), f"{args.save_path}/{checkpoint_name}")
    return model_saved

def train_model(encoder, model, dataloader_val, dataloader, dataset, dataset_val, args):
    """
    Main training loop.
    """
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    model_saved = {}

    for epoch in range(args.num_epochs):
        dataloader.dataset.shuffle_dataframe()  # Shuffle before each epoch

        # Train phase
        train_loss = compute_loss(dataloader, model, encoder, args, len(dataset), optimizer)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}")

        # Validation phase
        val_loss = compute_loss(dataloader_val, model, encoder, args, len(dataset_val))
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Val Loss: {val_loss:.4f}")

        # Save model if it's the best so far
        model_saved = save_best_model(val_loss, model, model_saved, epoch, args)
        
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    # Averaging model weights from saved checkpoints
    state_dicts = [torch.load(f"{args.save_path}/{ckpt}") for ckpt in model_saved]
    avg_state_dict = {k: sum([sd[k] for sd in state_dicts]) / len(state_dicts) for k in state_dicts[0]}
    if args.max_audio is not None:
        torch.save(avg_state_dict, f"{args.save_path}/{args.model}/epochs{args.num_epochs}_seed{args.seed}_{(args.max_audio-240)/16000}s.ckpt")
    else:
        torch.save(avg_state_dict, f"{args.save_path}/{args.model}/epochs{args.num_epochs}_seed{args.seed}.ckpt")

    # Clean up old checkpoints
    for ckpt in model_saved:
        os.remove(f"{args.save_path}/{ckpt}")
        print(f"Removed {ckpt}")
    
    return avg_state_dict
