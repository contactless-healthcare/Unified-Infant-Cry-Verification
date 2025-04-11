# src/model.py

import dill
from wespeaker.models.tdnn import XVEC
from wespeaker.models.whisper_PMFA import whisper_PMFA

def create_model(args):
    if args.model == 'PreTrainECAPA':
        with open(args.pretrain_ecapa_path, 'rb') as f:
            model = dill.load(f)
        for child_idx, child in enumerate(model.mods.embedding_model.children()):
            for param in child.parameters():
                param.requires_grad = True
        model = model.mods.embedding_model
    elif args.model == 'XVEC':
        model = XVEC(feat_dim=args.feat_dim, embed_dim=args.embed_dim)
    elif args.model == 'whisper':
        model = whisper_PMFA(output_size=args.feat_dim, embedding_dim=args.embed_dim)

    return model
