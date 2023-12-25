"""
Module Doc String
"""

import torch
import torch.nn.functional as F


def load_probe(path):
    toxic_classifier = torch.load(path)
    toxic_vector = toxic_classifier["mlp.weight"][1]
    return toxic_vector


def get_mlp_weights(model, probe, num_vecs):
    """
    Get mlp weights of interest.
    """
    scores = []
    for layer in range(model.config.n_layer):
        mlp_outs = model.transformer.h[layer].mlp.c_proj.weight
        cos_sims = F.cosine_similarity(
            mlp_outs, probe.unsqueeze(0).to(mlp_outs.device), dim=1
        )
        _topk = cos_sims.topk(k=300)
        _values = [x.item() for x in _topk.values]
        _idxs = [x.item() for x in _topk.indices]
        topk = list(zip(_values, _idxs, [layer] * _topk.indices.shape[0]))
        scores.extend(topk)

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    sorted_scores = sorted_scores[:num_vecs]
    #top_vecs = [
    #    model.transformer.h[x[2]].mlp.c_proj.weight[x[1]]
    #    for x in sorted_scores
    #]
    top_mlps = {}
    for score, mlp_idx, layer in sorted_scores:
        if layer not in top_mlps:
            top_mlps[layer] = []
        top_mlps[layer].append(mlp_idx)
    return top_mlps
