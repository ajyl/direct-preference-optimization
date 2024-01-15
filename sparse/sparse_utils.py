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
    # top_vecs = [
    #    model.transformer.h[x[2]].mlp.c_proj.weight[x[1]]
    #    for x in sorted_scores
    # ]
    top_mlps = {}
    for score, mlp_idx, layer in sorted_scores:
        if layer not in top_mlps:
            top_mlps[layer] = []
        top_mlps[layer].append(mlp_idx)
    return top_mlps


def reshape_group_mask(name, z_mask, group_mask, group_grad, config):
    """
    Construct mask.
    """
    n_heads = config.n_head
    d_model = config.n_embd
    d_mlp = config.n_inner if config.n_inner is not None else 4 * d_model
    vocab = config.vocab_size
    pos = config.n_positions

    if ".mlp." in name:
        if "c_fc.weight" in name:
            # z_mask: [d_model, d_mlp]
            # group_mask: [d_mlp]
            assert z_mask.shape == (d_model, d_mlp)
            _group_mask = group_mask.unsqueeze(0).repeat((z_mask.shape[0], 1))
            _group_grad = (
                group_grad.unsqueeze(0).repeat((z_mask.shape[0], 1))
                / z_mask.shape[0]
            )

        elif "c_fc.bias" in name:
            # z_mask: [d_mlp]
            # group_mask: [1]
            assert z_mask.shape == (d_mlp,)
            _group_mask = group_mask.repeat(z_mask.shape[0])
            _group_grad = group_grad.repeat(z_mask.shape[0]) / z_mask.shape[0]

        elif "c_proj.weight" in name:
            # z_mask: [d_mlp, d_model]
            # group_mask: [d_mlp]
            assert z_mask.shape == (d_mlp, d_model)
            _group_mask = group_mask.unsqueeze(-1).repeat((1, z_mask.shape[1]))
            _group_grad = (
                group_grad.unsqueeze(-1).repeat((1, z_mask.shape[1]))
                / z_mask.shape[1]
            )

        elif "c_proj.bias" in name:
            # z_mask: [d_model]
            # group_mask: [1]
            assert z_mask.shape == (d_model,)
            _group_mask = group_mask.repeat(z_mask.shape[0])
            _group_grad = group_grad.repeat(z_mask.shape[0]) / z_mask.shape[0]

        else:
            raise RuntimeError("Unexpected mask type.")

    elif ".attn." in name:
        if "c_attn.bias" in name:
            # z_mask: [d_model * 3]
            # group_mask: [1]
            assert z_mask.shape == (d_model * 3,)
            _group_mask = group_mask.repeat(z_mask.shape[0])
            _group_grad = group_grad.repeat(z_mask.shape[0]) / z_mask.shape[0]

        elif "c_attn.weight" in name:
            # z_mask: [d_model, d_model * 3]
            # group_mask: [d_model * 3]
            assert z_mask.shape == (d_model, d_model * 3)
            # Should be [3 * num_heads * d_attn]
            _group_mask = group_mask.unsqueeze(0).repeat((d_model, 1))
            _group_grad = (
                group_grad.unsqueeze(0).repeat((d_model, 1)) / d_model
            )

        elif "c_proj.bias" in name:
            # z_mask: [d_model]
            # group_mask: [1]
            assert z_mask.shape == (d_model,)
            _group_mask = group_mask.repeat(z_mask.shape[0])
            _group_grad = group_grad.repeat(z_mask.shape[0]) / z_mask.shape[0]

        elif "c_proj.weight" in name:
            # z_mask: [d_model, d_model]
            # group_mask: [d_model]
            assert z_mask.shape == (d_model, d_model)
            _group_mask = group_mask.unsqueeze(-1).repeat((1, z_mask.shape[1]))
            _group_grad = (
                group_grad.unsqueeze(-1).repeat((1, z_mask.shape[1]))
                / z_mask.shape[1]
            )

    elif "ln_1" in name or "ln_2" in name or "ln_f" in name:
        # z_mask: [d_model]
        # group_mask: [1]
        assert z_mask.shape == (d_model,)
        _group_mask = group_mask.repeat(z_mask.shape[0])
        _group_grad = group_grad.repeat(z_mask.shape[0]) / z_mask.shape[0]

    elif "wte" in name:
        # z_mask: [vocab, d_model]
        # group_mask: [vocab]
        assert z_mask.shape == (vocab, d_model)
        _group_mask = group_mask.unsqueeze(-1).repeat((1, z_mask.shape[1]))
        _group_grad = (
            group_grad.unsqueeze(-1).repeat((1, z_mask.shape[1]))
            / z_mask.shape[1]
        )

    elif "wpe" in name:
        # z_mask: [pos, d_model]
        # group_mask: [pos]
        assert z_mask.shape == (pos, d_model)
        _group_mask = group_mask.unsqueeze(-1).repeat((1, z_mask.shape[1]))
        _group_grad = (
            group_grad.unsqueeze(-1).repeat((1, z_mask.shape[1]))
            / z_mask.shape[1]
        )

    #if z_mask.shape != _group_mask.shape:
    #    breakpoint()
    # TODO: _group_mask device should be set elsewhere?
    return _group_mask, _group_grad


def concrete_stretched(alpha, l=-0.1, r=1.1):
    _u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
    _s = (torch.sigmoid(_u.log() - (1 - _u).log() + alpha)).detach()
    _s_bar = _s * (r - l) + l
    _t = _s_bar.clamp(0, 1000)
    _z = _t.clamp(-1000, 1)
    dz_dt = (_t < 1).float().to(alpha.device).detach()
    dt_du = (_s_bar > 0).float().to(alpha.device).detach()
    du_ds = r - l
    ds_dalpha = (_s * (1 - _s)).detach()
    dz_dalpha = dz_dt * dt_du * du_ds * ds_dalpha
    return _z.detach(), dz_dalpha.detach()
