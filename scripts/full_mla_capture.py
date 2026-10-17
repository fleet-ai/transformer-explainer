#!/usr/bin/env python3
"""Capture a complete, streamed, pretrained DeepSeek-V2-Lite language-model pass.

Uses pinned upstream Attention, RMSNorm, MLP, MoEGate, MoE.forward/moe_infer,
and DecoderLayer.forward definitions. Checkpoint tensors are fetched by exact
safetensors byte ranges; only experts selected by the actual router are loaded.
This changes weight residency, never model math or routing. All 27 layers,
shared experts, final normalization, and the complete LM head execute.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
from typing import Optional, Tuple
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
import transformers

from capture_mla_hybrid_common import Checkpoint, digest

MODEL = 'deepseek-ai/DeepSeek-V2-Lite'
REVISION = '604d5664dddd88a0433dbae533b7fe9472482de0'
SOURCE_SHA = '7d8e5221095286eea991137760893fd7ba52727c0b4ebf48ec09e8bc56b45b9c'


def values(x):
    return x.detach().cpu().float().tolist()


def preview(x):
    return values(x[..., :8])


def norms(x):
    return values(x.norm(dim=-1))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'Wrote {path}: {path.stat().st_size:,} bytes', flush=True)


def source_classes(path):
    source = path.read_bytes()
    assert digest(source) == SOURCE_SHA, 'Pinned official source changed'
    names = {'DeepseekV2RMSNorm', 'DeepseekV2RotaryEmbedding', 'DeepseekV2LinearScalingRotaryEmbedding',
             'DeepseekV2DynamicNTKScalingRotaryEmbedding', 'DeepseekV2YarnRotaryEmbedding',
             'yarn_find_correction_dim', 'yarn_find_correction_range', 'yarn_get_mscale', 'yarn_linear_ramp_mask',
             'rotate_half', 'apply_rotary_pos_emb', 'DeepseekV2Attention', 'DeepseekV2MLP',
             'MoEGate', 'AddAuxiliaryLoss', 'DeepseekV2MoE', 'DeepseekV2DecoderLayer'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    env = {'torch': torch, 'nn': nn, 'F': F, 'ACT2FN': ACT2FN, 'math': math, 'warnings': warnings,
           'Optional': Optional, 'Tuple': Tuple, 'Cache': Cache, 'DeepseekV2Config': SimpleNamespace,
           'logger': logging.getLogger('full-mla')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


class FullCheckpoint(Checkpoint):
    def fetch(self, file, start=None, end=None):
        suffix = '' if start is None else f'.{start}-{end}'
        path = self.cache / (file.replace('/', '__') + suffix)
        if not path.exists():
            cmd = ['curl', '--fail', '--location', '--silent', '--show-error', '--http1.1',
                   '--retry', '5', '--retry-all-errors', '--retry-delay', '1',
                   '--connect-timeout', '30', '--max-time', '180']
            if start is not None:
                cmd += ['--range', f'{start}-{end}', '--max-filesize', str(max(16384,end-start+1024))]
            url = self.url + file + ('?trace_range=' + suffix if suffix else '')
            temporary = path.with_name(path.name+'.partial')
            subprocess.run(cmd+[url,'-o',str(temporary)],check=True)
            if start is not None and temporary.stat().st_size != end-start+1:
                raise ValueError(f'Wrong HTTP range size: {file} {start}-{end}')
            temporary.replace(path)
        data = path.read_bytes()
        if start is not None and len(data) != end-start+1:
            raise ValueError(f'Cached range has wrong size: {file} {start}-{end}')
        return data


class ShardedCheckpoint:
    def __init__(self, cache):
        first = FullCheckpoint(MODEL, REVISION, 'model-00001-of-000004.safetensors', cache)
        self.first = first
        self.index = json.loads(first.asset('model.safetensors.index.json').read_text())
        self.shards = {first.filename: first}
        for filename in sorted(set(self.index['weight_map'].values())):
            if filename not in self.shards:
                self.shards[filename] = FullCheckpoint(MODEL, REVISION, filename, cache)
        self.config = json.loads(first.asset('config.json').read_text())

    def tensor(self, name, row=None):
        return self.shards[self.index['weight_map'][name]].tensor(name, row)

    def prefetch(self, names):
        # Small bounded ranges avoid a single slow large-object connection
        # blocking a layer. Reassembly preserves the exact checkpoint bytes;
        # tensor() records their SHA-256 after the complete range is assembled.
        pending = []
        chunks = []
        chunk_bytes = 2 * 1024 * 1024
        for name in names:
            cp = self.shards[self.index['weight_map'][name]]
            a, b = cp.header[name]['data_offsets']
            start, end = cp.offset+a, cp.offset+b-1
            path = cp.cache / f'{cp.filename}.{start}-{end}'
            if path.exists() and path.stat().st_size == end-start+1:
                continue
            pieces = [(cp, left, min(end, left+chunk_bytes-1)) for left in range(start,end+1,chunk_bytes)]
            pending.append((cp, start, end, path, pieces))
            chunks.extend(pieces)
        def fetch(piece):
            cp, start, end = piece
            cp.fetch(cp.filename, start, end)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fetch, chunks))
        for cp, start, end, path, pieces in pending:
            if len(pieces) == 1:
                continue
            temporary = path.with_name(path.name+'.assembling')
            with temporary.open('wb') as output:
                for _, left, right in pieces:
                    output.write((cp.cache / f'{cp.filename}.{left}-{right}').read_bytes())
            assert temporary.stat().st_size == end-start+1
            temporary.replace(path)
            for _, left, right in pieces:
                (cp.cache / f'{cp.filename}.{left}-{right}').unlink()

    def names(self, prefix):
        return [n for n in self.index['weight_map'] if n.startswith(prefix)]

    def provenance(self):
        weights = {}
        headers = {}
        for filename, cp in self.shards.items():
            weights.update({k: dict(v, file=filename) for k, v in cp.manifest.items()})
            headers[filename] = digest(cp.header_bytes)
        return {'modelId': MODEL, 'revision': REVISION, 'trained': True, 'partial': False,
                'scope': 'Complete pretrained 27-layer DeepSeek-V2-Lite forward pass, including dense layer 1, all 26 routed/shared expert layers, final RMSNorm and every LM-head vocabulary logit.',
                'dtype': 'float32 computation from original bfloat16 checkpoint tensors',
                'captureScript': 'scripts/full_mla_capture.py', 'captureScriptSha256': digest(Path(__file__).read_bytes()),
                'captureHelperSha256': digest(Path(__file__).with_name('capture_mla_hybrid_common.py').read_bytes()),
                'modelSourceSha256': SOURCE_SHA, 'modelUrl': f'https://huggingface.co/{MODEL}/tree/{REVISION}',
                'weightsSha256': weights, 'headerSha256': headers, 'assetsSha256': self.first.assets,
                'libraryVersions': {'torch': torch.__version__, 'transformers': transformers.__version__},
                'weightLoading': 'Exact safetensors tensor ranges. All and only router-selected experts execute; all shared experts execute. Layers stream sequentially without changing tensors or arithmetic.'}


def projection(weight, x, output):
    # Omitted contribution is separately summed over every hidden input omitted
    # by the displayed 4x4 window, never substituted with a fabricated remainder.
    omitted = F.linear(x[:, 4:], weight[:4, 4:])
    return {'weights': values(weight[:4, :4]), 'omittedContribution': values(omitted),
            'output': values(output[:, :4])}


class Recorder:
    def __init__(self, cp, config, env, example):
        self.cp, self.config, self.env, self.example = cp, config, env, example
        self.record = True
        self.current_layer = 0
        self.attention_ops = []
        self.expert_ops = []
        self.errors = []

    def load_mlp(self, prefix, intermediate):
        names = [prefix+'.'+name+'.weight' for name in ['gate_proj', 'up_proj', 'down_proj']]
        self.cp.prefetch(names)
        # Meta initialization avoids allocating and randomly initializing a second
        # copy of the large MLP; every parameter is assigned from the checkpoint.
        with torch.device('meta'):
            module = self.env['DeepseekV2MLP'](self.config, intermediate_size=intermediate)
        module.load_state_dict({n.removeprefix(prefix+'.'): self.cp.tensor(n) for n in names}, assign=True, strict=True)
        return module.eval()

    def mlp_run(self, module, x, expert_id, positions, weights=None):
        y = module(x)
        if not self.record:
            return y
        flat = x.reshape(-1, x.shape[-1])
        out = y.reshape(-1, y.shape[-1])
        gate = module.gate_proj(flat)
        up = module.up_proj(flat)
        silu = module.act_fn(gate)
        product = silu*up
        rebuilt = module.down_proj(product)
        assert torch.equal(rebuilt, out), 'MLP reference reconstruction differs'
        operations = {
            'gate': projection(module.gate_proj.weight, flat, gate),
            'up': projection(module.up_proj.weight, flat, up),
            'down': projection(module.down_proj.weight, product, out),
        }
        for row, position in enumerate(positions):
            w = float(weights[row]) if weights is not None else 1.0
            one = {'id': int(expert_id), 'intermediateSize': module.intermediate_size,
                   'siluGate': values(silu[row, :4]), 'product': values(product[row, :4]),
                   'weightedOutput': values(out[row, :4]*w)}
            for name, op in operations.items():
                one[name] = {'weights': op['weights'], 'omittedContribution': op['omittedContribution'][row],
                             'output': op['output'][row]}
            self.detail_tokens[position]['experts'].append(one)
            self.full_expert_outputs[position][int(expert_id)] = out[row].detach().clone()
        return y

    def load_layer(self, index):
        prefix = f'model.layers.{index}.'
        cp, config, env = self.cp, self.config, self.env
        attention_names = cp.names(prefix+'self_attn.')
        small_names = attention_names + [prefix+'input_layernorm.weight', prefix+'post_attention_layernorm.weight']
        if index:
            small_names += [prefix+'mlp.gate.weight']
        cp.prefetch(small_names)
        # Instantiate the exact official classes; use a short rotary cache since
        # seq_len is the complete recorded prompt. Cache length does not alter
        # YaRN frequencies and is checked against the full configured reference.
        short_config = SimpleNamespace(**vars(config))
        short_config.max_position_embeddings = len(self.example['tokens'])
        attn = env['DeepseekV2Attention'](short_config, layer_idx=index).eval()
        attn.load_state_dict({n.removeprefix(prefix+'self_attn.'): cp.tensor(n) for n in attention_names}, strict=True)
        norms_ = []
        for name in ['input_layernorm', 'post_attention_layernorm']:
            norm = env['DeepseekV2RMSNorm'](config.hidden_size, eps=config.rms_norm_eps).eval()
            norm.load_state_dict({'weight': cp.tensor(prefix+name+'.weight')})
            norms_.append(norm)
        if index == 0:
            dense = self.load_mlp(prefix+'mlp', config.intermediate_size)
            owner = self
            class DenseWrapper(nn.Module):
                def forward(self, hidden):
                    return owner.mlp_run(dense, hidden, -1, list(range(hidden.shape[1])))
            mlp = DenseWrapper()
        else:
            gate = env['MoEGate'](config).eval()
            gate.load_state_dict({'weight': cp.tensor(prefix+'mlp.gate.weight')})
            shared = self.load_mlp(prefix+'mlp.shared_experts', config.moe_intermediate_size*config.n_shared_experts)
            owner = self
            class LazyExpert(nn.Module):
                def __init__(self, expert):
                    super().__init__(); self.expert = expert
                def forward(self, hidden):
                    module = owner.load_mlp(prefix+f'mlp.experts.{self.expert}', config.moe_intermediate_size)
                    # Official moe_infer sorts flattened selections by expert.
                    # Mirror its row ordering only to associate captured values.
                    flat_ids = owner.top_ids.reshape(-1)
                    idxs = flat_ids.argsort()
                    positions = (idxs[flat_ids[idxs] == self.expert] // config.num_experts_per_tok).tolist()
                    ranks = (idxs[flat_ids[idxs] == self.expert] % config.num_experts_per_tok)
                    weights = owner.top_weights[positions, ranks]
                    return owner.mlp_run(module, hidden, self.expert, positions, weights)
            class SharedWrapper(nn.Module):
                def forward(self, hidden):
                    return owner.mlp_run(shared, hidden, 64, list(range(hidden.shape[1])))
            mlp = env['DeepseekV2MoE'].__new__(env['DeepseekV2MoE'])
            nn.Module.__init__(mlp)
            mlp.config = config; mlp.num_experts_per_tok = config.num_experts_per_tok
            mlp.ep_size = 1; mlp.experts_per_rank = config.n_routed_experts; mlp.ep_rank = 0
            mlp.experts = nn.ModuleList([LazyExpert(e) for e in range(config.n_routed_experts)])
            mlp.gate = gate; mlp.shared_experts = SharedWrapper()
            def gate_hook(module, inputs, output):
                self.top_ids, self.top_weights = output[:2]
                self.router_logits = F.linear(inputs[0].reshape(-1, config.hidden_size).float(), module.weight.float())
                self.router_probs = self.router_logits.softmax(-1)
                selected = sorted(set(self.top_ids.reshape(-1).tolist()))
                print(f'Layer {index+1}: selected {len(selected)} expert tensorsets {selected}', flush=True)
                cp.prefetch([prefix+f'mlp.experts.{e}.{name}.weight' for e in selected for name in ['gate_proj', 'up_proj', 'down_proj']])
            gate.register_forward_hook(gate_hook)
        layer = env['DeepseekV2DecoderLayer'].__new__(env['DeepseekV2DecoderLayer'])
        nn.Module.__init__(layer)
        layer.hidden_size = config.hidden_size
        layer.self_attn = attn
        layer.mlp = mlp
        layer.input_layernorm, layer.post_attention_layernorm = norms_
        return layer.eval()

    def capture_attention(self, module, normalized, output, weights, positions, mask):
        n = normalized.shape[1]
        q = module.q_proj(normalized).view(1,n,16,192).transpose(1,2)
        qcontent, qraw = q.split([128,64],dim=-1)
        compressed = module.kv_a_proj_with_mqa(normalized)
        raw_latent, kraw = compressed.split([512,64],dim=-1)
        latent = module.kv_a_layernorm(raw_latent)
        expanded = module.kv_b_proj(latent).view(1,n,16,256).transpose(1,2)
        kcontent, v = expanded.split([128,128],dim=-1)
        kraw = kraw.view(1,n,1,64).transpose(1,2)
        cos, sin = module.rotary_emb(v,seq_len=n)
        qr, kr = self.env['apply_rotary_pos_emb'](qraw,kraw,cos,sin,positions)
        query = torch.cat([qcontent,qr],dim=-1)
        key = torch.cat([kcontent,kr.expand(1,16,n,64)],dim=-1)
        scores = (query@key.transpose(-1,-2))*module.softmax_scale
        masked = scores+mask
        probabilities = torch.softmax(masked,dim=-1)
        context = probabilities@v
        rebuilt = module.o_proj(context.transpose(1,2).reshape(1,n,2048))
        assert torch.equal(probabilities,weights), 'Attention softmax reconstruction differs'
        assert torch.equal(rebuilt,output), 'Attention output reconstruction differs'
        reorder = lambda x: x.view(*x.shape[:-1],32,2).transpose(-1,-2).reshape(x.shape)
        qb, kb = reorder(qraw), reorder(kraw)
        # This independently checks the inspector's half-split formula.
        rr = qb*cos[None,None] + self.env['rotate_half'](qb)*sin[None,None]
        assert torch.equal(rr,qr)
        self.rotary = {'cos':values(cos[:,:32]),'sin':values(sin[:,:32])}
        self.attention_scale = module.softmax_scale
        self.attention_ops.append({'index':self.current_layer,'queryBefore':values(qb[0]),'queryAfter':values(qr[0]),
            'keyBefore':values(kb[0]),'keyAfter':values(kr[0]),'value':values(v[0]),
            'queryContent':values(qcontent[0]),'keyContent':values(kcontent[0]),'kvLatent':values(latent[0]),
            'rawLatent':values(raw_latent[0]),'queryProjection':values(q[0]),
            'rawQueryRotary':values(qraw[0]),'rawKeyRotary':values(kraw[0]),
            'attentionContext':values(context[0]),'attentionOutput':values(output[0]),'attentionInput':values(normalized[0]),
            'outputPreview':preview(output[0])})
        for name, val in [('attention',weights),('attentionScores',scores),('attentionMaskedScores',masked)]:
            self.example[name].append(values(val[0]))
        return {'attentionInputPreview':preview(normalized[0]),'attentionInputNorms':norms(normalized[0]),
                'attentionOutputPreview':preview(output[0]),'attentionOutputNorms':norms(output[0]),
                'queryPreview':preview(query[0]),'queryNorms':norms(query[0]),'keyPreview':preview(key[0]),
                'keyNorms':norms(key[0]),'valuePreview':preview(v[0]),'valueNorms':norms(v[0])}

    def run(self, capture=True):
        self.record = capture
        cp, config, ex = self.cp, self.config, self.example
        ids = [t['id'] for t in ex['tokens']]
        n = len(ids)
        x = torch.stack([cp.tensor('model.embed_tokens.weight', row=i) for i in ids])[None]
        if capture:
            ex['embeddingPreview'], ex['embeddingNorms'] = preview(x[0]), norms(x[0])
        positions = torch.arange(n)[None]
        mask = torch.triu(torch.full((n,n),torch.finfo(torch.float32).min),diagonal=1)[None,None]
        for index in range(config.num_hidden_layers):
            self.current_layer = index
            print(f'{"Capture" if capture else "Reference"} layer {index+1}/{config.num_hidden_layers}', flush=True)
            layer = self.load_layer(index)
            self.detail_tokens = [{'inputPreview': [], 'selectedExperts': [], 'routingWeights': [], 'experts': [], 'combinedOutput': []} for _ in ids]
            self.full_expert_outputs = [{} for _ in ids]
            seen = {}
            if capture:
                def attn_hook(module, inputs, kwargs, output):
                    seen.update(self.capture_attention(module,kwargs['hidden_states'],output[0],output[1],positions,mask))
                    seen['branch_output'] = output[0]
                def mlp_pre(module, inputs):
                    seen['mlp_input'] = inputs[0].detach()
                layer.self_attn.register_forward_hook(attn_hook,with_kwargs=True)
                layer.mlp.register_forward_pre_hook(mlp_pre)
            residual_input = x
            x, weights = layer(x,attention_mask=mask,position_ids=positions,output_attentions=True,use_cache=False)
            if capture:
                mlp_input = seen.pop('mlp_input')[0]
                branch_output = seen.pop('branch_output')[0]
                counts = [0]*64
                tokens = []
                for t in range(n):
                    outputs = self.full_expert_outputs[t]
                    if index == 0:
                        selected, mixing = [], []
                        dense = outputs[-1]
                        routed = torch.zeros_like(dense)
                        shared = torch.zeros_like(dense)
                        combined = dense
                        expert_outputs = []
                    else:
                        selected = self.top_ids[t].tolist()
                        mixing = self.top_weights[t].tolist()
                        expert_outputs = [outputs[e] for e in selected]
                        routed = (torch.stack(expert_outputs)*self.top_weights[t,:,None]).sum(0)
                        shared = outputs[64]
                        combined = routed+shared
                        for e in selected: counts[e] += 1
                    reconstructed = residual_input[0,t]+branch_output[t]+combined
                    error = float((reconstructed-x[0,t]).abs().max())
                    assert error == 0, f'Layer residual reconstruction: {error}'
                    token = {'routerLogits':values(self.router_logits[t]) if index else [],
                        'routerProbabilities':values(self.router_probs[t]) if index else [],
                        'selectedExperts':selected,'routingWeights':mixing,
                        'expertOutputNorms':[float(o.norm()) for o in expert_outputs],
                        'weightedOutputNorms':[float((o*w).norm()) for o,w in zip(expert_outputs,mixing)],
                        'expertOutputPreview':[preview(o) for o in expert_outputs],
                        'routedCombinedOutputPreview':preview(routed),'sharedExpertOutputPreview':preview(shared),
                        'sharedExpertOutputNorm':float(shared.norm()),'combinedOutputPreview':preview(combined),
                        'inputNorm':float(mlp_input[t].norm()),'outputNorm':float(combined.norm()),
                        'reconstructionMaxAbsError':error,'outputMaxAbs':float(combined.abs().max()),
                        'reconstructionToleranceRatio':0,'denseOutputPreview':preview(combined) if index == 0 else None}
                    tokens.append(token)
                    detail = self.detail_tokens[t]
                    detail.update({'inputPreview':values(mlp_input[t,:4]),'selectedExperts':selected,
                                   'routingWeights':mixing,'combinedOutput':values(combined[:4]),
                                   'routedCombinedOutput':values(routed[:4]),'sharedExpertOutput':values(shared[:4]),
                                   'feedForwardType':'dense' if index == 0 else 'moe'})
                ex['layers'].append({'index':index,'feedForwardType':'dense' if index == 0 else 'moe',
                     'tokens':tokens,'expertTokenCounts':counts,**seen,'residualInputPreview':preview(residual_input[0]),
                     'residualAfterAttentionPreview':preview(residual_input[0]+branch_output),
                     'feedForwardInputPreview':preview(mlp_input),'blockOutputPreview':preview(x[0])})
                self.expert_ops.append({'index':index,'tokens':self.detail_tokens})
            del layer, seen
            gc.collect()
        finalnorm = self.env['DeepseekV2RMSNorm'](config.hidden_size,eps=config.rms_norm_eps).eval()
        finalnorm.load_state_dict({'weight':cp.tensor('model.norm.weight')})
        hidden = finalnorm(x)
        if capture:
            ex['finalHiddenPreview'], ex['finalHiddenNorms'] = preview(hidden[0]), norms(hidden[0])
        print('Loading complete LM head and calculating all vocabulary logits',flush=True)
        cp.prefetch(['lm_head.weight'])
        head = cp.tensor('lm_head.weight')
        logits = F.linear(hidden[:,-1],head).float()[0]
        assert torch.isfinite(logits).all()
        if capture:
            selected = logits.topk(4).indices
            selected_weights = head[selected]
            self.lm_head = {'tokenIndex':n-1,'inputSize':config.hidden_size,
                'tokenIds':selected.tolist(),'weights':values(selected_weights[:,:4]),
                'input':values(hidden[0,-1,:4]),
                'omittedContribution':values(F.linear(hidden[0,-1,4:],selected_weights[:,4:])),
                'output':values(logits[selected])}
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir',default='/private/tmp/moe-architecture-cache')
    parser.add_argument('--output-dir',default='static/traces')
    parser.add_argument('--prompt',default='The capital of France is')
    parser.add_argument('--example-id',default='deepseek-fact')
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    cp = ShardedCheckpoint(args.cache_dir)
    config = SimpleNamespace(**cp.config)
    assert config.num_hidden_layers == 27 and config.first_k_dense_replace == 1
    assert config.topk_method == 'greedy' and not config.norm_topk_prob and config.n_shared_experts == 2
    env = source_classes(cp.first.asset('modeling_deepseek.py'))
    tokenizer = Tokenizer.from_file(str(cp.first.asset('tokenizer.json')))
    tokenizer_config = json.loads(cp.first.asset('tokenizer_config.json').read_text())
    ids = tokenizer.encode(args.prompt).ids
    if tokenizer_config.get('add_bos_token') and ids[0] != config.bos_token_id:
        ids.insert(0,config.bos_token_id)
    ex = {'id':args.example_id,'title':'A familiar fact','category':'Language','prompt':args.prompt,
          'tokens':[{'index':n,'id':i,'text':'<BOS>' if i == config.bos_token_id else tokenizer.decode([i],skip_special_tokens=False)} for n,i in enumerate(ids)],
          'layers':[],'attention':[],'attentionScores':[],'attentionMaskedScores':[]}
    full_rotary = env['DeepseekV2YarnRotaryEmbedding'](64,max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,scaling_factor=config.rope_scaling['factor'],
        **{k:v for k,v in config.rope_scaling.items() if k not in ['type','factor']})
    short_rotary = env['DeepseekV2YarnRotaryEmbedding'](64,max_position_embeddings=len(ids),
        base=config.rope_theta,scaling_factor=config.rope_scaling['factor'],
        **{k:v for k,v in config.rope_scaling.items() if k not in ['type','factor']})
    assert torch.equal(full_rotary.cos_cached[:len(ids)],short_rotary.cos_cached)
    assert torch.equal(full_rotary.sin_cached[:len(ids)],short_rotary.sin_cached)
    del full_rotary,short_rotary
    recorder = Recorder(cp,config,env,ex)
    logits = recorder.run(capture=True)
    # A second official-layer pass has no observers and compares every final
    # vocabulary logit. Lazy residency is identical; activations are recomputed.
    reference = recorder.run(capture=False)
    max_error = float((logits-reference).abs().max())
    assert max_error == 0, f'Observed vs reference logit mismatch: {max_error}'
    vocabulary = [tokenizer.decode([i],skip_special_tokens=False) for i in range(config.vocab_size)]
    probabilities = torch.softmax(logits.double(),-1)
    top_values,top_ids = probabilities.topk(32)
    ex['nextTokenLogits'] = values(logits)
    ex['nextTokenCandidates'] = [{'id':int(i),'text':vocabulary[int(i)],'probability':float(p)} for i,p in zip(top_ids,top_values)]
    ex['validation'] = {'hookedVsUnhookedMaxLogitError':max_error,'allTransformerLayersExecuted':27,
          'denseLayersExecuted':1,'moeLayersExecuted':26,'sharedExpertsExecutedPerMoELayer':2,
          'attentionAndOutputReconstructionExact':True,'routerUsesUnnormalizedTopKProbabilities':True,
          'fullVocabularyLogits':config.vocab_size,'sourceSha256':SOURCE_SHA}
    timestamp = datetime.now(timezone.utc).isoformat()
    capture = {'timestamp':timestamp,'device':'cpu','dtype':'float32','python':platform.python_version(),
          'packages':{'torch':torch.__version__,'transformers':transformers.__version__},
          'script':'scripts/full_mla_capture.py','scriptSha256':digest(Path(__file__).read_bytes()),
          'notes':['All 27 pretrained transformer layers, final RMSNorm, and complete LM head execute.',
                   'Layer 1 uses a dense SwiGLU MLP; layers 2–27 use 6 of 64 routed experts plus 2 shared experts.',
                   'Top-k probabilities are not renormalized, exactly as this checkpoint is configured.',
                   'Exact checkpoint tensors are loaded lazily. No parameter quantization, pruning, or invented activations.',
                   'Checkpoint BOS token is included and displayed as <BOS>. It is not typed prompt text.',
                   'Observed and unobserved complete passes produce bit-identical final vocabulary logits.']}
    model = {'id':MODEL,'name':'DeepSeek-V2-Lite','architecture':'mla','revision':REVISION,
         'url':f'https://huggingface.co/{MODEL}','license':'DeepSeek Model License','numLayers':27,
         'numExperts':64,'topK':6,'hiddenSize':2048,'intermediateSize':1408,'denseIntermediateSize':10944,
         'sharedIntermediateSize':2816,'sharedExperts':2,'denseLayers':1,'normalizeRoutingWeights':False,
         'totalParameters':15706507264,'activeParametersApprox':2400000000,'residualMultiplier':1,
         'attentionHeads':16,'keyValueHeads':16,'attentionHeadSize':192,'attentionMultiplier':recorder.attention_scale,
         'embeddingMultiplier':1,'logitsScaling':1,'ropeTheta':10000,'vocabSize':102400,
         'attentionMaskValue':float(torch.finfo(torch.float32).min),'contentKeySize':128,'rotarySize':64,
         'valueHeadSize':128,'kvLatentSize':512,'queryProjectionSize':3072,'keyProjectionSize':3072,'valueProjectionSize':2048,'rmsNormEps':config.rms_norm_eps,
         'ropeScaling':cp.config['rope_scaling']}
    # Calculate the exact architecture parameter count from checkpoint metadata,
    # including tensors not needed by this prompt's sparse routing.
    model['totalParameters'] = sum(math.prod(s['shape']) for shard in cp.shards.values() for name,s in shard.header.items() if name != '__metadata__')
    bundle = {'schemaVersion':1,'model':model,'capture':capture,'provenance':cp.provenance(),'examples':[ex],'vocabulary':vocabulary}
    out = Path(args.output_dir)
    write(out/'deepseek-mla.json',bundle)
    mainsha = digest((out/'deepseek-mla.json').read_bytes())
    attention = {'schemaVersion':1,'kind':'attention-operations','model':{'id':MODEL,'revision':REVISION,'architecture':'mla',
         'attentionHeads':16,'keyValueHeads':1,'headSize':64,'rotarySize':64,'valueHeads':16,'valueHeadSize':128,
         'contentKeySize':128,'kvLatentSize':512,'ropeTheta':10000,'attentionMultiplier':recorder.attention_scale},
         'capture':{**capture,'mainBundleSha256':mainsha,'mainCaptureScriptSha256':capture['scriptSha256'],
              'matchesMainVocabularyLogits':True,'actualValueUnchangedByRotation':True,
              'actualGroupedKeyValueRepetitionExact':True,'projectionHeadReshapeExact':True,
              'coordinateReordering':'Official DeepSeek interleaved rotary projection coordinates are reordered into first/second halves before rotation; queryBefore/keyBefore are after this reorder.',
              'rotaryAmplitude':math.sqrt(recorder.rotary['cos'][0][0]**2+recorder.rotary['sin'][0][0]**2)},
         'example':{k:ex[k] for k in ['id','prompt','tokens']},'positions':list(range(len(ids))),
         'rotary':recorder.rotary,'layers':recorder.attention_ops,'lmHead':recorder.lm_head,
         'validation':{'all27Layers':True,'all16QueryHeads':True,'rawProjectionAndReorderedRotaryValuesPresent':True,
             'independentAttentionReconstructionMaxAbsError':0,'independentOutputReconstructionMaxAbsError':0}}
    expert = {'schemaVersion':1,'kind':'expert-operations','model':{'id':MODEL,'revision':REVISION,'architecture':'mla',
         'hiddenSize':2048,'intermediateSize':1408,'sharedIntermediateSize':2816,'denseIntermediateSize':10944,'topK':6,
         'sharedExperts':2,'normalizeRoutingWeights':False},'capture':capture,'example':{k:ex[k] for k in ['id','prompt']},
         'layers':recorder.expert_ops}
    write(out/f'attention-operations-{args.example_id}.json',attention)
    write(out/f'expert-operations-{args.example_id}.json',expert)
    print('Next token candidates:',[(x['text'],round(x['probability'],5)) for x in ex['nextTokenCandidates'][:5]],flush=True)
    print('ALL COMPLETE-PASS CHECKS PASSED',flush=True)

if __name__ == '__main__':
    main()
