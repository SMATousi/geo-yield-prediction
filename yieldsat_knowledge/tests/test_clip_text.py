"""Exercise the real CLIP text projection with a tiny offline random model."""
import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, BatchEncoding, CLIPTextConfig, CLIPTextModelWithProjection
from yieldsat_knowledge.common import read
from yieldsat_knowledge.text_cache import build, DEFAULT_MODEL


def setup_clip(monkeypatch, length=4):
    torch.manual_seed(13)
    enc=CLIPTextModelWithProjection(CLIPTextConfig(vocab_size=20,hidden_size=16,
        intermediate_size=32,projection_dim=8,num_hidden_layers=1,num_attention_heads=2,
        max_position_embeddings=8,bos_token_id=1,eos_token_id=2,pad_token_id=0))
    calls=[]
    class Tokenizer:
        def __call__(self,texts,**kw):
            assert kw['truncation'] is False
            ids=torch.tensor([[1]+[3]*(length-2)+[2]]*len(texts))
            return BatchEncoding({'input_ids':ids,'attention_mask':torch.ones_like(ids)})
    tok=Tokenizer()
    def load_model(model,**kw):
        calls.append((model,kw));return enc
    monkeypatch.setattr(CLIPTextModelWithProjection,'from_pretrained',load_model)
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**kw:tok)
    return enc,tok,calls


def test_clip_uses_frozen_projected_eos_and_normalizes(tmp_path,monkeypatch):
    enc,tok,calls=setup_clip(monkeypatch)
    lib={'concepts':[{'id':'a','description':'test concept'}],
         'rules':[{'id':'r','statement':'test relationship'}]}
    build(lib,tmp_path/'text',revision='a'*40)
    assert calls==[(DEFAULT_MODEL,{'revision':'a'*40})]
    assert not enc.training and all(not p.requires_grad for p in enc.parameters())
    with torch.no_grad():
        v=enc(**tok(['test concept','test relationship'],truncation=False)).text_embeds
        expected=torch.nn.functional.normalize(v,dim=-1).numpy()
    with np.load(tmp_path/'text/vectors.npz') as cache:
        assert cache['ids'].tolist()==['a','r']
        np.testing.assert_allclose(cache['vectors'],expected,atol=1e-6)
    meta=read(tmp_path/'text/manifest.json')
    assert meta['encoder_class']=='CLIPTextModelWithProjection'
    assert meta['embedding_dim']==8 and meta['max_length']==8 and not meta['truncation']


def test_clip_rejects_overlong_qualifications_without_publishing(tmp_path,monkeypatch):
    setup_clip(monkeypatch,length=9)
    lib={'concepts':[{'id':'long_concept','description':'long'}],'rules':[]}
    with pytest.raises(ValueError,match='shorten and re-review.*long_concept'):
        build(lib,tmp_path/'text',revision='a'*40)
    assert not (tmp_path/'text').exists()
