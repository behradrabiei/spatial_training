"""CPU unit checks for longnav.utils.hamlet (no model, no GPU, runs on a login node).

Pins the read-out contract the rollout/replay equivalence relies on:
1. the memory read-out is exactly ``mem_embed`` at init (zero-init out_proj) and
   handles an empty history (turn 0);
2. the read-out for block q depends only on moment blocks < q, and the rollout
   form (history of q blocks, query q) equals row q of the replay form
   (all blocks, queries 0..T-1);
3. ``memory_window`` cuts the read-out's receptive field;
4. every parameter receives gradient from a single replay (DDP runs with
   find_unused_parameters=False);
5. ``splice_moment_tokens`` lays out [MEM] before the frame and the moment block
   after it (image-less chunks: both before the assistant header), and
   ``replace_rows_at`` writes per-block rows out of place with gradient to both sides.

Usage: PYTHONPATH=src python tests/hamlet_unit.py
"""
import torch

from longnav.utils.hamlet import (HamletModule, mem_ratio, moment_positions, replace_moment_rows,
                                  replace_rows_at, splice_moment_tokens)

H, N_MOMENT, N_MEM = 64, 8, 1
VS, VE, IMG, IM_START, IM_END, NL, ASSIST, USER, STAR = 151652, 151653, 151655, 151644, 151645, 198, 77091, 872, 334
MOMENT_IDS = list(range(1000, 1000 + N_MOMENT))
MEM_IDS = list(range(1000 + N_MOMENT, 1000 + N_MOMENT + N_MEM))


def make_module(**kw):
    torch.manual_seed(0)
    return HamletModule(H, n_moment=N_MOMENT, n_mem=N_MEM, d_mem=32, n_layers=kw.pop("n_layers", 2), n_heads=4, **kw)


def randomize_out_proj(mod, std=0.05):
    with torch.no_grad():
        mod.out_proj.weight.normal_(0, std)
        mod.out_proj.bias.normal_(0, std)


def check_readout():
    mod = make_module()
    moments = torch.randn(5, N_MOMENT, H)
    rows = mod(mode="readout", moments=moments, query_blocks=torch.arange(5))
    assert rows.shape == (5, N_MEM, H), rows.shape
    assert torch.equal(rows, mod.mem_embed.weight[None].expand_as(rows)), "read-out must be exactly mem_embed at init"
    assert mem_ratio(rows, mod(mode="mem_embeds")) == 0.0
    r0 = mod(mode="readout", moments=torch.zeros(0, N_MOMENT, H), query_blocks=torch.tensor([0]))
    assert r0.shape == (1, N_MEM, H)
    print("[1] ok: zero read-out at init, empty history handled")

    randomize_out_proj(mod)
    full = mod(mode="readout", moments=moments, query_blocks=torch.arange(5))
    for q in range(5):
        pert = moments.clone()
        pert[q:] = torch.randn_like(pert[q:])  # blocks >= q must not matter
        rq = mod(mode="readout", moments=pert, query_blocks=torch.tensor([q]))[0]
        assert torch.allclose(rq, full[q], atol=1e-5), f"read-out {q} depends on blocks >= {q}"
        rollout_form = mod(mode="readout", moments=moments[:q], query_blocks=torch.tensor([q]))[0]
        assert torch.allclose(rollout_form, full[q], atol=1e-5), f"rollout form != replay row {q}"
    pert = moments.clone()
    pert[0] += 1.0
    assert not torch.allclose(mod(mode="readout", moments=pert, query_blocks=torch.tensor([1]))[0], full[1]), \
        "read-out 1 should depend on block 0"
    print("[2] ok: strict causality, rollout form == replay form")

    modw = make_module(n_layers=1, memory_window=2)
    randomize_out_proj(modw)
    base = modw(mode="readout", moments=moments, query_blocks=torch.tensor([4]))[0]
    pert = moments.clone()
    pert[:2] += 1.0  # blocks 0,1 lie outside (4-2 .. 3]
    assert torch.allclose(modw(mode="readout", moments=pert, query_blocks=torch.tensor([4]))[0], base, atol=1e-5)
    pert = moments.clone()
    pert[2] += 1.0
    assert not torch.allclose(modw(mode="readout", moments=pert, query_blocks=torch.tensor([4]))[0], base)
    print("[3] ok: memory_window bounds the read-out")

    mod.zero_grad()
    mod(mode="readout", moments=moments, query_blocks=torch.arange(5)).sum().backward()
    # moment_embed feeds the LM input (replace_moment_rows), not the read-out; the smoke test covers it
    missing = [n for n, p in mod.named_parameters() if p.grad is None and not n.startswith("moment_embed")]
    assert not missing, f"parameters without gradient: {missing}"
    print(f"[4] ok: every read-out parameter ({sum(1 for _ in mod.parameters()) - 1}) receives gradient from one replay")


def check_splice():
    def chunk(with_image=True):
        ids = [13435, STAR, IM_END, NL, IM_START, USER, NL] + ([VS, IMG, IMG, IMG, VE] if with_image else []) \
              + [IM_END, NL, IM_START, ASSIST, NL, STAR]
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "mm_token_type_ids": torch.zeros_like(ids)}

    out = splice_moment_tokens(chunk(), MOMENT_IDS, VE, prefix_len=4, mem_ids=MEM_IDS, vision_start_id=VS)
    seq = out["input_ids"][0].tolist()
    assert seq[7:8] == MEM_IDS and seq[8] == VS and seq[12] == VE and seq[13:13 + N_MOMENT] == MOMENT_IDS, seq
    assert seq[-4:] == [IM_START, ASSIST, NL, STAR]
    assert out["attention_mask"].shape == out["input_ids"].shape == out["mm_token_type_ids"].shape
    assert int(out["mm_token_type_ids"].sum()) == 0 and int(out["attention_mask"].sum()) == len(seq)
    mpos = moment_positions(out["input_ids"][0], MOMENT_IDS)
    kpos = moment_positions(out["input_ids"][0], MEM_IDS)
    assert mpos.shape == (1, N_MOMENT) and kpos.shape == (1, N_MEM) and int(kpos[0, -1]) < int(mpos[0, 0])
    out2 = splice_moment_tokens(chunk(with_image=False), MOMENT_IDS, VE, prefix_len=4, mem_ids=MEM_IDS, vision_start_id=VS)
    seq2 = out2["input_ids"][0].tolist()
    assert seq2[-4:] == [IM_START, ASSIST, NL, STAR] and seq2[-4 - N_MOMENT:-4] == MOMENT_IDS \
        and seq2[-4 - N_MOMENT - N_MEM:-4 - N_MOMENT] == MEM_IDS, seq2
    print("[5a] ok: [MEM] before the frame, moment block after it; image-less chunk before the header")

    emb = torch.randn(1, len(seq), H, requires_grad=True)
    table = torch.randn(N_MOMENT, H, requires_grad=True)
    emb1 = replace_moment_rows(emb, out["input_ids"], MOMENT_IDS, table)
    assert torch.equal(emb1[0, mpos[0]], table) and torch.equal(emb1[0, :7], emb[0, :7])
    rows = torch.randn(1, N_MEM, H, requires_grad=True)
    emb2 = replace_rows_at(emb1, kpos, rows)
    assert torch.equal(emb2[0, kpos[0]], rows[0]) and torch.equal(emb2[0, mpos[0]], table)
    emb2.sum().backward()
    assert rows.grad is not None and table.grad is not None and emb.grad is not None
    assert float(emb.grad[0, kpos[0]].abs().sum()) == 0.0 and float(emb.grad[0, 0].abs().sum()) > 0
    print("[5b] ok: out-of-place row swaps carry gradient to the table, the rows and the untouched embeds")


if __name__ == "__main__":
    check_readout()
    check_splice()
    print("hamlet unit checks passed")
