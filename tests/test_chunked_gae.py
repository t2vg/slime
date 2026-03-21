import time
import pytest
import torch

from slime.utils.ppo_utils import chunked_gae, vanilla_gae


@pytest.mark.parametrize(
    "B,T",
    [
        (16, 4096),
        (32, 8192),
        (256, 128 * 1024),
    ],
)
@pytest.mark.parametrize("chunk_size", [64, 128, 256])
def test_gae_parallel_matches_serial(B, T, chunk_size):
    """
    Test that chunked_gae (parallel-scan) matches vanilla_gae (batch-serial)
    under various shapes, chunk sizes and dtypes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    rewards = torch.randn(B, T, device=device, dtype=torch.float32)
    values = torch.randn(B, T, device=device, dtype=torch.float32)

    gamma, lam = 0.99, 0.95

    # ---------- Serial ----------
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    adv_s, ret_s = vanilla_gae(rewards, values, gamma, lam)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    serial_time = t1 - t0

    # ---------- Parallel-scan ----------
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    adv_p, ret_p = chunked_gae(rewards, values, gamma, lam, chunk_size=chunk_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    parallel_time = t1 - t0

    # ---------- Accuracy ----------
    adv_err = (adv_s - adv_p).abs().max().item()
    ret_err = (ret_s - ret_p).abs().max().item()

    atol = 1e-5
    assert adv_err < atol, f"adv error too large: {adv_err}"
    assert ret_err < atol, f"ret error too large: {ret_err}"

    # ---------- logging ----------
    print(f"\n[GAE Test] B={B}, T={T}, chunk={chunk_size}")
    print(f"  Serial   : {serial_time:.6f} s")
    print(f"  Parallel : {parallel_time:.6f} s")
    print(f"  Speedup  : x{serial_time / parallel_time:.2f}")
    print(f"  Max diff adv={adv_err:.3e}, ret={ret_err:.3e}")


# ---------------------------------------------------------------------------
# Staged GAE tests: loss_mask defines independent stages
# ---------------------------------------------------------------------------

def _make_staged_mask(B, T, num_stages, device):
    """Create a loss_mask with *num_stages* contiguous active regions per sample,
    separated by inactive (0) gaps."""
    mask = torch.zeros(B, T, device=device, dtype=torch.float32)
    seg_len = T // (num_stages * 2)  # active length per stage
    for s in range(num_stages):
        start = s * 2 * seg_len + seg_len // 2
        end = start + seg_len
        mask[:, start:end] = 1.0
    return mask


@pytest.mark.parametrize("B,T", [(4, 512), (8, 1024)])
@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("num_stages", [2, 3, 5])
def test_staged_gae_chunked_matches_vanilla(B, T, chunk_size, num_stages):
    """chunked_gae with loss_mask must match vanilla_gae with the same mask."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    rewards = torch.randn(B, T, device=device, dtype=torch.float32)
    values = torch.randn(B, T, device=device, dtype=torch.float32)
    loss_mask = _make_staged_mask(B, T, num_stages, device)

    gamma, lam = 0.99, 0.95

    adv_s, ret_s = vanilla_gae(rewards, values, gamma, lam, loss_mask=loss_mask)
    adv_p, ret_p = chunked_gae(rewards, values, gamma, lam, chunk_size=chunk_size, loss_mask=loss_mask)

    adv_err = (adv_s - adv_p).abs().max().item()
    ret_err = (ret_s - ret_p).abs().max().item()

    atol = 1e-5
    assert adv_err < atol, f"staged adv error too large: {adv_err}"
    assert ret_err < atol, f"staged ret error too large: {ret_err}"

    print(f"\n[Staged GAE] B={B}, T={T}, chunk={chunk_size}, stages={num_stages}")
    print(f"  Max diff adv={adv_err:.3e}, ret={ret_err:.3e}")


@pytest.mark.parametrize("chunk_size", [32, 64])
def test_staged_gae_no_cross_stage_leakage(chunk_size):
    """Advantages in one stage must not be affected by rewards in another stage."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 2, 256
    gamma, lam = 0.99, 0.95

    # Stage 1: [20..80), Stage 2: [150..220)
    loss_mask = torch.zeros(B, T, device=device, dtype=torch.float32)
    loss_mask[:, 20:80] = 1.0
    loss_mask[:, 150:220] = 1.0

    torch.manual_seed(7)
    rewards_base = torch.randn(B, T, device=device, dtype=torch.float32)
    values = torch.randn(B, T, device=device, dtype=torch.float32)

    adv_base, _ = chunked_gae(rewards_base, values, gamma, lam,
                               chunk_size=chunk_size, loss_mask=loss_mask)

    # Perturb rewards only in stage 2 — stage 1 advantages must be unchanged
    rewards_perturbed = rewards_base.clone()
    rewards_perturbed[:, 150:220] += 100.0

    adv_perturbed, _ = chunked_gae(rewards_perturbed, values, gamma, lam,
                                    chunk_size=chunk_size, loss_mask=loss_mask)

    stage1_diff = (adv_base[:, 20:80] - adv_perturbed[:, 20:80]).abs().max().item()
    stage2_diff = (adv_base[:, 150:220] - adv_perturbed[:, 150:220]).abs().max().item()

    assert stage1_diff < 1e-6, f"Stage 1 leaked from stage 2: diff={stage1_diff}"
    assert stage2_diff > 1.0, "Stage 2 should have changed significantly"

    print(f"\n[Leakage Test] chunk={chunk_size}")
    print(f"  Stage 1 diff (should be ~0): {stage1_diff:.3e}")
    print(f"  Stage 2 diff (should be large): {stage2_diff:.3e}")
