"""
Step 6: Secure Aggregation via pairwise masking.

This implements the core mathematical construction behind Bonawitz et al.
2017 ("Practical Secure Aggregation for Privacy-Preserving Machine
Learning") -- the same foundation Flower's native SecAgg+ protocol uses --
rather than Flower's built-in implementation, which lives in a newer
ClientApp/ServerApp API that would require rewriting our existing,
working client/server code.

HOW IT WORKS
Every pair of clients (i, j) derives a shared pseudorandom mask. By
convention, the client earlier in CLIENTS adds the mask; the later one
subtracts it. When the server sums all 5 clients' masked updates, every
pairwise mask appears once as +mask and once as -mask, cancelling exactly.
The server is left with the true sum -- but never sees any individual
client's real update, masked or not, in isolation.

WHAT THIS SIMULATION SIMPLIFIES (be aware of this for your write-up)
Real Bonawitz et al. SecAgg derives each pairwise mask from a secret only
the two clients in that pair know (via real Diffie-Hellman key exchange,
so even the server can't derive it) and uses Shamir secret sharing so the
protocol survives clients dropping out mid-round. Here, MASTER_SECRET is a
single shared constant known to every simulated client -- a stand-in for
"the clients have already securely agreed on pairwise keys", since actually
running a key-exchange handshake between simulated processes adds
complexity without changing the core privacy property we're demonstrating:
the SERVER only ever sees the sum. This also means our current version
assumes full participation every round (no dropout) -- Step 9 revisits
this, since real SecAgg+'s threshold secret sharing exists specifically
to handle a client disappearing mid-protocol.

REQUIREMENT: equal client weighting. Masks only cancel in a SIMPLE sum.
If clients were weighted unequally (as in Step 5's early size-weighted or
log-weighted attempts), each client's mask would be scaled by a different
factor and would NOT cancel. This is precisely why Step 5 settled on equal
weighting (fedavg_weight=1000 for every client) -- it's a prerequisite for
SecAgg to work correctly, not just a fairness choice.
"""

import numpy as np

# Stand-in for securely agreed-upon pairwise keys -- see docstring above.
# In a real deployment this must NEVER be a constant known to the server.
MASTER_SECRET = b"fedshield-ids-secagg-demo-salt-v1"

# Must match the client set used everywhere else in the project.
CLIENTS = ["bank", "ecommerce", "hospital", "manufacturing", "university"]


def _pairwise_seed(client_a: str, client_b: str, round_num: int) -> int:
    """Deterministic seed both clients in a pair can independently derive."""
    a, b = sorted([client_a, client_b])  # order-independent: both sides agree
    key = f"{a}|{b}|{round_num}".encode() + MASTER_SECRET
    return int.from_bytes(np.frombuffer(
        __import__("hashlib").sha256(key).digest()[:8], dtype=np.uint64
    ), "little") % (2**32)


def _generate_masks(seed: int, shapes: list[tuple]) -> list[np.ndarray]:
    """Same seed -> same mask arrays, independent of who calls this."""
    rng = np.random.default_rng(seed)
    return [rng.normal(scale=100.0, size=shape).astype(np.float32) for shape in shapes]


def mask_parameters(client_name: str, parameters: list[np.ndarray],
                     round_num: int, participants: list[str] = CLIENTS) -> list[np.ndarray]:
    """
    Apply pairwise masks to this client's update before sending it to the
    server. Call this as the LAST step in fit(), right before returning.
    """
    shapes = [p.shape for p in parameters]
    masked = [p.copy() for p in parameters]

    for other in participants:
        if other == client_name:
            continue
        seed = _pairwise_seed(client_name, other, round_num)
        masks = _generate_masks(seed, shapes)
        # Consistent sign convention: earlier-in-list client adds, later subtracts.
        sign = 1 if sorted([client_name, other])[0] == client_name else -1
        masked = [m + sign * mask for m, mask in zip(masked, masks)]

    return masked


if __name__ == "__main__":
    # Sanity check: masked sum across all clients should equal the true sum,
    # to within floating-point precision -- proving the masks truly cancel.
    rng = np.random.default_rng(0)
    true_updates = {c: [rng.normal(size=(4, 3)), rng.normal(size=(3,))] for c in CLIENTS}

    masked_updates = {
        c: mask_parameters(c, true_updates[c], round_num=1) for c in CLIENTS
    }

    true_sum = [sum(true_updates[c][i] for c in CLIENTS) for i in range(2)]
    masked_sum = [sum(masked_updates[c][i] for c in CLIENTS) for i in range(2)]

    for i, (t, m) in enumerate(zip(true_sum, masked_sum)):
        max_diff = np.abs(t - m).max()
        print(f"Layer {i}: max difference between true sum and masked sum = {max_diff:.2e}")
        assert max_diff < 1e-4, "Masks did not cancel correctly!"
    print("\nMasks cancel correctly: the server-side sum is unaffected by masking,")
    print("but no individual client's masked update reveals their true update.")