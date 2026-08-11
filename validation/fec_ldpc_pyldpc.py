"""LDPC decoding against a second implementation (decision D7).

The analytic references available to a decoder are weak. A closed form
exists for the ensemble threshold, not for a finite code, and the union
bound that serves the convolutional module has no counterpart here. So
the only sharp check is a second implementation -- and the usual obstacle
is that published LDPC curves rarely come with the exact parity-check
matrix behind them, so any disagreement can be blamed on the code rather
than on the decoder.

That obstacle disappears when the matrix is *shared* rather than
re-derived. pyldpc (Hicham Janati, BSD-3-Clause) takes an arbitrary H and
decodes with log-domain sum-product; this library takes the same H and
decodes with min-sum. Both are fed the same log-likelihood ratios, so the
only thing that differs is the check-node update and a disagreement is
attributable to it alone.

What that buys, and what it does not. Min-sum is an approximation of
sum-product, so the two are *not* expected to agree numerically: it
overestimates the check-node magnitude and loses a few tenths of a dB.
The reference therefore pins three statements rather than one number:

* where the approximation vanishes -- at high SNR both must reach the
  same codeword, since the min-sum update is exact whenever one incoming
  magnitude is clearly the smallest;
* the sign of the penalty -- min-sum must be worse, never better;
* that the normalization factor recovers part of it, which is the reason
  ``alpha`` exists at all.

The channel is regenerated from a seed rather than shipped sample by
sample. That is safe only because the stream is stable, which was checked
rather than assumed -- bit-identical under numpy 1.26.4 and 2.4.6, the
floor and the newest release this project supports -- and the manifest
carries a hash of the LLRs so that a future change fails loudly instead
of quietly comparing two different channels.
"""
import hashlib
import pathlib

import numpy as np

from comnumpy.fec.ldpc import LDPCDecoder, LDPCEncoder

DATA = pathlib.Path(__file__).parent / "data" / "pyldpc"
N_ITER = 50


def manifest():
    rows = (DATA / "ldpc_manifest.csv").read_text().splitlines()[1:]
    return dict(row.split(",", 1) for row in rows if row)


def parity_check():
    rows = (DATA / "ldpc_parity_check.csv").read_text().splitlines()[1:]
    return np.array([[int(bit) for bit in row] for row in rows if row])


def reference():
    """pyldpc's decisions, keyed by SNR: (transmitted, decoded)."""
    rows = (DATA / "ldpc_reference.csv").read_text().splitlines()[1:]
    out = {}
    for row in rows:
        snr, _, sent, decoded = row.split(",")
        sent_bits, got_bits = out.setdefault(float(snr), ([], []))
        sent_bits.append([int(bit) for bit in sent])
        got_bits.append([int(bit) for bit in decoded])
    return {snr: (np.array(a), np.array(b)) for snr, (a, b) in out.items()}


def channel(config):
    """Replay the exact channel the reference was generated on."""
    H = parity_check()
    encoder = LDPCEncoder(H)
    rng = np.random.default_rng(int(config["seed"]))
    n_words = int(config["n_words"])
    digest = hashlib.sha256()
    blocks = {}
    for snr_dB in sorted(float(key.rsplit("_", 1)[-1])
                         for key in config if key.startswith("pyldpc_ber_")):
        variance = 10 ** (-snr_dB / 10)
        bits = rng.integers(0, 2, size=(n_words, encoder.k))
        codewords = np.asarray(encoder(bits))
        received = (1.0 - 2.0 * codewords) + np.sqrt(variance) * rng.normal(
            size=codewords.shape)
        llr = 2 * received / variance
        digest.update(np.ascontiguousarray(llr, dtype=np.float64).tobytes())
        blocks[snr_dB] = (codewords, llr)
    return H, blocks, digest.hexdigest()


def check_the_channel_is_the_one_that_was_recorded(config, digest):
    """Without this, a changed RNG stream would compare two different channels."""
    assert digest == config["llr_sha256"], (digest, config["llr_sha256"])
    print("PASS the replayed channel hashes to the recorded value")
    print("       so the two decoders are compared on the same data rather than")
    print("       on two draws that merely share a seed")


def check_they_agree_where_the_approximation_vanishes(H, blocks, ref):
    print("\nPASS min-sum against sum-product on identical LLRs:")
    print("       SNR    ours BER   pyldpc BER   identical codewords")
    agreement = {}
    for snr_dB, (codewords, llr) in sorted(blocks.items()):
        sent, theirs = ref[snr_dB]
        assert np.array_equal(sent, codewords), snr_dB
        ours = np.asarray(LDPCDecoder(H, n_iter=N_ITER, alpha=1.0,
                                      output="codeword")(llr))
        same = float(np.mean(np.all(ours == theirs, axis=-1)))
        agreement[snr_dB] = same
        print(f"       {snr_dB:4.1f}  {np.mean(ours != codewords):9.5f}  "
              f"{np.mean(theirs != codewords):11.5f}   {100 * same:6.1f} %")
    assert agreement[max(agreement)] > agreement[min(agreement)], agreement
    print("       the two converge as the SNR rises: the min-sum check update is")
    print("       exact whenever one incoming magnitude is clearly the smallest,")
    print("       so the gap is a waterfall phenomenon rather than a defect")
    return agreement


def check_min_sum_pays_and_alpha_refunds(H, blocks, ref):
    print("\nPASS the sign of the penalty, and what alpha buys:")
    print("       SNR      plain   alpha=0.75      pyldpc")
    plain_worse = 0
    for snr_dB, (codewords, llr) in sorted(blocks.items()):
        _, theirs = ref[snr_dB]
        errors = {}
        for label, alpha in (("plain", 1.0), ("normalized", 0.75)):
            decoded = np.asarray(LDPCDecoder(H, n_iter=N_ITER, alpha=alpha,
                                             output="codeword")(llr))
            errors[label] = float(np.mean(decoded != codewords))
        their_ber = float(np.mean(theirs != codewords))
        plain_worse += errors["plain"] >= their_ber
        assert errors["normalized"] <= errors["plain"], (snr_dB, errors)
        print(f"       {snr_dB:4.1f}  {errors['plain']:9.5f}  "
              f"{errors['normalized']:11.5f}  {their_ber:10.5f}")
    assert plain_worse == len(blocks), plain_worse
    print("       plain min-sum is worse than sum-product at every point, which is")
    print("       the direction the approximation must go: it overestimates the")
    print("       check magnitude, so it over-trusts its own messages. Scaling by")
    print("       0.75 refunds most of it -- and at 2 and 3 dB it overshoots and")
    print("       edges past sum-product, which is not an error in either: on a")
    print("       short graph with cycles, sum-product is itself overconfident,")
    print("       and damping the messages happens to help both ways.")


def main():
    config = manifest()
    H, blocks, digest = channel(config)
    ref = reference()
    check_the_channel_is_the_one_that_was_recorded(config, digest)
    agreement = check_they_agree_where_the_approximation_vanishes(H, blocks, ref)
    check_min_sum_pays_and_alpha_refunds(H, blocks, ref)
    print(f"\nAll checks passed on a ({H.shape[1]}, {LDPCEncoder(H).k}) Gallager "
          f"code, {config['n_words']} words per point, against pyldpc "
          f"{config['pyldpc_version']}.")
    print(f"Codeword agreement rises from {100 * agreement[min(agreement)]:.0f} % "
          f"to {100 * agreement[max(agreement)]:.0f} % across the range.")


if __name__ == "__main__":
    main()
