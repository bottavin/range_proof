# range_proof

This library implements the **range-proof** scheme described in [this HackMD note](https://hackmd.io/@dabo/B1U4kx8XI), built in Rust on top of the [arkworks](https://arkworks.rs) ecosystem. 

The library is generic over any pairing-friendly curve that implements
`ark_ec::pairing::Pairing`.

---

## How it works

The scheme encodes `v` into a polynomial `g(x)` via a binary-decomposition
recurrence evaluated at the `n`-th roots of unity, then uses three algebraic
constraints to prove well-formedness:

| Constraint | What it checks |
|---|---|
| **w₁** | `g(ω⁰) = f(ω⁰)` — the committed value equals the encoding base |
| **w₂** | Every evaluation of `g` is a bit (0 or 1) |
| **w₃** | The recurrence `g(ωⁱ) = 2·g(ωⁱ⁺¹) + zᵢ` holds everywhere |

All three constraints are combined into a single quotient polynomial `q(x)` and
opened via the **KZG10** polynomial commitment scheme over any supported pairing
curve `E`. The resulting `RangeProof<E>` is a constant-size, pairing-based proof.

### High-level API

```rust
prove_min<E: Pairing>(pp, min, values_poly, values, num_bits) -> Result<Vec<RangeProof<E>>>
prove_max<E: Pairing>(pp, max, values_poly, values, num_bits) -> Result<Vec<RangeProof<E>>>

min_verify<E: Pairing>(pp, min, values_com, proofs, num_bits) -> Result<bool>
max_verify<E: Pairing>(pp, max, values_com, proofs, num_bits) -> Result<bool>
```

Both the prover and verifier accept **batches** of values and run in parallel via
[rayon](https://github.com/rayon-rs/rayon).

---

## Repository structure

```
src/
├── lib.rs                   # crate root – re-exports public modules
├── range_proof.rs           # proof generation and verification (generic over E: Pairing)
├── utilities.rs             # polynomial helpers, binary encoding, Fiat-Shamir
└── ark_serde.rs             # serde serialisation for arkworks types
tests/
├── range_proof_test.rs      # comprehensive range proof tests (BLS12-381)
├── utilities_test.rs        # utility function tests
└── curves.rs                # cross-curve tests (BLS12-381 vs BN254)
```

---

## Prerequisites

| Tool | Version |
|---|---|
| Rust (stable) | ≥ 1.75 |
| Cargo | bundled with Rust |

No additional system libraries are required; all cryptographic dependencies are
pure Rust.

---

## Build

```bash
git clone https://github.com/<your-handle>/range_proof
cd range_proof
cargo build --release
```

---

## Run the tests

```bash
# All unit + integration tests (fast, single-threaded)
cargo test

# With output from println! (useful for the performance smoke tests)
cargo test -- --nocapture

# Run a specific test
cargo test test_max_verify_with_valid_values

# Run only the cross-curve tests
cargo test --test curves
```

> **Note:** The large-`n` tests (`n = 128`, `n = 256`) involve KZG setups and
> parallel proof generation and may take a few seconds on consumer hardware.

---

## Code coverage

Generate an LLVM coverage report with `cargo-llvm-cov`:

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --open
```

Current coverage snapshot:

| File | Lines |
|---|---|
| `range_proof.rs` | ~70 % |
| `lib.rs` | ~100 % |
| `utilities.rs` | ~100 % |
| `ark_serde.rs` | ~100 % (tested indirectly via serde round-trips) |

---

## Security notes

* The KZG setup produced by `KZG10::setup` is **not a trusted setup** suitable
  for production; replace it with a multi-party ceremony for real deployments.
* Proofs are **non-hiding** (no blinding factors).  If the committed value must
  remain secret against an adversary who also sees the proof, use the hiding
  variant of KZG10.
* The Fiat-Shamir heuristic is instantiated with SHA-256 over serialized
  compressed group elements, which is adequate for testing but should be
  reviewed for production use.
---

## Dependencies (key)

| Crate | Purpose |
|---|---|
| `ark-ec` | Generic pairing-curve traits (`Pairing`, `CurveGroup`) |
| `ark-ff` | Finite field arithmetic |
| `ark-poly` | Dense univariate polynomials and FFT domains |
| `ark-poly-commit` | KZG10 polynomial commitment scheme |
| `ark-serialize` | Canonical serialisation of group elements (Fiat-Shamir, serde) |
| `ark-std` | Arkworks RNG and utility abstractions |
| `ark-bls12-381` | BLS12-381 pairing curve (primary curve) |
| `ark-bn254` | BN254 pairing curve |
| `num-traits` | Generic numeric traits |
| `rand` | Random number generation |
| `rayon` | Data-parallel proof / verify |
| `sha2` | SHA-256 for Fiat-Shamir challenges |
| `serde` / `serde_json` | Proof serialisation |
| `anyhow` | Ergonomic error handling |

---

## License

This project is licensed under the MIT License.