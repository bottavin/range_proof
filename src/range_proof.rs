use super::utilities::*;
use anyhow::anyhow;
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use ark_ff::{Field, One, Zero};
use ark_poly::{
    univariate::DenseOrSparsePolynomial, univariate::DensePolynomial, DenseUVPolynomial, Polynomial,
};
use ark_poly_commit::{
    kzg10::{Powers, Randomness, UniversalParams, KZG10},
    PCCommitmentState,
};
use ark_std::borrow::Cow;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Convenience type aliases
// ---------------------------------------------------------------------------

/// The scalar field of the pairing curve `E`.
/// Used throughout in place of the concrete `Fr`.
type F<E> = <E as Pairing>::ScalarField;

/// The affine representation of the G1 group of `E`.
/// Used in place of the concrete `G1Affine`.
type G1Aff<E> = <<E as Pairing>::G1 as CurveGroup>::Affine;

/// The KZG10 polynomial commitment scheme instantiated over `E`.
pub type KZGScheme<E> = KZG10<E, DensePolynomial<F<E>>>;

/// Universal parameters (trusted-setup output) for curve `E`.
pub type Params<E> = UniversalParams<E>;

/// A KZG polynomial commitment for curve `E`.
pub type Commitment<E> = ark_poly_commit::kzg10::Commitment<E>;

/// A KZG evaluation proof for curve `E`.
pub type Proof<E> = ark_poly_commit::kzg10::Proof<E>;

// ---------------------------------------------------------------------------
// Serializable proof structure
// ---------------------------------------------------------------------------

/// A range proof for a single value, generic over the pairing curve `E`.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound = "E: Pairing")]
pub struct RangeProof<E: Pairing> {
    #[serde(with = "crate::ark_serde")]
    pub fcom: Commitment<E>,
    #[serde(with = "crate::ark_serde")]
    pub gcom: Commitment<E>,
    #[serde(with = "crate::ark_serde")]
    pub qcom: Commitment<E>,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_proof: Proof<E>,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_eval: F<E>,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_omega_proof: Proof<E>,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_omega_eval: F<E>,
    #[serde(with = "crate::ark_serde")]
    pub w_caret_rho_proof: Proof<E>,
    #[serde(with = "crate::ark_serde")]
    pub w_caret_rho_eval: F<E>,
}

pub type RangeProofVec<E> = Vec<RangeProof<E>>;

// ---------------------------------------------------------------------------
// Challenge derivation
// ---------------------------------------------------------------------------

/// Generates a Fiat-Shamir challenge scalar from a slice of KZG commitments.
/// # Type parameters
/// * `E` - The pairing curve; determines both the commitment type and the
///   scalar field of the returned challenge.
pub fn get_challenge_from_coms<E: Pairing>(coms: &[Commitment<E>]) -> F<E> {
    let affines: Vec<G1Aff<E>> = coms.iter().map(|c| c.0).collect();
    get_challenge(&affines)
}

// ---------------------------------------------------------------------------
// Polynomial helpers
// ---------------------------------------------------------------------------

/// Builds the quotient polynomial q(x) = R(x) / (x^n − 1), where
///
///   R(x) = w1(x) + τ·w2(x) + τ²·w3(x)
///
/// # Arguments
/// * `tau`      - Fiat-Shamir scalar τ.
/// * `omega`    - Primitive n-th root of unity.
/// * `num_bits` - Domain size n.
/// * `f`        - Difference polynomial f(x).
/// * `g`        - Encoding polynomial g(x).
pub fn get_quotient_polynomial<E: Pairing>(
    tau: &F<E>,
    omega: &F<E>,
    num_bits: usize,
    f: &DensePolynomial<F<E>>,
    g: &DensePolynomial<F<E>>,
) -> anyhow::Result<DensePolynomial<F<E>>> {
    let w1 = get_polynomial_w1::<E>(num_bits, f, g)?;
    let w2 = get_polynomial_w2::<E>(omega, num_bits, g)?;
    let w3 = get_polynomial_w3::<E>(omega, num_bits, g);

    // R(x) = w1 + τ·w2 + τ²·w3
    let tau_w2 = scalar_mul(&w2, tau);
    let tau_sq = tau.pow([2u64]);
    let tau_sq_w3 = scalar_mul(&w3, &tau_sq);

    let tmp = &w1 + &tau_w2;
    let dividend = &tmp + &tau_sq_w3;

    // denominator: x^n − 1
    let mut denom_coeffs = vec![F::<E>::zero(); num_bits + 1];
    denom_coeffs[0] = -F::<E>::one();
    denom_coeffs[num_bits] = F::<E>::one();
    let denominator = DensePolynomial::from_coefficients_vec(denom_coeffs);

    let (q, _) = DenseOrSparsePolynomial::from(&dividend)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&denominator))
        .ok_or_else(|| anyhow!("division failed"))?;

    Ok(q)
}

/// Builds the auxiliary polynomial
///
///   w1(x) = (g(x) − f(x)) · (x^n − 1) / (x − 1)
pub fn get_polynomial_w1<E: Pairing>(
    num_bits: usize,
    f: &DensePolynomial<F<E>>,
    g: &DensePolynomial<F<E>>,
) -> anyhow::Result<DensePolynomial<F<E>>> {
    // numerator: x^n − 1
    let mut num_coeffs = vec![F::<E>::zero(); num_bits + 1];
    num_coeffs[0] = -F::<E>::one();
    num_coeffs[num_bits] = F::<E>::one();
    let numerator = DensePolynomial::from_coefficients_vec(num_coeffs);

    // denominator: x − 1
    let denominator =
        DensePolynomial::from_coefficients_slice(&[-F::<E>::one(), F::<E>::one()]);
    let (quot, _) = DenseOrSparsePolynomial::from(&numerator)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&denominator))
        .ok_or_else(|| anyhow!("division failed"))?;

    let g_minus_f = g - f;
    Ok(&g_minus_f * &quot)
}

/// Builds the auxiliary polynomial
///
///   w2(x) = g(x)·(1 − g(x)) · (x^n − 1) / (x − ω^(n−1))
pub fn get_polynomial_w2<E: Pairing>(
    omega: &F<E>,
    num_bits: usize,
    g: &DensePolynomial<F<E>>,
) -> anyhow::Result<DensePolynomial<F<E>>> {
    // numerator: x^n − 1
    let mut num_coeffs = vec![F::<E>::zero(); num_bits + 1];
    num_coeffs[0] = -F::<E>::one();
    num_coeffs[num_bits] = F::<E>::one();
    let numerator = DensePolynomial::from_coefficients_vec(num_coeffs);

    // denominator: x − ω^(n−1)
    let denom = DensePolynomial::from_coefficients_slice(&[
        -omega.pow([(num_bits - 1) as u64]),
        F::<E>::one(),
    ]);
    let (quot, _) = DenseOrSparsePolynomial::from(&numerator)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&denom))
        .ok_or_else(|| anyhow!("division failed"))?;

    // 1 − g(x)
    let one_poly = DensePolynomial::from_coefficients_slice(&[F::<E>::one()]);
    let one_minus_g = &one_poly - g;
    let g_times_one_minus_g = g * &one_minus_g;

    Ok(&g_times_one_minus_g * &quot)
}

/// Builds the auxiliary polynomial
///
///   w3(x) = (g(x) − 2·g(x·ω)) · (1 − g(x) + 2·g(x·ω)) · (x − ω^(n−1))
pub fn get_polynomial_w3<E: Pairing>(
    omega: &F<E>,
    num_bits: usize,
    g: &DensePolynomial<F<E>>,
) -> DensePolynomial<F<E>> {
    // g(x·ω): scale each coefficient c_k by ω^k
    let g_x_omega = DensePolynomial::from_coefficients_vec(
        g.coeffs()
            .iter()
            .enumerate()
            .map(|(k, c)| *c * omega.pow([k as u64]))
            .collect(),
    );
    let two_g_x_omega = scalar_mul(&g_x_omega, &F::<E>::from(2u64));

    let one_poly = DensePolynomial::from_coefficients_slice(&[F::<E>::one()]);
    let one_minus_g = &one_poly - g;

    // (x − ω^(n−1))
    let omega_factor = DensePolynomial::from_coefficients_slice(&[
        -omega.pow([(num_bits - 1) as u64]),
        F::<E>::one(),
    ]);

    // fac1 = g(x) − 2·g(x·ω)
    let fac1 = g - &two_g_x_omega;
    // fac2 = 1 − g(x) + 2·g(x·ω)
    let fac2 = &one_minus_g + &two_g_x_omega;

    &fac1 * &fac2 * &omega_factor
}

/// Computes the linearisation polynomial
///
///   ŵ(x) = f(x) · (ρ^n − 1)/(ρ − 1)  +  q(x) · (ρ^n − 1)
pub fn get_w_caret<E: Pairing>(
    rho: &F<E>,
    num_bits: usize,
    f: &DensePolynomial<F<E>>,
    q: &DensePolynomial<F<E>>,
) -> DensePolynomial<F<E>> {
    let rho_n_minus_1 = rho.pow([num_bits as u64]) - F::<E>::one();
    let quot = rho_n_minus_1 / (*rho - F::<E>::one());

    let term1 = scalar_mul(f, &quot);
    let term2 = scalar_mul(q, &rho_n_minus_1);

    &term1 + &term2
}

/// Computes the commitment to ŵ homomorphically from the commitments to
/// f and q, without requiring access to the polynomials themselves:
///
///   ŵ_com = f_com · (ρ^n−1)/(ρ−1)  +  q_com · (ρ^n−1)
///
/// Uses projective arithmetic on `E::G1` rather than any concrete G1 type,
/// resolving the first concrete-group blocker from the original code.
pub fn compute_w_caret_com<E: Pairing>(
    f_com: &Commitment<E>,
    rho: &F<E>,
    q_com: &Commitment<E>,
    num_bits: usize,
) -> anyhow::Result<Commitment<E>> {
    if *rho == F::<E>::one() {
        return Err(anyhow!(
            "rho must not equal 1 (division by zero in (ρ^n−1)/(ρ−1))"
        ));
    }

    let rho_n = rho.pow([num_bits as u64]);
    let rho_n_minus_1 = rho_n - F::<E>::one();
    let quot = rho_n_minus_1 / (*rho - F::<E>::one());

    // Convert affine commitments to the generic projective group E::G1.
    let f_g1: E::G1 = f_com.0.into_group();
    let q_g1: E::G1 = q_com.0.into_group();
    let result = f_g1 * quot + q_g1 * rho_n_minus_1;

    Ok(ark_poly_commit::kzg10::Commitment(result.into_affine()))
}

// ---------------------------------------------------------------------------
// KZG parameter helpers
// ---------------------------------------------------------------------------

/// Extracts a `Powers` struct from `UniversalParams<E>` for use in KZG
/// commitments and openings.
pub fn get_powers_from_params<E: Pairing>(pp: &Params<E>) -> Powers<'_, E> {
    let max_degree = pp.powers_of_g.len() - 1;

    let gamma_g_vec: Vec<G1Aff<E>> = (0..=max_degree)
        .map(|i| {
            pp.powers_of_gamma_g
                .get(&i)
                .cloned()
                .unwrap_or_else(G1Aff::<E>::zero)
        })
        .collect();

    Powers {
        powers_of_g: Cow::Borrowed(&pp.powers_of_g),
        powers_of_gamma_g: Cow::Owned(gamma_g_vec),
    }
}

/// Builds a `VerifierKey<E>` from `UniversalParams<E>`.
fn make_verifier_key<E: Pairing>(
    pp: &Params<E>,
) -> ark_poly_commit::kzg10::VerifierKey<E> {
    ark_poly_commit::kzg10::VerifierKey {
        g: pp.powers_of_g[0],
        gamma_g: pp
            .powers_of_gamma_g
            .get(&0)
            .cloned()
            .unwrap_or_else(G1Aff::<E>::zero),
        h: pp.h,
        beta_h: pp.beta_h,
        prepared_h: pp.prepared_h.clone(),
        prepared_beta_h: pp.prepared_beta_h.clone(),
    }
}

// ---------------------------------------------------------------------------
// Single-value proof construction
// ---------------------------------------------------------------------------

/// Generates a single range proof showing that `z` encodes a value that lies
/// in the expected range relative to the polynomial `f`.
///
/// # Type parameters
/// * `E` - The pairing curve used for the KZG commitment scheme.
///
/// # Arguments
/// * `pp`       - KZG universal parameters (trusted setup).
/// * `omega`    - Primitive n-th root of unity (ω = roots[1]).
/// * `z`        - Non-negative witness value (e.g. value − min).
/// * `f`        - Difference polynomial with constant term `z`.
/// * `num_bits` - Domain size; must be a power of two.
pub fn single_value_proof<E: Pairing>(
    pp: &Params<E>,
    omega: &F<E>,
    z: &F<E>,
    f: &DensePolynomial<F<E>>,
    num_bits: usize,
) -> anyhow::Result<RangeProof<E>> {
    let powers = get_powers_from_params(pp);
    let zero_rand = Randomness::<F<E>, DensePolynomial<F<E>>>::empty();

    let g = build_encoding_polynomial(z, num_bits);
    let (f_com, _) = KZGScheme::<E>::commit(&powers, f, None, None)
        .map_err(|e| anyhow!("commit f failed: {:?}", e))?;
    let (g_com, _) = KZGScheme::<E>::commit(&powers, &g, None, None)
        .map_err(|e| anyhow!("commit g failed: {:?}", e))?;

    let tau = get_challenge_from_coms(&[f_com, g_com]);
    if tau == F::<E>::zero() {
        return Err(anyhow!(
            "tau must not equal 0 (eliminates the contribution of w2 and w3 \
             in R(x) = w1 + τ·w2 + τ²·w3)"
        ));
    }

    let q = get_quotient_polynomial::<E>(&tau, omega, num_bits, f, &g)?;

    let (q_com, _) = KZGScheme::<E>::commit(&powers, &q, None, None)
        .map_err(|e| anyhow!("commit q failed: {:?}", e))?;

    let rho = get_challenge_from_coms(&[f_com, g_com, q_com]);
    if rho == F::<E>::one() {
        return Err(anyhow!(
            "rho must not equal 1 (division by zero in (ρ^n−1)/(ρ−1))"
        ));
    }

    let w_caret = get_w_caret::<E>(&rho, num_bits, f, &q);

    let g_rho_proof = KZGScheme::<E>::open(&powers, &g, rho, &zero_rand)
        .map_err(|e| anyhow!("open g at rho failed: {:?}", e))?;
    let g_rho_omega_proof = KZGScheme::<E>::open(&powers, &g, rho * omega, &zero_rand)
        .map_err(|e| anyhow!("open g at rho*omega failed: {:?}", e))?;
    let w_caret_rho_proof = KZGScheme::<E>::open(&powers, &w_caret, rho, &zero_rand)
        .map_err(|e| anyhow!("open w_caret at rho failed: {:?}", e))?;

    let g_rho_eval = g.evaluate(&rho);
    let g_rho_omega_eval = g.evaluate(&(rho * omega));
    let w_caret_rho_eval = w_caret.evaluate(&rho);

    Ok(RangeProof {
        fcom: f_com,
        gcom: g_com,
        qcom: q_com,
        g_rho_proof,
        g_rho_eval,
        g_rho_omega_proof,
        g_rho_omega_eval,
        w_caret_rho_proof,
        w_caret_rho_eval,
    })
}

// ---------------------------------------------------------------------------
// Batch proof construction
// ---------------------------------------------------------------------------

/// Produces a range proof for each value in `values`, demonstrating that
/// `values[i] >= min`.
///
/// # Type parameters
/// * `E` - The pairing curve.
pub fn prove_min<E: Pairing>(
    pp: &Params<E>,
    min: &F<E>,
    values_poly: &[DensePolynomial<F<E>>],
    values: &[F<E>],
    num_bits: usize,
) -> anyhow::Result<RangeProofVec<E>> {
    let roots: Vec<F<E>> = generate_nth_roots_of_unity(num_bits);
    let min_poly = get_const_polynomial(min, num_bits);

    values
        .par_iter()
        .enumerate()
        .map(|(i, val)| {
            let z = *val - *min;
            let f = &values_poly[i] - &min_poly;
            single_value_proof(pp, &roots[1], &z, &f, num_bits)
                .map_err(|e| anyhow!("prove_min[{i}]: {e}"))
        })
        .collect()
}

/// Produces a range proof for each value in `values`, demonstrating that
/// `values[i] <= max`.
///
/// # Type parameters
/// * `E` - The pairing curve.
pub fn prove_max<E: Pairing>(
    pp: &Params<E>,
    max: &F<E>,
    values_poly: &[DensePolynomial<F<E>>],
    values: &[F<E>],
    num_bits: usize,
) -> anyhow::Result<RangeProofVec<E>> {
    let roots: Vec<F<E>> = generate_nth_roots_of_unity(num_bits);
    let max_poly = get_const_polynomial(max, num_bits);

    values
        .par_iter()
        .enumerate()
        .map(|(i, val)| {
            let z = *max - *val;
            let f = &max_poly - &values_poly[i];
            single_value_proof(pp, &roots[1], &z, &f, num_bits)
                .map_err(|e| anyhow!("prove_max[{i}]: {e}"))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Evaluates the combined constraint polynomial w at ρ:
///
///   w(ρ) = g(ρ)·(ρ^n−1)/(ρ−1)  +  τ·w2(ρ)  +  τ²·w3(ρ)  −  ŵ(ρ)
///
/// Returns `F::<E>::zero()` iff all three sub-constraints hold.
#[must_use]
pub fn eval_w<E: Pairing>(
    proof: &RangeProof<E>,
    rho: &F<E>,
    tau: &F<E>,
    omega_n_minus_1: &F<E>,
    num_bits: usize,
) -> F<E> {
    let g_rho = proof.g_rho_eval;
    let g_rho_omega = proof.g_rho_omega_eval;
    let w_rho = proof.w_caret_rho_eval;

    let rho_n_minus_1 = rho.pow([num_bits as u64]) - F::<E>::one();
    let rho_minus_omega_n = *rho - *omega_n_minus_1;

    // w1(ρ) = g(ρ) · (ρ^n − 1) / (ρ − 1)
    let w1 = g_rho * rho_n_minus_1 / (*rho - F::<E>::one());

    // τ · w2(ρ) = τ · g(ρ) · (1 − g(ρ)) · (ρ^n − 1) / (ρ − ω^(n−1))
    let tau_w2 =
        *tau * g_rho * (F::<E>::one() - g_rho) * rho_n_minus_1 / rho_minus_omega_n;

    // τ² · w3(ρ) = τ² · (g(ρ) − 2g(ρω)) · (1 − g(ρ) + 2g(ρω)) · (ρ − ω^(n−1))
    let two = F::<E>::from(2u64);
    let tau_sq_w3 = tau.pow([2u64])
        * (g_rho - g_rho_omega * two)
        * (F::<E>::one() - g_rho + g_rho_omega * two)
        * rho_minus_omega_n;

    w1 + tau_w2 + tau_sq_w3 - w_rho
}

/// Verifies a single range proof.
///
/// Returns `Ok(())` iff all checks pass, or `Err(reason)` naming the first
/// failing check.  Possible reason strings:
/// `"fcom"`, `"g_rho_proof"`, `"g_rho_omega_proof"`,
/// `"w_caret_com"`, `"w_caret_rho_proof"`, `"eval_w"`.
#[must_use = "ignoring the verification result defeats the purpose of the check"]
pub fn single_proof_verify<E: Pairing>(
    vk: &ark_poly_commit::kzg10::VerifierKey<E>,
    f_com: &Commitment<E>,
    proof: &RangeProof<E>,
    num_bits: usize,
    omega: &F<E>,
    omega_n_minus_1: &F<E>,
) -> Result<(), &'static str> {
    if f_com != &proof.fcom {
        return Err("fcom");
    }

    let rho = get_challenge_from_coms(&[*f_com, proof.gcom, proof.qcom]);

    if !matches!(
        KZGScheme::<E>::check(vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof),
        Ok(true)
    ) {
        return Err("g_rho_proof");
    }

    if !matches!(
        KZGScheme::<E>::check(
            vk,
            &proof.gcom,
            rho * omega,
            proof.g_rho_omega_eval,
            &proof.g_rho_omega_proof,
        ),
        Ok(true)
    ) {
        return Err("g_rho_omega_proof");
    }

    let w_caret_com = match compute_w_caret_com(f_com, &rho, &proof.qcom, num_bits) {
        Ok(c) => c,
        Err(_) => return Err("w_caret_com"),
    };

    if !matches!(
        KZGScheme::<E>::check(
            vk,
            &w_caret_com,
            rho,
            proof.w_caret_rho_eval,
            &proof.w_caret_rho_proof,
        ),
        Ok(true)
    ) {
        return Err("w_caret_rho_proof");
    }

    let tau = get_challenge_from_coms(&[*f_com, proof.gcom]);
    if eval_w(proof, &rho, &tau, omega_n_minus_1, num_bits) != F::<E>::zero() {
        return Err("eval_w");
    }

    Ok(())
}

/// Verifies range proofs for `values_com[i] >= min`.
#[must_use = "ignoring the verification result defeats the purpose of the check"]
pub fn min_verify<E: Pairing>(
    pp: &Params<E>,
    min: &F<E>,
    values_com: &[Commitment<E>],
    proofs: &[RangeProof<E>],
    num_bits: usize,
) -> anyhow::Result<bool> {
    let roots: Vec<F<E>> = generate_nth_roots_of_unity(num_bits);
    let omega = roots[1];
    let omega_n_minus_1 = roots[num_bits - 1];
    let powers = get_powers_from_params(pp);
    let vk = make_verifier_key(pp);

    let (min_com, _) =
        KZGScheme::<E>::commit(&powers, &get_const_polynomial(min, num_bits), None, None)
            .map_err(|e| anyhow!("min_verify: commit failed: {:?}", e))?;

    Ok(values_com.par_iter().enumerate().all(|(i, val)| {
        // Subtract min commitment from value commitment in projective space,
        // then convert back to affine — avoids relying on a concrete Sub impl
        // for affine points.
        let f_proj: E::G1 = val.0.into_group() - min_com.0.into_group();
        let f_com = ark_poly_commit::kzg10::Commitment(f_proj.into_affine());

        single_proof_verify(
            &vk,
            &f_com,
            &proofs[i],
            num_bits,
            &omega,
            &omega_n_minus_1,
        ).is_ok()
    }))
}

/// Verifies range proofs for `values_com[i] <= max`.
#[must_use = "ignoring the verification result defeats the purpose of the check"]
pub fn max_verify<E: Pairing>(
    pp: &Params<E>,
    max: &F<E>,
    values_com: &[Commitment<E>],
    proofs: &[RangeProof<E>],
    num_bits: usize,
) -> anyhow::Result<bool> {
    let roots: Vec<F<E>> = generate_nth_roots_of_unity(num_bits);
    let omega = roots[1];
    let omega_n_minus_1 = roots[num_bits - 1];
    let powers = get_powers_from_params(pp);
    let vk = make_verifier_key(pp);

    let (max_com, _) =
        KZGScheme::<E>::commit(&powers, &get_const_polynomial(max, num_bits), None, None)
            .map_err(|e| anyhow!("max_verify: commit failed: {:?}", e))?;

    Ok(values_com.par_iter().enumerate().all(|(i, val)| {
        let f_proj: E::G1 = max_com.0.into_group() - val.0.into_group();
        let f_com = ark_poly_commit::kzg10::Commitment(f_proj.into_affine());

        single_proof_verify(
            &vk,
            &f_com,
            &proofs[i],
            num_bits,
            &omega,
            &omega_n_minus_1,
        ).is_ok()
    }))
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

/// Returns the in-memory byte size of a `RangeProof<E>`.
///
/// Was `const fn` in the original (BLS12-381 only); now a regular generic
/// function since sizes vary by curve.
pub fn size_of_range_proof<E: Pairing>() -> usize {
    3 * std::mem::size_of::<Commitment<E>>()
        + 3 * std::mem::size_of::<Proof<E>>()
        + 3 * std::mem::size_of::<F<E>>()
}
