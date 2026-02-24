use super::utilities::*;
use anyhow::anyhow;
use ark_bls12_381::{Bls12_381, Fr, G1Affine, G1Projective};
use ark_ec::CurveGroup;
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
// Type aliases for clarity
// ---------------------------------------------------------------------------
pub type KZG = KZG10<Bls12_381, DensePolynomial<Fr>>;
pub type Params = UniversalParams<Bls12_381>;
pub type Commitment = ark_poly_commit::kzg10::Commitment<Bls12_381>;
pub type Proof = ark_poly_commit::kzg10::Proof<Bls12_381>;

// ---------------------------------------------------------------------------
// Serializable wrapper for our proof structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct RangeProof {
    #[serde(with = "crate::ark_serde")]
    pub fcom: Commitment,
    #[serde(with = "crate::ark_serde")]
    pub gcom: Commitment,
    #[serde(with = "crate::ark_serde")]
    pub qcom: Commitment,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_proof: Proof,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_eval: Fr,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_omega_proof: Proof,
    #[serde(with = "crate::ark_serde")]
    pub g_rho_omega_eval: Fr,
    #[serde(with = "crate::ark_serde")]
    pub w_caret_rho_proof: Proof,
    #[serde(with = "crate::ark_serde")]
    pub w_caret_rho_eval: Fr,
}

pub type RangeProofVec = Vec<RangeProof>;

/// Generates a challenge scalar from commitments
///
/// # Arguments
/// * coms - a vector of commitments.
///
/// # Returns
/// A field element that depends on coms.
pub fn get_challenge_from_coms(coms: &[Commitment]) -> Fr {
    let g1s: Vec<G1Projective> = coms.iter().map(|c| c.0.into()).collect();
    get_challenge(&g1s)
}

// ---------------------------------------------------------------------------
// Polynomial helpers for the range proof
// ---------------------------------------------------------------------------

/// Builds the quotient polynomial q(x), which is computed as
/// the quotient of a dividend polynomial R(x) divided by the
/// polynomial (x^n - 1).
///
///    q(x) = R(x) / (x^n − 1)
///
/// The dividend polynomial is constructed using auxiliary
/// polynomials w1(x), w2(x) and w3(x) which depend on the
/// input parameters tau, omega, n, f(x), and g(x).
///
///      R(x) = w1(x) + τ·w2(x) + τ²·w3(x)
///
/// # Arguments
/// * tau   - A field element used in the construction of the
///   dividend polynomial.
/// * omega - A primitive n-th root of unity.
/// * num_bits - The evaluation domain size: the number of binary digits used to
///   represent the secret value, and the number of roots of unity at which
///   the range constraints are checked.
/// * f     - A DensePolynomial<Fr> representing the polynomial f(x).
/// * g     - A DensePolynomial<Fr> representing the polynomial g(x).
///
/// # Returns
/// A DensePolynomial<Fr> representing the quotient polynomial q(x).
pub fn get_quotient_polynomial(
    tau: &Fr,
    omega: &Fr,
    num_bits: usize,
    f: &DensePolynomial<Fr>,
    g: &DensePolynomial<Fr>,
) -> anyhow::Result<DensePolynomial<Fr>> {
    let w1 = get_polynomial_w1(num_bits, f, g)?;
    let w2 = get_polynomial_w2(omega, num_bits, g)?;
    let w3 = get_polynomial_w3(omega, num_bits, g);

    // R(x) = w1 + τ·w2 + τ²·w3
    let tau_w2 = scalar_mul(&w2, tau);
    let tau_sq = tau.pow([2u64]);
    let tau_sq_w3 = scalar_mul(&w3, &tau_sq);

    let tmp = &w1 + &tau_w2;
    let dividend = &tmp + &tau_sq_w3;

    // denominator: x^n − 1
    let mut denom_coeffs = vec![Fr::zero(); num_bits + 1];
    denom_coeffs[0] = -Fr::one();
    denom_coeffs[num_bits] = Fr::one();
    let denominator = DensePolynomial::from_coefficients_vec(denom_coeffs);
    let (q, _) = DenseOrSparsePolynomial::from(&dividend)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&denominator))
        .ok_or_else(|| anyhow!("division failed"))?;

    Ok(q)
}

/// Builds the auxiliary polynomial w1(x).
///
/// w1(x) = (g(x) − f(x)) · (x^n − 1) / (x − 1)
///
/// # Arguments
/// * num_bits - The evaluation domain size: the number of binary digits used to
///   represent the secret value, and the number of roots of unity at which
///   the range constraints are checked.
/// * f - The polynomial f(x) as a DensePolynomial<Fr>.
/// * g - The encoding polynomial g(x) as a DensePolynomial<Fr>.
///
/// # Returns
/// A DensePolynomial<Fr> representing w1(x).
pub fn get_polynomial_w1(
    num_bits: usize,
    f: &DensePolynomial<Fr>,
    g: &DensePolynomial<Fr>,
) -> anyhow::Result<DensePolynomial<Fr>> {
    // (x^n − 1) / (x − 1)
    let mut num_coeffs = vec![Fr::zero(); num_bits + 1];
    num_coeffs[0] = -Fr::one();
    num_coeffs[num_bits] = Fr::one();
    let numerator = DensePolynomial::from_coefficients_vec(num_coeffs);
    let denominator = DensePolynomial::from_coefficients_slice(&[-Fr::one(), Fr::one()]);
    let (quot, _) = DenseOrSparsePolynomial::from(&numerator)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&denominator))
        .ok_or_else(|| anyhow!("division failed"))?;

    let g_minus_f = g - f;

    Ok(&g_minus_f * &quot)
}

/// Builds the auxiliary polynomial w2(x).
///
/// w2(x) = g(x)·(1 − g(x)) · (x^n − 1) / (x − ω^(n−1))
///
/// # Arguments
/// * omega - A primitive n-th root of unity.
/// * num_bits - The evaluation domain size: the number of binary digits used to
///   represent the secret value, and the number of roots of unity at which
///   the range constraints are checked.
/// * g     - A DensePolynomial<Fr> representing the polynomial g(x).
///
/// # Returns
///   A DensePolynomial<Fr> representing w2(x).
pub fn get_polynomial_w2(
    omega: &Fr,
    num_bits: usize,
    g: &DensePolynomial<Fr>,
) -> anyhow::Result<DensePolynomial<Fr>> {
    // (x^n − 1)
    let mut num_coeffs = vec![Fr::zero(); num_bits + 1];
    num_coeffs[0] = -Fr::one();
    num_coeffs[num_bits] = Fr::one();
    let numerator = DensePolynomial::from_coefficients_vec(num_coeffs);

    // (x − ω^(n−1))
    let denom =
        DensePolynomial::from_coefficients_slice(&[-omega.pow([(num_bits - 1) as u64]), Fr::one()]);
    let (quot, _) = DenseOrSparsePolynomial::from(&numerator)
        .divide_with_q_and_r(&DenseOrSparsePolynomial::from(&denom))
        .ok_or_else(|| anyhow!("division failed"))?;

    // 1 − g(x)
    let one_poly = DensePolynomial::from_coefficients_slice(&[Fr::one()]);
    let one_minus_g = &one_poly - g;

    let g_times_one_minus_g = g * &one_minus_g;

    Ok(&g_times_one_minus_g * &quot)
}

/// Builds the auxiliary polynomial w3(x).
///
/// w3(x) = (g(x) − 2·g(x·ω)) · (1 − g(x) + 2·g(x·ω)) · (x − ω^(n−1))
///
/// # Arguments
/// * omega - A primitive n-th root of unity.
/// * num_bits - The evaluation domain size: the number of binary digits used to
///   represent the secret value, and the number of roots of unity at which
///   the range constraints are checked.
/// * g     - A DensePolynomial<Fr> representing the polynomial g(x).
///
/// # Returns
/// A DensePolynomial<Fr> representing w3(x).
pub fn get_polynomial_w3(
    omega: &Fr,
    num_bits: usize,
    g: &DensePolynomial<Fr>,
) -> DensePolynomial<Fr> {
    // g(x·ω): multiply each coefficient c_k by ω^k
    let g_x_omega = DensePolynomial::from_coefficients_vec(
        g.coeffs()
            .iter()
            .enumerate()
            .map(|(k, c)| *c * omega.pow([k as u64]))
            .collect(),
    );
    let two_g_x_omega = scalar_mul(&g_x_omega, &Fr::from(2u64));

    let one_poly = DensePolynomial::from_coefficients_slice(&[Fr::one()]);
    let one_minus_g = &one_poly - g;

    // (x − ω^(n−1))
    let omega_factor =
        DensePolynomial::from_coefficients_slice(&[-omega.pow([(num_bits - 1) as u64]), Fr::one()]);

    // fac1 = g(x) − 2·g(x·ω)
    let fac1 = g - &two_g_x_omega;
    // fac2 = 1 − g(x) + 2·g(x·ω)
    let fac2 = &one_minus_g + &two_g_x_omega;

    &fac1 * &fac2 * &omega_factor
}

/// Computes the linearisation polynomial ŵ(x) from the difference polynomial
/// f(x) and the quotient polynomial q(x), at the Fiat-Shamir point ρ:
///
///    ŵ(x) = f(x) · (ρ^n − 1)/(ρ − 1)  +  q(x) · (ρ^n − 1)
///
/// The verifier can reconstruct the commitment to ŵ homomorphically from the
/// commitments to f and q, without knowing the polynomials themselves.
///
/// # Arguments
/// * rho - The Fiat-Shamir evaluation point ρ, derived from the commitments
///   to f, g, and q.  Must not equal 1 (would cause division by zero).
/// * num_bits - The evaluation domain size: the number of binary digits used to
///   represent the secret value, and the number of roots of unity at which
///   the range constraints are checked.
/// * f   - The DensePolynomial<Fr> representing the polynomial f(x).
/// * q   - The DensePolynomial<Fr> representing q(x).
///
/// # Returns
/// A vector of Fr representing the final polynomial w_caret(x).
pub fn get_w_caret(
    rho: &Fr,
    num_bits: usize,
    f: &DensePolynomial<Fr>,
    q: &DensePolynomial<Fr>,
) -> DensePolynomial<Fr> {
    let rho_n_minus_1 = rho.pow([num_bits as u64]) - Fr::one();
    let quot = rho_n_minus_1 / (rho - &Fr::one());

    let term1 = scalar_mul(f, &quot);
    let term2 = scalar_mul(q, &rho_n_minus_1);

    &term1 + &term2
}

/// Computes the commitment to w_caret directly from the commitments
/// to f and q (avoids re-evaluating the polynomial).
///
/// w_caret_com = f_com · (ρ^n−1)/(ρ−1) + q_com · (ρ^n−1)
///
/// # Arguments
/// * f_com - The commitment to the polynomial f(x).
/// * rho - A field element.
/// * q_com - The commitment to the polynomial q(x).
/// * num_bits - The degree (or order) of the polynomials involved.
///
/// # Returns
/// A Commitment to the linearisation polynomial ŵ(x), computed from the
/// commitments to f and q via the KZG homomorphic property.
pub fn compute_w_caret_com(
    f_com: &Commitment,
    rho: &Fr,
    q_com: &Commitment,
    num_bits: usize,
) -> anyhow::Result<Commitment> {
    if *rho == Fr::one() {
        return Err(anyhow!(
            "rho must not equal 1 (division by zero in (ρ^n−1)/(ρ−1))"
        ));
    }

    let rho_n = rho.pow([num_bits as u64]);
    let rho_n_minus_1 = rho_n - Fr::one();
    let quot = rho_n_minus_1 / (rho - &Fr::one());

    let f_g1: G1Projective = f_com.0.into();
    let q_g1: G1Projective = q_com.0.into();
    let result = f_g1 * quot + q_g1 * rho_n_minus_1;

    Ok(ark_poly_commit::kzg10::Commitment(result.into_affine()))
}

/// Extracts Powers from UniversalParams for use in KZG commitments and openings.
///
/// # Arguments
/// * pp   - The universal parameters
///
/// # Returns
/// A Powers struct containing borrowed powers_of_g and owned powers_of_gamma_g
/// converted from the BTreeMap in the universal parameters.
pub fn get_powers_from_params(pp: &Params) -> Powers<'_, Bls12_381> {
    let max_degree = pp.powers_of_g.len() - 1;

    // Create a vector for powers_of_gamma_g in the correct order
    let gamma_g_vec: Vec<G1Affine> = (0..=max_degree)
        .map(|i| {
            pp.powers_of_gamma_g
                .get(&i)
                .cloned()
                .unwrap_or_else(G1Affine::identity)
        })
        .collect();

    Powers {
        powers_of_g: Cow::Borrowed(&pp.powers_of_g),
        powers_of_gamma_g: Cow::Owned(gamma_g_vec),
    }
}

// ---------------------------------------------------------------------------
// Proof construction
// ---------------------------------------------------------------------------
fn make_verifier_key(pp: &Params) -> ark_poly_commit::kzg10::VerifierKey<Bls12_381> {
    ark_poly_commit::kzg10::VerifierKey {
        g: pp.powers_of_g[0],
        gamma_g: pp
            .powers_of_gamma_g
            .get(&0)
            .cloned()
            .unwrap_or_else(G1Affine::identity),
        h: pp.h,
        beta_h: pp.beta_h,
        prepared_h: pp.prepared_h.clone(),
        prepared_beta_h: pp.prepared_beta_h.clone(),
    }
}

/// Generates a single range proof showing that z encodes a value
/// that lies in the expected range relative to the given polynomial f.
///
/// # Arguments
/// * pp    - The KZG universal parameters (trusted setup).
/// * omega - A primitive n-th root of unity (ω = roots[1] for the
///   n-element domain).
/// * z     - The non-negative witness value (e.g. value − min).
/// * f     - The difference polynomial with constant term z
///   (e.g. values_poly[i] − min_poly).
/// * num_bits - The evaluation domain size; must be a power of two and
///   large enough to represent z in binary.
///
/// Returns:
/// A RangeProof struct containing the following components:
///   * fcom: The commitment to the input data vector f.
///   * gcom: The commitment to the encoding polynomial g, which
///     encodes the value z within the range.
///   * qcom: The commitment to the quotient polynomial q, which
///     is used to prove that the encoded value lies within
///     the range.
///   * g_rho_proof: A KZG proof for the polynomial g evaluated
///     at the point rho, which is used to prove part of
///     the range proof.
///   * g_rho_omega_proof: A KZG proof for the polynomial g evaluated
///     at the point rho*omega, providing further validation of
///     the proof.
///   * w_caret_rho_proof: A KZG proof for the polynomial w_caret,
///     which is another commitment involved in ensuring
///     the correctness of the range proof.
///
/// Workflow:
/// 1. Encoding Polynomial (g): The function builds an encoding
///    polynomial g that represents the value z and is used to
///    encode the range constraint.
/// 2. Commitments: The function computes commitments to the polynomials
///    f, and g using the KZG commitment scheme.
/// 3. Quotient Polynomial (q): The quotient polynomial q is computed,
///    which plays a key role in proving the value lies within the
///    range without revealing it.
/// 4. Commitments: Computes the commitment to the polynomial
///    q using the KZG commitment scheme.
/// 5. Opening the Polynomials: The function generates KZG proofs for
///    the evaluated polynomials (g and w_caret) at specific points
///    (rho and rho*omega) to ensure the correctness of the proof.
/// 6. Returning the Range Proof: Finally, the function returns a
///    RangeProof struct containing all the commitments and proofs
///    necessary for verification.
pub fn single_value_proof(
    pp: &Params,
    omega: &Fr,
    z: &Fr,
    f: &DensePolynomial<Fr>,
    num_bits: usize,
) -> anyhow::Result<RangeProof> {
    let powers = get_powers_from_params(pp);
    // Create empty randomness for non-hiding proofs using the trait method
    let zero_rand = Randomness::<Fr, DensePolynomial<Fr>>::empty();

    let g = build_encoding_polynomial(z, num_bits);
    let (f_com, _) =
        KZG::commit(&powers, f, None, None).map_err(|e| anyhow!("commit f failed: {:?}", e))?;
    let (g_com, _) =
        KZG::commit(&powers, &g, None, None).map_err(|e| anyhow!("commit g failed: {:?}", e))?;

    let tau = get_challenge_from_coms(&[f_com, g_com]);
    if tau == Fr::zero() {
        return Err(anyhow!("tau must not equal 0 (eliminate the contibution of w2 and w3 in R(x) = w1 + τ·w2 + τ²·w3)"));
    }

    let q = get_quotient_polynomial(&tau, omega, num_bits, f, &g)?;

    let (q_com, _) =
        KZG::commit(&powers, &q, None, None).map_err(|e| anyhow!("commit q failed: {:?}", e))?;

    let rho = get_challenge_from_coms(&[f_com, g_com, q_com]);
    if rho == Fr::one() {
        return Err(anyhow!(
            "tau must not equal 1 (division by zero in (ρ^n−1)/(ρ−1))"
        ));
    }

    let w_caret = get_w_caret(&rho, num_bits, f, &q);

    let g_rho_proof = KZG::open(&powers, &g, rho, &zero_rand)
        .map_err(|e| anyhow!("open g at rho failed: {:?}", e))?;
    let g_rho_omega_proof = KZG::open(&powers, &g, rho * omega, &zero_rand)
        .map_err(|e| anyhow!("open g at rho*omega failed: {:?}", e))?;
    let w_caret_rho_proof = KZG::open(&powers, &w_caret, rho, &zero_rand)
        .map_err(|e| anyhow!("open w_caret at rho failed: {:?}", e))?;

    // Compute evaluation values using the evaluate method from DenseUVPolynomial trait
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

/// Takes a minimum value min and a set of values, and produces a
/// set of range proofs for each value.
///
/// Each proof demonstrates that the corresponding value in values is
/// greater than or equal to the minimum value min.
///
/// The function constructs a polynomial representing the difference
/// between each value and the minimum, and then generates the
/// corresponding proof.
///
/// # Arguments
/// * pp    ` - The KZG universal parameters (trusted setup).
/// * min     - The minimum value that all values should be greater than or
///   equal to.
/// * values_poly - A vector of DensePolynomial<Fr>, where each polynomial
///   corresponds to a value. These polynomials represent the
///   secret values to be proven in the range proof.
/// * values - A vector of field elements representing the values
///   whose range proofs we want to generate.
/// * num_bits - The degree of the polynomial used in the range proof.
///
/// # Returns
/// A vector of RangeProof structs, each corresponding to a proof
/// for the values in the values vector.
pub fn prove_min(
    pp: &Params,
    min: &Fr,
    values_poly: &[DensePolynomial<Fr>],
    values: &[Fr],
    num_bits: usize,
) -> anyhow::Result<RangeProofVec> {
    let roots = generate_nth_roots_of_unity(num_bits);
    let min_poly = get_const_polynomial(min, num_bits);

    values
        .par_iter()
        .enumerate()
        .map(|(i, val)| {
            let z = val - min;
            let f = &values_poly[i] - &min_poly;
            single_value_proof(pp, &roots[1], &z, &f, num_bits)
                .map_err(|e| anyhow!("prove_min[{i}]: {e}"))
        })
        .collect()
}

/// Takes a maximum value max and a set of values, and produces a
/// set of range proofs for each value.
///
/// Each proof demonstrates that the corresponding value in values is less than or equal to
/// the maximum value max.
///
/// The function constructs a polynomial representing the difference
/// between the maximum and each value, and then generates the
/// corresponding proof.
///
/// # Arguments
/// * pp     - The universal parameters
/// * max    - The maximum value that all values should be less than or
///   equal to.
/// * values_poly - A vector of DensePolynomial<Fr>, where each polynomial
///   corresponds to a value. These polynomials represent the
///   secret values to be proven in the range proof.
/// * values - A vector of field elements representing the values
///   whose range proofs we want to generate.
/// * num_bits - The degree of the polynomial used in the range proof.
///
/// # Returns
/// A vector of RangeProof structs, each corresponding to a proof
/// for the values in the values vector.
pub fn prove_max(
    pp: &Params,
    max: &Fr,
    values_poly: &[DensePolynomial<Fr>],
    values: &[Fr],
    num_bits: usize,
) -> anyhow::Result<RangeProofVec> {
    let roots = generate_nth_roots_of_unity(num_bits);
    let max_poly = get_const_polynomial(max, num_bits);

    values
        .par_iter()
        .enumerate()
        .map(|(i, val)| {
            let z = max - val;
            let f = &max_poly - &values_poly[i];
            single_value_proof(pp, &roots[1], &z, &f, num_bits)
                .map_err(|e| anyhow!("prove_max[{i}]: {e}"))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Verify a single range proof for a given value.
///
/// # Arguments
/// * vk - The verifier key used for KZG verification.
/// * f_com - The commitment for the value in question (a KZG
///   commitment to the polynomial representing the value).
/// * proof - The range proof containing the necessary components to
///   prove that the value  committed in f_com is in the range.
/// * num_bits - The degree of the polynomial used in the range proof.
/// * omega - A field element used for evaluations in the proof.
/// * omega_n_minus_1 - the value omega^(num_bits-1)
/// * index - The index of the value in the array
///
/// # Returns
/// true if all four checks pass (commitment binding, g(ρ) opening,
/// g(ρω) opening, ŵ(ρ) opening, and the combined constraint w(ρ) = 0);
/// false as soon as any check fails. The failing step is reported
///  with an appropriate message.
#[must_use = "ignoring the verification result defeats the purpose of the check"]
pub fn single_proof_verify(
    vk: &ark_poly_commit::kzg10::VerifierKey<Bls12_381>,
    f_com: &Commitment,
    proof: &RangeProof,
    num_bits: usize,
    omega: &Fr,
    omega_n_minus_1: &Fr,
    index: usize,
) -> bool {
    if f_com != &proof.fcom {
        println!("{index}: Verification failed for proof.fcom");
        return false;
    }

    let rho = get_challenge_from_coms(&[*f_com, proof.gcom, proof.qcom]);

    // Verify g(ρ)
    if KZG::check(vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof).is_err() {
        println!("{index}: Verification failed for g_rho_proof");
        return false;
    }

    // Verify g(ρω)
    if KZG::check(
        vk,
        &proof.gcom,
        rho * omega,
        proof.g_rho_omega_eval,
        &proof.g_rho_omega_proof,
    )
    .is_err()
    {
        println!("{index}: Verification failed for g_rho_omega_proof");
        return false;
    }

    let w_caret_com = match compute_w_caret_com(f_com, &rho, &proof.qcom, num_bits) {
        Ok(c) => c,
        Err(e) => {
            println!("{index}: Verification failed for w_caret_com: {e}");
            return false;
        }
    };

    if KZG::check(
        vk,
        &w_caret_com,
        rho,
        proof.w_caret_rho_eval,
        &proof.w_caret_rho_proof,
    )
    .is_err()
    {
        println!("{index}: Verification failed for w_caret_rho_proof");
        return false;
    }

    let tau = get_challenge_from_coms(&[*f_com, proof.gcom]);
    if eval_w(proof, &rho, &tau, omega_n_minus_1, num_bits) != Fr::zero() {
        println!("{index}: Verification failed for w");
        return false;
    }

    true
}

/// Evaluates the combined constraint polynomial w at ρ to check
/// that all three sub-constraints hold simultaneously.
///
///     w(ρ) = g(ρ) · (ρ^n − 1)/(ρ − 1)  +  τ · w2(ρ)  +  τ² · w3(ρ)  −  ŵ(ρ)
///
///     w2(ρ) = g(ρ) · (1 − g(ρ)) · (ρ^n − 1)/[ρ − ω^(n − 1)]
///
///     w3(ρ) = (g(ρ) − 2 · g(ρ · ω)) · (1 − g(ρ) + 2 · g(ρ · ω)) · [ρ − ω^(n − 1)]
///
/// # Parameters
/// * proof: A reference to the range proof structure.
/// * rho: A field element representing ρ in the proof context.
/// * tau: A field element representing τ in the proof context.
/// * omega_n_minus_1: A field element representing omega^(n-1).
/// * num_bits: The size or degree used for the range proof.
///
/// # Returns
/// * Fr: The calculated field element representing the evaluation of w.
#[must_use]
pub fn eval_w(proof: &RangeProof, rho: &Fr, tau: &Fr, omega_n_minus_1: &Fr, num_bits: usize) -> Fr {
    let g_rho = proof.g_rho_eval;
    let g_rho_omega = proof.g_rho_omega_eval;
    let w_rho = proof.w_caret_rho_eval;

    // Evaluate (ρ^n − 1)
    let rho_n_minus_1 = rho.pow([num_bits as u64]) - Fr::one();
    // Evaluate ρ − ω^(n − 1)
    let rho_minus_omega_n = rho - omega_n_minus_1;

    // w1(ρ) = g(ρ) · (ρ^n − 1) / (ρ − 1)
    let w1 = g_rho * rho_n_minus_1 / (rho - &Fr::one());

    // τ · w2(ρ) = τ · g(ρ) · (1 − g(ρ)) · (ρ^n − 1) / (ρ − ω^(n−1))
    let tau_w2 = tau * &g_rho * (Fr::one() - g_rho) * rho_n_minus_1 / rho_minus_omega_n;

    // τ² · w3(ρ) = τ² · (g(ρ) − 2g(ρω)) · (1 − g(ρ) + 2g(ρω)) · (ρ − ω^(n−1))
    let tau_sq_w3 = tau.pow([2u64])
        * (g_rho - g_rho_omega * Fr::from(2u64))
        * (Fr::one() - g_rho + g_rho_omega * Fr::from(2u64))
        * rho_minus_omega_n;

    w1 + tau_w2 + tau_sq_w3 - w_rho
}

/// Verifies range proofs for `values_com[i] >= min`.
///
/// # Parameters
/// * pp: Params - The universal parameters
/// * min: Fr - The minimum threshold value that each element in
///   values_com must be greater than or equal to.
/// * values_com: A slice of KZG commitments to the secret values.
/// * proofs: A slice of RangeProofs, one per commitment.
/// * num_bits: usize - The evaluation domain size.
///
/// # Returns
/// * bool: true if all verifications succeed, false otherwise.
#[must_use = "ignoring the verification result defeats the purpose of the check"]
pub fn min_verify(
    pp: &Params,
    min: &Fr,
    values_com: &[Commitment],
    proofs: &[RangeProof],
    num_bits: usize,
) -> anyhow::Result<bool> {
    let roots = generate_nth_roots_of_unity(num_bits);
    let omega = roots[1];
    let omega_n_minus_1 = roots[num_bits - 1];
    let powers = get_powers_from_params(pp);
    // Extract verifier key from universal parameters
    let vk = make_verifier_key(pp);

    let (min_com, _) = KZG::commit(&powers, &get_const_polynomial(min, num_bits), None, None)
        .map_err(|e| anyhow!("min_verify: commit failed: {:?}", e))?;

    Ok(values_com.par_iter().enumerate().all(|(i, val)| {
        // Use the full path to the Commitment constructor
        let f_com = ark_poly_commit::kzg10::Commitment((val.0 - min_com.0).into());

        single_proof_verify(
            &vk,
            &f_com,
            &proofs[i],
            num_bits,
            &omega,
            &omega_n_minus_1,
            i,
        )
    }))
}

/// Verifies range proofs for `values_com[i] <= max`.
///
/// # Parameters
/// * pp: - The universal parameters
/// * max: Fr - The upper bound; each committed value must be
///   less than or equal to max.
/// * values_com: A slice of KZG commitments to the secret values.
/// * proofs: A slice of RangeProofs, one per commitment.
/// * num_bits: usize - The evaluation domain size.
///
/// # Returns
/// * bool: true if all verifications succeed, false otherwise.
#[must_use = "ignoring the verification result defeats the purpose of the check"]
pub fn max_verify(
    pp: &Params,
    max: &Fr,
    values_com: &[Commitment],
    proofs: &[RangeProof],
    num_bits: usize,
) -> anyhow::Result<bool> {
    let roots = generate_nth_roots_of_unity(num_bits);
    let omega = roots[1];
    let omega_n_minus_1 = roots[num_bits - 1];
    let powers = get_powers_from_params(pp);
    // Extract verifier key from universal parameters
    let vk = make_verifier_key(pp);

    let (max_com, _) = KZG::commit(&powers, &get_const_polynomial(max, num_bits), None, None)
        .map_err(|e| anyhow!("max_verify: commit failed: {:?}", e))?;

    Ok(values_com.par_iter().enumerate().all(|(i, val)| {
        let f_com = ark_poly_commit::kzg10::Commitment((max_com.0 - val.0).into());
        single_proof_verify(
            &vk,
            &f_com,
            &proofs[i],
            num_bits,
            &omega,
            &omega_n_minus_1,
            i,
        )
    }))
}

/// Returns the serialized byte size of a `RangeProof`.
///
/// Each `Commitment` and `Proof` is a compressed G1 point (48 bytes on
/// BLS12-381); each `Fr` evaluation is 32 bytes.
pub const fn size_of_range_proof() -> usize {
    3 * std::mem::size_of::<Commitment>()
        + 3 * std::mem::size_of::<Proof>()
        + 3 * std::mem::size_of::<Fr>()
}
