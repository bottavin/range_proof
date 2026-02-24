use ark_bls12_381::{Fr, G1Projective, G2Projective};
use ark_ff::{BigInt, BigInteger, One, PrimeField, Zero};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Radix2EvaluationDomain,
};
use ark_serialize::CanonicalSerialize;
use sha2::{Digest, Sha256};

/// Type alias for a BigInt of size 4 (using the BigInt struct
/// for arbitrary precision integers).
pub type BigInt4 = BigInt<4>;

/// Converts a usize to a BigInt4 (BigInteger with 4 limbs).
///
/// # Arguments
/// * n - The number to be converted into BigInt4.
///
/// # Returns
/// A BigInt4 representation of the input number.
pub fn create_big_int4(n: usize) -> BigInt4 {
    BigInt::from(n as u64)
}

/// Converts a field element z (of type Fr) into its n-bit binary
/// representation, with each bit expressed as Fr::zero() or Fr::one().
///
/// # Arguments
/// * z - The field element to be converted.
/// * n - The number of bits to represent the element z in binary.
///
/// # Returns
/// A vector of Fr elements where each element is either Fr::zero()
/// (bit 0) or Fr::one() (bit 1).
pub fn to_binary(z: &Fr, n: &usize) -> Vec<Fr> {
    let mut bits = z
        .into_bigint()
        .to_bits_le()
        .iter()
        .map(|&b| if b { Fr::one() } else { Fr::zero() })
        .collect::<Vec<Fr>>();
    // Adjust the length to n. If n is larger, pad with zeros.
    bits.resize(*n, Fr::zero());

    bits
}

/// Generates a Fiat-Shamir challenge scalar by hashing a slice of
/// G1Projective group elements with SHA-256.
///
/// The challenge is derived by hashing the string representation
/// of the elements using SHA-256, then converting the resulting hash
/// into a field element.
///
/// # Arguments
/// * elems - a vector of references to G1Projective elements.
///
/// # Returns
/// A field element derived from the SHA-256 digest of the serialised elements.
pub fn get_challenge(elems: &[G1Projective]) -> Fr {
    let mut hasher = Sha256::new();
    for elem in elems {
        let mut buf = Vec::new();
        elem.serialize_compressed(&mut buf).unwrap();
        hasher.update(&buf);
    }
    Fr::from_le_bytes_mod_order(&hasher.finalize())
}

/// Derives a challenge scalar from a list of G2 elements.
pub fn get_challenge_g2(polys: &[G2Projective]) -> Fr {
    let mut str_to_hash = String::new();

    for el in polys {
        str_to_hash += &el.to_string();
    }

    Fr::from_le_bytes_mod_order(&Sha256::digest(str_to_hash.as_bytes()))
}

/// Returns the constant polynomial f(x) = z (only the degree-0 term
/// is set; all higher-degree coefficients are zero) as a
/// `DensePolynomial<Fr>` of length n.
///
/// # Arguments
/// * z - The constant value to set as the coefficient of x^0.
/// * n - The degree of the polynomial (length of the vector).
///
/// # Returns
/// # Returns
/// A DensePolynomial<Fr> whose degree-0 coefficient equals z and all
/// higher-degree coefficients are zero.
pub fn get_const_polynomial(z: &Fr, n: usize) -> DensePolynomial<Fr> {
    let mut coeffs = vec![Fr::zero(); n];
    coeffs[0] = *z;

    DensePolynomial::from_coefficients_vec(coeffs)
}

/// Returns the roots of unity for the smallest power-of-two domain that
/// contains at least n elements: {1, ω, ω², …, ω^(m−1)} where
/// m = 2^⌈log₂ n⌉.
///
/// # Arguments
/// * n     - The minimum number of roots required; the domain is rounded up to
///   the next power of two by GeneralEvaluationDomain
///
/// # Returns
/// A Vec<Fr> of length m ≥ n containing the m-th roots of unity in order.
pub fn generate_nth_roots_of_unity(n: usize) -> Vec<Fr> {
    let domain = ark_poly::GeneralEvaluationDomain::<Fr>::new(n).unwrap();

    domain.elements().collect()
}

/// Builds the encoding polynomial for z via IFFT over the n-th roots of
/// unity.
///
/// # Arguments
/// * z     - The field element to encode as a polynomial.
/// * n     - The degree of the polynomial (the number of bits
///   in the binary representation).
///
/// # Returns
/// A vector of Fr representing the coefficients of the encoding
/// polynomial g(x).
///
/// # Note
/// The vector is rounded up to the next power of two, so the
/// returned polynomial has degree 2^⌈log₂ n⌉ − 1.
pub fn build_encoding_polynomial(z: &Fr, n: usize) -> DensePolynomial<Fr> {
    let g_values = get_encoding_polynomial(z, &n);
    let domain = Radix2EvaluationDomain::<Fr>::new(n)
        .expect("build_encoding_polynomial: n must be a power of two");
    ark_poly::evaluations::univariate::Evaluations::from_vec_and_domain(g_values, domain)
        .interpolate()
}

// ---------------------------------------------------------------------------
// Internal scalar-multiply helper
// ---------------------------------------------------------------------------
pub fn scalar_mul(p: &DensePolynomial<Fr>, s: &Fr) -> DensePolynomial<Fr> {
    DensePolynomial::from_coefficients_vec(p.coeffs().iter().map(|c| *c * s).collect())
}

/// Builds the encoding polynomial values g(ωⁱ) for i = 0…n−1 using
/// the recurrence:
///
///   g(ω^(n−1)) = z[n−1]
///   g(ωⁱ)      = 2·g(ω^(i+1)) + z[i]   for i = n−2, …, 0
///
/// # Arguments
/// * z - The field element to be encoded as a polynomial.
/// * n - The degree of the polynomial (the number of bits in
///   the binary representation).
///
/// # Returns
/// A Vec<Fr> of length n containing the evaluations
/// g(ω⁰), g(ω¹), …, g(ω^(n−1)) computed by the recurrence, where ω is
/// the primitive n-th root of unity.
pub fn get_encoding_polynomial(z: &Fr, n: &usize) -> Vec<Fr> {
    let z_digits = to_binary(z, n);
    let mut g = vec![Fr::zero(); *n];
    g[n - 1] = z_digits[n - 1];
    let two = Fr::from(2u64);

    for i in (0..n - 1).rev() {
        g[i] = g[i + 1] * two + z_digits[i];
    }

    g
}

/// Convenience constructor: Fr values from i64 literals.
pub fn create_fr_vec(values: &[i64]) -> Vec<Fr> {
    values.iter().map(|&v| Fr::from(v)).collect()
}

/// Convenience constructor: DensePolynomial from i64 coefficient literals.
pub fn create_poly(values: &[i64]) -> DensePolynomial<Fr> {
    DensePolynomial::from_coefficients_slice(&create_fr_vec(values))
}
