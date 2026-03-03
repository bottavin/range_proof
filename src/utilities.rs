use ark_ff::{BigInteger, PrimeField};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, GeneralEvaluationDomain,
    Radix2EvaluationDomain,
};
use ark_serialize::CanonicalSerialize;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Signed-integer → field-element helper
// ---------------------------------------------------------------------------

/// Converts any signed integer into a field element.
///
/// The conversion handles negative values correctly by mapping them to
/// `F::MODULUS - |v|`, which is the additive-inverse representation used
/// by prime-field implementations.
///
/// # Type parameters
/// * `F` - Any type implementing `PrimeField`.
/// * `T` - Any signed integer type that widens losslessly to `i128`
///   (i.e. `i8`, `i16`, `i32`, `i64`, `i128`).
pub fn int_to_field<F, T>(v: T) -> F
where
    F: PrimeField,
    T: Into<i128>,
{
    let n: i128 = v.into();
    if n >= 0 {
        F::from(n as u128)
    } else {
        // Negate: additive inverse of |n|
        -F::from((-n) as u128)
    }
}

// ---------------------------------------------------------------------------
// Binary decomposition
// ---------------------------------------------------------------------------

/// Converts a field element `z` into its `n`-bit little-endian binary
/// representation, with each bit expressed as `F::zero()` or `F::one()`.
///
/// # Arguments
/// * `z` - The field element to convert.
/// * `n` - The number of bits. If `n` is larger than the element's natural
///   bit-length the result is zero-padded; if smaller it is truncated.
///
/// # Returns
/// A `Vec<F>` of length `n` where entry `i` equals bit `i` of `z`.
pub fn to_binary<F: PrimeField>(z: &F, n: usize) -> Vec<F> {
    let mut bits = z
        .into_bigint()
        .to_bits_le()
        .iter()
        .map(|&b| if b { F::one() } else { F::zero() })
        .collect::<Vec<F>>();
    bits.resize(n, F::zero());
    bits
}

// ---------------------------------------------------------------------------
// Fiat-Shamir challenges
// ---------------------------------------------------------------------------

/// Generates a Fiat-Shamir challenge scalar by hashing any slice of
/// serialisable group (or curve) elements with SHA-256.
///
/// It works for any type that implements `CanonicalSerialize` and
/// returns any `PrimeField` element, so callers can use it with
/// G1, G2, or mixed inputs.
///
/// # Type parameters
/// * `G` - Any type implementing [`CanonicalSerialize`]
///   (e.g. `G1Projective`, `G2Projective`, `Fr`, …).
/// * `F` - The target field for the resulting challenge scalar.
///
/// # Arguments
/// * `elems` - A slice of references to group elements to hash.
///
/// # Returns
/// A field element derived from the SHA-256 digest of the compressed
/// serialisations of `elems`.
pub fn get_challenge<G, F>(elems: &[G]) -> F
where
    G: CanonicalSerialize,
    F: PrimeField,
{
    let mut hasher = Sha256::new();
    for elem in elems {
        let mut buf = Vec::new();
        elem.serialize_compressed(&mut buf).unwrap();
        hasher.update(&buf);
    }
    F::from_le_bytes_mod_order(&hasher.finalize())
}

// ---------------------------------------------------------------------------
// Polynomial helpers
// ---------------------------------------------------------------------------

/// Returns the constant polynomial `f(x) = z` (degree-0 coefficient `z`,
/// all higher coefficients zero) represented as a coefficient vector of
/// length `n`.
///
/// # Arguments
/// * `z` - The constant value for the `x^0` coefficient.
/// * `n` - Total number of coefficients (determines the allocated length,
///   not the mathematical degree).
///
/// # Returns
/// A `DensePolynomial<F>` whose only non-zero coefficient is `coeffs[0] = z`.
pub fn get_const_polynomial<F: PrimeField>(z: &F, n: usize) -> DensePolynomial<F> {
    let mut coeffs = vec![F::zero(); n];
    coeffs[0] = *z;
    DensePolynomial::from_coefficients_vec(coeffs)
}

/// Scalar-multiplies every coefficient of `p` by `s`.
pub fn scalar_mul<F: PrimeField>(p: &DensePolynomial<F>, s: &F) -> DensePolynomial<F> {
    DensePolynomial::from_coefficients_vec(p.coeffs().iter().map(|c| *c * s).collect())
}

// ---------------------------------------------------------------------------
// Roots of unity
// ---------------------------------------------------------------------------

/// Returns the roots of unity for the smallest power-of-two domain that
/// contains at least `n` elements: `{1, ω, ω², …, ω^(m−1)}` where
/// `m = 2^⌈log₂ n⌉`.
///
/// # Type parameters
/// * `F` - Any `PrimeField` whose multiplicative group contains a subgroup
///   of the required size (e.g. BLS12-381's `Fr`).
///
/// # Arguments
/// * `n` - The minimum number of roots required.
///
/// # Returns
/// A `Vec<F>` of length `m ≥ n`.
pub fn generate_nth_roots_of_unity<F: PrimeField>(n: usize) -> Vec<F> {
    let domain = GeneralEvaluationDomain::<F>::new(n).unwrap();
    domain.elements().collect()
}

// ---------------------------------------------------------------------------
// Encoding polynomial
// ---------------------------------------------------------------------------

/// Builds the encoding polynomial values `g(ωⁱ)` for `i = 0…n−1` using
/// the recurrence:
///
/// ```text
/// g(ω^(n−1)) = z[n−1]
/// g(ωⁱ)      = 2·g(ω^(i+1)) + z[i]   for i = n−2, …, 0
/// ```
///
/// # Type parameters
/// * `F` - Any `PrimeField`.
///
/// # Arguments
/// * `z` - The field element to encode.
/// * `n` - Number of bits / evaluation points.
///
/// # Returns
/// A `Vec<F>` of length `n` containing `g(ω⁰), g(ω¹), …, g(ω^(n−1))`.
pub fn get_encoding_polynomial<F: PrimeField>(z: &F, n: usize) -> Vec<F> {
    let z_digits = to_binary(z, n);
    let mut g = vec![F::zero(); n];
    g[n - 1] = z_digits[n - 1];
    let two = F::from(2u64);

    for i in (0..n - 1).rev() {
        g[i] = g[i + 1] * two + z_digits[i];
    }

    g
}

/// Builds the encoding polynomial for `z` via IFFT over the `n`-th roots
/// of unity.
///
/// # Type parameters
/// * `F` - Any `PrimeField` whose multiplicative group has a subgroup of
///   the required size (the field must support a `Radix2EvaluationDomain`
///   of size `n`, so `n` must be a power of two).
///
/// # Arguments
/// * `z` - The field element to encode.
/// * `n` - The domain size (must be a power of two).
///
/// # Returns
/// A `DensePolynomial<F>` whose evaluations over the `n`-th roots of unity
/// equal `get_encoding_polynomial(z, n)`.
pub fn build_encoding_polynomial<F: PrimeField>(z: &F, n: usize) -> DensePolynomial<F> {
    let g_values = get_encoding_polynomial(z, n);
    let domain = Radix2EvaluationDomain::<F>::new(n)
        .expect("build_encoding_polynomial: n must be a power of two");
    ark_poly::evaluations::univariate::Evaluations::from_vec_and_domain(g_values, domain)
        .interpolate()
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Builds a `Vec<F>` from a slice of any signed integer type.
///
/// Negative values are converted to their field additive-inverse
/// (`F::MODULUS - |v|`).
///
/// # Type parameters
/// * `F` - Any `PrimeField`.
/// * `T` - Any signed integer type that fits in `i128`
///   (`i8`, `i16`, `i32`, `i64`, `i128`).
pub fn create_field_vec<F, T>(values: &[T]) -> Vec<F>
where
    F: PrimeField,
    T: Copy + Into<i128>,
{
    values.iter().map(|&v| int_to_field(v)).collect()
}

/// Builds a `DensePolynomial<F>` from a slice of any signed integer type,
/// treating each entry as the coefficient of `x^i`.
///
/// # Type parameters
/// * `F` - Any `PrimeField`.
/// * `T` - Any signed integer type that fits in `i128`.
pub fn create_poly<F, T>(values: &[T]) -> DensePolynomial<F>
where
    F: PrimeField,
    T: Copy + Into<i128>,
{
    DensePolynomial::from_coefficients_slice(&create_field_vec(values))
}
