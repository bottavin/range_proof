#[cfg(test)]
mod test {
    use ark_bls12_381::Fr;
    use ark_bls12_381::{G1Projective, G2Projective};
    use ark_ff::Field;
    use ark_ff::{One, Zero};
    use ark_poly::{DenseUVPolynomial, Polynomial, univariate::DensePolynomial};
    use ark_std::{test_rng, UniformRand};
    use range_proof::utilities::*;

    fn rng() -> impl ark_std::rand::RngCore {
        ark_std::test_rng()
    }

    // -------------------------------------------------------------------------
    // int_to_field
    // -------------------------------------------------------------------------

    #[test]
    fn zero_maps_to_field_zero() {
        let result: Fr = int_to_field(0i64);
        assert_eq!(result, Fr::zero());
    }

    #[test]
    fn one_maps_to_field_one() {
        let result: Fr = int_to_field(1i64);
        assert_eq!(result, Fr::one());
    }

    #[test]
    fn positive_values_round_trip() {
        for v in [2i64, 42, 1_000_000, i64::MAX] {
            let f: Fr = int_to_field(v);
            // Re-derive via the canonical Fr::from path and compare.
            let expected = Fr::from(v as u64);
            assert_eq!(f, expected, "mismatch for v={v}");
        }
    }

    #[test]
    fn negative_one_is_additive_inverse_of_one() {
        let neg_one: Fr = int_to_field(-1i64);
        assert_eq!(neg_one + Fr::one(), Fr::zero());
    }

    #[test]
    fn negative_values_are_additive_inverses() {
        for v in [2i64, 100, 999] {
            let pos: Fr = int_to_field(v);
            let neg: Fr = int_to_field(-v);
            assert_eq!(pos + neg, Fr::zero(), "pos + neg ≠ 0 for v={v}");
        }
    }

    #[test]
    fn works_with_i8() {
        let f: Fr = int_to_field(-128i8);
        let expected: Fr = -Fr::from(128u64);
        assert_eq!(f, expected);
    }

    #[test]
    fn works_with_i32() {
        let f: Fr = int_to_field(-1_000_000i32);
        let expected: Fr = -Fr::from(1_000_000u64);
        assert_eq!(f, expected);
    }

    #[test]
    fn works_with_i128() {
        let large: i128 = 1_000_000_000_000_000_000;
        let f: Fr = int_to_field(large);
        let expected = Fr::from(large as u128);
        assert_eq!(f, expected);
    }


    // -------------------------------------------------------------------------
    // to_binary
    // -------------------------------------------------------------------------

    /// Reconstruct the integer from a little-endian bit vector.
    fn bits_to_u64(bits: &[Fr]) -> u64 {
        bits.iter().enumerate().fold(0u64, |acc, (i, &b)| {
            if b == Fr::one() { acc | (1u64 << i) } else { acc }
        })
    }

    #[test]
    fn zero_gives_all_zeros() {
        let bits = to_binary(&Fr::zero(), 8);
        assert_eq!(bits.len(), 8);
        assert!(bits.iter().all(|b| *b == Fr::zero()));
    }

    #[test]
    fn one_gives_lsb_set() {
        let bits = to_binary(&Fr::one(), 8);
        assert_eq!(bits[0], Fr::one());
        assert!(bits[1..].iter().all(|b| *b == Fr::zero()));
    }

    #[test]
    fn known_value_roundtrips() {
        for v in [0u64, 1, 2, 5, 13, 255, 1024, 0xDEAD_BEEF] {
            let z = Fr::from(v);
            let bits = to_binary(&z, 64);
            assert_eq!(bits_to_u64(&bits), v, "roundtrip failed for v={v}");
        }
    }

    #[test]
    fn output_length_equals_n() {
        let z = Fr::from(0xFF_u64);
        for n in [1, 4, 8, 16, 32] {
            assert_eq!(to_binary(&z, n).len(), n);
        }
    }

    #[test]
    fn padding_with_zeros_when_n_is_large() {
        // Fr::from(1) needs only 1 bit; ask for 32 → trailing 31 must be 0.
        let bits = to_binary(&Fr::one(), 32);
        assert!(bits[1..].iter().all(|b| *b == Fr::zero()));
    }

    #[test]
    fn truncation_when_n_is_small() {
        // 0b1010 = 10; with n=2 we keep only the two LSBs → [0, 1].
        let z = Fr::from(0b1010u64);
        let bits = to_binary(&z, 2);
        assert_eq!(bits[0], Fr::zero()); // bit 0 of 10 = 0
        assert_eq!(bits[1], Fr::one());  // bit 1 of 10 = 1
    }

    #[test]
    fn all_bits_are_zero_or_one() {
        let z = Fr::from(123456789u64);
        for b in to_binary(&z, 64) {
            assert!(b == Fr::zero() || b == Fr::one());
        }
    }

    #[test]
    fn test_to_binary_zero() {
        let n = 8;
        let result = to_binary(&Fr::zero(), n);
        assert_eq!(result, vec![Fr::zero(); n]);
    }

    #[test]
    fn test_to_binary_small_number() {
        // 5 in binary (LE): 1 0 1 0 0 0 0 0
        let result = to_binary(&Fr::from(5u64), 8);
        let expected = vec![
            Fr::one(),
            Fr::zero(),
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_to_binary_large_number() {
        // 12345 = 0b11000000111001 → LE bits
        let expected = vec![
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::one(),
            Fr::one(),
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::one(),
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
        ];
        assert_eq!(to_binary(&Fr::from(12345u64), 16), expected);
    }

    #[test]
    fn test_to_binary_large_number_small_n() {
        // 1234567 lower 16 bits (LE)
        let expected = vec![
            Fr::one(),
            Fr::one(),
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
            Fr::one(),
            Fr::zero(),
            Fr::one(),
            Fr::one(),
            Fr::zero(),
            Fr::one(),
            Fr::zero(),
            Fr::one(),
            Fr::one(),
        ];
        assert_eq!(to_binary(&Fr::from(1234567u64), 16), expected);
    }

    #[test]
    fn test_to_binary_large_n() {
        // 13 = 0b1101 padded to 18 bits
        let mut expected = vec![Fr::one(), Fr::zero(), Fr::one(), Fr::one()];
        expected.resize(18, Fr::zero());
        assert_eq!(to_binary(&Fr::from(13u64), 18), expected);
    }

    #[test]
    fn test_to_binary_small_n() {
        // 13 = 0b1101, take only 4 bits
        let expected = vec![Fr::one(), Fr::zero(), Fr::one(), Fr::one()];
        assert_eq!(to_binary(&Fr::from(13u64), 4), expected);
    }

    #[test]
    fn test_to_binary_edge_case() {
        assert_eq!(to_binary(&Fr::from(1u64), 1), vec![Fr::one()]);
    }

    #[test]
    fn test_to_binary_empty_case() {
        assert_eq!(to_binary(&Fr::from(5u64), 0), vec![]);
    }

    // -------------------------------------------------------------------------
    // get_challenge
    // -------------------------------------------------------------------------

    #[test]
    fn empty_slice_does_not_panic_and_is_deterministic() {
        let c1: Fr = get_challenge::<G1Projective, Fr>(&[]);
        let c2: Fr = get_challenge::<G1Projective, Fr>(&[]);
        assert_eq!(c1, c2);
    }

    #[test]
    fn same_inputs_produce_same_challenge() {
        let mut rng = rng();
        let pts: Vec<G1Projective> = (0..4).map(|_| G1Projective::rand(&mut rng)).collect();
        let c1 = get_challenge::<G1Projective, Fr>(&pts);
        let c2: Fr = get_challenge(&pts);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_inputs_produce_different_challenges() {
        let mut rng = rng();
        let pts_a: Vec<G1Projective> = (0..4).map(|_| G1Projective::rand(&mut rng)).collect();
        let pts_b: Vec<G1Projective> = (0..4).map(|_| G1Projective::rand(&mut rng)).collect();
        let ca: Fr = get_challenge(&pts_a);
        let cb: Fr = get_challenge(&pts_b);
        assert_ne!(ca, cb);
    }

    #[test]
    fn test_get_challenge_hash_returns_different_values() {
        let rng = &mut test_rng();
        let f = G1Projective::rand(rng);
        let g = G1Projective::rand(rng);
        let q = G1Projective::rand(rng);

        assert_ne!(get_challenge::<G1Projective, Fr>(&[f, g]), get_challenge(&[f, g, q]));
    }

    #[test]
    fn order_matters() {
        let mut rng = rng();
        let a = G1Projective::rand(&mut rng);
        let b = G1Projective::rand(&mut rng);
        let c_ab: Fr = get_challenge(&[a, b]);
        let c_ba: Fr = get_challenge(&[b, a]);
        assert_ne!(c_ab, c_ba);
    }

    #[test]
    fn result_is_nonzero_for_nontrivial_input() {
        let mut rng = rng();
        let pts: Vec<G1Projective> = (0..2).map(|_| G1Projective::rand(&mut rng)).collect();
        let c: Fr = get_challenge(&pts);
        // Negligible probability of being zero for a hash-derived value.
        assert_ne!(c, Fr::zero());
    }

    #[test]
    fn works_with_g2_elements() {
        let mut rng = rng();
        let pts: Vec<G2Projective> = (0..2).map(|_| G2Projective::rand(&mut rng)).collect();
        let c1: Fr = get_challenge(&pts);
        let c2: Fr = get_challenge(&pts);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_get_challenge_deterministic() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        let b = G2Projective::rand(rng);

        // Same inputs → same output
        assert_eq!(get_challenge::<G2Projective, Fr>(&[a, b]), get_challenge(&[a, b]));
    }

    #[test]
    fn test_get_challenge_differs_for_different_inputs() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        let b = G2Projective::rand(rng);

        // Different inputs → (almost certainly) different outputs
        assert_ne!(get_challenge::<G2Projective, Fr>(&[a]), get_challenge(&[b]));
    }

    #[test]
    fn test_get_challenge_length_matters() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        let b = G2Projective::rand(rng);

        // [a] vs [a, b] must differ
        assert_ne!(get_challenge::<G2Projective, Fr>(&[a]), get_challenge(&[a, b]));
    }

    #[test]
    fn test_get_challenge_returns_field_element() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        // Just verify we get a valid Fr (no panic, not zero with overwhelming probability)
        let ch: Fr = get_challenge(&[a]);
        // The probability of hitting exactly zero is negligible
        assert_ne!(ch, Fr::zero());
    }

    #[test]
    fn g1_and_g2_challenges_differ_for_equivalent_scalars() {
        // Challenges derived from G1 vs G2 elements should differ even if
        // computed from the same scalar multiple of the generator, because
        // compressed serialisation of G1 and G2 points have different formats.
        let mut rng = rng();
        let g1_pts = vec![G1Projective::rand(&mut rng)];
        let g2_pts = vec![G2Projective::rand(&mut rng)];
        let c_g1: Fr = get_challenge(&g1_pts);
        let c_g2: Fr = get_challenge(&g2_pts);
        // They live in the same field, but the serialised bytes differ in length
        // (48 vs 96 bytes), so the hashes—and therefore the challenges—differ.
        assert_ne!(c_g1, c_g2);
    }

    #[test]
    fn works_with_field_elements_as_input() {
        // Fr itself implements CanonicalSerialize, so get_challenge is usable
        // as a simple hash-to-field for arbitrary field elements too.
        let inputs = vec![Fr::from(1u64), Fr::from(2u64)];
        let c1: Fr = get_challenge(&inputs);
        let c2: Fr = get_challenge(&inputs);
        assert_eq!(c1, c2);
    }

    // -------------------------------------------------------------------------
    // get_const_polynomial
    // -------------------------------------------------------------------------

    #[test]
    fn evaluates_to_constant_at_every_point() {
        let z = Fr::from(42u64);
        let p = get_const_polynomial(&z, 4);
        for x in [Fr::zero(), Fr::one(), Fr::from(2u64), Fr::from(100u64)] {
            assert_eq!(p.evaluate(&x), z, "p({x:?}) ≠ {z:?}");
        }
    }

    #[test]
    fn zero_constant_gives_zero_polynomial() {
        let p: DensePolynomial<Fr> = get_const_polynomial(&Fr::zero(), 4);
        for x in [Fr::zero(), Fr::one(), Fr::from(7u64)] {
            assert_eq!(p.evaluate(&x), Fr::zero());
        }
    }

    #[test]
    fn coefficients_beyond_index_zero_are_zero() {
        let z = Fr::from(99u64);
        let p = get_const_polynomial(&z, 8);
        // coeffs() may be shorter than 8 after trailing-zero trimming by ark,
        // but the degree-0 coefficient must equal z.
        assert_eq!(p.coeffs()[0], z);
        for &c in p.coeffs().iter().skip(1) {
            assert_eq!(c, Fr::zero());
        }
    }

    #[test]
    fn degree_is_zero() {
        let p = get_const_polynomial(&Fr::from(5u64), 16);
        assert_eq!(p.degree(), 0);
    }

    // -------------------------------------------------------------------------
    // scalar_mul
    // -------------------------------------------------------------------------

    #[test]
    fn multiply_by_zero_gives_zero_polynomial() {
        let p = create_poly::<Fr, i64>(&[1, 2, 3]);
        let result = scalar_mul(&p, &Fr::zero());
        for x in [Fr::zero(), Fr::one(), Fr::from(5u64)] {
            assert_eq!(result.evaluate(&x), Fr::zero());
        }
    }

    #[test]
    fn test_scalar_mul_by_zero() {
        let p = create_poly(&[1, 2, 3]);
        let result = scalar_mul(&p, &Fr::zero());
        assert!(result.is_zero());
    }

    #[test]
    fn multiply_by_one_is_identity() {
        let p = create_poly::<Fr, i64>(&[1, -2, 3]);
        let result = scalar_mul(&p, &Fr::one());
        for x in [Fr::zero(), Fr::one(), Fr::from(5u64)] {
            assert_eq!(result.evaluate(&x), p.evaluate(&x));
        }
    }

    #[test]
    fn test_scalar_mul_by_one() {
        let p = create_poly(&[1, 2, 3]);
        let result = scalar_mul(&p, &Fr::one());
        assert_eq!(result, p);
    }

    #[test]
    fn test_scalar_mul_by_two() {
        let p = create_poly(&[1, 2, 3]);
        let result = scalar_mul(&p, &Fr::from(2u64));
        assert_eq!(
            result,
            DensePolynomial::from_coefficients_vec(vec![
                Fr::from(2u64),
                Fr::from(4u64),
                Fr::from(6u64),
            ])
        );
    }

    #[test]
    fn multiply_by_scalar_scales_evaluations() {
        // p(x) = 1 + 2x; s = 3; expected: 3 + 6x
        let p = create_poly::<Fr, i64>(&[1, 2]);
        let s = Fr::from(3u64);
        let result = scalar_mul(&p, &s);
        for x in [Fr::zero(), Fr::one(), Fr::from(7u64)] {
            assert_eq!(result.evaluate(&x), p.evaluate(&x) * s);
        }
    }

    #[test]
    fn coefficients_are_individually_scaled() {
        let p = create_poly::<Fr, i64>(&[4, 5, 6]);
        let s = Fr::from(2u64);
        let result = scalar_mul(&p, &s);
        assert_eq!(result.coeffs()[0], Fr::from(8u64));
        assert_eq!(result.coeffs()[1], Fr::from(10u64));
        assert_eq!(result.coeffs()[2], Fr::from(12u64));
    }

    // -------------------------------------------------------------------------
    // generate_nth_roots_of_unity
    // -------------------------------------------------------------------------

    #[test]
    fn first_root_is_one() {
        let roots = generate_nth_roots_of_unity::<Fr>(8);
        assert_eq!(roots.len(), 8);
        assert_eq!(roots[0], Fr::one());
    }

    #[test]
    fn test_generate_1st_root_of_unity() {
        let roots: Vec<Fr> = generate_nth_roots_of_unity(1);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], Fr::one());
    }

    #[test]
    fn test_generate_2nd_root_of_unity() {
        let roots: Vec<Fr> = generate_nth_roots_of_unity(2);
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0], Fr::one());
        // ω² = 1
        assert_eq!(Fr::one(), roots[1] * roots[1]);
    }

    #[test]
    fn length_is_at_least_n() {
        for n in [1usize, 2, 3, 4, 5, 8, 16] {
            let roots = generate_nth_roots_of_unity::<Fr>(n);
            assert!(roots.len() >= n);
        }
    }

    #[test]
    fn test_generate_3nd_root_of_unity() {
        // GeneralEvaluationDomain rounds up to the next power of two → 4 elements
        let roots: Vec<Fr> = generate_nth_roots_of_unity(3);
        assert_eq!(roots.len(), 4);
        assert_eq!(roots[0], Fr::one());
        assert_eq!(roots[2], roots[1] * roots[1]);
        assert_eq!(roots[3], roots[2] * roots[1]);
    }

    #[test]
    fn test_generate_nth_root_of_unity() {
        let roots: Vec<Fr> = generate_nth_roots_of_unity(10);
        // Domain rounds up to next power of two → 16
        assert_eq!(roots.len(), 16);
        assert_eq!(roots[0], Fr::one());
        for i in 2..roots.len() - 2 {
            assert_eq!(roots[i], roots[i - 1] * roots[1]);
        }
        // ω^(2^k) = 1
        assert_eq!(roots[roots.len() - 1] * roots[1], Fr::one());
    }

    #[test]
    fn length_is_a_power_of_two() {
        for n in [3usize, 5, 6, 7, 9, 10] {
            let roots = generate_nth_roots_of_unity::<Fr>(n);
            let len = roots.len();
            assert!(len.is_power_of_two(), "len={len} is not a power of two for n={n}");
        }
    }

    #[test]
    fn every_root_raised_to_domain_size_is_one() {
        let n = 8usize;
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let m = roots.len() as u64;
        for (i, &r) in roots.iter().enumerate() {
            let rm = r.pow([m]);
            assert_eq!(rm, Fr::one(), "ω^{i} ^ {m} ≠ 1");
        }
    }

    #[test]
    fn roots_are_distinct() {
        let roots = generate_nth_roots_of_unity::<Fr>(8);
        let unique: std::collections::HashSet<_> = roots.iter()
            .map(|r| format!("{r:?}"))
            .collect();
        assert_eq!(unique.len(), roots.len());
    }

    #[test]
    fn product_of_all_roots_is_negative_one_for_power_of_two_domain() {
        // For a domain of size m = 2^k > 1, the product of all m-th roots of
        // unity equals (−1), because their product equals the constant term of
        // x^m − 1, which is −1 when m > 1.
        let roots = generate_nth_roots_of_unity::<Fr>(8);
        let product = roots.iter().fold(Fr::one(), |acc, &r| acc * r);
        assert_eq!(product, -Fr::one());
    }

    #[test]
    fn test_generate_nth_root_of_unity_large_n() {
        let roots: Vec<Fr> = generate_nth_roots_of_unity(1_000_000);
        assert_eq!(roots[0], Fr::one());
        for i in 2..roots.len() - 2 {
            assert_eq!(roots[i], roots[i - 1] * roots[1]);
        }
        assert_eq!(roots[roots.len() - 1] * roots[1], Fr::one());
    }

    // -------------------------------------------------------------------------
    // get_encoding_polynomial
    // -------------------------------------------------------------------------

    /// Verify the recurrence directly: g[i] = 2·g[i+1] + bit[i].
    fn check_recurrence(z: &Fr, n: usize) {
        let bits = to_binary(z, n);
        let g = get_encoding_polynomial(z, n);
        let two = Fr::from(2u64);

        assert_eq!(g[n - 1], bits[n - 1], "last element should equal z's top bit");
        for i in 0..n - 1 {
            let expected = g[i + 1] * two + bits[i];
            assert_eq!(g[i], expected, "recurrence failed at i={i} for z={z:?}");
        }
    }

    #[test]
    fn zero_gives_all_zeros_poly() {
        let g = get_encoding_polynomial(&Fr::zero(), 8);
        assert!(g.iter().all(|&v| v == Fr::zero()));
    }

    #[test]
    fn test_get_encoding_polynomial_z_0() {
        let n = 3;
        assert_eq!(
            get_encoding_polynomial(&Fr::zero(), n),
            vec![Fr::zero(); n]
        );
    }

    #[test]
    fn test_get_encoding_polynomial_n_1() {
        let z = Fr::one();
        let result = get_encoding_polynomial(&z, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], z);
    }

    #[test]
    fn test_get_encoding_polynomial_n_2() {
        let result = get_encoding_polynomial(&Fr::from(2u64), 2);
        assert_eq!(result, vec![Fr::from(2u64), Fr::one()]);
    }

    #[test]
    fn test_get_encoding_polynomial_n_4() {
        let result = get_encoding_polynomial(&Fr::from(5u64), 4);
        assert_eq!(
            result,
            vec![Fr::from(5u64), Fr::from(2u64), Fr::one(), Fr::zero()]
        );
    }

    #[test]
    fn one_satisfies_recurrence() {
        check_recurrence(&Fr::one(), 8);
    }

    #[test]
    fn arbitrary_values_satisfy_recurrence() {
        for v in [2u64, 5, 42, 127, 255] {
            check_recurrence(&Fr::from(v), 8);
        }
    }

    #[test]
    fn output_length_equals_n_poly() {
        for n in [4usize, 8, 16] {
            let g = get_encoding_polynomial(&Fr::from(7u64), n);
            assert_eq!(g.len(), n);
        }
    }

    #[test]
    fn g0_encodes_full_integer() {
        // g[0] = 2^(n-1)·bit[n-1] + 2^(n-2)·bit[n-2] + … + bit[0] = z  (for z < 2^n).
        // So for small z the value at index 0 should equal z itself in the field.
        let z = Fr::from(13u64); // 0b00001101, fits in 8 bits
        let g = get_encoding_polynomial(&z, 8);
        assert_eq!(g[0], z);
    }

    #[test]
    fn test_get_encoding_polynomial_large_n() {
        let result = get_encoding_polynomial(&Fr::from(123456789u64), 30);
        let expected = vec![
            Fr::from(123456789u64),
            Fr::from(61728394u64),
            Fr::from(30864197u64),
            Fr::from(15432098u64),
            Fr::from(7716049u64),
            Fr::from(3858024u64),
            Fr::from(1929012u64),
            Fr::from(964506u64),
            Fr::from(482253u64),
            Fr::from(241126u64),
            Fr::from(120563u64),
            Fr::from(60281u64),
            Fr::from(30140u64),
            Fr::from(15070u64),
            Fr::from(7535u64),
            Fr::from(3767u64),
            Fr::from(1883u64),
            Fr::from(941u64),
            Fr::from(470u64),
            Fr::from(235u64),
            Fr::from(117u64),
            Fr::from(58u64),
            Fr::from(29u64),
            Fr::from(14u64),
            Fr::from(7u64),
            Fr::from(3u64),
            Fr::one(),
            Fr::zero(),
            Fr::zero(),
            Fr::zero(),
        ];
        assert_eq!(result, expected);
    }

    // -------------------------------------------------------------------------
    // get_encoding_polynomial / build_encoding_polynomial
    // -------------------------------------------------------------------------

    /// The polynomial built by IFFT must evaluate to get_encoding_polynomial
    /// values at the corresponding roots of unity.
    fn check_evaluations_match(z: &Fr, n: usize) {
        let poly = build_encoding_polynomial(z, n);
        let evals = get_encoding_polynomial(z, n);
        let roots = generate_nth_roots_of_unity::<Fr>(n);

        assert_eq!(roots.len(), poly.len(), "mismatch in number of roots vs poly coeffs for n={n}");
        assert_eq!(n, evals.len(), "mismatch in number of roots vs evals for n={n}");

        for (i, (omega_i, expected)) in roots.iter().zip(evals.iter()).enumerate() {
            let got = poly.evaluate(omega_i);
            assert_eq!(got, *expected, "mismatch at i={i} for z={z:?}, n={n}");
        }
    }

    /// Helper: evaluate `poly` at every m-th root of unity and return the values.
    fn eval_at_roots(poly: &DensePolynomial<Fr>, m: usize) -> Vec<Fr> {
        let roots = generate_nth_roots_of_unity(m);
        // generate_nth_roots_of_unity already rounds up, so roots.len() == m
        // when m is a power of two.
        roots.iter().map(|r| poly.evaluate(r)).collect()
    }

    #[test]
    fn zero_gives_zero_polynomial() {
        let poly = build_encoding_polynomial(&Fr::zero(), 8);
        for x in [Fr::zero(), Fr::from(1u64), Fr::from(42u64)] {
            assert_eq!(poly.evaluate(&x), Fr::zero());
        }
    }

    #[test]
    fn evaluations_match_for_one() {
        check_evaluations_match(&Fr::from(1u64), 8);
    }

    #[test]
    fn test_build_encoding_polynomial_exact_n1() {
        // n = 1 → domain size 1, g(1) = z
        let z = Fr::one();
        let poly = build_encoding_polynomial(&z, 1);
        // The polynomial must evaluate to z at ω⁰ = 1
        assert_eq!(poly.evaluate(&Fr::one()), z);
    }

    #[test]
    fn test_build_encoding_polynomial_exact_n2() {
        // z = 2 = 0b10; encoding values: [2, 1]
        let z = Fr::from(2u64);
        let m = 2;
        let poly = build_encoding_polynomial(&z, m);

        // Degree must be < m (domain size)
        assert!(poly.degree() < m);

        // Round-trip: evaluations at the 2nd roots of unity == get_encoding_polynomial
        let expected = get_encoding_polynomial(&z, m);
        assert_eq!(eval_at_roots(&poly, m), expected);
    }

    #[test]
    fn test_build_encoding_polynomial_rounds_up_n3() {
        // n = 3 → domain rounds to 4
        let z = Fr::from(5u64);
        let n = 3;
        let m = 4;
        let poly = build_encoding_polynomial(&z, n);

        // Degree is bounded by the rounded domain, not n
        assert!(poly.degree() < m);

        // The first n evaluations match get_encoding_polynomial(z, n);
        // the remaining (m - n) are padding determined by the IFFT.
        let enc = get_encoding_polynomial(&z, n);
        let evals = eval_at_roots(&poly, m);
        assert_eq!(&evals[..n], enc.as_slice());
    }

    #[test]
    fn test_build_encoding_polynomial_exact_n4() {
        // z = 5 = 0b0101; encoding values: [5, 2, 1, 0]
        let z = Fr::from(5u64);
        let m = 4;
        let poly = build_encoding_polynomial(&z, m);

        assert!(poly.degree() < m);
        let expected = get_encoding_polynomial(&z, m);
        assert_eq!(eval_at_roots(&poly, m), expected);
    }

    #[test]
    fn test_build_encoding_polynomial_exact_n8() {
        let z = Fr::from(42u64);
        let m = 8;
        let poly = build_encoding_polynomial(&z, m);

        assert!(poly.degree() < m);
        assert_eq!(eval_at_roots(&poly, m), get_encoding_polynomial(&z, m));
    }

    #[test]
    fn test_build_encoding_polynomial_rounds_up_n10() {
        // n = 10 → domain rounds to 16  (this is the user-provided test case)
        let z = Fr::from(30u64);
        let n = 10;
        let m = 16;
        let poly = build_encoding_polynomial(&z, n);

        // poly has at most m coefficients
        assert!(poly.coeffs().len() <= m);

        // Round-trip for the first n evaluation points
        let enc = get_encoding_polynomial(&z, n);
        let evals = eval_at_roots(&poly, m);
        assert_eq!(&evals[..n], enc.as_slice());
    }

    #[test]
    fn test_build_encoding_polynomial_n_10() {
        let z = Fr::from(30);
        check_evaluations_match(&z, 10);
    }

    #[test]
    fn test_build_encoding_polynomial_rounds_up_n30() {
        // The existing large-n test uses get_encoding_polynomial directly;
        // here we verify the IFFT round-trip for the same input.
        let z = Fr::from(123456789u64);
        let n = 30;
        let m = 32;
        let poly = build_encoding_polynomial(&z, n);

        assert!(poly.coeffs().len() <= m);

        let enc = get_encoding_polynomial(&z, n);
        let evals = eval_at_roots(&poly, m);
        assert_eq!(&evals[..n], enc.as_slice());
    }

    #[test]
    fn evaluations_match_for_arbitrary_values() {
        for v in [2u64, 5, 42, 100] {
            check_evaluations_match(&Fr::from(v), 8);
        }
    }

    #[test]
    fn degree_is_less_than_n() {
        let poly = build_encoding_polynomial(&Fr::from(7u64), 8);
        assert!(poly.degree() < 8);
    }

    #[test]
    fn works_for_n_equals_16() {
        check_evaluations_match(&Fr::from(12345u64), 16);
    }

    #[test]
    fn test_build_encoding_polynomial_largest_n_in_domain() {
        // n = 32: a large exact-power-of-two domain; verifies no off-by-one in
        // the Radix2 setup.
        let z = Fr::from(1_000_000u64);
        let m = 32;
        let poly = build_encoding_polynomial(&z, m);

        assert!(poly.degree() < m);
        assert_eq!(eval_at_roots(&poly, m), get_encoding_polynomial(&z, m));
    }

    // -------------------------------------------------------------------------
    // create_field_vec
    // -------------------------------------------------------------------------

    #[test]
    fn empty_slice_gives_empty_vec() {
        let v: Vec<Fr> = create_field_vec::<Fr, i64>(&[]);
        assert!(v.is_empty());
    }

    #[test]
    fn positive_values_correct() {
        let v: Vec<Fr> = create_field_vec(&[0i64, 1, 2, 42]);
        assert_eq!(v[0], Fr::zero());
        assert_eq!(v[1], Fr::one());
        assert_eq!(v[2], Fr::from(2u64));
        assert_eq!(v[3], Fr::from(42u64));
    }

    #[test]
    fn negative_values_vec_are_additive_inverses() {
        let v: Vec<Fr> = create_field_vec(&[-1i64, -2, -42]);
        assert_eq!(v[0] + Fr::one(), Fr::zero());
        assert_eq!(v[1] + Fr::from(2u64), Fr::zero());
        assert_eq!(v[2] + Fr::from(42u64), Fr::zero());
    }

    #[test]
    fn test_create_field_vec_zero() {
        let v:Vec<Fr> = create_field_vec(&[0]);
        assert_eq!(v, vec![Fr::zero()]);
    }

    #[test]
    fn works_with_i8_vec() {
        let v: Vec<Fr> = create_field_vec(&[127i8, -128i8]);
        assert_eq!(v[0], Fr::from(127u64));
        assert_eq!(v[1], -Fr::from(128u64));
    }

    #[test]
    fn works_with_i32_vec() {
        let v: Vec<Fr> = create_field_vec(&[i32::MIN, i32::MAX]);
        assert_eq!(v[0], -Fr::from(i32::MIN.unsigned_abs() as u64));
        assert_eq!(v[1], Fr::from(i32::MAX as u64));
    }

    #[test]
    fn length_matches_input() {
        let input = [1i64, 2, 3, 4, 5];
        let v: Vec<Fr> = create_field_vec(&input);
        assert_eq!(v.len(), input.len());
    }

    // -------------------------------------------------------------------------
    // create_poly
    // -------------------------------------------------------------------------

    #[test]
    fn constant_polynomial() {
        // f(x) = 5
        let p = create_poly::<Fr, i64>(&[5]);
        assert_eq!(p.evaluate(&Fr::zero()), Fr::from(5u64));
        assert_eq!(p.evaluate(&Fr::from(100u64)), Fr::from(5u64));
    }

    #[test]
    fn test_create_poly_constant() {
        let p = create_poly(&[5]);
        assert_eq!(p.evaluate(&Fr::zero()), Fr::from(5u64));
        assert_eq!(p.evaluate(&Fr::from(99u64)), Fr::from(5u64));
    }

    #[test]
    fn linear_polynomial() {
        // f(x) = 3 + 2x  →  f(1) = 5, f(2) = 7
        let p = create_poly::<Fr, i64>(&[3, 2]);
        assert_eq!(p.evaluate(&Fr::one()), Fr::from(5u64));
        assert_eq!(p.evaluate(&Fr::from(2u64)), Fr::from(7u64));
    }

    #[test]
    fn test_create_poly_quadratic() {
        // 1 + 0x + 1x²  ⟹ at x=3: 10
        let p = create_poly(&[1, 0, 1]);
        assert_eq!(p.evaluate(&Fr::from(3u64)), Fr::from(10u64));
    }

    #[test]
    fn quadratic_with_negative_coefficients() {
        // f(x) = -1 + 3x - x^2  →  f(0) = -1, f(1) = 1, f(3) = -1
        let p = create_poly::<Fr, i64>(&[-1, 3, -1]);
        assert_eq!(p.evaluate(&Fr::zero()), -Fr::one());
        assert_eq!(p.evaluate(&Fr::one()), Fr::one());
        assert_eq!(p.evaluate(&Fr::from(3u64)), -Fr::one());
    }

    #[test]
    fn zero_polynomial_evaluates_to_zero() {
        let p = create_poly::<Fr, i64>(&[0, 0, 0]);
        assert_eq!(p.evaluate(&Fr::from(42u64)), Fr::zero());
    }

    #[test]
    fn works_with_i8_coefficients() {
        // f(x) = -1 + x   (i8)
        let p = create_poly::<Fr, i8>(&[-1i8, 1i8]);
        assert_eq!(p.evaluate(&Fr::one()), Fr::zero()); // -1 + 1 = 0
        assert_eq!(p.evaluate(&Fr::from(2u64)), Fr::one()); // -1 + 2 = 1
    }

    #[test]
    fn coefficients_stored_in_correct_order() {
        // p = 7 + 0x + 3x^2
        let p = create_poly::<Fr, i64>(&[7, 0, 3]);
        assert_eq!(p.coeffs()[0], Fr::from(7u64));
        assert_eq!(p.coeffs()[2], Fr::from(3u64));
    }

    #[test]
    fn test_create_poly_empty() {
        let p: DensePolynomial<Fr> = create_poly::<Fr, i128>(&[]);
        assert!(p.is_zero());
    }
}
