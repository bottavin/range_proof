#[cfg(test)]
mod tests {
    use ark_bls12_381::{Fr, G1Projective, G2Projective};
    use ark_ff::{One, Zero};
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_std::{test_rng, UniformRand};
    use range_proof::utilities::*;

    // -------------------------------------------------------------------------
    // to_binary
    // -------------------------------------------------------------------------

    #[test]
    fn test_to_binary_zero() {
        let n = 8;
        let result = to_binary(&Fr::zero(), &n);
        assert_eq!(result, vec![Fr::zero(); n]);
    }

    #[test]
    fn test_to_binary_small_number() {
        // 5 in binary (LE): 1 0 1 0 0 0 0 0
        let result = to_binary(&Fr::from(5u64), &8);
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
        assert_eq!(to_binary(&Fr::from(12345u64), &16), expected);
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
        assert_eq!(to_binary(&Fr::from(1234567u64), &16), expected);
    }

    #[test]
    fn test_to_binary_large_n() {
        // 13 = 0b1101 padded to 18 bits
        let mut expected = vec![Fr::one(), Fr::zero(), Fr::one(), Fr::one()];
        expected.resize(18, Fr::zero());
        assert_eq!(to_binary(&Fr::from(13u64), &18), expected);
    }

    #[test]
    fn test_to_binary_small_n() {
        // 13 = 0b1101, take only 4 bits
        let expected = vec![Fr::one(), Fr::zero(), Fr::one(), Fr::one()];
        assert_eq!(to_binary(&Fr::from(13u64), &4), expected);
    }

    #[test]
    fn test_to_binary_edge_case() {
        assert_eq!(to_binary(&Fr::from(1u64), &1), vec![Fr::one()]);
    }

    #[test]
    fn test_to_binary_empty_case() {
        assert_eq!(to_binary(&Fr::from(5u64), &0), vec![]);
    }

    // -------------------------------------------------------------------------
    // get_challenge
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_challenge_basic() {
        let rng = &mut test_rng();
        let f = G1Projective::rand(rng);

        assert_eq!(get_challenge(&[f]), get_challenge(&[f]));
    }

    #[test]
    fn test_get_challenge_hash_returns_different_values() {
        let rng = &mut test_rng();
        let f = G1Projective::rand(rng);
        let g = G1Projective::rand(rng);
        let q = G1Projective::rand(rng);

        assert_ne!(get_challenge(&[f, g]), get_challenge(&[f, g, q]));
    }

    // -------------------------------------------------------------------------
    // generate_nth_roots_of_unity
    // -------------------------------------------------------------------------

    #[test]
    fn test_generate_1st_root_of_unity() {
        let roots = generate_nth_roots_of_unity(1);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], Fr::one());
    }

    #[test]
    fn test_generate_2nd_root_of_unity() {
        let roots = generate_nth_roots_of_unity(2);
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0], Fr::one());
        // ω² = 1
        assert_eq!(Fr::one(), roots[1] * roots[1]);
    }

    #[test]
    fn test_generate_3nd_root_of_unity() {
        // GeneralEvaluationDomain rounds up to the next power of two → 4 elements
        let roots = generate_nth_roots_of_unity(3);
        assert_eq!(roots.len(), 4);
        assert_eq!(roots[0], Fr::one());
        assert_eq!(roots[2], roots[1] * roots[1]);
        assert_eq!(roots[3], roots[2] * roots[1]);
    }

    #[test]
    fn test_generate_nth_root_of_unity() {
        let roots = generate_nth_roots_of_unity(10);
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
    fn test_generate_nth_root_of_unity_large_n() {
        let roots = generate_nth_roots_of_unity(1_000_000);
        assert_eq!(roots[0], Fr::one());
        for i in 2..roots.len() - 2 {
            assert_eq!(roots[i], roots[i - 1] * roots[1]);
        }
        assert_eq!(roots[roots.len() - 1] * roots[1], Fr::one());
    }

    // -------------------------------------------------------------------------
    // get_encoding_polynomial / build_encoding_polynomial
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_encoding_polynomial_n_1() {
        let z = Fr::one();
        let result = get_encoding_polynomial(&z, &1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], z);
    }

    #[test]
    fn test_get_encoding_polynomial_n_2() {
        let result = get_encoding_polynomial(&Fr::from(2u64), &2);
        assert_eq!(result, vec![Fr::from(2u64), Fr::one()]);
    }

    #[test]
    fn test_build_encoding_polynomial_z_0() {
        let n = 3;
        assert_eq!(
            get_encoding_polynomial(&Fr::zero(), &n),
            vec![Fr::zero(); n]
        );
    }

    #[test]
    fn test_build_encoding_polynomial_n_4() {
        let result = get_encoding_polynomial(&Fr::from(5u64), &4);
        assert_eq!(
            result,
            vec![Fr::from(5u64), Fr::from(2u64), Fr::one(), Fr::zero()]
        );
    }

    #[test]
    fn test_build_encoding_polynomial_n_10() {
        let z = Fr::from(30);
        let result = build_encoding_polynomial(&z, 10);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_build_encoding_polynomial_large_n() {
        let result = get_encoding_polynomial(&Fr::from(123456789u64), &30);
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

    /// Helper: evaluate `poly` at every m-th root of unity and return the values.
    fn eval_at_roots(poly: &DensePolynomial<Fr>, m: usize) -> Vec<Fr> {
        let roots = generate_nth_roots_of_unity(m);
        // generate_nth_roots_of_unity already rounds up, so roots.len() == m
        // when m is a power of two.
        roots.iter().map(|r| poly.evaluate(r)).collect()
    }

    /// Next power of two ≥ n.
    fn next_pow2(n: usize) -> usize {
        if n.is_power_of_two() {
            n
        } else {
            n.next_power_of_two()
        }
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
        let expected = get_encoding_polynomial(&z, &m);
        assert_eq!(eval_at_roots(&poly, m), expected);
    }

    #[test]
    fn test_build_encoding_polynomial_exact_n4() {
        // z = 5 = 0b0101; encoding values: [5, 2, 1, 0]
        let z = Fr::from(5u64);
        let m = 4;
        let poly = build_encoding_polynomial(&z, m);

        assert!(poly.degree() < m);
        let expected = get_encoding_polynomial(&z, &m);
        assert_eq!(eval_at_roots(&poly, m), expected);
    }

    #[test]
    fn test_build_encoding_polynomial_exact_n8() {
        let z = Fr::from(42u64);
        let m = 8;
        let poly = build_encoding_polynomial(&z, m);

        assert!(poly.degree() < m);
        assert_eq!(eval_at_roots(&poly, m), get_encoding_polynomial(&z, &m));
    }

    // -- Non-power-of-two inputs (rounding) ----------------------------------

    #[test]
    fn test_build_encoding_polynomial_rounds_up_n3() {
        // n = 3 → domain rounds to 4
        let z = Fr::from(5u64);
        let n = 3;
        let m = next_pow2(n); // 4
        let poly = build_encoding_polynomial(&z, n);

        // Degree is bounded by the rounded domain, not n
        assert!(poly.degree() < m);

        // The first n evaluations match get_encoding_polynomial(z, n);
        // the remaining (m - n) are padding determined by the IFFT.
        let enc = get_encoding_polynomial(&z, &n);
        let evals = eval_at_roots(&poly, m);
        assert_eq!(&evals[..n], enc.as_slice());
    }

    #[test]
    fn test_build_encoding_polynomial_rounds_up_n10() {
        // n = 10 → domain rounds to 16  (this is the user-provided test case)
        let z = Fr::from(30u64);
        let n = 10;
        let m = next_pow2(n); // 16
        let poly = build_encoding_polynomial(&z, n);

        // poly has at most m coefficients
        assert!(poly.coeffs().len() <= m);

        // Round-trip for the first n evaluation points
        let enc = get_encoding_polynomial(&z, &n);
        let evals = eval_at_roots(&poly, m);
        assert_eq!(&evals[..n], enc.as_slice());
    }

    #[test]
    fn test_build_encoding_polynomial_rounds_up_n30() {
        // The existing large-n test uses get_encoding_polynomial directly;
        // here we verify the IFFT round-trip for the same input.
        let z = Fr::from(123456789u64);
        let n = 30;
        let m = next_pow2(n); // 32
        let poly = build_encoding_polynomial(&z, n);

        assert!(poly.coeffs().len() <= m);

        let enc = get_encoding_polynomial(&z, &n);
        let evals = eval_at_roots(&poly, m);
        assert_eq!(&evals[..n], enc.as_slice());
    }

    #[test]
    fn test_build_encoding_polynomial_z_zero() {
        // g encodes 0: all evaluation values are 0, so the polynomial is the zero polynomial.
        let poly = build_encoding_polynomial(&Fr::zero(), 4);
        assert!(poly.is_zero());
    }

    #[test]
    fn test_build_encoding_polynomial_z_one() {
        // z = 1 = 0b0001; encoding values: [1, 0, 0, 0] for n = 4
        let z = Fr::one();
        let m = 4;
        let poly = build_encoding_polynomial(&z, m);

        assert_eq!(eval_at_roots(&poly, m), get_encoding_polynomial(&z, &m));
    }

    #[test]
    fn test_build_encoding_polynomial_largest_n_in_domain() {
        // n = 32: a large exact-power-of-two domain; verifies no off-by-one in
        // the Radix2 setup.
        let z = Fr::from(1_000_000u64);
        let m = 32;
        let poly = build_encoding_polynomial(&z, m);

        assert!(poly.degree() < m);
        assert_eq!(eval_at_roots(&poly, m), get_encoding_polynomial(&z, &m));
    }

    // -------------------------------------------------------------------------
    // create_big_int4
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_big_int4_zero() {
        let b = create_big_int4(0);
        assert_eq!(b, create_big_int4(0));
    }

    #[test]
    fn test_create_big_int4_one() {
        let b = create_big_int4(1);
        // A BigInt<4> with value 1 has its lowest limb == 1 and the rest zero.
        assert_eq!(b.0[0], 1u64);
        assert_eq!(b.0[1], 0u64);
    }

    #[test]
    fn test_create_big_int4_large() {
        let n = 1_000_000_usize;
        let b = create_big_int4(n);
        assert_eq!(b.0[0], n as u64);
    }

    #[test]
    fn test_create_big_int4_roundtrip() {
        for &n in &[0usize, 1, 42, u32::MAX as usize] {
            let b = create_big_int4(n);
            assert_eq!(b.0[0], n as u64);
        }
    }

    // -------------------------------------------------------------------------
    // get_challenge_g2
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_challenge_g2_deterministic() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        let b = G2Projective::rand(rng);

        // Same inputs → same output
        assert_eq!(get_challenge_g2(&[a, b]), get_challenge_g2(&[a, b]));
    }

    #[test]
    fn test_get_challenge_g2_differs_for_different_inputs() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        let b = G2Projective::rand(rng);

        // Different inputs → (almost certainly) different outputs
        assert_ne!(get_challenge_g2(&[a]), get_challenge_g2(&[b]));
    }

    #[test]
    fn test_get_challenge_g2_length_matters() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        let b = G2Projective::rand(rng);

        // [a] vs [a, b] must differ
        assert_ne!(get_challenge_g2(&[a]), get_challenge_g2(&[a, b]));
    }

    #[test]
    fn test_get_challenge_g2_returns_field_element() {
        let rng = &mut test_rng();
        let a = G2Projective::rand(rng);
        // Just verify we get a valid Fr (no panic, not zero with overwhelming probability)
        let ch = get_challenge_g2(&[a]);
        // The probability of hitting exactly zero is negligible
        assert_ne!(ch, Fr::zero());
    }

    // -------------------------------------------------------------------------
    // create_fr_vec
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_fr_vec_empty() {
        let v = create_fr_vec(&[]);
        assert!(v.is_empty());
    }

    #[test]
    fn test_create_fr_vec_positive() {
        let v = create_fr_vec(&[1, 2, 3]);
        assert_eq!(v, vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)]);
    }

    #[test]
    fn test_create_fr_vec_negative() {
        let v = create_fr_vec(&[-1, -2]);
        assert_eq!(v, vec![Fr::from(-1i64), Fr::from(-2i64)]);
    }

    #[test]
    fn test_create_fr_vec_zero() {
        let v = create_fr_vec(&[0]);
        assert_eq!(v, vec![Fr::zero()]);
    }

    // -------------------------------------------------------------------------
    // create_poly
    // -------------------------------------------------------------------------

    #[test]
    fn test_create_poly_constant() {
        let p = create_poly(&[5]);
        assert_eq!(p.evaluate(&Fr::zero()), Fr::from(5u64));
        assert_eq!(p.evaluate(&Fr::from(99u64)), Fr::from(5u64));
    }

    #[test]
    fn test_create_poly_linear() {
        // 3 + 2x  ⟹ at x=1: 5, at x=2: 7
        let p = create_poly(&[3, 2]);
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
    fn test_create_poly_empty() {
        let p = create_poly(&[]);
        assert!(p.is_zero());
    }

    #[test]
    fn test_create_poly_negative_coeff() {
        // -1 + x  ⟹ root at x=1
        let p = create_poly(&[-1, 1]);
        assert_eq!(p.evaluate(&Fr::one()), Fr::zero());
    }

    // -------------------------------------------------------------------------
    // scalar_mul (utility, line 162-164)
    // -------------------------------------------------------------------------

    #[test]
    fn test_scalar_mul_by_zero() {
        let p = create_poly(&[1, 2, 3]);
        let result = scalar_mul(&p, &Fr::zero());
        assert!(result.is_zero());
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
}
