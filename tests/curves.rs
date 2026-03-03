/// Cross-curve tests for the generic `E: Pairing` machinery.
///
/// These tests do *not* duplicate the exhaustive BLS12-381 suite.
/// Their sole purpose is to verify that the generic trait bounds resolve
/// correctly for a second curve — catching any accidental concrete-type
/// leakage or missing bound that BLS12-381 happened to satisfy implicitly.
#[cfg(test)]
mod cross_curve_tests {
    use ark_bls12_381::Bls12_381;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial};
    use ark_poly_commit::kzg10::KZG10;
    use ark_std::test_rng;

    use range_proof::range_proof::*;
    use range_proof::utilities::*;

    // -------------------------------------------------------------------------
    // Generic helpers used by every test below
    // -------------------------------------------------------------------------

    /// Minimal trusted-setup: degree `n` is enough for all tests here.
    fn setup<E: Pairing>(n: usize) -> Params<E> {
        KZG10::<E, DensePolynomial<<E as Pairing>::ScalarField>>::setup(
            n,
            false,
            &mut test_rng(),
        )
        .unwrap()
    }

    /// Field element from a small u64, independent of the concrete field type.
    fn fe<E: Pairing>(v: u64) -> <E as Pairing>::ScalarField {
        <E as Pairing>::ScalarField::from(v)
    }

    fn proof_roundtrip<E: Pairing>() {
        type F<E> = <E as Pairing>::ScalarField;

        let n = 4;
        let pp = setup::<E>(n * 2);
        let roots: Vec<F<E>> = generate_nth_roots_of_unity(n);
        let omega = roots[1];
        let omega_n_minus_1 = roots[n - 1];

        let z = fe::<E>(7);
        let f = DensePolynomial::from_coefficients_slice(&[z]);

        let proof = single_value_proof(&pp, &omega, &z, &f, n)
            .expect("single_value_proof failed");

        // Reconstruct the verifier key the same way the production code does.
        let powers = get_powers_from_params(&pp);
        let (f_com, _) =
            KZGScheme::<E>::commit(&powers, &f, None, None).expect("commit failed");

        let vk = {
            use ark_ec::{AffineRepr, CurveGroup};
            type G1Aff<E> = <<E as Pairing>::G1 as CurveGroup>::Affine;
            ark_poly_commit::kzg10::VerifierKey::<E> {
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
        };

        assert!(
            single_proof_verify(&vk, &f_com, &proof, n, &omega, &omega_n_minus_1).is_ok()
        );
    }

    #[test]
    fn test_proof_roundtrip_bls12_381() {
        proof_roundtrip::<Bls12_381>();
    }

    #[test]
    fn test_proof_roundtrip_bn254() {
        proof_roundtrip::<Bn254>();
    }

    // -------------------------------------------------------------------------
    // Test 2 — prove_min + min_verify end-to-end
    //
    // Exercises the homomorphic commitment subtraction path
    // (`into_group() - into_group()`) inside min_verify for both curves.
    // -------------------------------------------------------------------------

    fn prove_min_verify_roundtrip<E: Pairing>() {
        type F<E> = <E as Pairing>::ScalarField;

        // `build_encoding_polynomial` encodes z in exactly `n` bits, so every
        // z = val - min must satisfy z < 2^n.  The largest z here is 10,
        // which fits in 4 bits (2^4=16 > 10).
        let n = 4;
        let pp = setup::<E>(n * 2);
        let powers = get_powers_from_params(&pp);

        let min = fe::<E>(10);

        // z values are 1, 5, 10 — all < 2^4 = 16.
        let values: Vec<F<E>> = vec![fe::<E>(11), fe::<E>(15), fe::<E>(20)];
        let values_poly: Vec<DensePolynomial<F<E>>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        // Commitments to the original (un-shifted) values.
        let values_com: Vec<Commitment<E>> = values_poly
            .iter()
            .map(|p| {
                KZGScheme::<E>::commit(&powers, p, None, None)
                    .expect("commit failed")
                    .0
            })
            .collect();

        let proofs = prove_min(&pp, &min, &values_poly, &values, n)
            .expect("prove_min failed");

        assert!(
            min_verify(&pp, &min, &values_com, &proofs, n)
                .expect("min_verify returned Err"),
            "min_verify returned false for valid values"
        );
    }

    #[test]
    fn test_prove_min_verify_bls12_381() {
        prove_min_verify_roundtrip::<Bls12_381>();
    }

    #[test]
    fn test_prove_min_verify_bn254() {
        prove_min_verify_roundtrip::<Bn254>();
    }

    // -------------------------------------------------------------------------
    // Test 3 — get_w_caret / compute_w_caret_com consistency
    //
    // Commits to the w_caret polynomial directly and compares against the
    // homomorphic result from compute_w_caret_com.  Any concrete-type leakage
    // in the group arithmetic surfaces here.
    // -------------------------------------------------------------------------

    fn w_caret_com_consistency<E: Pairing>() {
        let n = 4;
        let pp = setup::<E>(n * 2);
        let powers = get_powers_from_params(&pp);

        // rho ≠ 1 (required by compute_w_caret_com).
        let rho = fe::<E>(3);

        let f = DensePolynomial::from_coefficients_slice(&[fe::<E>(5), fe::<E>(2)]);
        let q = DensePolynomial::from_coefficients_slice(&[fe::<E>(1), fe::<E>(4), fe::<E>(2)]);

        let (f_com, _) = KZGScheme::<E>::commit(&powers, &f, None, None).unwrap();
        let (q_com, _) = KZGScheme::<E>::commit(&powers, &q, None, None).unwrap();

        // Direct commitment to the polynomial.
        let w_caret_poly = get_w_caret::<E>(&rho, n, &f, &q);
        let (w_caret_direct, _) =
            KZGScheme::<E>::commit(&powers, &w_caret_poly, None, None).unwrap();

        // Homomorphic commitment from the two input commitments.
        let w_caret_hom =
            compute_w_caret_com(&f_com, &rho, &q_com, n).expect("compute_w_caret_com failed");

        assert_eq!(
            w_caret_direct, w_caret_hom,
            "direct and homomorphic w_caret commitments differ"
        );
    }

    #[test]
    fn test_w_caret_com_consistency_bls12_381() {
        w_caret_com_consistency::<Bls12_381>();
    }

    #[test]
    fn test_w_caret_com_consistency_bn254() {
        w_caret_com_consistency::<Bn254>();
    }

    // -------------------------------------------------------------------------
    // Test 4 — prove_max + max_verify end-to-end
    //
    // Mirrors test 2 for the max direction, covering the symmetric subtraction
    // path (max_com - val_com) inside max_verify.
    // -------------------------------------------------------------------------

    fn prove_max_verify_roundtrip<E: Pairing>() {
        type F<E> = <E as Pairing>::ScalarField;

        // `build_encoding_polynomial` encodes z in exactly `n` bits, so every
        // z = max - val must satisfy z < 2^n.  The largest z here is 40, which
        // needs at least 6 bits; we use n=8 (2^8=256) for comfortable headroom.
        let n = 8;
        let pp = setup::<E>(n * 2);
        let powers = get_powers_from_params(&pp);

        let max = fe::<E>(100);

        // z values are 100-60=40, 100-75=25, 100-100=0 — all < 2^8.
        let values: Vec<F<E>> = vec![fe::<E>(60), fe::<E>(75), fe::<E>(100)];
        let values_poly: Vec<DensePolynomial<F<E>>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let values_com: Vec<Commitment<E>> = values_poly
            .iter()
            .map(|p| {
                KZGScheme::<E>::commit(&powers, p, None, None)
                    .expect("commit failed")
                    .0
            })
            .collect();

        let proofs = prove_max(&pp, &max, &values_poly, &values, n)
            .expect("prove_max failed");

        assert!(
            max_verify(&pp, &max, &values_com, &proofs, n)
                .expect("max_verify returned Err"),
            "max_verify returned false for valid values"
        );
    }

    #[test]
    fn test_prove_max_verify_bls12_381() {
        prove_max_verify_roundtrip::<Bls12_381>();
    }

    #[test]
    fn test_prove_max_verify_bn254() {
        prove_max_verify_roundtrip::<Bn254>();
    }

    fn run_max_verify_perf<E: Pairing>(count: usize, n_bit: usize) {
        use rand::Rng;
        use rayon::prelude::*;
        use std::time::Instant;

        type F<E> = <E as Pairing>::ScalarField;

        if n_bit < usize::BITS as usize {
            assert!(
                (1usize << n_bit) > count,
                "n_bit={n_bit} too small: 2^n_bit={} ≤ count={count}",
                1usize << n_bit,
            );
        }

        let max_u64 = count as u64;
        let max = fe::<E>(max_u64);

        let pp = setup::<E>(n_bit * 2);
        let roots: Vec<F<E>> = generate_nth_roots_of_unity(n_bit);
        let omega = roots[1];

        let values: Vec<F<E>> = (0..count)
            .map(|_| fe::<E>(rand::thread_rng().gen_range(0..max_u64)))
            .collect();

        let powers = get_powers_from_params(&pp);
        let values_com: Vec<Commitment<E>> = values
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                KZGScheme::<E>::commit(&powers, &poly, None, None)
                    .expect("commit failed")
                    .0
            })
            .collect();

        // ---- Prover ----
        let t_prove = Instant::now();
        let proofs: Vec<RangeProof<E>> = values
            .par_iter()
            .map(|v| {
                let z = max - *v;
                let f = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f, n_bit)
                    .expect("single_value_proof failed")
            })
            .collect();
        let prove_elapsed = t_prove.elapsed();

        println!(
            "[{}] prover elapsed on {} values: {:?}",
            std::any::type_name::<E>(),
            count,
            prove_elapsed,
        );
        println!(
            "[{}] proof size: {} bytes",
            std::any::type_name::<E>(),
            size_of_range_proof::<E>(),
        );

        // ---- Verifier ----
        let t_verify = Instant::now();
        let ok = max_verify(&pp, &max, &values_com, &proofs, n_bit)
            .expect("max_verify returned Err");
        let verify_elapsed = t_verify.elapsed();

        println!(
            "[{}] max_verify elapsed on {} values: {:?}",
            std::any::type_name::<E>(),
            count,
            verify_elapsed,
        );

        assert!(ok, "max_verify returned false");
    }

    #[test]
    fn test_max_verify_perf_bls12_381() {
        run_max_verify_perf::<Bls12_381>(50, 16);
    }

    #[test]
    fn test_max_verify_perf_bn254() {
        run_max_verify_perf::<Bn254>(50, 16);
    }

    #[ignore] // This is a smoke test, not a regular unit test.
    #[test]
    fn test_max_verify_perf() {
        run_max_verify_perf::<Bls12_381>(300, 64);
        run_max_verify_perf::<Bls12_381>(300, 64);
        run_max_verify_perf::<Bls12_381>(300, 64);
        run_max_verify_perf::<Bn254>(300, 64);
        run_max_verify_perf::<Bn254>(300, 64);
        run_max_verify_perf::<Bn254>(300, 64);
    }
}