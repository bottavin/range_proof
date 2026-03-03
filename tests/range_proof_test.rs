#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr, G1Affine, G1Projective};
    use ark_ec::{AffineRepr, CurveGroup};
    use ark_ff::{Field, One, Zero};
    use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, Polynomial};
    use ark_poly_commit::kzg10::{Powers, VerifierKey, KZG10};
    use ark_std::{test_rng, UniformRand};
    use rand::rngs::OsRng;
    use range_proof::range_proof::*;
    use range_proof::utilities::*;
    use rayon::prelude::*;

    // -------------------------------------------------------------------------
    // Local type aliases — pin the generic types to Bls12_381 for all tests.
    // -------------------------------------------------------------------------

    /// Concrete KZG scheme used throughout the tests.
    type KZG = KZGScheme<Bls12_381>;

    // -------------------------------------------------------------------------
    // Shared helpers
    // -------------------------------------------------------------------------

    fn setup_params(n: usize) -> Params<Bls12_381> {
        KZG10::<Bls12_381, DensePolynomial<Fr>>::setup(n, false, &mut test_rng()).unwrap()
    }

    /// G1Projective → Commitment<Bls12_381>.
    fn to_commitment(g1: G1Projective) -> Commitment<Bls12_381> {
        ark_poly_commit::kzg10::Commitment(g1.into_affine())
    }

    /// Commitment<Bls12_381> → G1Projective (for assertions on group elements).
    /// Uses `into_group()` from `CurveGroup` to avoid the ambiguous `.into()`.
    fn from_commitment(com: &Commitment<Bls12_381>) -> G1Projective {
        com.0.into_group()
    }

    fn get_verifier_key(pp: &Params<Bls12_381>) -> VerifierKey<Bls12_381> {
        VerifierKey {
            g: pp.powers_of_g[0],
            gamma_g: pp
                .powers_of_gamma_g
                .get(&0)
                .cloned()
                .unwrap_or_else(G1Affine::zero),   // identity() → zero()
            h: pp.h,
            beta_h: pp.beta_h,
            prepared_h: pp.prepared_h.clone(),
            prepared_beta_h: pp.prepared_beta_h.clone(),
        }
    }

    fn valid_proof_and_parts(
        n: usize,
    ) -> (Params<Bls12_381>, Commitment<Bls12_381>, RangeProof<Bls12_381>, Fr, Fr) {
        let pp = setup_params(n * 2);
        let value = Fr::from(5u64);
        let min = Fr::from(0u64);
        let z = value - min;
        let poly = DensePolynomial::from_coefficients_slice(&[z]);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];

        let proof = single_value_proof(&pp, &omega, &z, &poly, n).unwrap();
        let f_com = proof.fcom;

        let omega_n_minus_1 = generate_nth_roots_of_unity::<Fr>(n)[n - 1];
        (pp, f_com, proof, omega, omega_n_minus_1)
    }

    // -------------------------------------------------------------------------
    // get_const_polynomial
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_const_polynomial_basic() {
        let result = get_const_polynomial(&Fr::from(5u64), 3);
        assert_eq!(result.coeffs, vec![Fr::from(5u64)]);
    }

    #[test]
    fn test_get_const_polynomial_n_is_1() {
        let result = get_const_polynomial(&Fr::from(7u64), 1);
        assert_eq!(result.coeffs, vec![Fr::from(7u64)]);
    }

    #[test]
    fn test_get_const_polynomial_zero_constant() {
        // DensePolynomial normalises the zero polynomial to empty coeffs.
        let result = get_const_polynomial(&Fr::zero(), 4);
        assert_eq!(result.coeffs, vec![]);
    }

    #[test]
    fn test_get_const_polynomial_large_n() {
        let result = get_const_polynomial(&Fr::from(3u64), 10);
        assert_eq!(result.coeffs, vec![Fr::from(3u64)]);
    }

    #[test]
    fn test_get_const_polynomial_n_equals_z() {
        let result = get_const_polynomial(&Fr::from(4u64), 4);
        assert_eq!(result.coeffs, vec![Fr::from(4u64)]);
    }

    #[test]
    fn test_get_const_polynomial_large_constant() {
        let result = get_const_polynomial(&Fr::from(100000u64), 5);
        assert_eq!(result.coeffs, vec![Fr::from(100000u64)]);
    }

    // -------------------------------------------------------------------------
    // get_quotient_polynomial
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_quotient_polynomial_n_2() {
        let tau = Fr::from(1u64);
        let omega = Fr::from(2u64);
        let n = 2;
        let f = DensePolynomial::from_coefficients_slice(&[Fr::from(1u64), Fr::from(2u64)]);
        let g = DensePolynomial::from_coefficients_slice(&[Fr::from(3u64), Fr::from(4u64)]);

        let q = get_quotient_polynomial::<Bls12_381>(&tau, &omega, n, &f, &g).unwrap();

        assert_eq!(q.coeffs, vec![Fr::from(154u64), Fr::from(-160i64)]);
    }

    #[test]
    fn test_get_quotient_polynomial_tau_zero() {
        let tau = Fr::zero();
        let omega = Fr::from(3u64);
        let n = 3;
        let f = DensePolynomial::from_coefficients_slice(&[Fr::from(1u64); 3]);
        let g = DensePolynomial::from_coefficients_slice(&[Fr::from(2u64); 3]);

        let q = get_quotient_polynomial::<Bls12_381>(&tau, &omega, n, &f, &g).unwrap();

        assert_eq!(q.coeffs, vec![Fr::from(2u64), Fr::from(1u64)]);
    }

    // -------------------------------------------------------------------------
    // get_polynomial_w1
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_polynomial_w1_basic() {
        let n = 2;
        let f = DensePolynomial::from_coefficients_slice(&[Fr::from(1u64), Fr::from(2u64)]);
        let g = DensePolynomial::from_coefficients_slice(&[Fr::from(3u64), Fr::from(4u64)]);

        let w1 = get_polynomial_w1::<Bls12_381>(n, &f, &g).unwrap();

        assert_eq!(
            w1.coeffs,
            vec![Fr::from(2u64), Fr::from(4u64), Fr::from(2u64)]
        );
    }

    #[test]
    fn test_get_polynomial_w1_n_1() {
        let n = 1;
        let f = DensePolynomial::from_coefficients_slice(&[Fr::from(1u64)]);
        let g = DensePolynomial::from_coefficients_slice(&[Fr::from(2u64)]);

        let w1 = get_polynomial_w1::<Bls12_381>(n, &f, &g).unwrap();

        assert_eq!(w1.coeffs, vec![Fr::one()]);
    }

    #[test]
    fn test_get_polynomial_w1_n_5() {
        let n = 5;
        let f = DensePolynomial::from_coefficients_slice(&[
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(5u64),
        ]);
        let g = DensePolynomial::from_coefficients_slice(&[
            Fr::from(5u64),
            Fr::from(4u64),
            Fr::from(3u64),
            Fr::from(2u64),
            Fr::from(1u64),
        ]);

        let w1 = get_polynomial_w1::<Bls12_381>(n, &f, &g).unwrap();

        let expected = vec![
            Fr::from(4u64),
            Fr::from(6u64),
            Fr::from(6u64),
            Fr::from(4u64),
            Fr::zero(),
            Fr::from(-4i64),
            Fr::from(-6i64),
            Fr::from(-6i64),
            Fr::from(-4i64),
        ];
        assert_eq!(w1.coeffs, expected);
    }

    #[test]
    fn test_get_polynomial_w1_null() {
        let n = 2;
        let f = DensePolynomial::from_coefficients_slice(&[Fr::zero(), Fr::zero()]);
        let g = DensePolynomial::from_coefficients_slice(&[Fr::zero(), Fr::zero()]);

        assert!(get_polynomial_w1::<Bls12_381>(n, &f, &g).unwrap().is_zero());
    }

    // -------------------------------------------------------------------------
    // get_polynomial_w2
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_polynomial_w2_basic() {
        let omega = Fr::from(2u64);
        let n = 3;
        let g = DensePolynomial::from_coefficients_slice(&[
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
        ]);

        let w2 = get_polynomial_w2::<Bls12_381>(&omega, n, &g).unwrap();

        let expected = vec![
            Fr::zero(),
            Fr::from(-32i64),
            Fr::from(-120i64),
            Fr::from(-222i64),
            Fr::from(-199i64),
            Fr::from(-48i64),
            Fr::from(-9i64),
        ];
        assert_eq!(w2.coeffs, expected);
    }

    #[test]
    fn test_get_polynomial_w2_g_zero() {
        let omega = Fr::from(3u64);
        let n = 3;
        let g = DensePolynomial::from_coefficients_slice(&[Fr::zero(); 3]);

        assert!(get_polynomial_w2::<Bls12_381>(&omega, n, &g).unwrap().is_zero());
    }

    #[test]
    fn test_get_polynomial_w2_higher_degree_g() {
        let omega = Fr::from(2u64);
        let n = 4;
        let g = DensePolynomial::from_coefficients_slice(&[
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
        ]);

        let w2 = get_polynomial_w2::<Bls12_381>(&omega, n, &g).unwrap();

        let expected = vec![
            Fr::zero(),
            Fr::from(-1024i64),
            Fr::from(-3712i64),
            Fr::from(-8656i64),
            Fr::from(-13882i64),
            Fr::from(-14023i64),
            Fr::from(-9944i64),
            Fr::from(-1241i64),
            Fr::from(-152i64),
            Fr::from(-16i64),
        ];
        assert_eq!(w2.coeffs, expected);
    }

    // -------------------------------------------------------------------------
    // get_polynomial_w3
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_polynomial_w3_basic() {
        let omega = Fr::from(2u64);
        let n = 3;
        let g = DensePolynomial::from_coefficients_slice(&[
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
        ]);

        let w3 = get_polynomial_w3::<Bls12_381>(&omega, n, &g);

        let expected = vec![
            Fr::from(8u64),
            Fr::from(70u64),
            Fr::from(378u64),
            Fr::from(909u64),
            Fr::from(1512u64),
            Fr::from(-441i64),
        ];
        assert_eq!(w3.coeffs, expected);
    }

    #[test]
    fn test_get_polynomial_w3_g_zero() {
        let omega = Fr::from(3u64);
        let n = 3;
        let g = DensePolynomial::from_coefficients_slice(&[Fr::zero(); 3]);

        assert!(get_polynomial_w3::<Bls12_381>(&omega, n, &g).is_zero());
    }

    #[test]
    fn test_get_polynomial_w3_g_one() {
        let omega = Fr::from(2u64);
        let n = 3;
        let g = DensePolynomial::from_coefficients_slice(&[Fr::from(1u64); 3]);

        let w3 = get_polynomial_w3::<Bls12_381>(&omega, n, &g);

        let expected = vec![
            Fr::from(8u64),
            Fr::from(34u64),
            Fr::from(111u64),
            Fr::from(138u64),
            Fr::from(154u64),
            Fr::from(-49i64),
        ];
        assert_eq!(w3.coeffs, expected);
    }

    #[test]
    fn test_get_polynomial_w3_higher_degree_g() {
        let omega = Fr::from(2u64);
        let n = 4;
        let g = DensePolynomial::from_coefficients_slice(&[
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
        ]);

        let w3 = get_polynomial_w3::<Bls12_381>(&omega, n, &g);

        let expected = vec![
            Fr::from(16u64),
            Fr::from(142u64),
            Fr::from(774u64),
            Fr::from(3357u64),
            Fr::from(8856u64),
            Fr::from(18999u64),
            Fr::from(26280u64),
            Fr::from(-3600i64),
        ];
        assert_eq!(w3.coeffs, expected);
    }

    #[test]
    fn test_get_polynomial_w3_large_n() {
        let omega = Fr::from(5u64);
        let n = 6;
        let g = DensePolynomial::from_coefficients_vec((0..n as u64).map(Fr::from).collect());

        let w3 = get_polynomial_w3::<Bls12_381>(&omega, n, &g);

        let expected = vec![
            Fr::zero(),
            Fr::from(28125u64),
            Fr::from(559366u64),
            Fr::from(7846696u64),
            Fr::from(87641239u64),
            Fr::from(836175079u64),
            Fr::from(6561091790u64),
            Fr::from(42460537865u64),
            Fr::from(223861555706u64),
            Fr::from(975553484954u64),
            Fr::from(3050469128085u64),
            Fr::from(-976250025i64),
        ];
        assert_eq!(w3.coeffs, expected);
    }

    // -------------------------------------------------------------------------
    // get_w_caret
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_w_caret_basic() {
        let ro = Fr::from(2u64);
        let n = 3;
        let f = DensePolynomial::from_coefficients_slice(&[
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
        ]);
        let q = DensePolynomial::from_coefficients_slice(&[
            Fr::from(4u64),
            Fr::from(5u64),
            Fr::from(6u64),
        ]);

        let wc = get_w_caret::<Bls12_381>(&ro, n, &f, &q);

        assert_eq!(
            wc.coeffs,
            vec![Fr::from(35u64), Fr::from(49u64), Fr::from(63u64)]
        );
    }

    #[test]
    fn test_get_w_caret_f_zero() {
        let ro = Fr::from(2u64);
        let n = 3;
        let f = DensePolynomial::zero();
        let q = DensePolynomial::from_coefficients_slice(&[
            Fr::from(4u64),
            Fr::from(5u64),
            Fr::from(6u64),
        ]);

        let wc = get_w_caret::<Bls12_381>(&ro, n, &f, &q);

        assert_eq!(
            wc.coeffs,
            vec![Fr::from(28u64), Fr::from(35u64), Fr::from(42u64)]
        );
    }

    #[test]
    fn test_get_w_caret_q_zero() {
        let ro = Fr::from(2u64);
        let n = 3;
        let f = DensePolynomial::from_coefficients_slice(&[
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
        ]);
        let q = DensePolynomial::zero();

        let wc = get_w_caret::<Bls12_381>(&ro, n, &f, &q);

        assert_eq!(
            wc.coeffs,
            vec![Fr::from(7u64), Fr::from(14u64), Fr::from(21u64)]
        );
    }

    #[test]
    fn test_get_w_caret_q_f_zero() {
        let ro = Fr::from(2u64);
        let n = 3;

        assert!(
            get_w_caret::<Bls12_381>(&ro, n, &DensePolynomial::zero(), &DensePolynomial::zero())
                .is_zero()
        );
    }

    #[test]
    fn test_get_w_caret_constant_polynomials() {
        let ro = Fr::from(2u64);
        let n = 3;
        let f = DensePolynomial::from_coefficients_slice(&[Fr::from(1u64); 3]);
        let q = DensePolynomial::from_coefficients_slice(&[Fr::from(2u64); 3]);

        let wc = get_w_caret::<Bls12_381>(&ro, n, &f, &q);

        assert_eq!(wc.coeffs, vec![Fr::from(21u64); 3]);
    }

    #[test]
    fn test_get_w_caret_large_n() {
        let ro = Fr::from(3u64);
        let n = 6;
        let f = DensePolynomial::from_coefficients_vec((0..n as u64).map(Fr::from).collect());
        let q = DensePolynomial::from_coefficients_vec((1..=n as u64).map(Fr::from).collect());

        let wc = get_w_caret::<Bls12_381>(&ro, n, &f, &q);

        let expected = vec![
            Fr::from(728u64),
            Fr::from(1820u64),
            Fr::from(2912u64),
            Fr::from(4004u64),
            Fr::from(5096u64),
            Fr::from(6188u64),
        ];
        assert_eq!(wc.coeffs, expected);
    }

    // -------------------------------------------------------------------------
    // compute_w_caret_com
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_w_caret_com_basic() {
        let rng = &mut test_rng();
        let f_com = to_commitment(G1Projective::rand(rng));
        let q_com = to_commitment(G1Projective::rand(rng));

        let result = compute_w_caret_com(&f_com, &Fr::from(2u64), &q_com, 3).unwrap();

        assert!(!from_commitment(&result).is_zero());
    }

    #[test]
    fn test_compute_w_caret_com_rho_equals_1() {
        let rng = &mut test_rng();
        let f_com = to_commitment(G1Projective::rand(rng));
        let q_com = to_commitment(G1Projective::rand(rng));

        let result = compute_w_caret_com(&f_com, &Fr::one(), &q_com, 4);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("rho must not equal 1"));
    }

    #[test]
    fn test_compute_w_caret_com_n_equals_0() {
        let rng = &mut test_rng();
        let f_com = to_commitment(G1Projective::rand(rng));
        let q_com = to_commitment(G1Projective::rand(rng));

        // ρ^0 − 1 = 0, so result should be the identity.
        let result = compute_w_caret_com(&f_com, &Fr::from(2u64), &q_com, 0).unwrap();

        assert!(from_commitment(&result).is_zero());
    }

    #[test]
    fn test_compute_w_caret_com_n_equals_1() {
        let rng = &mut test_rng();
        let f_com = to_commitment(G1Projective::rand(rng));
        // q_com = identity → contributes nothing.
        let q_com = to_commitment(G1Projective::zero());

        // With n = 1 and ρ ≠ 1:  (ρ−1)/(ρ−1) = 1, so w_caret_com = f_com.
        let result = compute_w_caret_com(&f_com, &Fr::from(2u64), &q_com, 1).unwrap();

        assert_eq!(from_commitment(&result), from_commitment(&f_com));
    }

    #[test]
    fn test_compute_w_caret_com_large_n() {
        let rng = &mut test_rng();
        let f_com = to_commitment(G1Projective::rand(rng));
        let q_com = to_commitment(G1Projective::rand(rng));

        let result = compute_w_caret_com(&f_com, &Fr::from(2u64), &q_com, 1_000_000).unwrap();

        assert!(!from_commitment(&result).is_zero());
    }

    /// The commitment to w_caret computed linearly must equal KZG::commit on the
    /// polynomial itself.
    #[test]
    fn test_compute_w_caret_com_matches_polynomial_commit() {
        let rho = Fr::from(2u64);
        let n = 4;
        let pp = setup_params(n * 2);
        let powers = get_powers_from_params(&pp);

        let f = DensePolynomial::from_coefficients_slice(&[Fr::from(4u64), Fr::zero(), Fr::zero()]);
        let q = DensePolynomial::from_coefficients_slice(&[
            Fr::from(4u64),
            Fr::from(2u64),
            Fr::from(1u64),
        ]);

        let (f_com, _) = KZG::commit(&powers, &f, None, None).unwrap();
        let (q_com, _) = KZG::commit(&powers, &q, None, None).unwrap();

        let w_caret = get_w_caret::<Bls12_381>(&rho, n, &f, &q);
        let (expected, _) = KZG::commit(&powers, &w_caret, None, None).unwrap();

        let result = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_compute_w_caret_com_consistent_with_get_w_caret() {
        let n = 8;
        let pp = setup_params(n * 2);
        let powers = get_powers_from_params(&pp);
        let rho = Fr::from(3u64);

        let f = DensePolynomial::from_coefficients_slice(&[Fr::from(5u64)]);
        let q = DensePolynomial::from_coefficients_slice(&[Fr::from(2u64), Fr::from(1u64)]);

        let (f_com, _) =
            KZG10::<Bls12_381, DensePolynomial<Fr>>::commit(&powers, &f, None, None).unwrap();
        let (q_com, _) =
            KZG10::<Bls12_381, DensePolynomial<Fr>>::commit(&powers, &q, None, None).unwrap();

        let w_caret_poly = get_w_caret::<Bls12_381>(&rho, n, &f, &q);
        let (w_caret_com_direct, _) =
            KZG10::<Bls12_381, DensePolynomial<Fr>>::commit(&powers, &w_caret_poly, None, None)
                .unwrap();
        let w_caret_com_computed = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();

        assert_eq!(w_caret_com_direct, w_caret_com_computed);
    }

    // -------------------------------------------------------------------------
    // single_value_proof — helpers
    // -------------------------------------------------------------------------

    /// Re-derives every intermediate value that `single_value_proof` computes
    /// internally, so tests can assert on individual `RangeProof` fields.
    ///
    /// **`n` must be a power of two.**
    fn derive_proof_components(
        powers: &Powers<'_, Bls12_381>,
        omega: &Fr,
        z: &Fr,
        f: &DensePolynomial<Fr>,
        n: usize,
    ) -> (
        Commitment<Bls12_381>,   // f_com
        Commitment<Bls12_381>,   // g_com
        Commitment<Bls12_381>,   // q_com
        Fr,                      // rho
        DensePolynomial<Fr>,     // w_caret
        DensePolynomial<Fr>,     // g
    ) {
        let g = build_encoding_polynomial(z, n);

        let (f_com, _) = KZG::commit(powers, f, None, None).unwrap();
        let (g_com, _) = KZG::commit(powers, &g, None, None).unwrap();

        let tau = get_challenge_from_coms(&[f_com, g_com]);
        let q = get_quotient_polynomial::<Bls12_381>(&tau, omega, n, f, &g).unwrap();
        let (q_com, _) = KZG::commit(powers, &q, None, None).unwrap();

        let rho = get_challenge_from_coms(&[f_com, g_com, q_com]);
        let w_caret = get_w_caret::<Bls12_381>(&rho, n, f, &q);

        (f_com, g_com, q_com, rho, w_caret, g)
    }

    // -------------------------------------------------------------------------
    // single_value_proof — tests
    // All `n` values must be powers of two (build_encoding_polynomial uses FFT).
    // -------------------------------------------------------------------------

    #[test]
    fn test_single_value_proof_basic() {
        let n = 8; // was 5 – not a power of two
        let pp = setup_params(n * 2);
        let powers = get_powers_from_params(&pp);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let z = Fr::rand(&mut OsRng);
        let f =
            DensePolynomial::from_coefficients_vec((0..n).map(|_| Fr::rand(&mut OsRng)).collect());

        let (f_com, g_com, q_com, rho, w_caret, g) =
            derive_proof_components(&powers, &omega, &z, &f, n);
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        assert_eq!(proof.fcom, f_com);
        assert_eq!(proof.gcom, g_com);
        assert_eq!(proof.qcom, q_com);
        assert_eq!(proof.g_rho_eval, g.evaluate(&rho));
        assert_eq!(proof.g_rho_omega_eval, g.evaluate(&(rho * omega)));
        assert_eq!(proof.w_caret_rho_eval, w_caret.evaluate(&rho));

        let vk = get_verifier_key(&pp);
        let w_caret_com = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();
        assert!(
            KZG::check(&vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof).is_ok(),
            "g_rho_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &proof.gcom,
                rho * omega,
                proof.g_rho_omega_eval,
                &proof.g_rho_omega_proof
            )
            .is_ok(),
            "g_rho_omega_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &w_caret_com,
                rho,
                proof.w_caret_rho_eval,
                &proof.w_caret_rho_proof
            )
            .is_ok(),
            "w_caret_rho_proof failed KZG::check"
        );
    }

    #[test]
    fn test_single_value_proof_with_zeroes() {
        let n = 4; // was 5 – not a power of two
        let pp = setup_params(n * 2);
        let powers = get_powers_from_params(&pp);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let z = Fr::zero();
        let f = DensePolynomial::zero();

        let (f_com, g_com, q_com, rho, w_caret, g) =
            derive_proof_components(&powers, &omega, &z, &f, n);
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        assert_eq!(proof.fcom, f_com);
        assert_eq!(proof.gcom, g_com);
        assert_eq!(proof.qcom, q_com);
        assert_eq!(proof.g_rho_eval, g.evaluate(&rho));
        assert_eq!(proof.g_rho_omega_eval, g.evaluate(&(rho * omega)));
        assert_eq!(proof.w_caret_rho_eval, w_caret.evaluate(&rho));

        let vk = get_verifier_key(&pp);
        let w_caret_com = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();
        assert!(
            KZG::check(&vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof).is_ok(),
            "g_rho_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &proof.gcom,
                rho * omega,
                proof.g_rho_omega_eval,
                &proof.g_rho_omega_proof
            )
            .is_ok(),
            "g_rho_omega_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &w_caret_com,
                rho,
                proof.w_caret_rho_eval,
                &proof.w_caret_rho_proof
            )
            .is_ok(),
            "w_caret_rho_proof failed KZG::check"
        );
    }

    #[test]
    fn test_single_value_proof_large_input() {
        let n = 128; // was 100 – not a power of two
        let pp = setup_params(256);
        let powers = get_powers_from_params(&pp);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let z = Fr::rand(&mut OsRng);
        let f =
            DensePolynomial::from_coefficients_vec((0..n).map(|_| Fr::rand(&mut OsRng)).collect());

        let (f_com, g_com, q_com, rho, w_caret, g) =
            derive_proof_components(&powers, &omega, &z, &f, n);
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        assert_eq!(proof.fcom, f_com);
        assert_eq!(proof.gcom, g_com);
        assert_eq!(proof.qcom, q_com);
        assert_eq!(proof.g_rho_eval, g.evaluate(&rho));
        assert_eq!(proof.g_rho_omega_eval, g.evaluate(&(rho * omega)));
        assert_eq!(proof.w_caret_rho_eval, w_caret.evaluate(&rho));

        let vk = get_verifier_key(&pp);
        let w_caret_com = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();
        assert!(
            KZG::check(&vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof).is_ok(),
            "g_rho_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &proof.gcom,
                rho * omega,
                proof.g_rho_omega_eval,
                &proof.g_rho_omega_proof
            )
            .is_ok(),
            "g_rho_omega_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &w_caret_com,
                rho,
                proof.w_caret_rho_eval,
                &proof.w_caret_rho_proof
            )
            .is_ok(),
            "w_caret_rho_proof failed KZG::check"
        );
    }

    #[test]
    fn test_single_value_proof_valid() {
        let n = 8;
        let pp = setup_params(128);
        let powers = get_powers_from_params(&pp);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let z = Fr::rand(&mut OsRng);
        let f = DensePolynomial::from_coefficients_vec(vec![Fr::rand(&mut OsRng); n]);

        let (f_com, g_com, q_com, rho, w_caret, g) =
            derive_proof_components(&powers, &omega, &z, &f, n);
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        assert_eq!(proof.fcom, f_com);
        assert_eq!(proof.gcom, g_com);
        assert_eq!(proof.qcom, q_com);
        assert_eq!(proof.g_rho_eval, g.evaluate(&rho));
        assert_eq!(proof.g_rho_omega_eval, g.evaluate(&(rho * omega)));
        assert_eq!(proof.w_caret_rho_eval, w_caret.evaluate(&rho));

        let vk = get_verifier_key(&pp);
        let w_caret_com = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();
        assert!(
            KZG::check(&vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof).is_ok(),
            "g_rho_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &proof.gcom,
                rho * omega,
                proof.g_rho_omega_eval,
                &proof.g_rho_omega_proof
            )
            .is_ok(),
            "g_rho_omega_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &w_caret_com,
                rho,
                proof.w_caret_rho_eval,
                &proof.w_caret_rho_proof
            )
            .is_ok(),
            "w_caret_rho_proof failed KZG::check"
        );
    }

    #[test]
    fn test_single_value_proof_empty_polynomial() {
        let n = 4;
        let pp = setup_params(n * 2);
        let powers = get_powers_from_params(&pp);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let z = Fr::rand(&mut OsRng);
        let f = DensePolynomial::zero();

        let (f_com, g_com, q_com, rho, w_caret, g) =
            derive_proof_components(&powers, &omega, &z, &f, n);
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        assert_eq!(proof.fcom, f_com);
        assert_eq!(proof.gcom, g_com);
        assert_eq!(proof.qcom, q_com);
        assert_eq!(proof.g_rho_eval, g.evaluate(&rho));
        assert_eq!(proof.g_rho_omega_eval, g.evaluate(&(rho * omega)));
        assert_eq!(proof.w_caret_rho_eval, w_caret.evaluate(&rho));

        let vk = get_verifier_key(&pp);
        let w_caret_com = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();
        assert!(
            KZG::check(&vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof).is_ok(),
            "g_rho_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &proof.gcom,
                rho * omega,
                proof.g_rho_omega_eval,
                &proof.g_rho_omega_proof
            )
            .is_ok(),
            "g_rho_omega_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &w_caret_com,
                rho,
                proof.w_caret_rho_eval,
                &proof.w_caret_rho_proof
            )
            .is_ok(),
            "w_caret_rho_proof failed KZG::check"
        );
    }

    #[test]
    fn test_single_value_proof_consistency() {
        // Calling proof twice on the same input must produce identical results
        // because the challenge derivation is purely deterministic.
        let n = 8;
        let pp = setup_params(128);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let z = Fr::rand(&mut OsRng);
        let f = DensePolynomial::from_coefficients_vec(vec![Fr::rand(&mut OsRng); n]);

        let proof1 = single_value_proof(&pp, &omega, &z, &f, n).unwrap();
        let proof2 = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        assert_eq!(proof1.fcom, proof2.fcom);
        assert_eq!(proof1.gcom, proof2.gcom);
        assert_eq!(proof1.qcom, proof2.qcom);
        assert_eq!(proof1.g_rho_proof, proof2.g_rho_proof);
        assert_eq!(proof1.g_rho_omega_proof, proof2.g_rho_omega_proof);
        assert_eq!(proof1.w_caret_rho_proof, proof2.w_caret_rho_proof);

        let vk = get_verifier_key(&pp);
        let rho = get_challenge_from_coms(&[proof1.fcom, proof1.gcom, proof1.qcom]);
        let w_caret_com = compute_w_caret_com(&proof1.fcom, &rho, &proof1.qcom, n).unwrap();
        assert!(
            KZG::check(
                &vk,
                &proof1.gcom,
                rho,
                proof1.g_rho_eval,
                &proof1.g_rho_proof
            )
            .is_ok(),
            "proof1 g_rho_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &proof1.gcom,
                rho * omega,
                proof1.g_rho_omega_eval,
                &proof1.g_rho_omega_proof
            )
            .is_ok(),
            "proof1 g_rho_omega_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &w_caret_com,
                rho,
                proof1.w_caret_rho_eval,
                &proof1.w_caret_rho_proof
            )
            .is_ok(),
            "proof1 w_caret_rho_proof failed KZG::check"
        );
    }

    #[test]
    fn test_get_challenge_from_coms_deterministic() {
        let n = 4;
        let pp = setup_params(n * 2);
        let powers = get_powers_from_params(&pp);
        let poly = DensePolynomial::from_coefficients_slice(&[Fr::from(7u64)]);
        let (com, _) =
            KZG10::<Bls12_381, DensePolynomial<Fr>>::commit(&powers, &poly, None, None).unwrap();

        let c1: Fr = get_challenge_from_coms(&[com]);
        let c2: Fr = get_challenge_from_coms(&[com]);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_get_challenge_from_coms_varies_with_input() {
        let rng = &mut test_rng();
        let com_a = to_commitment(G1Projective::rand(rng));
        let com_b = to_commitment(G1Projective::rand(rng));
        assert_ne!(
            get_challenge_from_coms::<Bls12_381>(&[com_a]),
            get_challenge_from_coms::<Bls12_381>(&[com_b])
        );
    }

    #[test]
    fn test_single_value_proof_large_range() {
        let n = 128;
        let pp = setup_params(n * 2);
        let powers = get_powers_from_params(&pp);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let z = Fr::rand(&mut OsRng);
        let f = DensePolynomial::from_coefficients_vec(vec![Fr::rand(&mut OsRng); n]);

        let (f_com, g_com, q_com, rho, w_caret, g) =
            derive_proof_components(&powers, &omega, &z, &f, n);
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        assert_eq!(proof.fcom, f_com);
        assert_eq!(proof.gcom, g_com);
        assert_eq!(proof.qcom, q_com);
        assert_eq!(proof.g_rho_eval, g.evaluate(&rho));
        assert_eq!(proof.g_rho_omega_eval, g.evaluate(&(rho * omega)));
        assert_eq!(proof.w_caret_rho_eval, w_caret.evaluate(&rho));

        let vk = get_verifier_key(&pp);
        let w_caret_com = compute_w_caret_com(&f_com, &rho, &q_com, n).unwrap();
        assert!(
            KZG::check(&vk, &proof.gcom, rho, proof.g_rho_eval, &proof.g_rho_proof).is_ok(),
            "g_rho_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &proof.gcom,
                rho * omega,
                proof.g_rho_omega_eval,
                &proof.g_rho_omega_proof
            )
            .is_ok(),
            "g_rho_omega_proof failed KZG::check"
        );
        assert!(
            KZG::check(
                &vk,
                &w_caret_com,
                rho,
                proof.w_caret_rho_eval,
                &proof.w_caret_rho_proof
            )
            .is_ok(),
            "w_caret_rho_proof failed KZG::check"
        );
    }

    // -------------------------------------------------------------------------
    // prove_min
    // -------------------------------------------------------------------------

    #[test]
    fn test_prove_min_basic() {
        let n = 4;
        let pp = setup_params(8);
        let min = Fr::from(10u64);
        let values = vec![Fr::from(15u64), Fr::from(20u64), Fr::from(25u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let proofs = prove_min(&pp, &min, &values_poly, &values, n).unwrap();

        assert_eq!(proofs.len(), 3);
        for p in &proofs {
            assert!(!from_commitment(&p.fcom).is_zero());
            assert!(!from_commitment(&p.gcom).is_zero());
            assert!(!from_commitment(&p.qcom).is_zero());
        }
    }

    #[test]
    fn test_prove_min_invalid_n() {
        let pp = setup_params(8);
        let min = Fr::from(10u64);
        let values = vec![Fr::from(15u64), Fr::from(20u64), Fr::from(25u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let result = prove_min(&pp, &min, &values_poly, &values, 5);

        assert!(result.is_err(), "Expected an error for non-power-of-two n");

        let err = result.unwrap_err();
        let err_string = err.to_string();

        assert!(
            err_string.contains("n must be a power of two")
                || err_string.contains("TooManyCoefficients")
                || err_string.contains("commit q failed")
                || err_string.contains("division failed")
                || err_string.contains("prove_min["),
            "Expected error about invalid parameters, got: {}",
            err_string
        );
    }

    #[test]
    fn test_prove_min_valid_n_powers_of_two() {
        let pp = setup_params(16);
        let min = Fr::from(10u64);
        let values = vec![Fr::from(15u64), Fr::from(20u64), Fr::from(25u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        for &n in &[2usize, 4, 8] {
            let proofs = prove_min(&pp, &min, &values_poly, &values, n).unwrap();
            assert_eq!(proofs.len(), values.len());
        }
    }

    #[test]
    fn test_prove_min_large_n() {
        let n = 128;
        let pp = setup_params(n * 2);
        let min = Fr::from(10u64);
        let values: Vec<Fr> = (0..n).map(|i| Fr::from(i as u64 + 10)).collect();
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let proofs = prove_min(&pp, &min, &values_poly, &values, n).unwrap();

        assert_eq!(proofs.len(), n);

        for (i, p) in proofs.iter().enumerate() {
            if i == 0 {
                assert!(
                    from_commitment(&p.fcom).is_zero(),
                    "fcom should be zero when value equals min at index {}",
                    i
                );
            } else {
                assert!(
                    !from_commitment(&p.fcom).is_zero(),
                    "fcom should be non-zero when value > min at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_prove_min_all_values_less_than_min() {
        let n = 4;
        let pp = setup_params(16);
        let min = Fr::from(10u64);
        let values = vec![Fr::from(5u64), Fr::from(7u64), Fr::from(9u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let proofs = prove_min(&pp, &min, &values_poly, &values, n).unwrap();

        assert_eq!(proofs.len(), 3);

        let vk = get_verifier_key(&pp);
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let omega_n_minus_1 = roots[n - 1];

        let values_com: Vec<Commitment<Bls12_381>> = values
            .iter()
            .map(|&v| {
                let poly = DensePolynomial::from_coefficients_slice(&[v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let min_poly = get_const_polynomial(&min, n);
        let (min_com, _) =
            KZG::commit(&get_powers_from_params(&pp), &min_poly, None, None).unwrap();

        for (i, (val_com, proof)) in values_com.iter().zip(proofs.iter()).enumerate() {
            // Compute f_com = val_com - min_com in projective space.
            let f_com = to_commitment(val_com.0.into_group() - min_com.0.into_group());

            let result = single_proof_verify(&vk, &f_com, proof, n, &omega, &omega_n_minus_1);

            assert!(
                result.is_err(),
                "Proof for value {} should NOT verify (value < min)",
                values[i]
            );
        }
    }

    #[test]
    fn test_prove_min_empty_values() {
        let pp = setup_params(16);
        let min = Fr::from(10u64);

        let proofs = prove_min(&pp, &min, &[], &[], 4).unwrap();

        assert_eq!(proofs.len(), 0);
    }

    #[test]
    fn test_prove_min_and_verify() {
        let n = 8;
        let pp = setup_params(n * 2);
        let min = Fr::from(10u64);
        let values = vec![Fr::from(10u64), Fr::from(15u64), Fr::from(20u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|v| DensePolynomial::from_coefficients_slice(&[*v]))
            .collect();

        let powers = get_powers_from_params(&pp);
        let values_com: Vec<Commitment<Bls12_381>> = values_poly
            .iter()
            .map(|p| {
                KZG10::<Bls12_381, DensePolynomial<Fr>>::commit(&powers, p, None, None)
                    .unwrap()
                    .0
            })
            .collect();

        let proofs = prove_min(&pp, &min, &values_poly, &values, n).unwrap();
        assert!(min_verify(&pp, &min, &values_com, &proofs, n).unwrap());
    }

    // -------------------------------------------------------------------------
    // prove_max
    // -------------------------------------------------------------------------

    #[test]
    fn test_basic_proof() {
        let n = 2;
        let pp = setup_params(8);
        let max = Fr::from(10u64);
        let values = vec![Fr::from(5u64), Fr::from(8u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let proofs = prove_max(&pp, &max, &values_poly, &values, n).unwrap();

        assert_eq!(proofs.len(), n);
        for p in &proofs {
            assert!(!from_commitment(&p.fcom).is_zero());
            assert!(!from_commitment(&p.gcom).is_zero());
            assert!(!from_commitment(&p.qcom).is_zero());
        }
    }

    #[test]
    fn test_max_less_than_values() {
        let n = 2;
        let pp = setup_params(8);
        let max = Fr::from(3u64);
        let values = vec![Fr::from(5u64), Fr::from(8u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let proofs = prove_max(&pp, &max, &values_poly, &values, n).unwrap();

        assert_eq!(proofs.len(), 2);

        let vk = get_verifier_key(&pp);
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let omega_n_minus_1 = roots[n - 1];

        let values_com: Vec<Commitment<Bls12_381>> = values
            .iter()
            .map(|&v| {
                let poly = DensePolynomial::from_coefficients_slice(&[v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let max_poly = get_const_polynomial(&max, n);
        let (max_com, _) =
            KZG::commit(&get_powers_from_params(&pp), &max_poly, None, None).unwrap();

        for (val_com, proof) in values_com.iter().zip(proofs.iter()) {
            // Compute f_com = max_com - val_com in projective space.
            let f_com = to_commitment(max_com.0.into_group() - val_com.0.into_group());

            let result = single_proof_verify(&vk, &f_com, proof, n, &omega, &omega_n_minus_1);
            assert!(result.is_err(), "Proof should NOT verify for out-of-range values");
        }
    }

    #[test]
    fn test_max_equals_value() {
        let pp = setup_params(8);
        let max = Fr::from(5u64);
        let values = vec![Fr::from(5u64), Fr::from(8u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let proofs = prove_max(&pp, &max, &values_poly, &values, 2).unwrap();

        assert_eq!(proofs.len(), 2);

        assert!(
            from_commitment(&proofs[0].fcom).is_zero(),
            "fcom should be zero when value equals max"
        );
        assert!(
            from_commitment(&proofs[0].gcom).is_zero(),
            "gcom should be zero when value equals max"
        );

        assert!(
            !from_commitment(&proofs[1].fcom).is_zero(),
            "fcom should be non-zero when value < max"
        );
        assert!(
            !from_commitment(&proofs[1].gcom).is_zero(),
            "gcom should be non-zero when value < max"
        );
    }

    #[test]
    fn test_empty_values_vector() {
        let pp = setup_params(8);
        let max = Fr::from(10u64);

        let proofs = prove_max(&pp, &max, &[], &[], 2).unwrap();

        assert!(proofs.is_empty());
    }

    #[test]
    fn test_large_degree_polynomial() {
        let n = 16;
        let pp = setup_params(n * 2);
        let max = Fr::from(100u64);
        let values = vec![Fr::from(50u64), Fr::from(60u64), Fr::from(90u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|&v| DensePolynomial::from_coefficients_slice(&[v]))
            .collect();

        let proofs = prove_max(&pp, &max, &values_poly, &values, n).unwrap();

        assert_eq!(proofs.len(), 3);
        for p in &proofs {
            assert!(!from_commitment(&p.fcom).is_zero());
            assert!(!from_commitment(&p.gcom).is_zero());
            assert!(!from_commitment(&p.qcom).is_zero());
        }
    }

    #[test]
    fn test_invalid_polynomial_size() {
        // values has 2 entries but values_poly has only 1 → index out of bounds.
        let n = 2;
        let pp = setup_params(n * 2);
        let max = Fr::from(10u64);
        let values_poly = vec![DensePolynomial::from_coefficients_slice(&[Fr::from(5u64)])];
        let values = vec![Fr::from(5u64), Fr::from(8u64)];

        let result = std::panic::catch_unwind(|| {
            let _ = prove_max(&pp, &max, &values_poly, &values, n).unwrap();
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_prove_max_and_verify() {
        let n = 8;
        let pp = setup_params(n * 2);
        let max = Fr::from(100u64);
        let values = vec![Fr::from(50u64), Fr::from(75u64), Fr::from(100u64)];
        let values_poly: Vec<DensePolynomial<Fr>> = values
            .iter()
            .map(|v| DensePolynomial::from_coefficients_slice(&[*v]))
            .collect();

        let powers = get_powers_from_params(&pp);
        let values_com: Vec<Commitment<Bls12_381>> = values_poly
            .iter()
            .map(|p| {
                KZG10::<Bls12_381, DensePolynomial<Fr>>::commit(&powers, p, None, None)
                    .unwrap()
                    .0
            })
            .collect();

        let proofs = prove_max(&pp, &max, &values_poly, &values, n).unwrap();
        assert!(max_verify(&pp, &max, &values_com, &proofs, n).unwrap());
    }

    // -------------------------------------------------------------------------
    // eval_w
    // -------------------------------------------------------------------------

    /// Build a dummy RangeProof<Bls12_381> with given scalar evaluations.
    fn dummy_proof(g_rho: Fr, g_rho_omega: Fr, w_caret_rho: Fr) -> RangeProof<Bls12_381> {
        let default_w: G1Affine = G1Projective::default().into_affine();
        RangeProof {
            fcom: to_commitment(G1Projective::default()),
            gcom: to_commitment(G1Projective::default()),
            qcom: to_commitment(G1Projective::default()),
            g_rho_proof: Proof { w: default_w, random_v: None },
            g_rho_eval: g_rho,
            g_rho_omega_proof: Proof { w: default_w, random_v: None },
            g_rho_omega_eval: g_rho_omega,
            w_caret_rho_proof: Proof { w: default_w, random_v: None },
            w_caret_rho_eval: w_caret_rho,
        }
    }

    #[test]
    fn test_eval_w_basic() {
        let n = 3;
        let omega = Fr::from(3u64);
        let omega_n_minus_1 = omega.pow([(n - 1) as u64]);
        let tau = Fr::from(3u64);
        let rho = Fr::from(2u64);

        let proof = dummy_proof(Fr::from(17u64), Fr::from(61u64), Fr::from(702125u64));

        assert_eq!(eval_w(&proof, &rho, &tau, &omega_n_minus_1, n), Fr::zero());
    }

    #[test]
    fn test_eval_w_zero_tau() {
        // With tau = 0 only the w1 term survives.
        let n = 5;
        let omega = Fr::from(3u64);
        let omega_n_minus_1 = omega.pow([(n - 1) as u64]);
        let rho = Fr::from(2u64);

        let proof = dummy_proof(Fr::from(17u64), Fr::from(61u64), Fr::from(527u64));

        assert_eq!(
            eval_w(&proof, &rho, &Fr::zero(), &omega_n_minus_1, n),
            Fr::zero()
        );
    }

    #[test]
    fn test_eval_w_large_n() {
        let n = 16;
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let z = Fr::from(7u64);
        let omega: Fr = roots[1];
        let omega_n_minus_1 = omega.pow([(n - 1) as u64]);
        let tau = Fr::from(3u64);
        let rho = Fr::from(2u64);

        let mut f_coeffs = vec![Fr::zero(); n];
        f_coeffs[0] = z;
        let f = DensePolynomial::from_coefficients_vec(f_coeffs);
        let g = build_encoding_polynomial(&z, n);
        let q = get_quotient_polynomial::<Bls12_381>(&tau, &omega, n, &f, &g).unwrap();
        let w_caret = get_w_caret::<Bls12_381>(&rho, n, &f, &q);

        let proof = dummy_proof(
            g.evaluate(&rho),
            g.evaluate(&(rho * omega)),
            w_caret.evaluate(&rho),
        );

        assert_eq!(eval_w(&proof, &rho, &tau, &omega_n_minus_1, n), Fr::zero());
    }

    #[test]
    fn test_eval_w_returns_zero_for_valid_proof() {
        let n = 8;
        let (_, _, proof, _, omega_n_minus_1) = valid_proof_and_parts(n);

        let rho = get_challenge_from_coms(&[proof.fcom, proof.gcom, proof.qcom]);
        let tau = get_challenge_from_coms(&[proof.fcom, proof.gcom]);

        assert_eq!(eval_w(&proof, &rho, &tau, &omega_n_minus_1, n), Fr::zero());
    }

    // -------------------------------------------------------------------------
    // single_proof_verify
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_proof_verification() {
        let n = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::from(13u64);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_commitment() {
        let n = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::rand(&mut OsRng);
        let f = get_const_polynomial(&z, n);
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();
        let f_com = to_commitment(G1Projective::default());

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert_eq!(result, Err("fcom"));
    }

    #[test]
    fn test_invalid_g_rho_proof() {
        let n = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::rand(&mut OsRng);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let mut proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        proof.g_rho_proof = Proof {
            w: G1Projective::default().into_affine(),
            random_v: None,
        };
        proof.g_rho_eval = Fr::rand(&mut OsRng);

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert_eq!(result, Err("g_rho_proof"));
    }

    #[test]
    fn test_invalid_g_rho_omega_proof() {
        let n = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::rand(&mut OsRng);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let mut proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        proof.g_rho_omega_proof = Proof {
            w: G1Projective::default().into_affine(),
            random_v: None,
        };
        proof.g_rho_omega_eval = Fr::rand(&mut OsRng);

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert_eq!(result, Err("g_rho_omega_proof"));
    }

    #[test]
    fn test_invalid_w_caret_rho_proof() {
        let n = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::rand(&mut OsRng);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let mut proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        proof.w_caret_rho_proof = Proof {
            w: G1Projective::default().into_affine(),
            random_v: None,
        };
        proof.w_caret_rho_eval = Fr::rand(&mut OsRng);

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert_eq!(result, Err("w_caret_rho_proof"));
    }

    #[test]
    fn test_invalid_fcom() {
        let n = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::rand(&mut OsRng);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let mut proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        proof.fcom = to_commitment(G1Projective::default());

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert_eq!(result, Err("fcom"));
    }

    #[test]
    fn test_invalid_gcom() {
        let n = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::rand(&mut OsRng);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let mut proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        proof.gcom = to_commitment(G1Projective::default());

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert_eq!(result, Err("g_rho_proof"));
    }

    #[test]
    fn test_invalid_qcom() {
        let n: usize = 4;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::rand(&mut OsRng);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        let result = single_proof_verify(&vk, &f_com, &proof, 12, &omega, &roots[n - 1]);

        assert_eq!(result, Err("w_caret_rho_proof"));
    }

    #[test]
    fn test_edge_case_n_equals_2() {
        let n: usize = 2;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::from(3u64);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_large_n() {
        let n: usize = 128;
        let pp = setup_params(n * 2);
        let vk = get_verifier_key(&pp);
        let z = Fr::from(7u64);
        let f = get_const_polynomial(&z, n);
        let (f_com, _) = KZG::commit(&get_powers_from_params(&pp), &f, None, None).unwrap();
        let roots = generate_nth_roots_of_unity::<Fr>(n);
        let omega = roots[1];
        let proof = single_value_proof(&pp, &omega, &z, &f, n).unwrap();

        let result = single_proof_verify(&vk, &f_com, &proof, n, &omega, &roots[n - 1]);

        assert!(result.is_ok());
    }

    #[test]
    fn test_single_proof_verify_valid() {
        let n = 8;
        let (pp, f_com, proof, omega, omega_n_minus_1) = valid_proof_and_parts(n);
        let vk = get_verifier_key(&pp);
        assert!(single_proof_verify(
            &vk,
            &f_com,
            &proof,
            n,
            &omega,
            &omega_n_minus_1
        ).is_ok());
    }

    #[test]
    fn test_single_proof_verify_wrong_fcom() {
        let n = 8;
        let (pp, _f_com, proof, omega, omega_n_minus_1) = valid_proof_and_parts(n);
        let vk = get_verifier_key(&pp);

        let rng = &mut test_rng();
        let wrong_com = to_commitment(G1Projective::rand(rng));

        assert_eq!(
            single_proof_verify(&vk, &wrong_com, &proof, n, &omega, &omega_n_minus_1),
            Err("fcom")
        );
    }

    #[test]
    fn test_single_proof_verify_tampered_g_rho_eval() {
        let n = 8;
        let (pp, f_com, mut proof, omega, omega_n_minus_1) = valid_proof_and_parts(n);
        let vk = get_verifier_key(&pp);

        proof.g_rho_eval += Fr::one();

        assert_eq!(
            single_proof_verify(&vk, &f_com, &proof, n, &omega, &omega_n_minus_1),
            Err("g_rho_proof")
        );
    }

    #[test]
    fn test_single_proof_verify_tampered_g_rho_omega_eval() {
        let n = 8;
        let (pp, f_com, mut proof, omega, omega_n_minus_1) = valid_proof_and_parts(n);
        let vk = get_verifier_key(&pp);

        proof.g_rho_omega_eval += Fr::one();

        assert_eq!(
            single_proof_verify(&vk, &f_com, &proof, n, &omega, &omega_n_minus_1),
            Err("g_rho_omega_proof")
        );
    }

    #[test]
    fn test_single_proof_verify_tampered_w_caret_eval() {
        let n = 8;
        let (pp, f_com, mut proof, omega, omega_n_minus_1) = valid_proof_and_parts(n);
        let vk = get_verifier_key(&pp);

        proof.w_caret_rho_eval += Fr::one();

        assert_eq!(
            single_proof_verify(&vk, &f_com, &proof, n, &omega, &omega_n_minus_1),
            Err("w_caret_rho_proof")
        );
    }

    #[test]
    fn test_single_proof_verify_tampered_qcom() {
        let n = 8;
        let (pp, f_com, proof, omega, omega_n_minus_1) = valid_proof_and_parts(n);
        let vk = get_verifier_key(&pp);
        let tampered_omega = omega_n_minus_1+Fr::one();

        assert_eq!(
            single_proof_verify(&vk, &f_com, &proof, n, &omega, &tampered_omega),
            Err("eval_w")
        );
    }

    // -------------------------------------------------------------------------
    // min_verify
    // -------------------------------------------------------------------------

    #[test]
    fn test_min_verify_basic_case() {
        let n = 2;
        let pp = setup_params(n * 2);
        let min = Fr::from(4u64);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];

        let val_poly = DensePolynomial::from_coefficients_slice(&[Fr::from(4u64)]);
        let (val_com, _) =
            KZG::commit(&get_powers_from_params(&pp), &val_poly, None, None).unwrap();

        let z = Fr::from(4u64) - min;
        let f_min = DensePolynomial::from_coefficients_slice(&[z]);
        let proof = single_value_proof(&pp, &omega, &z, &f_min, n).unwrap();

        assert!(min_verify(&pp, &min, &[val_com], &[proof], n).unwrap());
    }

    #[test]
    fn test_min_verify_n_equals_4_all_valid() {
        let n = 4;
        let pp = setup_params(n * 2);
        let min = Fr::from(2u64);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let val: Vec<Fr> = (2..n as u64 + 2).map(Fr::from).collect();

        let values_com: Vec<Commitment<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let proofs: Vec<RangeProof<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let z = *v - min;
                let f_min = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f_min, n).unwrap()
            })
            .collect();

        assert!(min_verify(&pp, &min, &values_com, &proofs, n).unwrap());
    }

    #[test]
    fn test_min_verify_n_equals_8_some_invalid() {
        let n = 8;
        let pp = setup_params(n * 2);
        let min = Fr::from(50u64);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let val = vec![
            Fr::from(60u64),
            Fr::from(40u64), // below threshold — verification must reject
            Fr::from(80u64),
            Fr::from(90u64),
            Fr::from(70u64),
            Fr::from(50u64),
            Fr::from(55u64),
            Fr::from(65u64),
        ];

        let values_com: Vec<Commitment<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let proofs: Vec<RangeProof<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let z = *v - min;
                let f_min = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f_min, n).unwrap()
            })
            .collect();

        assert!(!min_verify(&pp, &min, &values_com, &proofs, n).unwrap());
    }

    #[test]
    fn test_min_verify_large_n() {
        let n = 128;
        let pp = setup_params(n * 2);
        let min = Fr::from(50u64);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let val: Vec<Fr> = vec![Fr::from(51u64); n];

        let values_com: Vec<Commitment<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let proofs: Vec<RangeProof<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let z = *v - min;
                let f_min = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f_min, n).unwrap()
            })
            .collect();

        assert!(min_verify(&pp, &min, &values_com, &proofs, n).unwrap());
    }

    // -------------------------------------------------------------------------
    // max_verify
    // -------------------------------------------------------------------------

    #[test]
    fn test_max_verify_with_valid_values() {
        let n = 8;
        let pp = setup_params(n * 2);
        let max = Fr::from(90u64);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let val = vec![
            Fr::from(60u64),
            Fr::from(40u64),
            Fr::from(80u64),
            Fr::from(90u64),
            Fr::from(70u64),
            Fr::from(50u64),
            Fr::from(55u64),
            Fr::from(65u64),
        ];

        let values_com: Vec<Commitment<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let proofs: Vec<RangeProof<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let z = max - *v;
                let f_max = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f_max, n).unwrap()
            })
            .collect();

        assert!(max_verify(&pp, &max, &values_com, &proofs, n).unwrap());
    }

    #[test]
    fn test_max_verify_with_invalid_values() {
        let n = 4;
        let pp = setup_params(n * 2);
        let max = Fr::from(100u64);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let val = vec![
            Fr::from(60u64),
            Fr::from(150u64), // exceeds max — verification must reject
            Fr::from(80u64),
            Fr::from(65u64),
        ];

        let values_com: Vec<Commitment<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let proofs: Vec<RangeProof<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let z = max - *v;
                let f_max = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f_max, n).unwrap()
            })
            .collect();

        assert!(!max_verify(&pp, &max, &values_com, &proofs, n).unwrap());
    }

    #[test]
    fn test_max_verify_large_n() {
        let n = 256;
        let pp = setup_params(n * 2);
        let max = Fr::from(1000u64);
        let omega = generate_nth_roots_of_unity::<Fr>(n)[1];
        let val: Vec<Fr> = (0..n as u64).map(Fr::from).collect();

        let values_com: Vec<Commitment<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let proofs: Vec<RangeProof<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let z = max - *v;
                let f_max = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f_max, n).unwrap()
            })
            .collect();

        assert!(max_verify(&pp, &max, &values_com, &proofs, n).unwrap());
    }

    // -------------------------------------------------------------------------
    // Performance smoke test
    // -------------------------------------------------------------------------

    fn run_test_max_verify(count: usize, n_bit: usize) {
        use rand::Rng;
        use std::time::Instant;

        let pp = setup_params(n_bit * 2);
        let roots = generate_nth_roots_of_unity::<Fr>(n_bit);
        let omega = roots[1];
        let max_u64 = count as u64;
        let max_fr = Fr::from(max_u64);

        let val: Vec<Fr> = (0..count)
            .map(|_| Fr::from(rand::thread_rng().gen_range(0..max_u64)))
            .collect();

        let values_com: Vec<Commitment<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let poly = DensePolynomial::from_coefficients_slice(&[*v]);
                let (com, _) =
                    KZG::commit(&get_powers_from_params(&pp), &poly, None, None).unwrap();
                com
            })
            .collect();

        let start = Instant::now();
        let proofs: Vec<RangeProof<Bls12_381>> = val
            .par_iter()
            .map(|v| {
                let z = max_fr - *v;
                let f_max = DensePolynomial::from_coefficients_slice(&[z]);
                single_value_proof(&pp, &omega, &z, &f_max, n_bit).unwrap()
            })
            .collect();
        println!("prover elapsed on {} values: {:?}", count, start.elapsed());
        // size_of_range_proof is now a generic function; instantiate for Bls12_381.
        println!("proof size: {} bytes", size_of_range_proof::<Bls12_381>());

        let start = Instant::now();
        let result = max_verify(&pp, &max_fr, &values_com, &proofs, n_bit).unwrap();
        println!(
            "max_verify elapsed on {} values: {:?}",
            count,
            start.elapsed()
        );

        assert!(result);
    }

    #[test]
    fn test_max_proof_perf() {
        run_test_max_verify(50, 16);
    }
}
