pub(crate) mod ark_serde;
pub mod range_proof;
pub mod utilities;

#[cfg(test)]
mod ark_serde_tests {
    use ark_bls12_381::{Bls12_381, Fr, G1Projective};
    use ark_ec::CurveGroup;
    use ark_poly::univariate::DensePolynomial;
    use ark_poly_commit::kzg10::{Commitment, KZG10};
    use ark_std::{test_rng, UniformRand};

    /// Helper: round-trip a Commitment through serde_json via ark_serde.
    /// Commitment<E> has no built-in serde impl, so we wrap the inner
    /// G1Affine point (which CanonicalSerialize covers) in a newtype.
    fn commitment_roundtrip(
        com: &ark_poly_commit::kzg10::Commitment<Bls12_381>,
    ) -> ark_poly_commit::kzg10::Commitment<Bls12_381> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Wrapper(#[serde(with = "crate::ark_serde")] ark_bls12_381::G1Affine);

        let json = serde_json::to_string(&Wrapper(com.0)).expect("serialise Commitment");
        let restored: Wrapper = serde_json::from_str(&json).expect("deserialise Commitment");
        ark_poly_commit::kzg10::Commitment::<Bls12_381>(restored.0)
    }

    /// Helper: round-trip an Fr through serde_json using ark_serde.
    /// We test it via a struct that embeds #[serde(with = "crate::ark_serde")].
    fn fr_roundtrip(fr: Fr) -> Fr {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Wrapper(#[serde(with = "crate::ark_serde")] Fr);

        let json = serde_json::to_string(&Wrapper(fr)).expect("serialise Fr");
        serde_json::from_str::<Wrapper>(&json)
            .expect("deserialise Fr")
            .0
    }

    // NOTE: because ark_serde is pub(crate), the easiest path from integration
    // tests is to exercise it indirectly through RangeProof serde (which uses
    // it internally via the derive macros).

    #[test]
    fn test_commitment_serde_roundtrip_identity() {
        use ark_bls12_381::Bls12_381;
        use ark_ec::CurveGroup;

        let pp =
            KZG10::<Bls12_381, DensePolynomial<Fr>>::setup(16, false, &mut test_rng()).unwrap();
        let g1: G1Projective = pp.powers_of_g[0].into();
        let com = Commitment(g1.into_affine());

        let restored = commitment_roundtrip(&com);
        assert_eq!(com, restored);
    }

    #[test]
    fn test_commitment_serde_roundtrip_random() {
        let rng = &mut test_rng();
        for _ in 0..5 {
            let g1 = G1Projective::rand(rng);
            let com = Commitment(g1.into_affine());
            assert_eq!(com, commitment_roundtrip(&com));
        }
    }

    #[test]
    fn test_fr_serde_roundtrip_zero() {
        use ark_ff::Zero;
        assert_eq!(Fr::zero(), fr_roundtrip(Fr::zero()));
    }

    #[test]
    fn test_fr_serde_roundtrip_one() {
        use ark_ff::One;
        assert_eq!(Fr::one(), fr_roundtrip(Fr::one()));
    }

    #[test]
    fn test_fr_serde_roundtrip_random() {
        let rng = &mut test_rng();
        for _ in 0..10 {
            let fr = Fr::rand(rng);
            assert_eq!(fr, fr_roundtrip(fr));
        }
    }

    #[test]
    fn test_deserialize_bad_bytes_returns_error() {
        // Fr is a 32-byte field element for BLS12-381.
        // All-0xFF is larger than the field modulus, so ark rejects it with
        // Validate::Yes.  Using the wrong length (33 bytes) is a second
        // layer of defence — either property alone is enough to force an error.
        #[derive(serde::Deserialize)]
        struct Wrapper(#[serde(with = "crate::ark_serde")] Fr);

        // 33 × 0xFF: wrong length *and* every byte exceeds the field modulus.
        let bad_bytes: Vec<u8> = vec![0xFFu8; 33];
        let bad_json = serde_json::to_string(&bad_bytes).unwrap();
        let result = serde_json::from_str::<Wrapper>(&bad_json);
        assert!(
            result.is_err(),
            "expected deserialisation to fail on garbage input"
        );
    }
}
