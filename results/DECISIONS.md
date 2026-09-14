# Pre-registered decisions (ralplan v3 + iteration 2), computed by scripts/decisions.py

Disclosure (pre-registration timeline):
- Two smoke runs of scripts/fixed_instrument.py on E10b, all 8 cutoffs, bands 0-9 with floors: log_dirs/prereg_smoke/e19_smoke.json (22:14, 32 images, sc_fix@{0.05,0.1,0.3}) and e19_smoke0.json (22:15, 16 images, sc_fix@{0,0.05,0.1,0.3}), hashes in log_dirs/PREREG.txt. Both predate every frozen S5 rule.
- The S5 rule changed form after those smokes: v2 (22:10) required a gap >= 0.2 at r = 0.1, 0.2 and 0.3; v3 (22:19) uses the r=0.3-minus-r=0.1 difference as primary. Because the E10b values were seen, the dial verdict is confirmed only if the two untouched seeds E13_seed2027/2028 also pass; otherwise it is exploratory.
- Iteration 2 (22:41) added the S1 fixed-instrument co-primary; iteration 3 replaced a by-name exclusion of scale 0.25 with the mean-A < 0.2 rule after scale 0.25 and 0.35 moving-instrument SC were known, before E19 scored any scale directory.
- S1 ran across code versions: E18a_scale0.1 before the models.py edit (22:20); E18a_scale0.2 and 0.35 with an intermediate models/pipelines.py (class-name check); E18a_scale0.6 onward with the final pipelines.py (22:40:56).
## S1 C2 matched curve (strength vs cutoff)
pipeline regression check: r=0.2: 0.3987 vs seeds [0.3948, 0.4142] +-0.03 -> ok; r=0.3: 0.2297 vs seeds [0.2201, 0.2407] +-0.03 -> ok; scale SC monotone in s across E12/E18a (7/7 present): True
sampling FID sd from 5 seed pairs = 0.910; delta = 1.820
[moving instrument, pre-registered v3] dial curve: (0.05, 0.7469, 51.58), (0.1, 0.6211, 54.68), (0.15, 0.5081, 54.09), (0.2, 0.4019, 53.45), (0.25, 0.3165, 53.43), (0.3, 0.2347, 52.77), (0.5, 0.1259, 53.97), (0.7, 0.1110, 53.69)
  [moving instrument, pre-registered v3] E18a_scale0.1: SC 0.0165 below dial range; FID 117.58 vs 53.69+1.82 -> dial dominates
  [moving instrument, pre-registered v3] E18a_scale0.2: SC 0.0300 below dial range; FID 110.20 vs 53.69+1.82 -> dial dominates
  [moving instrument, pre-registered v3] E12_scale0.25: SC 0.0408 below dial range; FID 104.95 vs 53.69+1.82 -> dial dominates
  [moving instrument, pre-registered v3] E18a_scale0.35: SC 0.0808 below dial range; FID 95.62 vs 53.69+1.82 -> dial dominates
  [moving instrument, pre-registered v3] E12_scale0.5: SC 0.2146 FID 76.46; dial FID at same SC 52.99; diff +23.46 -> dial lower FID
  [moving instrument, pre-registered v3] E18a_scale0.6: SC 0.3373 FID 66.39; dial FID at same SC 53.44; diff +12.95 -> dial lower FID
  [moving instrument, pre-registered v3] E12_scale0.75: SC 0.4893 FID 58.90; dial FID at same SC 53.98; diff +4.92 -> dial lower FID
[fixed r_m=0.1 (co-primary)] dial curve: (0.05, 0.6443, 51.58), (0.1, 0.6128, 54.68), (0.15, 0.4799, 54.09), (0.2, 0.3769, 53.45), (0.25, 0.3088, 53.43), (0.3, 0.2522, 52.77), (0.5, 0.1766, 53.97), (0.7, 0.1403, 53.69)
  [fixed r_m=0.1 (co-primary)] E18a_scale0.1: SC 0.0140 below dial range; FID 117.58 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.1 (co-primary)] E18a_scale0.2: SC 0.0272 below dial range; FID 110.20 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.1 (co-primary)] E12_scale0.25: SC 0.0382 below dial range; FID 104.95 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.1 (co-primary)] E18a_scale0.35: SC 0.0775 below dial range; FID 95.62 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.1 (co-primary)] E12_scale0.5: SC 0.2073 FID 76.46; dial FID at same SC 53.49; diff +22.97 -> dial lower FID
  [fixed r_m=0.1 (co-primary)] E18a_scale0.6: SC 0.3296 FID 66.39; dial FID at same SC 53.44; diff +12.95 -> dial lower FID
  [fixed r_m=0.1 (co-primary)] E12_scale0.75: SC 0.4811 FID 58.90; dial FID at same SC 54.10; diff +4.80 -> dial lower FID
[fixed r_m=0.05 (robustness)] dial curve: (0.05, 0.7309, 51.58), (0.1, 0.5596, 54.68), (0.15, 0.4449, 54.09), (0.2, 0.3733, 53.45), (0.25, 0.3340, 53.43), (0.3, 0.2980, 52.77), (0.5, 0.2399, 53.97), (0.7, 0.2000, 53.69)
  [fixed r_m=0.05 (robustness)] E18a_scale0.1: SC 0.0100 below dial range; FID 117.58 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.05 (robustness)] E18a_scale0.2: SC 0.0204 below dial range; FID 110.20 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.05 (robustness)] E12_scale0.25: SC 0.0282 below dial range; FID 104.95 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.05 (robustness)] E18a_scale0.35: SC 0.0588 below dial range; FID 95.62 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.05 (robustness)] E12_scale0.5: SC 0.1725 below dial range; FID 76.46 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.05 (robustness)] E18a_scale0.6: SC 0.2769 FID 66.39; dial FID at same SC 53.21; diff +13.18 -> dial lower FID
  [fixed r_m=0.05 (robustness)] E12_scale0.75: SC 0.4159 FID 58.90; dial FID at same SC 53.83; diff +5.06 -> dial lower FID
[fixed r_m=0.3 (robustness)] dial curve: (0.05, 0.3056, 51.58), (0.1, 0.2956, 54.68), (0.15, 0.2844, 54.09), (0.2, 0.2730, 53.45), (0.25, 0.2602, 53.43), (0.3, 0.2276, 52.77), (0.5, 0.1205, 53.97), (0.7, 0.0949, 53.69)
  [fixed r_m=0.3 (robustness)] E18a_scale0.1: SC 0.0101 below dial range; FID 117.58 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.3 (robustness)] E18a_scale0.2: SC 0.0147 below dial range; FID 110.20 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.3 (robustness)] E12_scale0.25: SC 0.0179 below dial range; FID 104.95 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.3 (robustness)] E18a_scale0.35: SC 0.0324 below dial range; FID 95.62 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.3 (robustness)] E12_scale0.5: SC 0.0781 below dial range; FID 76.46 vs 53.69+1.82 -> dial dominates
  [fixed r_m=0.3 (robustness)] E18a_scale0.6: SC 0.1287 FID 66.39; dial FID at same SC 53.88; diff +12.51 -> dial lower FID
  [fixed r_m=0.3 (robustness)] E12_scale0.75: SC 0.2064 FID 58.90; dial FID at same SC 53.01; diff +5.89 -> dial lower FID
  write-up E18a_scale0.1: moving=dial, fixed0.1=dial -> C2 sentence allowed
  write-up E18a_scale0.2: moving=dial, fixed0.1=dial -> C2 sentence allowed
  write-up E12_scale0.25: moving=dial, fixed0.1=dial -> C2 sentence allowed
  write-up E18a_scale0.35: moving=dial, fixed0.1=dial -> C2 sentence allowed
  write-up E12_scale0.5: moving=dial, fixed0.1=dial -> C2 sentence allowed
  write-up E18a_scale0.6: moving=dial, fixed0.1=dial -> C2 sentence allowed
  write-up E12_scale0.75: moving=dial, fixed0.1=dial -> C2 sentence allowed
## S2 VAE survival
- sdxl (stabilityai/sdxl-vae/, 4ch, n=500): r=0.1 0.8790 (+0.0333), r=0.3 0.6268 (+0.0686) -> C3 holds for the SDXL VAE too
- flux16 (ostris/Flex.1-alpha/vae, 16ch, n=500): r=0.1 0.9705 (+0.1247), r=0.3 0.8983 (+0.3401) -> narrow C3 to 4-channel SD-family VAEs
  sd15 reference: r=0.1 0.8457, r=0.3 0.5582
## S3 out-of-domain photos (SC_vs_drawing, same script/prompt/cutoffs)
OOD [0.4506, 0.3106, 0.1944, 0.1384]  in-domain [0.5981, 0.5057, 0.4407, 0.4142]
slope OOD 0.3122, in-domain 0.1839, ratio 1.698; 3*sem 0.0458; monotone(2 sem tol) True -> dial carries out of domain
  descriptive: adapter off at r=0.1 SC_vs_drawing 0.0145
## S4 E18 skip-only trained from scratch
E18 SC@0.1 0.5765 (FID 58.19739696588226); bounds 0.5951/0.654/0.2658 -> case (ii): paths complementary; placement dominates at matched capacity
## S5 band selectivity (fixed instrument)
- dial r=0.3 vs r=0.1, E10b: gap -0.494 (valid bands [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
- dial r=0.3 vs r=0.1, E13_seed2027: gap -0.479 (valid bands [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
- dial r=0.3 vs r=0.1, E13_seed2028: gap -0.488 (valid bands [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  dial verdict: selective (confirmed on held-out seeds)
- E16_spec_scale*: not evaluable (anchor E2_r0.1/gen_r0.0 does not exist)
- E18a_scale0.1: mean A 0.053 < 0.2 -> descriptive only; ratio gap -0.036
- E18a_scale0.2: mean A 0.074 < 0.2 -> descriptive only; ratio gap -0.036
- E12_scale0.25: mean A 0.087 < 0.2 -> descriptive only; ratio gap -0.030
- E18a_scale0.35: mean A 0.148 < 0.2 -> descriptive only; ratio gap -0.027
- E12_scale0.5: mean A 0.309; ratio[0.1,0.3) − ratio[0.3,1) = +0.041
- E18a_scale0.6: mean A 0.480; ratio[0.1,0.3) − ratio[0.3,1) = +0.078
- E12_scale0.75: mean A 0.721; ratio[0.1,0.3) − ratio[0.3,1) = +0.079
  strength verdict: broadband (3 discriminative points)
-> dial band-selective, strength broadband
## S6 conditional slot
E2_r0.1 seed 2027: SC@0.1 0.7038, FID 49.84 (bounds SC>0.654, FID<50.09) -> specialist advantage replicated
