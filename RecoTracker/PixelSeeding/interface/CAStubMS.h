#ifndef RecoTracker_PixelSeeding_interface_CAStubMS_h
#define RecoTracker_PixelSeeding_interface_CAStubMS_h

namespace caStubMS {

  // Multiple-scattering spread of the direction of a track that crossed one outer-tracker layer,
  // written as a factor times the track's own curvature so that no momentum estimate is needed:
  //   theta0 = (13.6 MeV / p) sqrt(f) (1 + 0.038 ln f)   (PDG RPP, passage of particles through matter),
  //   f = x/X0 ~ 0.03 for one OT layer and its gap, and kappa = 0.3 B / (100 pT) = 0.0114 / pT per cm
  //   at B = 3.8 T, so for p ~ pT   theta0 = 1.19 sqrt(f) (1 + 0.038 ln f) |kappa| = 0.18 |kappa|.
  // At 1 GeV that is 2 mrad, an order of magnitude above the hit-precision term of a stub bend, so a
  // stub-direction cut that leaves it out rejects real low-pT tracks.
  inline constexpr float kThetaPerCurv = 0.18f;

}  // namespace caStubMS

#endif  // RecoTracker_PixelSeeding_interface_CAStubMS_h
