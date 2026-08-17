#ifndef RecoTracker_PixelSeeding_interface_TripletDumpSoA_h
#define RecoTracker_PixelSeeding_interface_TripletDumpSoA_h

// Per-built-triplet training-dataset row, captured on device by Kernel_connect ONLY in
// CA_TRIPLET_DUMP builds (the buffer is allocated, written, and emitted under #ifdef
// CA_TRIPLET_DUMP, so production builds carry nothing). It is the multithread-safe, ROOT-robust
// nanoAOD capture path. Columns 0..17 are the 18 BASE features fed to the triplet DNN (exact order +
// formulas == BASE_FEATURES in RecoTracker/PixelSeeding/test/train_triplet_dnn_v2.py; the 11
// DERIVED features are recomputed offline from these + lay1/2/3). h1/h2/h3 are the three global
// MERGED-hit indices (truth join key).

#include <alpaka/alpaka.hpp>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace caStructures {

  GENERATE_SOA_LAYOUT(TripletDumpLayout,
                      // --- 18 base features (DNN input order) ---
                      SOA_COLUMN(float, absCurvature),
                      SOA_COLUMN(float, tipTimesCurvature),
                      SOA_COLUMN(float, dca),
                      SOA_COLUMN(float, curvatureStubs),
                      SOA_COLUMN(float, curvatureStubsErrSquared),
                      SOA_COLUMN(float, curvature13),
                      SOA_COLUMN(float, dPhi12),
                      SOA_COLUMN(float, dPhi13),
                      SOA_COLUMN(float, dPhi23),
                      SOA_COLUMN(float, dr12),
                      SOA_COLUMN(float, dr13),
                      SOA_COLUMN(float, r1),
                      SOA_COLUMN(float, r2),
                      SOA_COLUMN(float, r3),
                      SOA_COLUMN(float, z1),
                      SOA_COLUMN(float, z2),
                      SOA_COLUMN(float, z3),
                      SOA_COLUMN(float, nStubs),
                      // Signed curvature (not a base feature, but the offline derived features
                      // stubCirclePull/stubCircleRatio/curv13Resid need the sign, as accept() does;
                      // absCurvature loses it).
                      SOA_COLUMN(float, curvature),
                      // --- layers (for the layGap12/23 derived features) ---
                      SOA_COLUMN(int32_t, lay1),
                      SOA_COLUMN(int32_t, lay2),
                      SOA_COLUMN(int32_t, lay3),
                      // --- merged-hit indices (truth join key) ---
                      SOA_COLUMN(uint32_t, h1),
                      SOA_COLUMN(uint32_t, h2),
                      SOA_COLUMN(uint32_t, h3),
                      // The in-kernel DNN score score(feat) for this triplet (-1 if not evaluated),
                      // for the in-kernel-vs-offline consistency check.
                      SOA_COLUMN(float, inKernelScore),
                      // Number of valid rows (== device_nTriplets). The collection is allocated at the
                      // full capacity (tripletsN_); only rows [0, nValid) are written by Kernel_connect.
                      SOA_SCALAR(uint32_t, nValid))

  using TripletDumpSoA = TripletDumpLayout<>;
  using TripletDumpSoAView = TripletDumpSoA::View;
  using TripletDumpSoAConstView = TripletDumpSoA::ConstView;

}  // namespace caStructures

#endif  // RecoTracker_PixelSeeding_interface_TripletDumpSoA_h
