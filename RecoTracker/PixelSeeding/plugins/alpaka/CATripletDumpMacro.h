#ifndef RecoTracker_PixelSeeding_plugins_alpaka_CATripletDumpMacro_h
#define RecoTracker_PixelSeeding_plugins_alpaka_CATripletDumpMacro_h

// Single toggle for the built-triplet training-dataset dump: one row per built triplet carrying
// the in-kernel triplet feature vector, used to train the inline triplet DNN. It is OFF -- the
// #define below is commented out -- and must stay that way in production: every dump-related
// parameter and branch (the device feature capture in Kernel_connect and its launch site, the
// per-event Triplet nano product surfaced from CAHitNtupletGenerator and emplaced by
// CAHitNtuplet) is #ifdef'd on this macro, so a default build compiles none of it.
//
// This macro lives in its own minimal header (no device/DNN dependencies) so it can be
// included cheaply by the producer (CAHitNtuplet.cc), the generator (CAHitNtupletGenerator
// .h/.cc) and the kernels host header (CAHitNtupletGeneratorKernels.h) -- all of which need
// to see the toggle to (de)activate the OUTPUT half -- WITHOUT pulling in the heavy
// CATripletCuts.h. CATripletCuts.h includes this header so flipping the toggle in one place
// still activates the in-kernel DEVICE-CAPTURE half too.
//
// Full recipe: RecoTracker/PixelSeeding/test/models/README.md.
// #define CA_TRIPLET_DUMP

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_CATripletDumpMacro_h
