#ifndef RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
#define RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "RecoTracker/PixelTrackFitting/interface/FitResult.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

#include "CAStructures.h"

namespace riemannFit {

  // Sizing constant, not a tuning knob: `stride` below is this value, and every fit-side Eigen Map
  // stride derives from it. Changing it re-lays every per-lane buffer.
  constexpr uint32_t maxNumberOfConcurrentFits = 8 * 1024;
  constexpr uint32_t stride = maxNumberOfConcurrentFits;
  using Matrix3x4d = Eigen::Matrix<double, 3, 4>;
  using Map3x4d = Eigen::Map<Matrix3x4d, 0, Eigen::Stride<3 * stride, stride> >;
  using Matrix6x4f = Eigen::Matrix<float, 6, 4>;
  using Map6x4f = Eigen::Map<Matrix6x4f, 0, Eigen::Stride<6 * stride, stride> >;

  // Stride-parameterized fit-buffer maps. The lane stride S = the launch's concurrent-fit count: the
  // per-fit buffers pack lane l's entries at base+l with an S-lane inter-element stride, so a launch
  // that runs only K << maxNumberOfConcurrentFits fits can allocate S=K-sized buffers instead of the
  // global 8192-wide ones.
  // S defaults to the global stride, which is what the main fit and every other caller use.
  // hits
  template <int N>
  using Matrix3xNd = Eigen::Matrix<double, 3, N>;
  template <int N, uint32_t S = stride>
  using Map3xNdS = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * S, S> >;
  template <int N>
  using Map3xNd = Map3xNdS<N, stride>;
  // errors
  template <int N>
  using Matrix6xNf = Eigen::Matrix<float, 6, N>;
  template <int N, uint32_t S = stride>
  using Map6xNfS = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * S, S> >;
  template <int N>
  using Map6xNf = Map6xNfS<N, stride>;
  // fast fit
  template <uint32_t S = stride>
  using Map4dS = Eigen::Map<Vector4d, 0, Eigen::InnerStride<S> >;
  using Map4d = Map4dS<stride>;

  template <auto Start, auto End, auto Inc, class F>  //a compile-time bounded for loop
  constexpr void rolling_fits(F &&f) {
    if constexpr (Start < End) {
      f(std::integral_constant<decltype(Start), Start>());
      rolling_fits<Start + Inc, End, Inc>(f);
    }
  }

}  // namespace riemannFit

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TrackerTraits>
  class HelixFit {
  public:
    using HitView = ::reco::TrackingRecHitView;
    using HitConstView = ::reco::TrackingRecHitConstView;
    using OutputSoAView = ::reco::TrackSoAView;

    using Tuples = caStructures::SequentialContainer;
    using TupleMultiplicity = caStructures::GenericContainer;

    explicit HelixFit(float bf, bool fitNas4) : bField_(bf), fitNas4_(fitNas4) {}
    ~HelixFit() { deallocate(); }

    void setBField(double bField) { bField_ = bField; }

    // Device pointer to the BL-fit Geant4 material-map grid (kSize floats), provided by the EventSetup
    // BLMaterialMap condition (copied to the device once per IOV). Set before launchBrokenLineKernels;
    // it is forwarded into Kernel_BLFit and read by prepareBrokenLineData/segmentXX0. Not owned.
    void setMaterialMap(const float *rhoMap) { rhoMap_ = rhoMap; }
    // Device pointer to the normalized (Bz,Br) r-z field map (blBFieldMap::kNValues floats), provided by
    // the EventSetup BLBFieldMap condition. When set, the fit uses a per-track hit-averaged effective
    // field (blEffectiveBField) in the curvature->pT conversion and in the MS/dE/dx momentum; when null
    // the fit uses the scalar bField at the origin everywhere. Not owned. Read by the CA main fit
    // (Kernel_BLFit) only under fitCorrections_: the Highland
    // variance the material map feeds is divided by a momentum that only the true bending field gives,
    // so the field belongs to the same correctness package as the material map.
    // Set by the producers that run the fit.
    void setBFieldMap(const float *bMap) { bMap_ = bMap; }
    // Host copy of the tuple-multiplicity assoc's finalized per-N-bin cumulative offsets (see
    // CAHitNtupletGeneratorKernels::tupleMultiplicityOffsets), already read back by the caller at the
    // producer's acquire/produce boundary. Optional: when set, launchBrokenLineKernels uses each N-bin's
    // population to elide empty chunk/bin launches, with no D2H and no host wait of its own. Left null,
    // the elision no-ops and the fit runs to the cap. Not owned; must stay valid
    // for the duration of the fit launch.
    void setHostTupleMultiplicityOffsets(const uint32_t *off) { hostTupleMultiplicityOffsets_ = off; }
    // Fit correctness package (producer parameter useFitCorrections; see the block at the head of
    // RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h). Read only by the CA main fit. It also
    // gates the CA main fit's use
    // of the (Bz,Br) map set by setBFieldMap (see there): on, the fit's material AND its bending field are
    // the measured ones; off, the flat 0.06/16 material and the origin scalar field, i.e. upstream.
    void setFitCorrections(bool on) { fitCorrections_ = on; }

    // Enable a one-shot device-side dump of the first N fitted tracks at the
    // end of each launchBrokenLineKernels call.  Prints (phi0, d0, kappa,
    // cotTheta, z0) as well as the derived pT [GeV] and eta.  OFF by default.
    void setVerboseDump(bool on, uint32_t nToPrint = 10) {
      verboseDump_ = on;
      verboseDumpN_ = nToPrint;
    }
    void launchRiemannKernels(const caStructures::HitsMultiView &hv,
                              const ::reco::CAModulesConstView &fr,
                              uint32_t nhits,
                              uint32_t maxNumberOfTuples,
                              Queue &queue);
    // ONE sweep of the N-binned BLFastFit+BLFit kernels: the factorized fast BrokenLine fit that every
    // CA iteration and every topology runs on its own tracks. The General Broken Lines fit runs once per
    // track downstream, in the merger (refitExtended / refitMergedTwins below).
    void launchBrokenLineKernels(const caStructures::HitsMultiView &hv,
                                 const ::reco::CAModulesConstView &fr,
                                 uint32_t nhits,
                                 uint32_t maxNumberOfTuples,
                                 Queue &queue);

    void allocate(TupleMultiplicity const *tupleMultiplicity,
                  OutputSoAView &helix_fit_results,
                  Tuples const *__restrict__ foundNtuplets);
    void deallocate();

  private:
    static constexpr uint32_t maxNumberOfConcurrentFits_ = riemannFit::maxNumberOfConcurrentFits;

    // fowarded
    Tuples const *tuples_ = nullptr;
    TupleMultiplicity const *tupleMultiplicity_ = nullptr;
    OutputSoAView outputSoa_;
    float bField_;
    const float *rhoMap_ = nullptr;  // BL material-map device grid (EventSetup condition; not owned)
    const float *bMap_ = nullptr;    // normalized (Bz,Br) r-z field map (EventSetup condition; not owned)
    bool fitCorrections_ = false;    // CA main fit's correctness package (useFitCorrections)
    // Tuple-multiplicity per-N-bin cumulative offsets, pre-read by the caller into host memory (see
    // setHostTupleMultiplicityOffsets). Not owned; null means the fit runs to the cap.
    const uint32_t *hostTupleMultiplicityOffsets_ = nullptr;

    // One-shot post-fit device dump of the first verboseDumpN_ tracks (see setVerboseDump). OFF by default.
    bool verboseDump_ = false;
    uint32_t verboseDumpN_ = 10;

    const bool fitNas4_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
