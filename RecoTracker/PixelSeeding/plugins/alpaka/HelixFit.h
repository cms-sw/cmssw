#ifndef RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
#define RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/FitResult.h"
#include "Geometry/CommonTopologies/interface/SimpleSeedingLayersTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/alpaka/FrameSoACollection.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/FrameSoALayout.h"

#include "CAStructures.h"

namespace riemannFit {

  // TODO: Can this be taken from TrackerTraits or somewhere else?
  // in case of memory issue can be made smaller
  constexpr uint32_t maxNumberOfConcurrentFits = 32 * 1024;
  constexpr uint32_t stride = maxNumberOfConcurrentFits;
  using Matrix3x4d = Eigen::Matrix<double, 3, 4>;
  using Map3x4d = Eigen::Map<Matrix3x4d, 0, Eigen::Stride<3 * stride, stride> >;
  using Matrix6x4f = Eigen::Matrix<float, 6, 4>;
  using Map6x4f = Eigen::Map<Matrix6x4f, 0, Eigen::Stride<6 * stride, stride> >;

  // hits
  template <int N>
  using Matrix3xNd = Eigen::Matrix<double, 3, N>;
  template <int N>
  using Map3xNd = Eigen::Map<Matrix3xNd<N>, 0, Eigen::Stride<3 * stride, stride> >;
  // errors
  template <int N>
  using Matrix6xNf = Eigen::Matrix<float, 6, N>;
  template <int N>
  using Map6xNf = Eigen::Map<Matrix6xNf<N>, 0, Eigen::Stride<6 * stride, stride> >;
  // fast fit
  using Map4d = Eigen::Map<Vector4d, 0, Eigen::InnerStride<stride> >;

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
    using TrackingRecHitSoAs = TrackingRecHitSoA<TrackerTraits>;

    using HitView = TrackingRecHitSoAView<TrackerTraits>;
    using HitConstView = TrackingRecHitSoAConstView<TrackerTraits>;

    using Tuples = typename reco::TrackSoA<TrackerTraits>::HitContainer;
    using OutputSoAView = reco::TrackSoAView<TrackerTraits>;

    using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;

    using ParamsOnDevice = pixelCPEforDevice::ParamsOnDeviceT<TrackerTraits>;

    explicit HelixFit(float bf, bool fitNas4) : bField_(bf), fitNas4_(fitNas4) {}
    ~HelixFit() { deallocate(); }

    void setBField(double bField) { bField_ = bField; }
    void launchRiemannKernels(
        const HitConstView &hv, const FrameSoAConstView &fr, uint32_t nhits, uint32_t maxNumberOfTuples, Queue &queue);
    void launchBrokenLineKernels(
        const HitConstView &hv, const FrameSoAConstView &fr, uint32_t nhits, uint32_t maxNumberOfTuples, Queue &queue);

    void allocate(TupleMultiplicity const *tupleMultiplicity, OutputSoAView &helix_fit_results);
    void deallocate();

  private:
    static constexpr uint32_t maxNumberOfConcurrentFits_ = riemannFit::maxNumberOfConcurrentFits;

    // fowarded
    Tuples const *tuples_ = nullptr;
    TupleMultiplicity const *tupleMultiplicity_ = nullptr;
    OutputSoAView outputSoa_;
    float bField_;

    const bool fitNas4_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
