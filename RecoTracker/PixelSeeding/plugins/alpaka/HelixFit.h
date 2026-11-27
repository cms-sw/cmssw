#ifndef RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
#define RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h

#include <alpaka/alpaka.hpp>

#include <Eigen/Core>

#include "DataFormats/TrackSoA/interface/alpaka/TrackUtilities.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "RecoTracker/PixelTrackFitting/interface/BLBFieldMap.h"
#include "RecoTracker/PixelTrackFitting/interface/BLMaterialMap.h"
#include "RecoTracker/PixelTrackFitting/interface/FitResult.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/PixelSeeding/interface/CAGeometrySoA.h"

#include "CAStructures.h"

namespace riemannFit {

  // Number of tuples fitted per round (the fits' lanes).
  constexpr uint32_t maxNumberOfConcurrentFits = 8 * 1024;
  constexpr uint32_t stride = maxNumberOfConcurrentFits;

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

}  // namespace riemannFit

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Configuration of the BrokenLine fit. The maps are read only with fitCorrections, and are not owned.
  struct BLFitConfig {
    const blMaterialMap::Map *rhoMap = nullptr;
    const blBFieldMap::Map *bMap = nullptr;
    bool fitCorrections = false;
  };

  template <typename TrackerTraits>
  class HelixFit {
  public:
    // Hit view of the fit (see CAStructures.h).
    using HitsMultiView = caStructures::HitsViewT<TrackerTraits>;

    using HitView = ::reco::TrackingRecHitView;
    using HitConstView = ::reco::TrackingRecHitConstView;
    using OutputSoAView = ::reco::TrackSoAView;

    using Tuples = caStructures::SequentialContainer;
    using TupleMultiplicity = caStructures::GenericContainer;

    explicit HelixFit(float bf) : bField_(bf) {}
    ~HelixFit() { deallocate(); }

    void setBField(double bField) { bField_ = bField; }

    void setFitConfig(const BLFitConfig &cfg) { fitCfg_ = cfg; }
    // The outer-tracker rechits that lowerHitIdx and upperHitIdx of the vector hits point to. An empty view gives no
    // partners.
    void setPartnerRecHits(const ::reco::OTRecHitsConstView &otRecHits) { partnerRecHits_ = otRecHits; }
    // Host copy of the tuple-multiplicity offsets, not owned. They set the rounds and the lane-buffer sizes, so they
    // must be the offsets of the container that the fit reads.
    void setHostTupleMultiplicityOffsets(const uint32_t *off) { hostTupleMultiplicityOffsets_ = off; }

    void launchRiemannKernels(const HitsMultiView &hv, const ::reco::CAModulesConstView &fr, Queue &queue);
    // One fused fast-fit launch and one fused fit launch per round (BrokenLineFitKernels.h).
    void launchBrokenLineKernels(const HitsMultiView &hv, const ::reco::CAModulesConstView &fr, Queue &queue);

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
    BLFitConfig fitCfg_;
    ::reco::OTRecHitsConstView partnerRecHits_{};             // see setPartnerRecHits
    const uint32_t *hostTupleMultiplicityOffsets_ = nullptr;  // see setHostTupleMultiplicityOffsets
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // RecoTracker_PixelSeeding_plugins_alpaka_HelixFit_h
