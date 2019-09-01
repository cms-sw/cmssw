#ifndef TrackProducerAlgorithm_h
#define TrackProducerAlgorithm_h

/** \class TrackProducerAlgorithm
 *  This class calls the Final Fit and builds the Tracks then produced by the TrackProducer or by the TrackRefitter
 *
 *  \author cerati
 */

#include "AlgoProductTraits.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

class MagneticField;
class TrackingGeometry;
class Propagator;
class Trajectory;
class TrajectoryStateOnSurface;

struct FitterCloner {
  std::unique_ptr<TrajectoryFitter> fitter;
  TkClonerImpl hitCloner;

  FitterCloner(const TrajectoryFitter *theFitter, const TransientTrackingRecHitBuilder *builder)
      : fitter(theFitter->clone()),
        hitCloner(static_cast<TkTransientTrackingRecHitBuilder const *>(builder)->cloner()) {
    fitter->setHitCloner(&hitCloner);
  }
};

template <class T>
class TrackProducerAlgorithm : public AlgoProductTraits<T> {
public:
  using Base = AlgoProductTraits<T>;
  using TrackCollection = typename Base::TrackCollection;
  using TrackView = typename Base::TrackView;
  using AlgoProductCollection = typename Base::AlgoProductCollection;

  using SeedRef = edm::RefToBase<TrajectorySeed>;
  using VtxConstraintAssociationCollection =
      edm::AssociationMap<edm::OneToOne<std::vector<T>, std::vector<VertexConstraint> > >;

public:
  /// Constructor
  TrackProducerAlgorithm(const edm::ParameterSet &conf)
      : algo_(reco::TrackBase::algoByName(conf.getParameter<std::string>("AlgorithmName"))),
        originalAlgo_(reco::TrackBase::undefAlgorithm),
        stopReason_(0),
        reMatchSplitHits_(false),
        usePropagatorForPCA_(false) {
    geometricInnerState_ = (conf.exists("GeometricInnerState") ? conf.getParameter<bool>("GeometricInnerState") : true);
    if (conf.exists("reMatchSplitHits"))
      reMatchSplitHits_ = conf.getParameter<bool>("reMatchSplitHits");
    if (conf.exists("usePropagatorForPCA"))
      usePropagatorForPCA_ = conf.getParameter<bool>("usePropagatorForPCA");
  }

  /// Destructor
  ~TrackProducerAlgorithm() {}

  /// Run the Final Fit taking TrackCandidates as input
  void runWithCandidate(const TrackingGeometry *,
                        const MagneticField *,
                        const TrackCandidateCollection &,
                        const TrajectoryFitter *,
                        const Propagator *,
                        const TransientTrackingRecHitBuilder *,
                        const reco::BeamSpot &,
                        AlgoProductCollection &);

  /// Run the Final Fit taking Tracks as input (for Refitter)
  void runWithTrack(const TrackingGeometry *,
                    const MagneticField *,
                    const TrackView &,
                    const TrajectoryFitter *,
                    const Propagator *,
                    const TransientTrackingRecHitBuilder *,
                    const reco::BeamSpot &,
                    AlgoProductCollection &);

  /// Run the Final Fit taking TrackMomConstraintAssociation as input (Refitter with momentum constraint)
  void runWithMomentum(const TrackingGeometry *,
                       const MagneticField *,
                       const TrackMomConstraintAssociationCollection &,
                       const TrajectoryFitter *,
                       const Propagator *,
                       const TransientTrackingRecHitBuilder *,
                       const reco::BeamSpot &,
                       AlgoProductCollection &);

  /// Run the Final Fit taking TrackVtxConstraintAssociation as input (Refitter with vertex constraint)
  ///   currently hit sorting is disabled - will work (only) with standard tracks
  void runWithVertex(const TrackingGeometry *,
                     const MagneticField *,
                     const VtxConstraintAssociationCollection &,
                     const TrajectoryFitter *,
                     const Propagator *,
                     const TransientTrackingRecHitBuilder *,
                     const reco::BeamSpot &,
                     AlgoProductCollection &);

  /// Run the Final Fit taking TrackParamConstraintAssociation as input (Refitter with complete track parameters constraint)
  ///   currently hit sorting is disabled - will work (only) with standard tracks
  void runWithTrackParameters(const TrackingGeometry *,
                              const MagneticField *,
                              const TrackParamConstraintAssociationCollection &,
                              const TrajectoryFitter *,
                              const Propagator *,
                              const TransientTrackingRecHitBuilder *,
                              const reco::BeamSpot &,
                              AlgoProductCollection &);

  /// Construct Tracks to be put in the event
  bool buildTrack(const TrajectoryFitter *,
                  const Propagator *,
                  AlgoProductCollection &,
                  TransientTrackingRecHit::RecHitContainer &,
                  TrajectoryStateOnSurface &,
                  const TrajectorySeed &,
                  float,
                  const reco::BeamSpot &,
                  SeedRef seedRef = SeedRef(),
                  int qualityMask = 0,
                  signed char nLoops = 0);

private:
  reco::TrackBase::TrackAlgorithm algo_;
  reco::TrackBase::TrackAlgorithm originalAlgo_;
  reco::TrackBase::AlgoMask algoMask_;
  uint8_t stopReason_;

  bool reMatchSplitHits_;
  bool geometricInnerState_;
  bool usePropagatorForPCA_;

  TrajectoryStateOnSurface getInitialState(const T *theT,
                                           TransientTrackingRecHit::RecHitContainer &hits,
                                           const TrackingGeometry *theG,
                                           const MagneticField *theMF);
};

#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.icc"

template <>
bool TrackProducerAlgorithm<reco::Track>::buildTrack(const TrajectoryFitter *,
                                                     const Propagator *,
                                                     AlgoProductCollection &,
                                                     TransientTrackingRecHit::RecHitContainer &,
                                                     TrajectoryStateOnSurface &,
                                                     const TrajectorySeed &,
                                                     float,
                                                     const reco::BeamSpot &,
                                                     SeedRef seedRef,
                                                     int qualityMask,
                                                     signed char nLoops);

template <>
bool TrackProducerAlgorithm<reco::GsfTrack>::buildTrack(const TrajectoryFitter *,
                                                        const Propagator *,
                                                        AlgoProductCollection &,
                                                        TransientTrackingRecHit::RecHitContainer &,
                                                        TrajectoryStateOnSurface &,
                                                        const TrajectorySeed &,
                                                        float,
                                                        const reco::BeamSpot &,
                                                        SeedRef seedRef,
                                                        int qualityMask,
                                                        signed char nLoops);

#endif
