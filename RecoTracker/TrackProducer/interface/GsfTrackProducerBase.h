#ifndef GsfTrackProducerBase_h
#define GsfTrackProducerBase_h

/** \class GsfTrackProducerBase
 *  Produce Tracks from TrackCandidates
 *
 *  \author cerati
 */

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateMode.h"

// #include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class TrajectoryStateOnSurface;
class Propagator;
class TransverseImpactPointExtrapolator;
class TrajectoryStateClosestToBeamLineBuilder;

class GsfTrackProducerBase : public TrackProducerBase<reco::GsfTrack> {
public:

  /// Constructor
  explicit GsfTrackProducerBase(bool trajectoryInEvent, bool split) :
  TrackProducerBase<reco::GsfTrack>(trajectoryInEvent),
    useSplitting(split){}
  
  /// Put produced collections in the event
  virtual void putInEvt(edm::Event&,
			const Propagator* prop,
			const MeasurementTracker* measTk,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::GsfTrackCollection>&,
			std::auto_ptr<reco::TrackExtraCollection>&,
			std::auto_ptr<reco::GsfTrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&, TransientTrackingRecHitBuilder const*,
			const reco::BeamSpot&, const TrackerTopology *ttopo);


protected:
  void fillStates (TrajectoryStateOnSurface tsos, std::vector<reco::GsfComponent5D>& states) const;
  void fillMode (reco::GsfTrack& track, const TrajectoryStateOnSurface innertsos,
		 const Propagator& gsfProp,
		 const TransverseImpactPointExtrapolator& tipExtrapolator,
		 TrajectoryStateClosestToBeamLineBuilder& tscblBuilder,
		 const reco::BeamSpot& bs) const;

private:
  /// local parameters rescaled with q/p from mode
  void localParametersFromQpMode (const TrajectoryStateOnSurface tsos,
				  AlgebraicVector5& parameters,
				  AlgebraicSymMatrix55& covariance) const;
  /// position, momentum and estimated deltaP at an intermediate measurement (true if successful)
  bool computeModeAtTM (const TrajectoryMeasurement& tm,
			reco::GsfTrackExtra::Point& position,
			reco::GsfTrackExtra::Vector& momentum,
			Measurement1D& deltaP) const;

private:
bool useSplitting;

};

     
#endif
