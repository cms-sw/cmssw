#ifndef KfTrackProducerBase_h
#define KfTrackProducerBase_h

/** \class KfTrackProducerBase
 *  Produce Tracks from TrackCandidates
 *
 *  $Date: 2009/07/03 00:55:13 $
 *  $Revision: 1.3 $
 *  \author cerati
 */

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class Trajectory;

class KfTrackProducerBase : public TrackProducerBase<reco::Track> {
public:

  /// Constructor
  explicit KfTrackProducerBase(bool trajectoryInEvent, bool split) :
    TrackProducerBase<reco::Track>(trajectoryInEvent),useSplitting(split) {}

  /// Put produced collections in the event
  virtual void putInEvt(edm::Event&,
			const Propagator* prop,
			const MeasurementTracker* measTk,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::TrackCollection>&,
			std::auto_ptr<reco::TrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&);


  //  void setSecondHitPattern(Trajectory* traj, reco::Track& track);
 private:
  bool useSplitting;

};

#endif
