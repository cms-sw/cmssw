#ifndef KfTrackProducerBase_h
#define KfTrackProducerBase_h

/** \class KfTrackProducerBase
 *  Produce Tracks from TrackCandidates
 *
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
			AlgoProductCollection&, TransientTrackingRecHitBuilder const*,
                        const TrackerTopology *ttopo,
			//allow to fill different tracks collections if necessary ::
			//0: not needed
			//1: Before DAF
			//2: After DAF	
			int BeforeOrAfter = 0);


  //  void setSecondHitPattern(Trajectory* traj, reco::Track& track);
 private:
  bool useSplitting;

};

#endif
