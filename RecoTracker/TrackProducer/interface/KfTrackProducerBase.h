#ifndef KfTrackProducerBase_h
#define KfTrackProducerBase_h

/** \class KfTrackProducerBase
 *  Produce Tracks from TrackCandidates
 *
 *  $Date: 2007/10/06 08:04:11 $
 *  $Revision: 1.1 $
 *  \author cerati
 */

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class KfTrackProducerBase : public TrackProducerBase<reco::Track> {
public:

  /// Constructor
  explicit KfTrackProducerBase(bool trajectoryInEvent, bool split) :
    TrackProducerBase<reco::Track>(trajectoryInEvent),useSplitting(split) {}

  /// Put produced collections in the event
  virtual void putInEvt(edm::Event&,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::TrackCollection>&,
			std::auto_ptr<reco::TrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&);
  
 private:
  bool useSplitting;

};

#endif
