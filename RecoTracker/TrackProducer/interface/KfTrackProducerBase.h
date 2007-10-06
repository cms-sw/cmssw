#ifndef KfTrackProducerBase_h
#define KfTrackProducerBase_h

/** \class KfTrackProducerBase
 *  Produce Tracks from TrackCandidates
 *
 *  $Date: 2007/03/27 07:12:05 $
 *  $Revision: 1.7 $
 *  \author cerati
 */

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class KfTrackProducerBase : public TrackProducerBase<reco::Track> {
public:

  /// Constructor
  explicit KfTrackProducerBase(bool trajectoryInEvent = false) :
    TrackProducerBase<reco::Track>(trajectoryInEvent) {}

  /// Put produced collections in the event
  virtual void putInEvt(edm::Event&,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::TrackCollection>&,
			std::auto_ptr<reco::TrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&);

};

#endif
