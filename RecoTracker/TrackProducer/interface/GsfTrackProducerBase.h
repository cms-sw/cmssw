#ifndef GsfTrackProducerBase_h
#define GsfTrackProducerBase_h

/** \class GsfTrackProducerBase
 *  Produce Tracks from TrackCandidates
 *
 *  $Date: 2007/10/09 17:41:02 $
 *  $Revision: 1.5 $
 *  \author cerati
 */

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfComponent5D.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

// #include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class GsfTrackProducerBase : public TrackProducerBase<reco::GsfTrack> {
public:

  /// Constructor
  explicit GsfTrackProducerBase(bool trajectoryInEvent, bool split) :
    TrackProducerBase<reco::GsfTrack>(trajectoryInEvent),
    useSplitting(split){}

  /// Put produced collections in the event
  virtual void putInEvt(edm::Event&,
			std::auto_ptr<TrackingRecHitCollection>&,
			std::auto_ptr<reco::GsfTrackCollection>&,
			std::auto_ptr<reco::TrackExtraCollection>&,
			std::auto_ptr<reco::GsfTrackExtraCollection>&,
			std::auto_ptr<std::vector<Trajectory> >&,
			AlgoProductCollection&);


protected:
  void fillStates (TrajectoryStateOnSurface tsos, std::vector<reco::GsfComponent5D>& states) const;

private:
bool useSplitting;

};

     
#endif
