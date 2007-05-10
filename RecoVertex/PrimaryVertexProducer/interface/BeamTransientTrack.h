#ifndef VertexReco_BeamTrackTransientTrack_h
#define VertexReco_BeamTrackTransientTrack_h

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

class BeamTransientTrack: public reco::TransientTrack {
 public:
  TrajectoryStateClosestToPoint beamState;
  inline BeamTransientTrack(const reco::TransientTrack &t, const GlobalPoint &  beamPosition):
    reco::TransientTrack(t),beamState(t.trajectoryStateClosestToPoint(beamPosition)){}
  inline BeamTransientTrack(const BeamTransientTrack & bt):reco::TransientTrack(bt),beamState(bt.beamState){}
};
#endif
