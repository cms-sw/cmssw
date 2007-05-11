#ifndef VertexReco_BeamTrackTransientTrack_h
#define VertexReco_BeamTrackTransientTrack_h

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class BeamTransientTrack: public reco::TransientTrack {
 private:
  TrajectoryStateClosestToPoint beamTrajectoryState;
  double z0;
  double d0Significance;

 public:
  inline BeamTransientTrack(const reco::TransientTrack &t, const GlobalPoint &  beamPosition):
    reco::TransientTrack(t),beamTrajectoryState(t.trajectoryStateClosestToPoint(beamPosition)){
    z0=beamTrajectoryState.perigeeParameters().longitudinalImpactParameter();
    double d0=beamTrajectoryState.perigeeParameters().transverseImpactParameter();
    double sd0=beamTrajectoryState.perigeeError().transverseImpactParameterError();
    if(sd0>0){
      d0Significance=d0/sd0;
    }else{
      edm::LogWarning ("RecoVertex/PrimaryVertexProducer") 
	<< "Track with zero impact parameter error\n";
      d0Significance=100.;
    }
  }
  inline BeamTransientTrack(const BeamTransientTrack & bt):
    reco::TransientTrack(bt),beamTrajectoryState(bt.beamTrajectoryState){
    z0=bt.z0; d0Significance=bt.d0Significance;
  }
  inline TrajectoryStateClosestToPoint beamState()const{return beamTrajectoryState;}
  inline double zBeam()const{return z0;}
  inline double impactParameterSignificance()const{return d0Significance;}
};
#endif
