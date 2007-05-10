#ifndef VertexReco_BeamTrackTransientTrack_h
#define VertexReco_BeamTrackTransientTrack_h

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"

class BeamTransientTrack: public reco::TransientTrack {
 private:
  TrajectoryStateClosestToPoint beamTrajectoryState;
  double d0,sd0,z0;
 public:
  inline BeamTransientTrack(const reco::TransientTrack &t, const GlobalPoint &  beamPosition):
    reco::TransientTrack(t),beamTrajectoryState(t.trajectoryStateClosestToPoint(beamPosition)){
    std::cout << "position= " << beamPosition << std::endl;
    std::cout<< "created:\n " << beamTrajectoryState.theState() << std::endl;
    //z0=beamTrajectoryState.perigeeParameters().longitudinalImpactParameter();
    //d0=beamTrajectoryState.perigeeParameters().transverseImpactParameter();
    //sd0=beamTrajectoryState.perigeeError().transverseImpactParameterError();
    d0=t.impactPointTSCP().position().perp();
    sd0=sqrt(t.impactPointTSCP().perigeeError().covarianceMatrix()(4,4));
    z0=t.impactPointTSCP().position().z();
    std::cout << "z,d,s " << z0 << " " << d0 << " " << sd0 <<std::endl;
  }
  inline BeamTransientTrack(const BeamTransientTrack & bt):
    reco::TransientTrack(bt),beamTrajectoryState(bt.beamTrajectoryState){
    z0=bt.z0; sd0=bt.sd0; d0=bt.d0; 
    std::cout<<"copied:\n " << beamTrajectoryState.theState() << std::endl;
    std::cout << "z,d,s " << z0 << " " << d0 << " " << sd0 <<std::endl;
  }
  inline TrajectoryStateClosestToPoint beamState()const{return beamTrajectoryState;}
  inline double zBeam()const{return z0;}
  inline double impactParameterSignificance()const{return d0/sd0;}
};
#endif
