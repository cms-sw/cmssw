#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace reco;

TransientTrack::TransientTrack( const Track & tk ) : 
  //  Track(tk), tk_(&tk), tkr_(0), stateAtVertexAvailable(false) 
  Track(tk), tkr_(0), stateAtVertexAvailable(false) 
{
  std::cout << "construct from Track" << std::endl;
  originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), covariance(), GlobalPoint(0.,0.,0.));
  std::cout << "construct from Track OK" << std::endl;
}


TransientTrack::TransientTrack( const TrackRef & tk ) : 
  //  Track(*tk), tk_(&(*tk)), tkr_(&tk), stateAtVertexAvailable(false) 
  Track(*tk), tkr_(&tk), stateAtVertexAvailable(false) 
{
  std::cout << "construct from TrackRef" << std::endl;
  originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), covariance(), GlobalPoint(0.,0.,0.));
  std::cout << "construct from TrackRef OK" << std::endl;
}


TransientTrack::TransientTrack( const TransientTrack & tt ) :
  //  Track(tt.persistentTrack()), tk_(&tt.persistentTrack()),
    //  tkr_(tt.persistentTrackRef()), stateAtVertexAvailable(false) 
  Track(tt), tkr_(tt.persistentTrackRef()), stateAtVertexAvailable(false) 
{
  std::cout << "construct from TransientTrack" << std::endl;
  originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), covariance(), GlobalPoint(0.,0.,0.));
  std::cout << "construct from TransientTrack OK" << std::endl;
}


TransientTrack& TransientTrack::operator=(const TransientTrack & tt)
{
  std::cout << "assign op." << std::endl;
  if (this == &tt) return *this;
  //
  //  std::cout << tt.tk_ << std::endl;
  std::cout << "assign base." << std::endl;
  Track::operator=(tt);
  std::cout << "done assign base." << std::endl;
  //  tk_ = &(tt.persistentTrack());
  //  tk_ = tt.tk_;
  std::cout << "assign ref." << std::endl;
  tkr_ = tt.persistentTrackRef();
  std::cout << "done assign ref." << std::endl;
  originalTSCP = tt.originalTSCP;
  stateAtVertexAvailable = tt.stateAtVertexAvailable;
  theStateAtVertex = tt.theStateAtVertex;
  std::cout << "assign op. OK" << std::endl;
  
  return *this;
}


TrajectoryStateOnSurface TransientTrack::impactPointState() const
{
  if (!stateAtVertexAvailable) calculateStateAtVertex();
  return theStateAtVertex;
}


void TransientTrack::calculateStateAtVertex() const
{
  //  edm::LogInfo("TransientTrack") 
  //    << "initial state validity:" << originalTSCP.theState() << "\n";

  theStateAtVertex = TransverseImpactPointExtrapolator().extrapolate(
     originalTSCP.theState(), originalTSCP.position());
  //  edm::LogInfo("TransientTrack") 
  //    << "extrapolated state validity:" 
  //    << theStateAtVertex.isValid() << "\n";
  
  stateAtVertexAvailable = true;
}

