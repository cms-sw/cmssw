#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace reco;

TransientTrack::TransientTrack( const Track & tk , const MagneticField* field) : 
  //  Track(tk), tk_(&tk), tkr_(0), stateAtVertexAvailable(false) 
  Track(tk), tkr_(0), theField(field), stateAtVertexAvailable(false) 
{
  originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}


TransientTrack::TransientTrack( const TrackRef & tk , const MagneticField* field) : 
  //  Track(*tk), tk_(&(*tk)), tkr_(&tk), stateAtVertexAvailable(false) 
  Track(*tk), tkr_(&tk), theField(field), stateAtVertexAvailable(false) 
{
  originalTSCP = TrajectoryStateClosestToPoint
    (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}


TransientTrack::TransientTrack( const TransientTrack & tt ) :
  Track(tt), tkr_(tt.persistentTrackRef()), theField(tt.field()), stateAtVertexAvailable(false) 
{
//   std::cout << "construct from TransientTrack" << std::endl;
  originalTSCP = tt.impactPointTSCP();
  if (tt.stateAtVertexAvailable) theStateAtVertex= tt.impactPointState();
//   originalTSCP = TrajectoryStateClosestToPoint
//     (parameters(), covariance(), GlobalPoint(0.,0.,0.), theField);
//   std::cout << "construct from TransientTrack OK" << std::endl;
}


TransientTrack& TransientTrack::operator=(const TransientTrack & tt)
{
//   std::cout << "assign op." << std::endl;
  if (this == &tt) return *this;
  //
  //  std::cout << tt.tk_ << std::endl;
//   std::cout << "assign base." << std::endl;
  Track::operator=(tt);
//   std::cout << "done assign base." << std::endl;
  //  tk_ = &(tt.persistentTrack());
  //  tk_ = tt.tk_;
//   std::cout << "assign ref." << std::endl;
  tkr_ = tt.persistentTrackRef();
//   std::cout << "done assign ref." << std::endl;
  originalTSCP = tt.originalTSCP;
  stateAtVertexAvailable = tt.stateAtVertexAvailable;
  theStateAtVertex = tt.theStateAtVertex;
  theField = tt.field();
//   std::cout << "assign op. OK" << std::endl;
  
  return *this;
}

void TransientTrack::setES(const edm::EventSetup& setup) {

  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 

}


TrajectoryStateOnSurface TransientTrack::impactPointState() const
{
  if (!stateAtVertexAvailable) calculateStateAtVertex();
  return theStateAtVertex;
}

TrajectoryStateOnSurface TransientTrack::outermostMeasurementState() const
{
    math::XYZPoint outPosXYZ = (*tkr_)->outerPosition();
    math::XYZVector outMomXYZ = (*tkr_)->outerMomentum();

    GlobalPoint outPos(outPosXYZ.x(),outPosXYZ.y(),outPosXYZ.z());
    GlobalVector outMom(outMomXYZ.x(),outMomXYZ.y(),outMomXYZ.z());
 //  edm::LogInfo("TransientTrack")
 //     << "outermost meas: pos: " <<outPos <<" mom: "<<outMom; 

    GlobalTrajectoryParameters par(outPos, outMom, (*tkr_)->charge(), theField);
    FreeTrajectoryState fts(par);
    trackingRecHit_iterator last = (*tkr_)->recHitsEnd();
    last--;
 //  edm::LogInfo("TransientTrack")
//      << "id "<<(*last)->geographicalId().rawId();

    TrajectoryStateOnSurface result(fts,theTrackingGeometry->idToDet((*last)->geographicalId())->surface());

   // if (result.isValid())
   // edm::LogInfo("TransientTrack")
  //    << "outermost TSOS result: pos: " <<result.globalPosition() <<" mom: "<<result.globalMomentum();

    return result;

}

TrajectoryStateOnSurface TransientTrack::innermostMeasurementState() const
{
    math::XYZPoint inPosXYZ = (*tkr_)->innerPosition();
    math::XYZVector inMomXYZ = (*tkr_)->innerMomentum();

    GlobalPoint inPos(inPosXYZ.x(),inPosXYZ.y(),inPosXYZ.z());
    GlobalVector inMom(inMomXYZ.x(),inMomXYZ.y(),inMomXYZ.z());
 //   edm::LogInfo("TransientTrack")
 //     << "innermost meas: pos: " <<inPos <<" mom: "<<inMom;

    GlobalTrajectoryParameters par(inPos, inMom, (*tkr_)->charge(), theField);

    FreeTrajectoryState fts(par);
 //  edm::LogInfo("TransientTrack")
//      << "id "<<(*last)->geographicalId().rawId();

    TrajectoryStateOnSurface result(fts,theTrackingGeometry->idToDet((*(*tkr_)->recHitsBegin())->geographicalId())->surface());

  //  if (result.isValid())
   // edm::LogInfo("TransientTrack")
   //   << "innermost TSOS result: pos: " <<result.globalPosition() <<" mom: "<<result.globalMomentum();

    return result;

}

void TransientTrack::calculateStateAtVertex() const
{
  //  edm::LogInfo("TransientTrack") 
  //    << "initial state validity:" << originalTSCP.theState() << "\n";
  TransverseImpactPointExtrapolator tipe(theField);
  theStateAtVertex = tipe.extrapolate(
     originalTSCP.theState(), originalTSCP.position());
 //   edm::LogInfo("TransientTrack") 
//      << "extrapolated state validity:" 
//      << theStateAtVertex.isValid() << "\n";
  
  stateAtVertexAvailable = true;
}

