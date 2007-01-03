#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace reco;

TransientTrack::TransientTrack() : 
  Track(), tkr_(), theField(0), initialTSOSAvailable(false),
  initialTSCPAvailable(false)
{
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}

TransientTrack::TransientTrack( const Track & tk , const MagneticField* field) : 
  //  Track(tk), tk_(&tk), tkr_(0), initialTSOSAvailable(false) 
  Track(tk), tkr_(), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false) 
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}


TransientTrack::TransientTrack( const TrackRef & tk , const MagneticField* field) : 
  //  Track(*tk), tk_(&(*tk)), tkr_(&tk), initialTSOSAvailable(false) 
  Track(*tk), tkr_(tk), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false)
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(*tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}

TransientTrack::TransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(tk), tkr_(), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), theTrackingGeometry(tg)
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}

TransientTrack::TransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(*tk), tkr_(tk), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), theTrackingGeometry(tg)
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(*tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}


TransientTrack::TransientTrack( const TransientTrack & tt ) :
  Track(tt), tkr_(tt.persistentTrackRef()), theField(tt.field()), 
  initialFTS(tt.initialFreeState()), initialTSOSAvailable(false),
  initialTSCPAvailable(false)
{
//   std::cout << "construct from TransientTrack" << std::endl;
//   initialTSCP = tt.impactPointTSCP();
  if (tt.initialTSOSAvailable) initialTSOS= tt.impactPointState();
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), covariance(), GlobalPoint(0.,0.,0.), theField);
//   std::cout << "construct from TransientTrack OK" << std::endl;
}


TransientTrack& TransientTrack::operator=(const TransientTrack & tt) {
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
  initialTSOSAvailable =  tt.initialTSOSAvailable;
  initialTSCPAvailable = tt.initialTSCPAvailable;
  initialTSCP = tt.initialTSCP;
  initialTSOS = tt.initialTSOS;
  theField = tt.field();
  initialFTS = tt.initialFreeState();
//   std::cout << "assign op. OK" << std::endl;
  
  return *this;
}

void TransientTrack::setES(const edm::EventSetup& setup) {

  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 

}

void TransientTrack::setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& tg) {

  theTrackingGeometry = tg;

}


TrajectoryStateOnSurface TransientTrack::impactPointState() const
{
  if (!initialTSOSAvailable) calculateTSOSAtVertex();
  return initialTSOS;
}

TrajectoryStateClosestToPoint TransientTrack::impactPointTSCP() const
{
  if (!initialTSCPAvailable) {
    initialTSCP = builder(initialFTS, initialFTS.position());
    initialTSCPAvailable = true;
  }
  return initialTSCP;
}

TrajectoryStateOnSurface TransientTrack::outermostMeasurementState() const
{
    TrajectoryStateTransform theTransform;
    return theTransform.outerStateOnSurface((*this),*theTrackingGeometry,theField);

}

TrajectoryStateOnSurface TransientTrack::innermostMeasurementState() const
{

    TrajectoryStateTransform theTransform;
    return theTransform.innerStateOnSurface((*this),*theTrackingGeometry,theField);


}

void TransientTrack::calculateTSOSAtVertex() const
{
  TransverseImpactPointExtrapolator tipe(theField);
  initialTSOS = tipe.extrapolate(initialFTS, initialFTS.position());
 //   edm::LogInfo("TransientTrack") 
//      << "extrapolated state validity:" 
//      << initialTSOS.isValid() << "\n";
  initialTSOSAvailable = true;
}

