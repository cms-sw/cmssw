#include "TrackingTools/TransientTrack/interface/GsfTransientTrack.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GsfTools/interface/GsfPropagatorAdapter.h"
#include "TrackingTools/GsfTools/interface/MultiTrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include <iostream>

using namespace reco;
using namespace std;

GsfTransientTrack::GsfTransientTrack() : 
  GsfTrack(), tkr_(), theField(0), initialTSOSAvailable(false),
  initialTSCPAvailable(false)
{
  init();
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}

GsfTransientTrack::GsfTransientTrack( const GsfTrack & tk , const MagneticField* field) : 
  //  GsfTrack(tk), tk_(&tk), tkr_(0), initialTSOSAvailable(false) 
  GsfTrack(tk), tkr_(), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false) 
{
  init();
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}


GsfTransientTrack::GsfTransientTrack( const GsfTrackRef & tk , const MagneticField* field) : 
  //  GsfTrack(*tk), tk_(&(*tk)), tkr_(&tk), initialTSOSAvailable(false) 
  GsfTrack(*tk), tkr_(tk), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false)
{
  init();
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(*tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}

GsfTransientTrack::GsfTransientTrack( const GsfTrack & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  GsfTrack(tk), tkr_(), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), theTrackingGeometry(tg)
{
  init();
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}

GsfTransientTrack::GsfTransientTrack( const GsfTrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  GsfTrack(*tk), tkr_(tk), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), theTrackingGeometry(tg)
{
  init();
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(*tk, field);
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), pt(), covariance(), GlobalPoint(0.,0.,0.), theField);
}


GsfTransientTrack::GsfTransientTrack( const GsfTransientTrack & tt ) :
  GsfTrack(tt), tkr_(tt.persistentTrackRef()), theField(tt.field()), 
  initialFTS(tt.initialFreeState()), initialTSOSAvailable(false),
  initialTSCPAvailable(false)
{
  init();
//   std::cout << "construct from GsfTransientTrack" << std::endl;
//   initialTSCP = tt.impactPointTSCP();
  if (tt.initialTSOSAvailable) initialTSOS= tt.impactPointState();
//   initialTSCP = TrajectoryStateClosestToPoint
//     (parameters(), covariance(), GlobalPoint(0.,0.,0.), theField);
//   std::cout << "construct from GsfTransientTrack OK" << std::endl;
}


// GsfTransientTrack& GsfTransientTrack::operator=(const TransientTrack & tt) {
// //   std::cout << "assign op." << std::endl;
//   if (this == &tt) return *this;
//   //
//   //  std::cout << tt.tk_ << std::endl;
// //   std::cout << "assign base." << std::endl;
//   Track::operator=(tt);
// //   std::cout << "done assign base." << std::endl;
//   //  tk_ = &(tt.persistentTrack());
//   //  tk_ = tt.tk_;
// //   std::cout << "assign ref." << std::endl;
//   tkr_ = tt.persistentTrackRef();
//   initialTSOSAvailable =  tt.initialTSOSAvailable;
//   initialTSCPAvailable = tt.initialTSCPAvailable;
//   initialTSCP = tt.initialTSCP;
//   initialTSOS = tt.initialTSOS;
//   theField = tt.field();
//   initialFTS = tt.initialFreeState();
// //   std::cout << "assign op. OK" << std::endl;
//   
//   return *this;
// }

void GsfTransientTrack::init()
{
  thePropagator = 
    new GsfPropagatorAdapter(AnalyticalPropagator(theField, alongMomentum));
  theTIPExtrapolator = new TransverseImpactPointExtrapolator(*thePropagator);
}


void GsfTransientTrack::setES(const edm::EventSetup& setup) {

  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 

}

void GsfTransientTrack::setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& tg) {

  theTrackingGeometry = tg;

}


TrajectoryStateOnSurface GsfTransientTrack::impactPointState() const
{
  if (!initialTSOSAvailable) calculateTSOSAtVertex();
  return initialTSOS;
}

TrajectoryStateClosestToPoint GsfTransientTrack::impactPointTSCP() const
{
  if (!initialTSCPAvailable) {
    initialTSCP = builder(initialFTS, initialFTS.position());
    initialTSCPAvailable = true;
  }
  return initialTSCP;
}

TrajectoryStateOnSurface GsfTransientTrack::outermostMeasurementState() const
{
    MultiTrajectoryStateTransform theTransform;
    return theTransform.outerStateOnSurface((*this),*theTrackingGeometry,theField);
}

TrajectoryStateOnSurface GsfTransientTrack::innermostMeasurementState() const
{
    MultiTrajectoryStateTransform theTransform;
    return theTransform.innerStateOnSurface((*this),*theTrackingGeometry,theField);
}

void GsfTransientTrack::calculateTSOSAtVertex() const
{
  TransverseImpactPointExtrapolator tipe(theField);
  initialTSOS = tipe.extrapolate(initialFTS, initialFTS.position());
 //   edm::LogInfo("GsfTransientTrack") 
//      << "extrapolated state validity:" 
//      << initialTSOS.isValid() << "\n";
  initialTSOSAvailable = true;
}

TrajectoryStateOnSurface 
GsfTransientTrack::stateOnSurface(const GlobalPoint & point) const
{
  return theTIPExtrapolator->extrapolate(innermostMeasurementState(), point);
}


TrajectoryStateClosestToPoint 
GsfTransientTrack::trajectoryStateClosestToPoint( const GlobalPoint & point ) const
{
  return builder(stateOnSurface(point), point);
}
