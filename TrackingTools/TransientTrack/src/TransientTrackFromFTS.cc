#include "TrackingTools/TransientTrack/interface/TransientTrackFromFTS.h"
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

TransientTrackFromFTS::TransientTrackFromFTS() : 
  theField(0), initialTSOSAvailable(false), initialTSCPAvailable(false),
  trackAvailable(false)
{}

TransientTrackFromFTS::TransientTrackFromFTS(const FreeTrajectoryState & fts) :
  initialFTS(fts), theField(&(initialFTS.parameters().magneticField())),
  initialTSOSAvailable(false), initialTSCPAvailable(false), trackAvailable(false)
{}


TransientTrackFromFTS::TransientTrackFromFTS(const FreeTrajectoryState & fts,
	const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  initialFTS(fts), theField(&(initialFTS.parameters().magneticField())),
  initialTSOSAvailable(false), initialTSCPAvailable(false), trackAvailable(false),
  theTrackingGeometry(tg)
{}


TransientTrackFromFTS::TransientTrackFromFTS( const TransientTrackFromFTS & tt ) :
  initialFTS(tt.initialFreeState()), theField(tt.field()), initialTSOSAvailable(false),
  initialTSCPAvailable(false), trackAvailable(false)
{
  if (tt.initialTSOSAvailable) {
    initialTSOS= tt.impactPointState();
    initialTSOSAvailable = true;
  }
  if (tt.initialTSCPAvailable) {
    initialTSCP= tt.impactPointTSCP();
    initialTSCPAvailable = true;
  }
}



void TransientTrackFromFTS::setES(const edm::EventSetup& setup)
{
  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 
}

void
TransientTrackFromFTS::setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& tg)
{
  theTrackingGeometry = tg;
}


TrajectoryStateOnSurface TransientTrackFromFTS::impactPointState() const
{
  if (!initialTSOSAvailable) calculateTSOSAtVertex();
  return initialTSOS;
}

TrajectoryStateClosestToPoint TransientTrackFromFTS::impactPointTSCP() const
{
  if (!initialTSCPAvailable) {
    initialTSCP = builder(initialFTS, initialFTS.position());
    initialTSCPAvailable = true;
  }
  return initialTSCP;
}

TrajectoryStateOnSurface TransientTrackFromFTS::outermostMeasurementState() const
{
  throw cms::Exception("LogicError") << 
    "TransientTrack built from a FreeTrajectoryState (TransientTrackFromFTS) can not have an outermostMeasurementState";
}

TrajectoryStateOnSurface TransientTrackFromFTS::innermostMeasurementState() const
{
  throw cms::Exception("LogicError") << 
    "TransientTrack built from a FreeTrajectoryState (TransientTrackFromFTS) can not have an innermostMeasurementState";
}

void TransientTrackFromFTS::calculateTSOSAtVertex() const
{
  TransverseImpactPointExtrapolator tipe(theField);
  initialTSOS = tipe.extrapolate(initialFTS, initialFTS.position());
  initialTSOSAvailable = true;
}

TrajectoryStateOnSurface 
TransientTrackFromFTS::stateOnSurface(const GlobalPoint & point) const
{
  TransverseImpactPointExtrapolator tipe(theField);
  return tipe.extrapolate(initialFTS, point);
}

const Track & TransientTrackFromFTS::track() const
{
  if (!trackAvailable) {
    GlobalPoint v = initialFTS.position();
    math::XYZPoint  pos( v.x(), v.y(), v.z() );
    GlobalVector p = initialFTS.momentum();
    math::XYZVector mom( p.x(), p.y(), p.z() );

    theTrack = Track(0., 0., pos, mom, initialFTS.charge(),
	initialFTS.curvilinearError());
    trackAvailable = true;
  }
  return theTrack;
}
