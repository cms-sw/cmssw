#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include <iostream>

using namespace reco;

TrackTransientTrack::TrackTransientTrack() : 
  Track(), tkr_(), theField(0), initialTSOSAvailable(false),
  initialTSCPAvailable(false), blStateAvailable(false)
{
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const MagneticField* field) : 
  Track(tk), tkr_(), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), blStateAvailable(false)
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(tk, field);
}


TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const MagneticField* field) : 
  Track(*tk), tkr_(tk), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), blStateAvailable(false)
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(*tk, field);
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(tk), tkr_(), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), blStateAvailable(false), theTrackingGeometry(tg)
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(tk, field);
}

TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(*tk), tkr_(tk), theField(field), initialTSOSAvailable(false),
  initialTSCPAvailable(false), blStateAvailable(false), theTrackingGeometry(tg)
{
  TrajectoryStateTransform theTransform;
  initialFTS = theTransform.initialFreeState(*tk, field);
}


TrackTransientTrack::TrackTransientTrack( const TrackTransientTrack & tt ) :
  Track(tt), tkr_(tt.persistentTrackRef()), theField(tt.field()), 
  initialFTS(tt.initialFreeState()), initialTSOSAvailable(false),
  initialTSCPAvailable(false)
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

void TrackTransientTrack::setES(const edm::EventSetup& setup) {

  setup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry); 

}

void TrackTransientTrack::setTrackingGeometry(const edm::ESHandle<GlobalTrackingGeometry>& tg) {

  theTrackingGeometry = tg;

}

void TrackTransientTrack::setBeamSpot(const BeamSpot& beamSpot)
{
  theBeamSpot = beamSpot;
}

TrajectoryStateOnSurface TrackTransientTrack::impactPointState() const
{
  if (!initialTSOSAvailable) calculateTSOSAtVertex();
  return initialTSOS;
}

TrajectoryStateClosestToPoint TrackTransientTrack::impactPointTSCP() const
{
  if (!initialTSCPAvailable) {
    initialTSCP = builder(initialFTS, initialFTS.position());
    initialTSCPAvailable = true;
  }
  return initialTSCP;
}

TrajectoryStateOnSurface TrackTransientTrack::outermostMeasurementState() const
{
    TrajectoryStateTransform theTransform;
    return theTransform.outerStateOnSurface((*this),*theTrackingGeometry,theField);
}

TrajectoryStateOnSurface TrackTransientTrack::innermostMeasurementState() const
{
    TrajectoryStateTransform theTransform;
    return theTransform.innerStateOnSurface((*this),*theTrackingGeometry,theField);
}

void TrackTransientTrack::calculateTSOSAtVertex() const
{
  TransverseImpactPointExtrapolator tipe(theField);
  initialTSOS = tipe.extrapolate(initialFTS, initialFTS.position());
  initialTSOSAvailable = true;
}

TrajectoryStateOnSurface 
TrackTransientTrack::stateOnSurface(const GlobalPoint & point) const
{
  TransverseImpactPointExtrapolator tipe(theField);
  return tipe.extrapolate(initialFTS, point);
}

TrajectoryStateClosestToBeamLine TrackTransientTrack::stateAtBeamLine() const
{
  if (!blStateAvailable) {
    TrajectoryStateClosestToBeamLineBuilder blsBuilder;
    trajectoryStateClosestToBeamLine = blsBuilder(initialFTS, theBeamSpot);
    blStateAvailable = true;
  }
  return trajectoryStateClosestToBeamLine;
}

