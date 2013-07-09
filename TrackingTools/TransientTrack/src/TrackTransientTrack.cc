#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include <iostream>

using namespace reco;

TrackTransientTrack::TrackTransientTrack() : 
  Track(), tkr_(), theField(0), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)
{
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const MagneticField* field) : 
  Track(tk), tkr_(), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(tk, field);
}


TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const MagneticField* field) : 
  Track(*tk), tkr_(tk), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(*tk, field);
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(tk), tkr_(), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset), theTrackingGeometry(tg)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(tk, field);
}

TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(*tk), tkr_(tk), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset), theTrackingGeometry(tg)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(*tk, field);
}


TrackTransientTrack::TrackTransientTrack( const TrackTransientTrack & tt ) :
  Track(tt), tkr_(tt.persistentTrackRef()), theField(tt.field()), 
  initialFTS(tt.initialFreeState()), m_TSOS(kUnset), m_TSCP(kUnset)
{
  if (kSet == tt.m_TSOS.load()) {
    initialTSOS= tt.impactPointState();
    m_TSOS.store(kSet);
  }
  if (kSet == tt.m_TSCP.load()) {
    initialTSCP= tt.impactPointTSCP();
    m_TSCP.store(kSet);
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
  m_SCTBL = kUnset;
}

TrajectoryStateOnSurface TrackTransientTrack::impactPointState() const
{
  if(kSet == m_TSOS.load()) return initialTSOS;
  TransverseImpactPointExtrapolator tipe(theField);
  auto tmp = tipe.extrapolate(initialFTS, initialFTS.position());
  char expected = kUnset;
  if(m_TSOS.compare_exchange_strong(expected, kSetting)) {
    initialTSOS = tmp;
    m_TSOS.store(kSet);
    return initialTSOS;
  }
  return tmp;
}

TrajectoryStateClosestToPoint TrackTransientTrack::impactPointTSCP() const
{
  if(kSet == m_TSCP.load()) return initialTSCP;
  auto tmp = builder(initialFTS, initialFTS.position());
  char expected = kUnset;
  if(m_TSCP.compare_exchange_strong(expected, kSetting)) {
    initialTSCP = tmp;
    m_TSCP.store(kSet);
    return initialTSCP;
  }
  return tmp;
}

TrajectoryStateOnSurface TrackTransientTrack::outermostMeasurementState() const
{
    
    return trajectoryStateTransform::outerStateOnSurface((*this),*theTrackingGeometry,theField);
}

TrajectoryStateOnSurface TrackTransientTrack::innermostMeasurementState() const
{
    
    return trajectoryStateTransform::innerStateOnSurface((*this),*theTrackingGeometry,theField);
}

TrajectoryStateOnSurface 
TrackTransientTrack::stateOnSurface(const GlobalPoint & point) const
{
  TransverseImpactPointExtrapolator tipe(theField);
  return tipe.extrapolate(initialFTS, point);
}

TrajectoryStateClosestToBeamLine TrackTransientTrack::stateAtBeamLine() const
{
  if(kSet == m_SCTBL.load()) return trajectoryStateClosestToBeamLine;
  TSCBLBuilderNoMaterial blsBuilder;
  const auto tmp = blsBuilder(initialFTS, theBeamSpot);
  char expected = kUnset;
  if(m_SCTBL.compare_exchange_strong(expected, kSetting)) {
      trajectoryStateClosestToBeamLine = tmp;
      m_SCTBL.store(kSet);
      return trajectoryStateClosestToBeamLine;
  }
  return tmp;
}

