#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include <iostream>

/*
 * ThreadSafe statement:
 * This class is using mutable member data: initialTSOS, initialTSCP,
 * trajectoryStateClosestToBeamLine. To guarantee thread safeness we
 * rely on helper member data: m_TSOS, m_TSCP and m_SCTBL, respectively.
 * Each time we'll change mutable member data we rely on specific order of the
 * operator= and the store. It is important since C++11 will guarantee that
 * the value changed by the operator= will be seen by all threads as occuring
 * before the call to store and therefore the kSet == m_TSOS.load is always
 * guaranteed to be true if and only if the thread will see the most recent
 * value of initialTSOS
 */

using namespace reco;

TrackTransientTrack::TrackTransientTrack() : 
  Track(), tkr_(), hasTime(false), timeExt_(0.), dtErrorExt_(0.), theField(nullptr), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)

{
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const MagneticField* field) : 
  Track(tk), tkr_(), hasTime(false), timeExt_(0.), dtErrorExt_(0.), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(tk, field);
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const double time,
                                          const double dtime,
                                          const MagneticField* field) : 
  Track(tk), tkr_(), hasTime(true), timeExt_(time), dtErrorExt_(dtime), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(tk, field);
}


TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const MagneticField* field) : 
  Track(*tk), tkr_(tk), hasTime(false), timeExt_(0.), dtErrorExt_(0.), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(*tk, field);
}

TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const double time,
                                          const double dtime,
                                          const MagneticField* field) : 
  Track(*tk), tkr_(tk), hasTime(true), timeExt_(time), dtErrorExt_(dtime), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(*tk, field);
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(tk), tkr_(), hasTime(false), timeExt_(0.), dtErrorExt_(0.), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset), theTrackingGeometry(tg)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(tk, field);
}

TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const MagneticField* field, const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(*tk), tkr_(tk), hasTime(false), timeExt_(0.), dtErrorExt_(0.), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset), theTrackingGeometry(tg)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(*tk, field);
}

TrackTransientTrack::TrackTransientTrack( const Track & tk , const double time,
                                          const double dtime,
                                          const MagneticField* field, 
                                          const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(tk), tkr_(), hasTime(true), timeExt_(time), dtErrorExt_(dtime), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset), theTrackingGeometry(tg)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(tk, field);
}

TrackTransientTrack::TrackTransientTrack( const TrackRef & tk , const double time, 
                                          const double dtime,
                                          const MagneticField* field, 
                                          const edm::ESHandle<GlobalTrackingGeometry>& tg) :
  Track(*tk), tkr_(tk), hasTime(true), timeExt_(time), dtErrorExt_(dtime), theField(field), m_TSOS(kUnset), m_TSCP(kUnset), m_SCTBL(kUnset), theTrackingGeometry(tg)
{
  
  initialFTS = trajectoryStateTransform::initialFreeState(*tk, field);
}


TrackTransientTrack::TrackTransientTrack( const TrackTransientTrack & tt ) :
  Track(tt), tkr_(tt.persistentTrackRef()), 
  hasTime(tt.hasTime), timeExt_(tt.timeExt()), dtErrorExt_(tt.dtErrorExt()),
  theField(tt.field()), 
  initialFTS(tt.initialFreeState()), m_TSOS(kUnset), m_TSCP(kUnset)
{
  // see ThreadSafe statement above about the order of operator= and store
  if (kSet == tt.m_TSOS.load()) {
    initialTSOS= tt.impactPointState();
    m_TSOS.store(kSet);
  }
  // see ThreadSafe statement above about the order of operator= and store
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
  // see ThreadSafe statement above about the order of operator= and store
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
  // see ThreadSafe statement above about the order of operator= and store
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
  // see ThreadSafe statement above about the order of operator= and store
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

