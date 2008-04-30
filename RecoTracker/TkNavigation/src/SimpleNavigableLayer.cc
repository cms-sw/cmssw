#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

using namespace std;

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       const BarrelDetLayer* bl,
				       DLC& result) const
{
  TSOS propState = propagator(dir).propagate( fts, bl->specificSurface());
  if ( !propState.isValid()) return false;
 
  //if requested check that the layer is crossed on the right side
  if (theCheckCrossingSide){
    bool backTobackTransverse = (fts.position().x()*propState.globalPosition().y() - fts.position().y()*propState.globalPosition().x())<0;
    bool backToback = propState.globalPosition().basicVector().dot(fts.position().basicVector())<0;

    if (backTobackTransverse || backToback ){
      LogTrace("TkNavigation") << "Crossing over prevented!\nStaring from (x,y,z,r) (" 
			       << fts.position().x()<<","<< fts.position().y()<<","<< fts.position().z()<<","<<fts.position().perp()
			       << ") going to TSOS (x,y,z,r)" 
			       << propState.globalPosition().x()<<","<< propState.globalPosition().y()<<","<< propState.globalPosition().z()<<","<<propState.globalPosition().perp()<<")";
      return false;
    
    /*
    //we have to check the crossing side only if we are going to something smaller
    if (fts.position().perp()>bl->specificSurface().radius() || 
    fabs(fts.position().z())>bl->surface().bounds().length()/2. ){
    if (propState.globalPosition().basicVector().dot(fts.position().basicVector())<0){
    LogTrace("TkNavigation") << "Crossing over prevented!\nStaring from (x,y,z,r) (" 
    << fts.position().x()<<","<< fts.position().y()<<","<< fts.position().z()<<","<<fts.position().perp()
    << ") going to TSOS (x,y,z,r)" 
    << propState.globalPosition().x()<<","<< propState.globalPosition().y()<<","<< propState.globalPosition().z()<<","<<propState.globalPosition().perp()<<")";; 
    return false;
    }
    } 	
    */
    }}

  const Bounds& bounds( bl->specificSurface().bounds());
  float length = bounds.length() / 2.f;

  // take into account the thickness of the layer
  float deltaZ = bounds.thickness()/2. / 
    fabs( tan( propState.globalDirection().theta()));

  // take into account the error on the predicted state
  const float nSigma = theEpsilon;  // temporary reuse of epsilon
  if (propState.hasError()) {
    deltaZ += nSigma * sqrt( fts.cartesianError().position().czz());
  }

  // cout << "SimpleNavigableLayer BarrelDetLayer deltaZ = " << deltaZ << endl;

  float zpos = propState.globalPosition().z();
  if ( fabs( zpos) < length + deltaZ) result.push_back( bl);

  if ( fabs( zpos) < length - deltaZ) return true;
  else return false;
}

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       const ForwardDetLayer* fl,
				       DLC& result) const
{
  TSOS propState = propagator(dir).propagate( fts, fl->specificSurface());
  if ( !propState.isValid()) return false;

  //if requested avoids crossing over the tracker 
  if (theCheckCrossingSide){
    bool backTobackTransverse = (fts.position().x()*propState.globalPosition().y() - fts.position().y()*propState.globalPosition().x())<0;
    bool backToback = propState.globalPosition().basicVector().dot(fts.position().basicVector())<0;

    if (backTobackTransverse || backToback ){
      LogTrace("TkNavigation") << "Crossing over prevented!\nStaring from (x,y,z,r) (" 
			       << fts.position().x()<<","<< fts.position().y()<<","<< fts.position().z()<<","<<fts.position().perp()
			       << ") going to TSOS (x,y,z,r)" 
			       << propState.globalPosition().x()<<","<< propState.globalPosition().y()<<","<< propState.globalPosition().z()<<","<<propState.globalPosition().perp()<<")";; 
      return false;
    
  //	if (fts.position().z()*propState.globalPosition().z() < 0) return false;
    }}


  float rpos = propState.globalPosition().perp();
  float innerR = fl->specificSurface().innerRadius();
  float outerR = fl->specificSurface().outerRadius();
 
  // take into account the thickness of the layer
  float deltaR = fl->surface().bounds().thickness()/2. *
    fabs( tan( propState.localDirection().theta()));

  // take into account the error on the predicted state
  const float nSigma = theEpsilon;
  if (propState.hasError()) {
    LocalError err = propState.localError().positionError();
    // ignore correlation for the moment...
    deltaR += nSigma * sqrt(err.xx() + err.yy());
  }

  // cout << "SimpleNavigableLayer BarrelDetLayer deltaR = " << deltaR << endl;

  if ( innerR-deltaR < rpos && rpos < outerR+deltaR) result.push_back( fl);
  
  if ( innerR+deltaR < rpos && rpos < outerR-deltaR) return true;
  else return false;
}

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       const DLC& layers,
				       DLC& result) const
{

  // cout << "Entering SimpleNavigableLayer::wellInside" << endl;

  for (DLC::const_iterator i = layers.begin(); i != layers.end(); i++) {
    const BarrelDetLayer* bl = dynamic_cast<const BarrelDetLayer*>(*i);
    if ( bl != 0) {
      if (wellInside( fts, dir, bl, result)) return true;
    }
    else {
      const ForwardDetLayer* fl = dynamic_cast<const ForwardDetLayer*>(*i);
      if ( fl == 0) edm::LogError("TkNavigation") << "dynamic_cast<const ForwardDetLayer*> failed" ;
      if (wellInside( fts, dir, fl, result)) return true;
    }
  }
  return false;
}

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       ConstBDLI begin, ConstBDLI end,
				       DLC& result) const
{
  for ( ConstBDLI i = begin; i < end; i++) {
    if (wellInside( fts, dir, *i, result)) return true;
  }
  return false;
}

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       ConstFDLI begin, ConstFDLI end,
				       DLC& result) const
{
  for ( ConstFDLI i = begin; i < end; i++) {
    if (wellInside( fts, dir, *i, result)) return true;
  }
  return false;
}

Propagator& SimpleNavigableLayer::propagator( PropagationDirection dir) const
{
#ifndef CMS_NO_MUTABLE
  thePropagator.setPropagationDirection(dir);
  return thePropagator;
#else
  SimpleNavigableLayer* mthis = const_cast<SimpleNavigableLayer*>(this);
  mthis->thePropagator.setPropagationDirection(dir);
  return mthis->thePropagator;
#endif
}

void SimpleNavigableLayer::pushResult( DLC& result, const BDLC& tmp) const 
{
  for ( ConstBDLI i = tmp.begin(); i != tmp.end(); i++) {
    result.push_back(*i);
  }
}

void SimpleNavigableLayer::pushResult( DLC& result, const FDLC& tmp) const 
{
  for ( ConstFDLI i = tmp.begin(); i != tmp.end(); i++) {
    result.push_back(*i);
  }
}

