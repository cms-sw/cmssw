#include "RecoTracker/TkNavigation/interface/SimpleNavigableLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include <set>
#include <TrackingTools/TrackAssociator/interface/DetIdInfo.h>

using namespace std;

TrajectoryStateOnSurface SimpleNavigableLayer::crossingState(const FreeTrajectoryState& fts,
							     PropagationDirection dir) const{
  TSOS propState;
  //self propagating. step one: go close to the center
  GlobalPoint initialPoint = fts.position();
  TransverseImpactPointExtrapolator middle;
  GlobalPoint center(0,0,0);
  propState = middle.extrapolate(fts, center, propagator(dir));
  if ( !propState.isValid()) return TrajectoryStateOnSurface();
  
  FreeTrajectoryState & dest = *propState.freeState();
  GlobalPoint middlePoint = dest.position();
  const float toCloseToEachOther2 = 1e-4*1e-4;
  if unlikely( (middlePoint-initialPoint).mag2() < toCloseToEachOther2){
    LogDebug("SimpleNavigableLayer")<<"initial state and PCA are identical. Things are bound to fail. Do not add the link.";
    return TrajectoryStateOnSurface();
  }
  
 
  /*
  std::string dirS;
  if (dir==alongMomentum) dirS = "alongMomentum";
  else if (dir==oppositeToMomentum) dirS = "oppositeToMomentum";
  else dirS = "anyDirection";
  */

  LogDebug("SimpleNavigableLayer")<<"self propagating("<< dir <<") from:\n"
				  <<fts<<"\n"
				  <<dest<<"\n"
				  <<" and the direction is: "<<dir;
  
  //second propagation to go on the other side of the barrel
  //propState = propagator(dir).propagate( dest, detLayer()->specificSurface());
  propState = propagator(dir).propagate( dest, detLayer()->surface());
  if ( !propState.isValid()) return TrajectoryStateOnSurface();
  
  FreeTrajectoryState & dest2 = *propState.freeState();
  GlobalPoint finalPoint = dest2.position();
  LogDebug("SimpleNavigableLayer")<<"second propagation("<< dir <<") to: \n"
				  <<dest2;
  double finalDot = (middlePoint - initialPoint).basicVector().dot((finalPoint-middlePoint).basicVector());
  if unlikely(finalDot<0){ // check that before and after are in different side.
    LogDebug("SimpleNavigableLayer")<<"switch side back: ABORT.";
    return TrajectoryStateOnSurface();
  }
  return propState;
}

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       const BarrelDetLayer* bl,
				       DLC& result) const
{

  TSOS propState = (bl==detLayer()) ?
    crossingState(fts,dir)
    :
    propagator(dir).propagate( fts, bl->specificSurface());

  if ( !propState.isValid()) return false;
 
  //if requested check that the layer is crossed on the right side
  if (theCheckCrossingSide){
    bool backTobackTransverse = (fts.position().x()*propState.globalPosition().x() + fts.position().y()*propState.globalPosition().y())<0;
    bool backToback = propState.globalPosition().basicVector().dot(fts.position().basicVector())<0;

    if (backTobackTransverse || backToback ){
      LogTrace("TkNavigation") << "Crossing over prevented!\nStaring from (x,y,z,r) (" 
			       << fts.position().x()<<","<< fts.position().y()<<","<< fts.position().z()<<","<<fts.position().perp()
			       << ") going to TSOS (x,y,z,r)" 
			       << propState.globalPosition().x()<<","<< propState.globalPosition().y()
			       <<","<< propState.globalPosition().z()<<","<<propState.globalPosition().perp()<<")";
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
  float length = bounds.length()*0.5f;

  // take into account the thickness of the layer
  float deltaZ = 0.5f*bounds.thickness() *
    std::abs(propState.globalDirection().z())/propState.globalDirection().perp();


  // take into account the error on the predicted state
  const float nSigma = theEpsilon;  // temporary reuse of epsilon
  if (propState.hasError()) {
    deltaZ += nSigma * sqrt( fts.cartesianError().position().czz());
  }

  // cout << "SimpleNavigableLayer BarrelDetLayer deltaZ = " << deltaZ << endl;

  float zpos = propState.globalPosition().z();
  if ( std::abs( zpos) < length + deltaZ) result.push_back( bl);

  return std::abs( zpos) < length - deltaZ;
}

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       const ForwardDetLayer* fl,
				       DLC& result) const
{
  TSOS propState = propagator(dir).propagate( fts, fl->specificSurface());
  if ( !propState.isValid()) return false;

  if (fl==detLayer()){
    LogDebug("SimpleNavigableLayer")<<"self propagating from:\n"
				    <<fts<<"\n to \n"
				    <<*propState.freeState();
  }

  //if requested avoids crossing over the tracker 
  if (theCheckCrossingSide){
    bool backTobackTransverse = (fts.position().x()*propState.globalPosition().x() + fts.position().y()*propState.globalPosition().y())<0;
    bool backToback = propState.globalPosition().basicVector().dot(fts.position().basicVector())<0;

    if (backTobackTransverse || backToback ){
      LogTrace("TkNavigation") << "Crossing over prevented!\nStaring from (x,y,z,r) (" 
			       << fts.position().x()<<","<< fts.position().y()<<","<< fts.position().z()<<","<<fts.position().perp()
			       << ") going to TSOS (x,y,z,r)" 
			       << propState.globalPosition().x()<<","<< propState.globalPosition().y()
			       <<","<< propState.globalPosition().z()<<","<<propState.globalPosition().perp()<<")";; 
      return false;
    
  //	if (fts.position().z()*propState.globalPosition().z() < 0) return false;
    }}


  float rpos = propState.globalPosition().perp();
  float innerR = fl->specificSurface().innerRadius();
  float outerR = fl->specificSurface().outerRadius();
 
  // take into account the thickness of the layer
  float deltaR = 0.5f*fl->surface().bounds().thickness() *
    propState.localDirection().perp()/std::abs(propState.localDirection().z());

  // take into account the error on the predicted state
  const float nSigma = theEpsilon;
  if (propState.hasError()) {
    LocalError err = propState.localError().positionError();
    // ignore correlation for the moment...
    deltaR += nSigma * sqrt(err.xx() + err.yy());
  }

  // cout << "SimpleNavigableLayer BarrelDetLayer deltaR = " << deltaR << endl;

  if ( innerR-deltaR < rpos && rpos < outerR+deltaR) result.push_back( fl);
  return ( innerR+deltaR < rpos && rpos < outerR-deltaR);
}

bool SimpleNavigableLayer::wellInside( const FreeTrajectoryState& fts,
				       PropagationDirection dir,
				       const DLC& layers,
				       DLC& result) const
{
  for (auto l : layers) {
    if (l->isBarrel()) {
	const BarrelDetLayer * bl = reinterpret_cast<const BarrelDetLayer *>(l);
	if (wellInside( fts, dir, bl, result)) return true;
      }
    else {
      const ForwardDetLayer* fl = reinterpret_cast<const ForwardDetLayer*>(l);
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


std::vector< const DetLayer * > SimpleNavigableLayer::compatibleLayers (const FreeTrajectoryState &fts, 
									PropagationDirection timeDirection,
									int& counter) const {
  typedef std::vector<const DetLayer*> Lvect;
  typedef std::set<const DetLayer *> Lset;  

  //initiate the first iteration
  Lvect && someLayers = nextLayers(fts,timeDirection);
  if (someLayers.empty()) {
    LogDebug("SimpleNavigableLayer")  <<"Number of compatible layers: "<< 0;
    return someLayers;
  }


  Lset collect; //a container of unique instances. to avoid duplicates
  Lset layerToTry, nextLayerToTry;//set used for iterations
  layerToTry.insert(someLayers.begin(),someLayers.end());
  
  while (!layerToTry.empty() && (counter++)<=150){
    LogDebug("SimpleNavigableLayer")
      <<counter<<"] going to check on : "<<layerToTry.size()<<" next layers.";
    //clear this set first, it will be swaped with layerToTry
    nextLayerToTry.clear();
    for (auto toTry : layerToTry){
      //add the layer you tried.
      LogDebug("SimpleNavigableLayer")
	<<counter<<"] adding layer with pointer: "<<(toTry)
	<<" first detid: "<<DetIdInfo::info((toTry)->basicComponents().front()->geographicalId());
      if (!collect.insert(toTry).second) continue;
      
      //find the next layers from it
      Lvect && nextLayers = (toTry)->nextLayers(fts,timeDirection);
      LogDebug("SimpleNavigableLayer")
	<<counter<<"] this layer has : "<<nextLayers.size()<<" next layers.";
      nextLayerToTry.insert(nextLayers.begin(),nextLayers.end());
    } // layerToTry
    //swap now that you where to go next.
    layerToTry.swap(nextLayerToTry);
  }
  if(counter>=150) {
    edm::LogWarning("SimpleNavigableLayer") << "WARNING: compatibleLayers() more than 150 iterations!!! Bailing out..";
    counter = -1;
    return Lvect();
  }

  LogDebug("SimpleNavigableLayer")
   <<"Number of compatible layers: "<< collect.size();
  
  return Lvect(collect.begin(),collect.end());

}
