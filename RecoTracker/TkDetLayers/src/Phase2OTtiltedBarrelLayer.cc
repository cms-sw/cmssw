#include "Phase2OTtiltedBarrelLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "Phase2EndcapLayerBuilder.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

Phase2OTtiltedBarrelLayer::Phase2OTtiltedBarrelLayer(std::vector<const Phase2OTBarrelRod*>& innerRods,
			                             std::vector<const Phase2OTBarrelRod*>& outerRods,
                                                     vector<const Phase2EndcapRing*>& negRings,
                                                     vector<const Phase2EndcapRing*>& posRings) :
  Phase2OTBarrelLayer(innerRods,outerRods),
  theNegativeRingsComps(negRings.begin(),negRings.end()),
  thePositiveRingsComps(posRings.begin(),posRings.end())
{

  std::vector<const GeometricSearchDet*> theComps;
  theComps.assign(Phase2OTBarrelLayer::components().begin(),Phase2OTBarrelLayer::components().end());
  theComps.insert(theComps.end(),negRings.begin(),negRings.end());
  theComps.insert(theComps.end(),posRings.begin(),posRings.end());

  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin();
      it!=theComps.end();it++){  
    theBasicComps.insert(theBasicComps.end(),	
			 (**it).basicComponents().begin(),
			 (**it).basicComponents().end());
  }

  BarrelDetLayer::initialize();
  theCylinder = cylinder( theComps );

#ifdef EDM_ML_DEBUG
  //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "==== DEBUG Phase2OTtiltedBarrelLayer =====" ; 
  LogTrace("TkDetLayers") << "Phase2OTtiltedBarrelLayer Cyl r,lenght: "
                          << theCylinder->radius() << " , "
                          << theCylinder->bounds().length();

  for (vector<const GeometricSearchDet*>::const_iterator i=theNegativeRingsComps.begin();
       i != theNegativeRingsComps.end(); i++){
    LogTrace("TkDetLayers") << "negative rings in Phase2OT tilted barrel pos z,perp,eta,phi: " 
			    << (**i).position().z()    << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta()  << " , " 
			    << (**i).position().phi()  ;
  }
  
  for (vector<const GeometricSearchDet*>::const_iterator i=Phase2OTBarrelLayer::components().begin();
       i != Phase2OTBarrelLayer::components().end(); i++){
    LogTrace("TkDetLayers") << "rods in Phase2OT tilted barrel pos z,perp,eta,phi: " 
			    << (**i).position().z()    << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta()  << " , " 
			    << (**i).position().phi()  ;
  }

  for (vector<const GeometricSearchDet*>::const_iterator i=thePositiveRingsComps.begin();
       i != thePositiveRingsComps.end(); i++){
    LogTrace("TkDetLayers") << "positive rings in Phase2OT tilted barrel pos z,perp,eta,phi: " 
			    << (**i).position().z()    << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta()  << " , " 
			    << (**i).position().phi()  ;
  }
  LogTrace("TkDetLayers") << "==== end DEBUG Phase2OTtiltedBarrelLayer =====" ; 
  //----------------------------------- 
#endif
}

Phase2OTtiltedBarrelLayer::~Phase2OTtiltedBarrelLayer(){

  vector<const GeometricSearchDet*>::const_iterator i;
  for (i=theNegativeRingsComps.begin(); i!=theNegativeRingsComps.end(); i++) {
    delete *i;
  }
  for (i=thePositiveRingsComps.begin(); i!=thePositiveRingsComps.end(); i++) {
    delete *i;
  }

} 

void 
Phase2OTtiltedBarrelLayer::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					   const MeasurementEstimator& est,
					   std::vector<DetGroup> & result) const {
  vector<DetGroup> closestResultRods;
  vector<DetGroup> closestResultNeg;
  vector<DetGroup> closestResultPos;
  Phase2OTBarrelLayer::groupedCompatibleDetsV(tsos, prop, est, closestResultRods);
  for(auto ring : theNegativeRingsComps){
    ring->groupedCompatibleDetsV(tsos, prop, est, closestResultNeg);
  }
  for(auto ring : thePositiveRingsComps){
    ring->groupedCompatibleDetsV(tsos, prop, est, closestResultPos);
  }

  result.assign(closestResultRods.begin(),closestResultRods.end());
  result.insert(result.end(),closestResultPos.begin(),closestResultPos.end());
  result.insert(result.end(),closestResultNeg.begin(),closestResultNeg.end());

#ifdef EDM_ML_DEBUG
  LogDebug("TkDetLayers") << "==== output di Phase2OTtiltedBarrelLayer =====" ; 
  if(closestResultRods.size() != 0){
    for (auto gr : closestResultRods) {
      LogTrace("TkDetLayers") << "New Rod group:";
      for (auto dge : gr) {
        LogTrace("TkDetLayers") << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" tsos at:" <<dge.trajectoryState().globalPosition();
      }
    }
  }
  if(closestResultNeg.size() != 0){
    for (auto gr : closestResultNeg) {
      LogTrace("TkDetLayers") << "New negative group:";
      for (auto dge : gr) {
        LogTrace("TkDetLayers") << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" tsos at:" <<dge.trajectoryState().globalPosition();
      }
    }
  }
  if(closestResultPos.size() != 0){
    for (auto gr : closestResultPos) {
      LogTrace("TkDetLayers") << "New positive group:";
      for (auto dge : gr) {
        LogTrace("TkDetLayers") << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" tsos at:" <<dge.trajectoryState().globalPosition();
      }
    }
  }

  if(result.size() != 0){
    for (auto gr : result) {
      LogTrace("TkDetLayers") << "Total group:";
      for (auto dge : gr) {
        LogTrace("TkDetLayers") << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" tsos at:" <<dge.trajectoryState().globalPosition();
      }
    }
  } else {
      LogTrace("TkDetLayers") << "result size is zero"; 
  }
#endif
  
}
