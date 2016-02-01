#include "Phase2OTtiltedBarrelLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "Phase2OTEndcapLayerBuilder.h"

//#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

Phase2OTtiltedBarrelLayer::Phase2OTtiltedBarrelLayer(std::vector<const Phase2OTBarrelRod*>& innerRods,
			                             std::vector<const Phase2OTBarrelRod*>& outerRods,
                                                     vector<const Phase2OTEndcapRing*>& negRings, 
                                                     vector<const Phase2OTEndcapRing*>& posRings) : 
  Phase2OTBarrelLayer(innerRods,outerRods),
  theNegativeRingsComps(negRings.begin(),negRings.end()),
  thePositiveRingsComps(posRings.begin(),posRings.end())
{
  std::cout << "yes, we are in the place where we should be ... Phase2OTtiltedBarrelLayer::Phase2OTtiltedBarrelLayer" << std::endl;
  thePhase2OTBarrelLayer = new Phase2OTBarrelLayer(innerRods,outerRods);

  std::vector<const GeometricSearchDet*> theComps;
  theComps.assign(thePhase2OTBarrelLayer->components().begin(),thePhase2OTBarrelLayer->components().end());
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
/*
  theInnerBinFinder = BinFinderType(theInnerComps.front()->position().phi(),
				    theInnerComps.size());
  theOuterBinFinder = BinFinderType(theOuterComps.front()->position().phi(),
				    theOuterComps.size());
*/  
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
  
  for (vector<const GeometricSearchDet*>::const_iterator i=thePhase2OTBarrelLayer->components().begin();
       i != thePhase2OTBarrelLayer->components().end(); i++){
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
/*  
namespace {

  bool groupSortByR(DetGroupElement i,DetGroupElement j) { return (i.det()->position().perp()<j.det()->position().perp()); }

}
*/
void 
Phase2OTtiltedBarrelLayer::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					   const MeasurementEstimator& est,
					   std::vector<DetGroup> & result) const {
  std::cout << "yes, we are in the place where we should be ... Phase2OTtiltedBarrelLayer::groupedCompatibleDetsV" << std::endl;
  vector<DetGroup> closestResultRods;
  vector<DetGroup> closestResultNeg;
  vector<DetGroup> closestResultPos;
  thePhase2OTBarrelLayer->groupedCompatibleDetsV(tsos, prop, est, closestResultRods);
  for(auto ring : theNegativeRingsComps){
    ring->groupedCompatibleDetsV(tsos, prop, est, closestResultNeg);
  }
  for(auto ring : thePositiveRingsComps){
    ring->groupedCompatibleDetsV(tsos, prop, est, closestResultPos);
  }

  result.assign(closestResultRods.begin(),closestResultRods.end());
  result.insert(result.end(),closestResultPos.begin(),closestResultPos.end());
  result.insert(result.end(),closestResultNeg.begin(),closestResultNeg.end());
//  SubLayerCrossings  crossings;
//  crossings = computeCrossings( tsos, prop.propagationDirection());
//  if(! crossings.isValid()) return;  
//  LogDebug("TkDetLayers") << "closest cross pos,r:" << crossings.closest().position()<<","<<crossings.closest().position().perp() 
//  	                  << "  other cross pos,r:" << crossings.other().position()<<","<<crossings.other().position().perp()   << std::endl;

/*
  addClosest( tsos, prop, est, crossings.closest(), closestResult);
  if (closestResult.empty()) {
    addClosest( tsos, prop, est, crossings.other(), result);
    return;
  }

  DetGroupElement closestGel( closestResult.front().front());
  float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);

  //std::cout << "before searchNeighbors closestResult size:" << (closestResult.size()?closestResult.front().size():0) << std::endl;
  
  searchNeighbors( tsos, prop, est, crossings.closest(), window,
		   closestResult, false);
  
  vector<DetGroup> nextResult;
  searchNeighbors( tsos, prop, est, crossings.other(), window,
		   nextResult, true);

  //std::cout << "closestResult size:" << (closestResult.size()?closestResult.front().size():0)
  //	    << " nextResult size:" << (nextResult.size()?nextResult.front().size():0) << std::endl;
  
  int crossingSide = LayerCrossingSide().barrelSide( closestGel.trajectoryState(), prop);
  DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result, 
					  crossings.closestIndex(), crossingSide);

  //std::cout << "Phase2OTtiltedBarrelLayer::groupedCompatibleDetsV - result size=" << result.size() << std::endl;
*/
  LogDebug("TkDetLayers") << "==== output di Phase2OTtiltedBarrelLayer =====" ; 
  if(closestResultRods.size() != 0){
    for (auto gr : closestResultRods) {
      LogTrace("TkDetLayers") << "New Rod group:";
      for (auto dge : gr) {
        LogTrace("TkDetLayers") << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" tsos at:" <<dge.trajectoryState().globalPosition();
      }
    }
  } else {
        LogTrace("TkDetLayers") << "result size is zero";
  }

  if(closestResultNeg.size() != 0){
    for (auto gr : closestResultNeg) {
      LogTrace("TkDetLayers") << "New negative group:";
      for (auto dge : gr) {
        LogTrace("TkDetLayers") << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" tsos at:" <<dge.trajectoryState().globalPosition();
      }
    }
  } else {
      LogTrace("TkDetLayers") << "result size is zero"; 
  }
  if(closestResultPos.size() != 0){
    for (auto gr : closestResultPos) {
      LogTrace("TkDetLayers") << "New positive group:";
      for (auto dge : gr) {
        LogTrace("TkDetLayers") << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" tsos at:" <<dge.trajectoryState().globalPosition();
      }
    }
  } else {
      LogTrace("TkDetLayers") << "result size is zero"; 
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

  
}


// private methods for the implementation of groupedCompatibleDets()

/*
SubLayerCrossings Phase2OTtiltedBarrelLayer::computeCrossings( const TrajectoryStateOnSurface& startingState,
						      PropagationDirection propDir) const
{
  GlobalPoint startPos( startingState.globalPosition());
  GlobalVector startDir( startingState.globalMomentum());
  double rho( startingState.transverseCurvature());
  
  HelixBarrelCylinderCrossing innerCrossing( startPos, startDir, rho,
					     propDir,*theInnerCylinder);

  if (!innerCrossing.hasSolution()) return SubLayerCrossings();
  //{
  //edm::LogInfo(TkDetLayers) << "ERROR in Phase2OTtiltedBarrelLayer: inner cylinder not crossed by track" ;
  //throw DetLayerException("TkRodBarrelLayer: inner subRod not crossed by track");
  //}

  GlobalPoint gInnerPoint( innerCrossing.position());
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.phi());
  float innerDist = theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.phi();
  SubLayerCrossing innerSLC( 0, innerIndex, gInnerPoint);

  HelixBarrelCylinderCrossing outerCrossing( startPos, startDir, rho,
					     propDir,*theOuterCylinder);

  if (!outerCrossing.hasSolution()) return SubLayerCrossings();
  //if (!outerCrossing.hasSolution()) {
  //  throw DetLayerException("Phase2OTtiltedBarrelLayer: inner cylinder not crossed by track");
  //}

  GlobalPoint gOuterPoint( outerCrossing.position());
  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.phi());
  float outerDist = theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.phi() ;
  SubLayerCrossing outerSLC( 1, outerIndex, gOuterPoint);
  
  innerDist *= PhiLess()( theInnerBinFinder.binPosition(innerIndex),gInnerPoint.phi()) ? -1. : 1.; 
  outerDist *= PhiLess()( theOuterBinFinder.binPosition(outerIndex),gOuterPoint.phi()) ? -1. : 1.; 
  if (innerDist < 0.) { innerDist += 2.*Geom::pi();}
  if (outerDist < 0.) { outerDist += 2.*Geom::pi();}
  

  if (innerDist < outerDist) {
    return SubLayerCrossings( innerSLC, outerSLC, 0);
  }
  else {
    return SubLayerCrossings( outerSLC, innerSLC, 1);
  } 
}
bool Phase2OTtiltedBarrelLayer::addClosest( const TrajectoryStateOnSurface& tsos,
				   const Propagator& prop,
				   const MeasurementEstimator& est,
				   const SubLayerCrossing& crossing,
				   vector<DetGroup>& result) const
{
  const vector<const GeometricSearchDet*>& sub( subLayer( crossing.subLayerIndex()));
  const GeometricSearchDet* det(sub[crossing.closestDetIndex()]);
  return CompatibleDetToGroupAdder::add( *det, tsos, prop, est, result);
}
*/
/*
float Phase2OTtiltedBarrelLayer::computeWindowSize( const GeomDet* det, 
					   const TrajectoryStateOnSurface& tsos, 
					   const MeasurementEstimator& est) const
{
  double xmax = 
    est.maximalLocalDisplacement(tsos, det->surface()).x();
  return calculatePhiWindow( xmax, *det, tsos);
}


double Phase2OTtiltedBarrelLayer::calculatePhiWindow( double Xmax, const GeomDet& det,
					     const TrajectoryStateOnSurface& state) const
{

  LocalPoint startPoint = state.localPosition();
  LocalVector shift( Xmax , 0. , 0.);
  LocalPoint shift1 = startPoint + shift;
  LocalPoint shift2 = startPoint + (-shift); 
  //LocalPoint shift2( startPoint); //original code;
  //shift2 -= shift;

  double phi1 = det.surface().toGlobal(shift1).phi();
  double phi2 = det.surface().toGlobal(shift2).phi();
  double phiStart = state.globalPosition().phi();
  double phiWin = min(fabs(phiStart-phi1),fabs(phiStart-phi2));

  return phiWin;
}


void Phase2OTtiltedBarrelLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
					const Propagator& prop,
					const MeasurementEstimator& est,
					const SubLayerCrossing& crossing,
					float window, 
					vector<DetGroup>& result,
					bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<const GeometricSearchDet*>& sLayer( subLayer( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if ( PhiLess()( gCrossingPos.phi(), sLayer[closestIndex]->position().phi())) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  const BinFinderType& binFinder = (crossing.subLayerIndex()==0 ? theInnerBinFinder : theOuterBinFinder);

  CompatibleDetToGroupAdder adder;
  int quarter = sLayer.size()/4;
  for (int idet=negStartIndex; idet >= negStartIndex - quarter; idet--) {
    const GeometricSearchDet* neighborRod = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborRod, window)) break;
    if (!adder.add( *neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + quarter; idet++) {
    const GeometricSearchDet* neighborRod = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborRod, window)) break;
    if (!adder.add( *neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}

bool Phase2OTtiltedBarrelLayer::overlap( const GlobalPoint& gpos, const GeometricSearchDet& gsdet, float phiWin) const
{
  GlobalPoint crossPoint(gpos);

  // introduce offset (extrapolated point and true propagated point differ by 0.0003 - 0.00033, 
  // due to thickness of Rod of 1 cm) 
  const float phiOffset = 0.00034;  //...TOBE CHECKED LATER...
  phiWin += phiOffset;

  // detector phi range
  
  pair<float,float> phiRange(crossPoint.phi()-phiWin, crossPoint.phi()+phiWin);

  return rangesIntersect(phiRange, gsdet.surface().phiSpan(), PhiLess());

} 


*/
