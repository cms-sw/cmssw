#include "PixelBlade.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "BladeShapeBuilderFromDet.h"
#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

PixelBlade::~PixelBlade(){}

PixelBlade::PixelBlade(vector<const GeomDet*>& frontDets,
		       vector<const GeomDet*>& backDets):		       
  theFrontDets(frontDets), theBackDets(backDets) 
{
  theDets.assign(theFrontDets.begin(),theFrontDets.end());
  theDets.insert(theDets.end(),theBackDets.begin(),theBackDets.end());

  theDiskSector      = BladeShapeBuilderFromDet()(theDets);  
  theFrontDiskSector = BladeShapeBuilderFromDet()(theFrontDets);
  theBackDiskSector  = BladeShapeBuilderFromDet()(theBackDets);   


  //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "DEBUG INFO for PixelBlade" ;
  LogDebug("TkDetLayers") << "Blade z, perp, innerRadius, outerR: " 
			  << this->position().z() << " , "
			  << this->position().perp() << " , "
			  << theDiskSector->innerRadius() << " , "
			  << theDiskSector->outerRadius() ;

  for(vector<const GeomDet*>::const_iterator it=theFrontDets.begin(); 
      it!=theFrontDets.end(); it++){
    LogDebug("TkDetLayers") << "frontDet phi,z,r: " 
			    << (*it)->position().phi() << " , "
			    << (*it)->position().z()   << " , "
			    << (*it)->position().perp() ;;
  }

  for(vector<const GeomDet*>::const_iterator it=theBackDets.begin(); 
      it!=theBackDets.end(); it++){
    LogDebug("TkDetLayers") << "backDet phi,z,r: " 
			    << (*it)->position().phi() << " , "
			    << (*it)->position().z()   << " , "
			    << (*it)->position().perp() ;
  }
  //-----------------------------------

}


const vector<const GeometricSearchDet*>& 
PixelBlade::components() const{
  throw DetLayerException("TOBRod doesn't have GeometricSearchDet components"); 
}

pair<bool, TrajectoryStateOnSurface>
PixelBlade::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
			const MeasurementEstimator&) const{
  edm::LogError("TkDetLayers") << "temporary dummy implementation of PixelBlade::compatible()!!" ;
  return pair<bool,TrajectoryStateOnSurface>();
}



void
PixelBlade::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					   const MeasurementEstimator& est,
					   std::vector<DetGroup> & result) const{
 SubLayerCrossings  crossings; 
  crossings = computeCrossings( tsos, prop.propagationDirection());
  if(! crossings.isValid()) return;

  vector<DetGroup> closestResult;
  addClosest( tsos, prop, est, crossings.closest(), closestResult);

  if (closestResult.empty()){
    vector<DetGroup> nextResult;
    addClosest( tsos, prop, est, crossings.other(), nextResult);
    if(nextResult.empty())    return;
    
    DetGroupElement nextGel( nextResult.front().front());  
    int crossingSide = LayerCrossingSide().endcapSide( nextGel.trajectoryState(), prop);

    DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result,
					    crossings.closestIndex(), crossingSide);   
  }
  else {
    DetGroupElement closestGel( closestResult.front().front());
    float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);
    
    searchNeighbors( tsos, prop, est, crossings.closest(), window,
		     closestResult, false);
    
    vector<DetGroup> nextResult;
    searchNeighbors( tsos, prop, est, crossings.other(), window,
		     nextResult, true);
    
    int crossingSide = LayerCrossingSide().endcapSide( closestGel.trajectoryState(), prop);
    DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result,
					    crossings.closestIndex(), crossingSide);
  }
}

SubLayerCrossings 
PixelBlade::computeCrossings( const TrajectoryStateOnSurface& startingState,
			      PropagationDirection propDir) const
{
  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition());
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum());
  double rho( startingState.transverseCurvature());

  HelixArbitraryPlaneCrossing crossing( startPos, startDir, rho, propDir);

  pair<bool,double> innerPath = crossing.pathLength( *theFrontDiskSector);
  if (!innerPath.first) return SubLayerCrossings();

  GlobalPoint gInnerPoint( crossing.position(innerPath.second));
  //Code for use of binfinder
  //int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.perp());  
  //float innerDist = fabs( theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.z());
  int innerIndex = findBin(gInnerPoint.perp(),0);
  float innerDist = fabs( findPosition(innerIndex,0).perp() - gInnerPoint.perp());
  SubLayerCrossing innerSLC( 0, innerIndex, gInnerPoint);

  pair<bool,double> outerPath = crossing.pathLength( *theBackDiskSector);
  if (!outerPath.first) return SubLayerCrossings();

  GlobalPoint gOuterPoint( crossing.position(outerPath.second));
  //Code for use of binfinder
  //int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.perp());
  //float outerDist = fabs( theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.perp());
  int outerIndex  = findBin(gOuterPoint.perp(),1);
  float outerDist = fabs( findPosition(outerIndex,1).perp() - gOuterPoint.perp());
  SubLayerCrossing outerSLC( 1, outerIndex, gOuterPoint);

  if (innerDist < outerDist) {
    return SubLayerCrossings( innerSLC, outerSLC, 0);
  }
  else {
    return SubLayerCrossings( outerSLC, innerSLC, 1);
  } 
}




bool 
PixelBlade::addClosest( const TrajectoryStateOnSurface& tsos,
			const Propagator& prop,
			const MeasurementEstimator& est,
			const SubLayerCrossing& crossing,
			vector<DetGroup>& result) const
{

  const vector<const GeomDet*>& sBlade( subBlade( crossing.subLayerIndex()));
  return CompatibleDetToGroupAdder().add( *sBlade[crossing.closestDetIndex()], 
					  tsos, prop, est, result);
}


float PixelBlade::computeWindowSize( const GeomDet* det, 
				     const TrajectoryStateOnSurface& tsos, 
				     const MeasurementEstimator& est) const
{
  return
    est.maximalLocalDisplacement(tsos, det->surface()).x();
}




void PixelBlade::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				  const Propagator& prop,
				  const MeasurementEstimator& est,
				  const SubLayerCrossing& crossing,
				  float window, 
				  vector<DetGroup>& result,
				  bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<const GeomDet*>& sBlade( subBlade( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if (gCrossingPos.perp() < sBlade[closestIndex]->surface().position().perp()) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  typedef CompatibleDetToGroupAdder Adder;
  for (int idet=negStartIndex; idet >= 0; idet--) {
    if (!overlap( gCrossingPos, *sBlade[idet], window)) break;
    if (!Adder::add( *sBlade[idet], tsos, prop, est, result)) break;
  }
  for (int idet=posStartIndex; idet < static_cast<int>(sBlade.size()); idet++) {
    if (!overlap( gCrossingPos, *sBlade[idet], window)) break;
    if (!Adder::add( *sBlade[idet], tsos, prop, est, result)) break;
  }
}



bool PixelBlade::overlap( const GlobalPoint& crossPoint, const GeomDet& det, float window) const
{
  // check if the z window around TSOS overlaps with the detector theDet (with a 1% margin added)
  
  //   const float tolerance = 0.1;
  const float relativeMargin = 1.01;

  LocalPoint localCrossPoint( det.surface().toLocal(crossPoint));
  //   if (fabs(localCrossPoint.z()) > tolerance) {
  //     edm::LogInfo(TkDetLayers) << "PixelBlade::overlap calculation assumes point on surface, but it is off by "
  // 	 << localCrossPoint.z() ;
  //   }

  float localX = localCrossPoint.x();
  float detHalfLength = det.surface().bounds().length()/2.;

  //   edm::LogInfo(TkDetLayers) << "PixelBlade::overlap: Det at " << det.position() << " hit at " << localY 
  //        << " Window " << window << " halflength "  << detHalfLength ;
  
  if ( ( fabs(localX)-window) < relativeMargin*detHalfLength ) { // FIXME: margin hard-wired!
    return true;
  } else {
    return false;
  }
}

int 
PixelBlade::findBin( float R,int diskSectorIndex) const 
{
  vector<const GeomDet*> localDets = diskSectorIndex==0 ? theFrontDets : theBackDets;
  
  int theBin = 0;
  float rDiff = fabs( R - localDets.front()->surface().position().perp());;
  for (vector<const GeomDet*>::const_iterator i=localDets.begin(); i !=localDets.end(); i++){
    float testDiff = fabs( R - (**i).surface().position().perp());
    if ( testDiff < rDiff) {
      rDiff = testDiff;
      theBin = i - localDets.begin();
    }
  }
  return theBin;
}



GlobalPoint 
PixelBlade::findPosition(int index,int diskSectorType) const 
{
  vector<const GeomDet*> diskSector = diskSectorType == 0 ? theFrontDets : theBackDets; 
  return (diskSector[index])->surface().position();
}

