#include "TOBRod.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"


using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

class DetZLess {
public:
  bool operator()(const GeomDet* a,const GeomDet* b) 
  {
    return (a->position().z() < b->position().z());
  } 
};


TOBRod::TOBRod(vector<const GeomDet*>& innerDets,
	       vector<const GeomDet*>& outerDets):
  theInnerDets(innerDets),theOuterDets(outerDets)
{
  theDets.assign(theInnerDets.begin(),theInnerDets.end());
  theDets.insert(theDets.end(),theOuterDets.begin(),theOuterDets.end());


  RodPlaneBuilderFromDet planeBuilder;
  setPlane( planeBuilder( theDets));
  theInnerPlane = planeBuilder( theInnerDets);
  theOuterPlane = planeBuilder( theOuterDets);


  sort(theDets.begin(),theDets.end(),DetZLess());
  sort(theInnerDets.begin(),theInnerDets.end(),DetZLess());
  sort(theOuterDets.begin(),theOuterDets.end(),DetZLess());
  theInnerBinFinder = BinFinderType(theInnerDets.begin(), theInnerDets.end());
  theOuterBinFinder = BinFinderType(theOuterDets.begin(), theOuterDets.end());


 
  LogDebug("TkDetLayers") << "==== DEBUG TOBRod =====" ; 
  for (vector<const GeomDet*>::const_iterator i=theInnerDets.begin();
       i != theInnerDets.end(); i++){
    LogDebug("TkDetLayers") << "inner TOBRod's Det pos z,perp,eta,phi: " 
			    << (**i).position().z() << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta() << " , " 
			    << (**i).position().phi() ;
  }
  
  for (vector<const GeomDet*>::const_iterator i=theOuterDets.begin();
       i != theOuterDets.end(); i++){
    LogDebug("TkDetLayers") << "outer TOBRod's Det pos z,perp,eta,phi: " 
			    << (**i).position().z() << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta() << " , " 
			    << (**i).position().phi() ;
  }
  LogDebug("TkDetLayers") << "==== end DEBUG TOBRod =====" ; 
  


}

TOBRod::~TOBRod(){
  
} 


const vector<const GeometricSearchDet*>& 
TOBRod::components() const{
  throw DetLayerException("TOBRod doesn't have GeometricSearchDet components");
}
  
pair<bool, TrajectoryStateOnSurface>
TOBRod::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		    const MeasurementEstimator&) const{
  edm::LogError("TkDetLayers") << "temporary dummy implementation of TOBRod::compatible()!!" ;
  return pair<bool,TrajectoryStateOnSurface>();
}





void
TOBRod::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
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
    int crossingSide = LayerCrossingSide().barrelSide( nextGel.trajectoryState(), prop);
    DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result, 
					   crossings.closestIndex(), crossingSide);   
  } else {
  
    DetGroupElement closestGel( closestResult.front().front());
    float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);

    searchNeighbors( tsos, prop, est, crossings.closest(), window,
     		     closestResult, false);

    vector<DetGroup> nextResult;
    searchNeighbors( tsos, prop, est, crossings.other(), window,
		     nextResult, true);

    int crossingSide = LayerCrossingSide().barrelSide( closestGel.trajectoryState(), prop);
    DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result, 
					    crossings.closestIndex(), crossingSide);
  }
}


SubLayerCrossings 
TOBRod::computeCrossings( const TrajectoryStateOnSurface& startingState,
			  PropagationDirection propDir) const
{
  GlobalPoint startPos( startingState.globalPosition());
  GlobalVector startDir( startingState.globalMomentum());
  double rho( startingState.transverseCurvature());

  HelixBarrelPlaneCrossingByCircle crossing( startPos, startDir, rho, propDir);

  pair<bool,double> innerPath = crossing.pathLength( *theInnerPlane);
  if (!innerPath.first) return SubLayerCrossings();

  GlobalPoint gInnerPoint( crossing.position(innerPath.second));
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.z());
  float innerDist = fabs( theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.z());
  SubLayerCrossing innerSLC( 0, innerIndex, gInnerPoint);

  pair<bool,double> outerPath = crossing.pathLength( *theOuterPlane);
  if (!outerPath.first) return SubLayerCrossings();

  GlobalPoint gOuterPoint( crossing.position(outerPath.second));
  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.z());
  float outerDist = fabs( theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.z());
  SubLayerCrossing outerSLC( 1, outerIndex, gOuterPoint);

  if (innerDist < outerDist) {
    return SubLayerCrossings( innerSLC, outerSLC, 0);
  }
  else {
    return SubLayerCrossings( outerSLC, innerSLC, 1);
  } 
}




bool 
TOBRod::addClosest( const TrajectoryStateOnSurface& tsos,
		    const Propagator& prop,
		    const MeasurementEstimator& est,
		    const SubLayerCrossing& crossing,
		    vector<DetGroup>& result) const
{

  const vector<const GeomDet*>& sRod( subRod( crossing.subLayerIndex()));
  return CompatibleDetToGroupAdder::add( *sRod[crossing.closestDetIndex()], 
					 tsos, prop, est, result);
}


float TOBRod::computeWindowSize( const GeomDet* det, 
				 const TrajectoryStateOnSurface& tsos, 
				 const MeasurementEstimator& est) const
{
  return
    est.maximalLocalDisplacement(tsos, det->surface()).y();
}




void TOBRod::searchNeighbors( const TrajectoryStateOnSurface& tsos,
			      const Propagator& prop,
			      const MeasurementEstimator& est,
			      const SubLayerCrossing& crossing,
			      float window, 
			      vector<DetGroup>& result,
			      bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<const GeomDet*>& sRod( subRod( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if (gCrossingPos.z() < sRod[closestIndex]->surface().position().z()) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  typedef CompatibleDetToGroupAdder Adder;
  for (int idet=negStartIndex; idet >= 0; idet--) {
    if (!overlap( gCrossingPos, *sRod[idet], window)) break;
    if (!Adder::add( *sRod[idet], tsos, prop, est, result)) break;
  }
  for (int idet=posStartIndex; idet < static_cast<int>(sRod.size()); idet++) {
    if (!overlap( gCrossingPos, *sRod[idet], window)) break;
    if (!Adder::add( *sRod[idet], tsos, prop, est, result)) break;
  }
}



bool TOBRod::overlap( const GlobalPoint& crossPoint, const GeomDet& det, float window) const
{
  // check if the z window around TSOS overlaps with the detector theDet (with a 1% margin added)
  
  //   const float tolerance = 0.1;
  const float relativeMargin = 1.01;

  LocalPoint localCrossPoint( det.surface().toLocal(crossPoint));
  //   if (fabs(localCrossPoint.z()) > tolerance) {
  //     edm::LogInfo(TkDetLayers) << "TOBRod::overlap calculation assumes point on surface, but it is off by "
  // 	 << localCrossPoint.z() ;
  //   }

  float localY = localCrossPoint.y();
  float detHalfLength = det.surface().bounds().length()/2.;

  //   edm::LogInfo(TkDetLayers) << "TOBRod::overlap: Det at " << det.position() << " hit at " << localY 
  //        << " Window " << window << " halflength "  << detHalfLength ;
  
  if ( ( fabs(localY)-window) < relativeMargin*detHalfLength ) { // FIXME: margin hard-wired!
    return true;
  } else {
    return false;
  }
}

