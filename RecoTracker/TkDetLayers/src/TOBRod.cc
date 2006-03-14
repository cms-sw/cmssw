#include "RecoTracker/TkDetLayers/interface/TOBRod.h"
#include "TrackingTools/DetLayers/interface/RodPlaneBuilderFromDet.h"


#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"

#include "Utilities/General/interface/CMSexception.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossingByCircle.h"


typedef GeometricSearchDet::DetWithState DetWithState;

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

  theInnerBinFinder = BinFinderType(theInnerDets.begin(), theInnerDets.end());
  theOuterBinFinder = BinFinderType(theOuterDets.begin(), theOuterDets.end());


}

TOBRod::~TOBRod(){
  
} 

vector<const GeomDet*> 
TOBRod::basicComponents() const{
  return theDets;
}

vector<const GeometricSearchDet*> 
TOBRod::components() const{
  cout << "temporary dummy implementation of TOBRod::components()!!" << endl;
  return vector<const GeometricSearchDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TOBRod::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		    const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TOBRod::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TOBRod::compatibleDets( const TrajectoryStateOnSurface& startingState,
			const Propagator& prop, 
			const MeasurementEstimator& est) const{
  
  // standard implementation of compatibleDets() for class which have 
  // groupedCompatibleDets implemented.
  // This code should be moved in a common place intead of being 
  // copied many times.
  
  vector<DetWithState> result;  
  vector<DetGroup> vectorGroups = groupedCompatibleDets(startingState,prop,est);
  for(vector<DetGroup>::const_iterator itDG=vectorGroups.begin();
      itDG!=vectorGroups.end();itDG++){
    for(vector<DetGroupElement>::const_iterator itDGE=itDG->begin();
	itDGE!=itDG->end();itDGE++){
      result.push_back(DetWithState(itDGE->det(),itDGE->trajectoryState()));
    }
  }
  return result;  
}


vector<DetGroup> 
TOBRod::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est) const{
  
  vector<DetGroup> closestResult;
  SubLayerCrossings  crossings; 
  try{
    crossings = computeCrossings( tsos, prop.propagationDirection());  
  }
  catch(Genexception& err){ //In ORCA, it was a DetLogicError exception
    cout << "Aie, got an exception in DetRodTwoR::groupedCompatibleDets:" 
	 << err.what() << endl;
    return closestResult;
  }    
  addClosest( tsos, prop, est, crossings.closest(), closestResult);

  if (closestResult.empty()){
    vector<DetGroup> nextResult;
    addClosest( tsos, prop, est, crossings.other(), nextResult);
    if(nextResult.empty())    return nextResult;

    DetGroupElement nextGel( nextResult.front().front());  
    int crossingSide = LayerCrossingSide().barrelSide( nextGel.trajectoryState(), prop);
    DetGroupMerger merger;
    return  merger.orderAndMergeTwoLevels( closestResult, nextResult, 
					   crossings.closestIndex(), crossingSide);   
  }
  
  DetGroupElement closestGel( closestResult.front().front());
  float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);

  searchNeighbors( tsos, prop, est, crossings.closest(), window,
		   closestResult, false);

  vector<DetGroup> nextResult;
  searchNeighbors( tsos, prop, est, crossings.other(), window,
		   nextResult, true);

  int crossingSide = LayerCrossingSide().barrelSide( closestGel.trajectoryState(), prop);
  DetGroupMerger merger;
  return merger.orderAndMergeTwoLevels( closestResult, nextResult, 
					crossings.closestIndex(), crossingSide);

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
  if (!innerPath.first) {
    cout << "ERROR in DetRodTwoR: inner subRod not crossed by track" << endl;
    //throw DetLogicError("DetRodTwoR: inner subRod not crossed by track");
    throw Genexception("DetRodTwoR: inner subRod not crossed by track");
  }
  GlobalPoint gInnerPoint( crossing.position(innerPath.second));
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.z());
  float innerDist = fabs( theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.z());
  SubLayerCrossing innerSLC( 0, innerIndex, gInnerPoint);

  pair<bool,double> outerPath = crossing.pathLength( *theOuterPlane);
  if (!outerPath.first) {
    cout << "ERROR in DetRodTwoR: outer subRod not crossed by track" << endl;
    //throw DetLogicError("DetRodTwoR: outer subRod not crossed by track");
    throw Genexception("DetRodTwoR: outer subRod not crossed by track");
  }
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
  return CompatibleDetToGroupAdder().add( *sRod[crossing.closestDetIndex()], 
					  tsos, prop, est, result);
}


float TOBRod::computeWindowSize( const GeomDet* det, 
				 const TrajectoryStateOnSurface& tsos, 
				 const MeasurementEstimator& est) const
{
  return
    est.maximalLocalDisplacement(tsos, dynamic_cast<const BoundPlane&>(det->surface())).y();
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

  CompatibleDetToGroupAdder adder;
  for (int idet=negStartIndex; idet >= 0; idet--) {
    if (!overlap( gCrossingPos, *sRod[idet], window)) break;
    if (!adder.add( *sRod[idet], tsos, prop, est, result)) break;
  }
  for (int idet=posStartIndex; idet < static_cast<int>(sRod.size()); idet++) {
    if (!overlap( gCrossingPos, *sRod[idet], window)) break;
    if (!adder.add( *sRod[idet], tsos, prop, est, result)) break;
  }
}



bool TOBRod::overlap( const GlobalPoint& crossPoint, const GeomDet& det, float window) const
{
  // check if the z window around TSOS overlaps with the detector theDet (with a 1% margin added)
  
  //   const float tolerance = 0.1;
  const float relativeMargin = 1.01;

  LocalPoint localCrossPoint( det.surface().toLocal(crossPoint));
  //   if (fabs(localCrossPoint.z()) > tolerance) {
  //     cout << "TOBRod::overlap calculation assumes point on surface, but it is off by "
  // 	 << localCrossPoint.z() << endl;
  //   }

  float localY = localCrossPoint.y();
  float detHalfLength = det.surface().bounds().length()/2.;

  //   cout << "TOBRod::overlap: Det at " << det.position() << " hit at " << localY 
  //        << " Window " << window << " halflength "  << detHalfLength << endl;
  
  if ( ( fabs(localY)-window) < relativeMargin*detHalfLength ) { // FIXME: margin hard-wired!
    return true;
  } else {
    return false;
  }
}

