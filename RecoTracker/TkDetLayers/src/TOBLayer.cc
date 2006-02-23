#include "RecoTracker/TkDetLayers/interface/TOBLayer.h"
#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"

#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TOBLayer::TOBLayer(vector<const TOBRod*>& innerRods,
		   vector<const TOBRod*>& outerRods) : 
  theInnerRods(innerRods.begin(),innerRods.end()), 
  theOuterRods(outerRods.begin(),outerRods.end())
{
  theRods.assign(theInnerRods.begin(),theInnerRods.end());
  theRods.insert(theRods.end(),theOuterRods.begin(),theOuterRods.end());

  theInnerCylinder = cylinder( theInnerRods);
  theOuterCylinder = cylinder( theOuterRods);

  theInnerBinFinder = BinFinderType(theInnerRods.front()->position().phi(),
				    theInnerRods.size());
  theOuterBinFinder = BinFinderType(theOuterRods.front()->position().phi(),
				    theOuterRods.size());

  
  BarrelDetLayer::initialize();
}


TOBLayer::~TOBLayer(){
  vector<const TOBRod*>::const_iterator i;
  for (i=theRods.begin(); i!=theRods.end(); i++) {
    delete *i;
  }
} 

vector<const GeomDet*> 
TOBLayer::basicComponents() const{
  cout << "temporary dummy implementation of TOBLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}

vector<const GeometricSearchDet*> 
TOBLayer::components() const{
  vector<const GeometricSearchDet*> tmp;
  for(vector<const TOBRod*>::const_iterator it=theRods.begin();it!=theRods.end();it++){
    tmp.push_back(*it);
  }
  return tmp;
}

  
pair<bool, TrajectoryStateOnSurface>
TOBLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TOBLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TOBLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TOBLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
				 const Propagator& prop,
				 const MeasurementEstimator& est) const{

  SubLayerCrossings crossings = computeCrossings( tsos, prop.propagationDirection());

  vector<DetGroup> closestResult;
  addClosest( tsos, prop, est, crossings.closest(), closestResult);
  if (closestResult.empty()){
    vector<DetGroup> nextResult;
    addClosest( tsos, prop, est, crossings.other(), nextResult);
    return nextResult;
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



// private methods for the implementation of groupedCompatibleDets()

SubLayerCrossings TOBLayer::computeCrossings( const TrajectoryStateOnSurface& startingState,
					      PropagationDirection propDir) const
{
  cout << "dummy implementation of TOBLayer::computeCrossings" << endl;
  GlobalPoint startPos( startingState.globalPosition());
  GlobalVector startDir( startingState.globalMomentum());
  double rho( startingState.transverseCurvature());

  //  HelixBarrelCylinderCrossing crossing( startPos, startDir, rho, propDir);
  HelixBarrelCylinderCrossing innerCrossing( startPos, startDir, rho,
					     propDir,*theInnerCylinder);
  if (!innerCrossing.hasSolution()) {
    cout << "ERROR in TkRodBarrelLayer: inner cylinder not crossed by track" << endl;
    //throw DetLogicError("TkRodBarrelLayer: inner subRod not crossed by track");
  }

  GlobalPoint gInnerPoint( innerCrossing.position());
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.phi());
  float innerDist = theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.phi();
  SubLayerCrossing innerSLC( 0, innerIndex, gInnerPoint);

  HelixBarrelCylinderCrossing outerCrossing( startPos, startDir, rho,
					     propDir,*theOuterCylinder);
  if (!outerCrossing.hasSolution()) {
    cout << "ERROR in TkRodBarrelLayer: outer cylinder not crossed by track" << endl;
    //throw DetLogicError("TkRodBarrelLayer: inner subRod not crossed by track");
  }

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

bool TOBLayer::addClosest( const TrajectoryStateOnSurface& tsos,
			   const Propagator& prop,
			   const MeasurementEstimator& est,
			   const SubLayerCrossing& crossing,
			   vector<DetGroup>& result) const
{
  cout << "dummy implementation of TOBLayer::addClosest" << endl;
  return false;
}

float TOBLayer::computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est) const
{
  cout << "dummy implementation of TOBLayer::computeWindowSize" << endl;
  return 0;
}

void TOBLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est,
				const SubLayerCrossing& crossing,
				float window, 
				vector<DetGroup>& result,
				bool checkClosest) const
{
  cout << "dummy implementation of TOBLayer::searchNeighbors" << endl;
}


BoundCylinder* TOBLayer::cylinder( const vector<const TOBRod*>& rods) const 
{
  vector<const GeomDet*> tmp;
  for (vector<const TOBRod*>::const_iterator it=rods.begin(); it!=rods.end(); it++) {    
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    tmp.insert(tmp.end(),tmp2.begin(),tmp2.end());
  }
  return CylinderBuilderFromDet()( tmp.begin(), tmp.end());
}






/*

bool TkRodBarrelLayer::addClosest( const TrajectoryStateOnSurface& tsos,
				   const Propagator& prop,
				   const MeasurementEstimator& est,
				   const SubLayerCrossing& crossing,
				   vector<DetGroup>& result) const
{
//   cout << "Entering TkRodBarrelLayer::addClosest" << endl;

  const vector<DetRod*>& sub( subLayer( crossing.subLayerIndex()));
  const Det* det(sub[crossing.closestDetIndex()]);
  return CompatibleDetToGroupAdder().add( *det, tsos, prop, est, result);
}

void TkRodBarrelLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
					const Propagator& prop,
					const MeasurementEstimator& est,
					const SubLayerCrossing& crossing,
					float window, 
					vector<DetGroup>& result,
					bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<DetRod*>& sLayer( subLayer( crossing.subLayerIndex()));
 
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
    const DetRod* neighborRod = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborRod, window)) break;
    if (!adder.add( *neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + quarter; idet++) {
    const DetRod* neighborRod = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborRod, window)) break;
    if (!adder.add( *neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}

float TkRodBarrelLayer::computeWindowSize( const Det* det, 
					   const TrajectoryStateOnSurface& tsos, 
					   const MeasurementEstimator& est) const
{
  double xmax = est.maximalLocalDisplacement(tsos, dynamic_cast<const BoundPlane&>(det->surface())).x();
  return calculatePhiWindow( xmax, *det, tsos);
}

bool TkRodBarrelLayer::overlap( const GlobalPoint& gpos, const DetRod& rod, float phiWin) const
{
  GlobalPoint crossPoint(gpos);

  // introduce offset (extrapolated point and true propagated point differ by 0.0003 - 0.00033, 
  // due to thickness of Rod of 1 cm)
  const float phiOffset = 0.00034;
  phiWin += phiOffset;

  // detector phi range
  GlobalDetRodRangeZPhi rodRange( rod.specificSurface());
  pair<float,float> phiRange(crossPoint.phi()-phiWin, crossPoint.phi()+phiWin);

//   // debug
//   cout << endl;
//   cout << " overlapInPhi: position, det phi range " 
//        << "("<< rod.position().perp() << ", " << rod.position().phi() << ")  "
//        << rodRange.phiRange().first << " " << rodRange.phiRange().second << endl;
//   cout << " overlapInPhi: cross point phi, window " << crossPoint.phi() << " " << phiWin << endl;
//   cout << " overlapInPhi: search window: " << crossPoint.phi()-phiWin << "  " << crossPoint.phi()+phiWin << endl;

  if ( rangesIntersect(phiRange, rodRange.phiRange(), PhiLess())) {
//     cout << " overlapInPhi:  Ranges intersect " << endl;
    return true;
  } else {
//     cout << "  overlapInPhi: Ranges DO NOT intersect " << endl;
    return false;
  }
} 
*/



