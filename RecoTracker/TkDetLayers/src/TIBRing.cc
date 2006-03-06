#include "RecoTracker/TkDetLayers/interface/TIBRing.h"

#include "Utilities/General/interface/CMSexception.h"

#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/simple_stat.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelPlaneCrossing2OrderLocal.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TIBRing::TIBRing(vector<const GeomDet*>& theGeomDets):
  theDets(theGeomDets.begin(),theGeomDets.end())
{
  //checkRadius( first, last);
  //sort( theDets.begin(), theDets.end(), DetLessPhi());
  //checkPeriodicity( theDets.begin(), theDets.end());

  theBinFinder = BinFinderType( theDets.front()->surface().position().phi(),
				theDets.size());

  theCylinder = CylinderBuilderFromDet()( theDets.begin(), theDets.end());

  computeHelicity();
}



void TIBRing::checkRadius(vector<const GeomDet*>::const_iterator first,
			  vector<const GeomDet*>::const_iterator last)
{
  // check radius range
  float rMin = 10000.;
  float rMax = 0.;
  for (vector<const GeomDet*>::const_iterator i=first; i!=last; i++) {
    float r = (**i).surface().position().perp();
    if (r < rMin) rMin = r;
    if (r > rMax) rMax = r;
  }
  if (rMax - rMin > 0.1) throw Genexception(
    "TIBRing construction failed: detectors not at constant radius");
}


void TIBRing::checkPeriodicity(vector<const GeomDet*>::const_iterator first,
			       vector<const GeomDet*>::const_iterator last)
{
  vector<double> adj_diff(last-first-1);
  for (int i=0; i < static_cast<int>(adj_diff.size()); i++) {
    vector<const GeomDet*>::const_iterator curent = first + i;
    adj_diff[i] = (**(curent+1)).surface().position().phi() - 
                  (**curent).surface().position().phi();
  }
  double step = stat_mean( adj_diff);
  double phi_step = 2.*Geom::pi()/(last-first);  

  if ( fabs(step-phi_step)/phi_step > 0.01) {
    int ndets = last-first;
    cout << "TIBRing Warning: not periodic. ndets=" << ndets << endl;
    for (int j=0; j<ndets; j++) {
      cout << "Dets(r,phi): (" << theDets[j]->surface().position().perp() 
	   << "," << theDets[j]->surface().position().phi() << ") " << endl;
    }
    cout << endl;
    throw Genexception( "Error: TIBRing is not periodic");
  }
}

void TIBRing::computeHelicity() {

  const GeomDet& det = *theDets.front();
  GlobalVector radial = det.surface().position() - GlobalPoint(0,0,0);
  GlobalVector normal = det.surface().toGlobal( LocalVector(0,0,1));
//   cout << "BarrelDetRing::computeHelicity: phi(normal) " << normal.phi()
//        << " phi(radial) " << radial.phi() << endl;
  if (PhiLess()( normal.phi(), radial.phi())) {
    theHelicity = 0;  // smaller phi angles mean "inner" group
  }
  else {
    theHelicity = 1;  // smaller phi angles mean "outer" group
  }
}


TIBRing::~TIBRing(){

} 

vector<const GeomDet*> 
TIBRing::basicComponents() const{
  return theDets;
}
  
pair<bool, TrajectoryStateOnSurface>
TIBRing::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TIBRing::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TIBRing::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TIBRing::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est) const
{
  vector<DetGroup> result;  //to clean out
  vector<DetGroup> closestResult;
  SubRingCrossings  crossings; 
  try{
    crossings = computeCrossings( tsos, prop.propagationDirection());
  }
  catch(Genexception& err){
    cout << "Aie, got a Genexception in TIBRing::groupedCompatibleDets:" 
	 << err.what() << endl;
    return closestResult;
  }
  CompatibleDetToGroupAdder adder;
  adder.add( *theDets[theBinFinder.binIndex(crossings.closestIndex)], 
	     tsos, prop, est, closestResult);
  
  if(closestResult.empty()){
    vector<DetGroup> nextResult;
    adder.add( *theDets[theBinFinder.binIndex(crossings.nextIndex)], 
	       tsos, prop, est, nextResult);
    return nextResult;
  }      

  DetGroupElement closestGel( closestResult.front().front());
  float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);

  //vector<DetGroup> result;
  float detWidth = closestGel.det()->surface().bounds().width();
  if (crossings.nextDistance < detWidth + window) {
    vector<DetGroup> nextResult;
    if (adder.add( *theDets[theBinFinder.binIndex(crossings.nextIndex)], 
		   tsos, prop, est, nextResult)) {
      int crossingSide = LayerCrossingSide().barrelSide( tsos, prop);
      DetGroupMerger merger;
      if (crossings.closestIndex < crossings.nextIndex) {
	result = merger.orderAndMergeTwoLevels( closestResult, nextResult, 
						theHelicity, crossingSide);
      }
      else {
	result = merger.orderAndMergeTwoLevels( nextResult, closestResult, 
						theHelicity, crossingSide);
      }
    }
    else {
      result = closestResult;
    }
  }
  
  // only loop over neighbors (other than closest and next) if window is BIG
  if (window > 0.5*detWidth) {
    searchNeighbors( tsos, prop, est, crossings, window, result);
  } 
  return result;
}

void TIBRing::searchNeighbors( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       const SubRingCrossings& crossings,
			       float window, 
			       vector<DetGroup>& result) const
{
  CompatibleDetToGroupAdder adder;
  int crossingSide = LayerCrossingSide().barrelSide( tsos, prop);
  DetGroupMerger merger;
  
  int negStart = min( crossings.closestIndex, crossings.nextIndex) - 1;
  int posStart = max( crossings.closestIndex, crossings.nextIndex) + 1;
  
  int quarter = theDets.size()/4;
  for (int idet=negStart; idet >= negStart - quarter+1; idet--) {
    const GeomDet* neighbor = theDets[theBinFinder.binIndex(idet)];
    // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
    // maybe also add shallow crossing angle test here???
    vector<DetGroup> tmp;
    if (!adder.add( *neighbor, tsos, prop, est, tmp)) break;
    result = merger.orderAndMergeTwoLevels( tmp, result, theHelicity, crossingSide);
  }
  for (int idet=posStart; idet < posStart + quarter-1; idet++) {
    const GeomDet* neighbor = theDets[theBinFinder.binIndex(idet)];
    // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
    // maybe also add shallow crossing angle test here???
    vector<DetGroup> tmp;
    if (!adder.add( *neighbor, tsos, prop, est, tmp)) break;
    result = merger.orderAndMergeTwoLevels( result, tmp, theHelicity, crossingSide);
  }
}


TIBRing::SubRingCrossings 
TIBRing::computeCrossings( const TrajectoryStateOnSurface& startingState,
    PropagationDirection propDir) const
{
  typedef HelixBarrelPlaneCrossing2OrderLocal    Crossing;
  typedef MeasurementEstimator::Local2DVector Local2DVector;

  GlobalPoint startPos( startingState.globalPosition());
  GlobalVector startDir( startingState.globalMomentum());
  double rho( startingState.transverseCurvature());

  HelixBarrelCylinderCrossing cylCrossing( startPos, startDir, rho,
					   propDir,specificSurface());
  if (!cylCrossing.hasSolution()) {
    cout << "ERROR in TIBRing: cylinder not crossed by track" << endl;
    throw Genexception("TIBRing: inner subRod not crossed by track");
  }

  GlobalPoint  cylPoint( cylCrossing.position());
  GlobalVector cylDir( cylCrossing.direction());
  int closestIndex = theBinFinder.binIndex(cylPoint.phi());

  const BoundPlane& closestPlane( dynamic_cast<const BoundPlane&>( 
    theDets[closestIndex]->surface()));

  LocalPoint closestPos = Crossing( cylPoint, cylDir, rho, closestPlane).position();
  float closestDist = closestPos.x(); // use fact that local X perp to global Z 

  //int next = cylPoint.phi() - closestPlane.position().phi() > 0 ? closest+1 : closest-1;
  int nextIndex = PhiLess()( closestPlane.position().phi(), cylPoint.phi()) ? 
    closestIndex+1 : closestIndex-1;

  const BoundPlane& nextPlane( dynamic_cast<const BoundPlane&>( 
    theDets[ theBinFinder.binIndex(nextIndex)]->surface()));
  LocalPoint nextPos = Crossing( cylPoint, cylDir, rho, nextPlane).position();
  float nextDist = nextPos.x();

  if (fabs(closestDist) < fabs(nextDist)) {
    return SubRingCrossings( closestIndex, nextIndex, nextDist);
  }
  else {
    return SubRingCrossings( nextIndex, closestIndex, closestDist);
  }
}

float TIBRing::computeWindowSize( const GeomDet* det, 
				  const TrajectoryStateOnSurface& tsos, 
				  const MeasurementEstimator& est) const
{
  return est.maximalLocalDisplacement(tsos, dynamic_cast<const BoundPlane&>(det->surface())).x();
}



