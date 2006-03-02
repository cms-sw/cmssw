#include "RecoTracker/TkDetLayers/interface/TECLayer.h"
#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"

#include "Utilities/General/interface/CMSexception.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"

typedef GeometricSearchDet::DetWithState DetWithState;

TECLayer::TECLayer(vector<const TECPetal*>& innerPetals,
		   vector<const TECPetal*>& outerPetals) : 
  theFrontPetals(innerPetals.begin(),innerPetals.end()), 
  theBackPetals(outerPetals.begin(),outerPetals.end())
{
  thePetals.assign(theFrontPetals.begin(),theFrontPetals.end());
  thePetals.insert(thePetals.end(),theBackPetals.begin(),theBackPetals.end());


  //This should be no necessary. TO BE CHECKED
  //sort(theFrontPetals.begin(), theFrontPetals.end(), PetalLessPhi());
  //sort(theBackPetals.begin(), theBackPetals.end(), PetalLessPhi());

  // building disk for front and back petals
  /* ---- TO BE CHANGED AS SOON AS  THE PROBLEM WITH GLUEDGEOMDET IS FIXED
  theLayerDisk = computeDisk( thePetals );
  theFrontDisk = computeDisk( theFrontPetals);
  theBackDisk  = computeDisk( theBackPetals);

  // set up the bin finders
  theFrontBinFinder = BinFinderPhi(theFrontPetals.front()->position().phi(),
				   theFrontPetals.size());
  theBackBinFinder  = BinFinderPhi(theBackPetals.front()->position().phi(),
				   theBackPetals.size());  
  ------ */
}



TECLayer::~TECLayer(){
  vector<const TECPetal*>::const_iterator i;
  for (i=thePetals.begin(); i!=thePetals.end(); i++) {
    delete *i;
  }
} 

vector<const GeomDet*> 
TECLayer::basicComponents() const{
  cout << "temporary dummy implementation of TECLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}

vector<const GeometricSearchDet*> 
TECLayer::components() const{
  cout << "temporary dummy implementation of TECLayer::components()!!" << endl;
  return vector<const GeometricSearchDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TECLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TECLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TECLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TECLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
				 const Propagator& prop,
				 const MeasurementEstimator& est) const
{
  vector<DetGroup> closestResult;
  SubLayerCrossings  crossings; 
  try{
    crossings = computeCrossings( tsos, prop.propagationDirection());  
  }
  catch(Genexception& err){
    cout << "Aie, got a Genexception in TECLayer::groupedCompatibleDets:" 
	 << err.what() << endl;
    return closestResult;
  }    
  addClosest( tsos, prop, est, crossings.closest(), closestResult); 

  // this differs from other groupedCompatibleDets logic, which DON'T check next in such cases!!!
  if(closestResult.empty()){
    vector<DetGroup> nextResult;
    addClosest( tsos, prop, est, crossings.other(), nextResult);     
    if(nextResult.empty())       return nextResult;
    

    DetGroupElement nextGel( nextResult.front().front());  
    int crossingSide = LayerCrossingSide().endcapSide( nextGel.trajectoryState(), prop);
    DetGroupMerger merger;
    return  merger.orderAndMergeTwoLevels( closestResult, nextResult, 
					   crossings.closestIndex(), crossingSide);   
  }  
  
  DetGroupElement closestGel( closestResult.front().front());  
  float phiWindow = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est); 
  searchNeighbors( tsos, prop, est, crossings.closest(), phiWindow,
		   closestResult, false); 
  vector<DetGroup> nextResult;  
  searchNeighbors( tsos, prop, est, crossings.other(), phiWindow,
		   nextResult, true); 
  
  int crossingSide = LayerCrossingSide().endcapSide( closestGel.trajectoryState(), prop);
  DetGroupMerger merger;
  return merger.orderAndMergeTwoLevels( closestResult, nextResult, 
					crossings.closestIndex(), crossingSide);
}


SubLayerCrossings TECLayer::computeCrossings(const TrajectoryStateOnSurface& startingState,
					     PropagationDirection propDir) const
{
  double rho( startingState.transverseCurvature());
  
  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition() );
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum() );
  HelixForwardPlaneCrossing crossing(startPos,startDir,rho,propDir);

  pair<bool,double> frontPath = crossing.pathLength( *theFrontDisk);
  if (!frontPath.first) {
    cout << "ERROR in TECLayer: front disk not crossed by track" << endl;
    throw Genexception("TECLayer: front disk not crossed by track");
  }

  GlobalPoint gFrontPoint(crossing.position(frontPath.second));

  int frontIndex = theFrontBinFinder.binIndex(gFrontPoint.phi());
  float frontDist = theFrontPetals[frontIndex]->position().phi()  - gFrontPoint.phi(); 
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);



  pair<bool,double> backPath = crossing.pathLength( *theBackDisk);
  if (!backPath.first) {
    cout << "ERROR in TECLayer: back disk not crossed by track" << endl;
    throw Genexception("TECLayer: back disk not crossed by track");
  }


  GlobalPoint gBackPoint( crossing.position(backPath.second));
  int backIndex = theBackBinFinder.binIndex(gBackPoint.phi());
  float backDist = theBackPetals[backIndex]->position().phi()  - gBackPoint.phi(); 
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);

  
  // 0ss: frontDisk has index=0, backDisk has index=1
  frontDist *= PhiLess()( theFrontPetals[frontIndex]->position().phi(),gFrontPoint.phi()) ? -1. : 1.; 
  backDist  *= PhiLess()( theBackPetals[backIndex]->position().phi(),gBackPoint.phi()) ? -1. : 1.;
  if (frontDist < 0.) { frontDist += 2.*Geom::pi();}
  if ( backDist < 0.) { backDist  += 2.*Geom::pi();}

  if (frontDist < backDist) {
    return SubLayerCrossings( frontSLC, backSLC, 0);
  }
  else {
    return SubLayerCrossings( backSLC, frontSLC, 1);
  } 
}

bool TECLayer::addClosest( const TrajectoryStateOnSurface& tsos,
			   const Propagator& prop,
			   const MeasurementEstimator& est,
			   const SubLayerCrossing& crossing,
			   vector<DetGroup>& result) const
{
  const vector<const TECPetal*>& sub( subLayer( crossing.subLayerIndex()));
  const GeometricSearchDet* det(sub[crossing.closestDetIndex()]);
  return CompatibleDetToGroupAdder().add( *det, tsos, prop, est, result); 
}

void TECLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est,
				const SubLayerCrossing& crossing,
				float window, 
				vector<DetGroup>& result,
				bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();
  
  const vector<const TECPetal*>& sLayer( subLayer( crossing.subLayerIndex()));
 
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

  const BinFinderPhi& binFinder = (crossing.subLayerIndex()==0 ? theFrontBinFinder : theBackBinFinder);

  CompatibleDetToGroupAdder adder;
  int half = sLayer.size()/2;  // to check if dets are called twice....
  for (int idet=negStartIndex; idet >= negStartIndex - half; idet--) {
    const TECPetal* neighborPetal = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborPetal, window)) break;
    if (!adder.add( *neighborPetal, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + half; idet++) {
    const TECPetal* neighborPetal = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborPetal, window)) break;
    if (!adder.add( *neighborPetal, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}

float TECLayer::computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est) const
{
  const BoundPlane& startPlane( dynamic_cast<const BoundPlane&>(det->surface()) );  
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement( tsos, startPlane);
  return calculatePhiWindow( maxDistance, tsos, startPlane);
}


bool TECLayer::overlap( const GlobalPoint& gpos, const TECPetal& petal, float phiWin) const
{
  pair<float,float> phiRange(gpos.phi()-phiWin,gpos.phi()+phiWin);
  pair<float,float> petalPhiRange(petal.position().phi() - petal.specificSurface().phiExtension()/2.,
				  petal.position().phi() + petal.specificSurface().phiExtension()/2.);


  if ( rangesIntersect(phiRange, petalPhiRange, PhiLess())) {
//     cout << " overlapInPhi:  Ranges intersect " << endl;
    return true;
  } else {
//     cout << "  overlapInPhi: Ranges DO NOT intersect " << endl;
    return false;
  }
} 



BoundDisk*
TECLayer::computeDisk( vector<const TECPetal*>& petals) const
{
  // Attention: it is assumed that the petals do belong to one layer, and are all
  // of the same rmin/rmax extension !!  
  
  float rmin = petals.front()->specificSurface().innerRadius();
  float rmax = petals.front()->specificSurface().outerRadius();
  
  float theZmax(petals.front()->position().z());
  float theZmin(theZmax);
  for ( vector<const TECPetal*>::const_iterator i = petals.begin(); i != petals.end(); i++ ) {
    float zmin = (**i).position().z() - (**i).surface().bounds().thickness()/2.;
    float zmax = (**i).position().z() + (**i).surface().bounds().thickness()/2.;
    theZmax = max( theZmax, zmax);
    theZmin = min( theZmin, zmin);
  }

  float zPos = (theZmax+theZmin)/2.;
  PositionType pos(0.,0.,zPos);
  RotationType rot;

  return new BoundDisk( pos, rot,SimpleDiskBounds(rmin, rmax,    
						  theZmin-zPos, theZmax-zPos));
}




float 
TECLayer::calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
			      const TrajectoryStateOnSurface& ts, 
			      const BoundPlane& plane) const
{
  vector<GlobalPoint> corners(4);
  vector<LocalPoint> lcorners(4);
  LocalPoint start = ts.localPosition();
  lcorners[0] = LocalPoint( start.x()+maxDistance.x(), start.y()+maxDistance.y());  
  lcorners[1] = LocalPoint( start.x()-maxDistance.x(), start.y()+maxDistance.y());
  lcorners[2] = LocalPoint( start.x()-maxDistance.x(), start.y()-maxDistance.y());
  lcorners[3] = LocalPoint( start.x()+maxDistance.x(), start.y()-maxDistance.y());
  
  for( int i = 0; i<4; i++) {
    corners[i] = plane.toGlobal( lcorners[i]);
  }
  float phimin = corners[0].phi();
  float phimax = phimin;
  for ( int i = 1; i<4; i++) {
    float cPhi = corners[i].phi();
    if ( PhiLess()( cPhi, phimin)) { phimin = cPhi; }
    if ( PhiLess()( phimax, cPhi)) { phimax = cPhi; }
  }
  float phiWindow = phimax - phimin;
  if ( phiWindow < 0.) { phiWindow +=  2.*Geom::pi();}

  return phiWindow;
}
