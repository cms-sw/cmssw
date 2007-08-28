#include "RecoTracker/TkDetLayers/interface/TECLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

TECLayer::TECLayer(vector<const TECPetal*>& innerPetals,
		   vector<const TECPetal*>& outerPetals) : 
  theFrontComps(innerPetals.begin(),innerPetals.end()), 
  theBackComps(outerPetals.begin(),outerPetals.end())
{
  theComps.assign(theFrontComps.begin(),theFrontComps.end());
  theComps.insert(theComps.end(),theBackComps.begin(),theBackComps.end());

  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin();
      it!=theComps.end();it++){  
    theBasicComps.insert(theBasicComps.end(),	
			 (**it).basicComponents().begin(),
			 (**it).basicComponents().end());
  }


  //This should be no necessary. TO BE CHECKED
  //sort(theFrontPetals.begin(), theFrontPetals.end(), PetalLessPhi());
  //sort(theBackPetals.begin(), theBackPetals.end(), PetalLessPhi());

  // building disk for front and back petals
  setSurface( computeDisk( theComps ) );
  theFrontDisk = computeDisk( theFrontComps);
  theBackDisk  = computeDisk( theBackComps);

  // set up the bin finders
  theFrontBinFinder = BinFinderPhi(theFrontComps.front()->position().phi(),
  				   theFrontComps.size());
  theBackBinFinder  = BinFinderPhi(theBackComps.front()->position().phi(),
				   theBackComps.size());  

  //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "DEBUG INFO for TECLayer" << "\n"
			  << "TECLayer z,perp, innerRadius, outerR: " 
			  << this->position().z()    << " , "
			  << this->position().perp() << " , "
			  << this->specificSurface().innerRadius() << " , "
			  << this->specificSurface().outerRadius() ;
  

  for(vector<const GeometricSearchDet*>::const_iterator it=theFrontComps.begin(); 
      it!=theFrontComps.end(); it++){
    LogDebug("TkDetLayers") << "frontPetal phi,z,r: " 
	 << (*it)->surface().position().phi() << " , "
	 << (*it)->surface().position().z() <<   " , "
	 << (*it)->surface().position().perp() ;
  }

  for(vector<const GeometricSearchDet*>::const_iterator it=theBackComps.begin(); 
      it!=theBackComps.end(); it++){
    LogDebug("TkDetLayers") << "backPetal phi,z,r: " 
	 << (*it)->surface().position().phi() << " , "
	 << (*it)->surface().position().z() <<   " , "
	 << (*it)->surface().position().perp() ;
  }
  //----------------------------------- 


}



TECLayer::~TECLayer(){
  vector<const GeometricSearchDet*>::const_iterator i;
  for (i=theComps.begin(); i!=theComps.end(); i++) {
    delete *i;
  }
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
  crossings = computeCrossings( tsos, prop.propagationDirection());
  if(! crossings.isValid()) return closestResult;

  addClosest( tsos, prop, est, crossings.closest(), closestResult); 
  LogDebug("TkDetLayers") << "in TECLayer, closestResult.size(): " << closestResult.size();

  // this differs from other groupedCompatibleDets logic, which DON'T check next in such cases!!!
  if(closestResult.empty()){
    vector<DetGroup> nextResult;
    addClosest( tsos, prop, est, crossings.other(), nextResult);   
    LogDebug("TkDetLayers") << "in TECLayer, nextResult.size(): " << nextResult.size();
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
  if (!frontPath.first) SubLayerCrossings();


  GlobalPoint gFrontPoint(crossing.position(frontPath.second));
  LogDebug("TkDetLayers") 
    << "in TECLayer,front crossing point: r,z,phi: (" 
    << gFrontPoint.perp() << ","
    << gFrontPoint.z() << "," 
    << gFrontPoint.phi() << ")" << endl;
  

  int frontIndex = theFrontBinFinder.binIndex(gFrontPoint.phi());
  float frontDist = theFrontComps[frontIndex]->position().phi()  - gFrontPoint.phi(); 
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);



  pair<bool,double> backPath = crossing.pathLength( *theBackDisk);
  if (!backPath.first) SubLayerCrossings();


  GlobalPoint gBackPoint( crossing.position(backPath.second));
  LogDebug("TkDetLayers") 
    << "in TECLayer,back crossing point: r,z,phi: (" 
    << gBackPoint.perp() << "," 
    << gFrontPoint.z() << "," 
    << gBackPoint.phi() << ")" << endl;


  int backIndex = theBackBinFinder.binIndex(gBackPoint.phi());
  float backDist = theBackComps[backIndex]->position().phi()  - gBackPoint.phi(); 
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);

  
  // 0ss: frontDisk has index=0, backDisk has index=1
  frontDist *= PhiLess()( theFrontComps[frontIndex]->position().phi(),gFrontPoint.phi()) ? -1. : 1.; 
  backDist  *= PhiLess()( theBackComps[backIndex]->position().phi(),gBackPoint.phi()) ? -1. : 1.;
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
  const vector<const GeometricSearchDet*>& sub( subLayer( crossing.subLayerIndex()));
  const GeometricSearchDet* det(sub[crossing.closestDetIndex()]);

  LogDebug("TkDetLayers")  
    << "in TECLayer, adding petal at r,z,phi: (" 
    << det->position().perp() << "," 
    << det->position().z() << "," 
    << det->position().phi() << ")" << endl;

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

  const BinFinderPhi& binFinder = (crossing.subLayerIndex()==0 ? theFrontBinFinder : theBackBinFinder);

  CompatibleDetToGroupAdder adder;
  int half = sLayer.size()/2;  // to check if dets are called twice....
  for (int idet=negStartIndex; idet >= negStartIndex - half; idet--) {
    const GeometricSearchDet* neighborPetal = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborPetal, window)) break;
    if (!adder.add( *neighborPetal, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + half; idet++) {
    const GeometricSearchDet* neighborPetal = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborPetal, window)) break;
    if (!adder.add( *neighborPetal, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}

float TECLayer::computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est) const
{
  const BoundPlane& startPlane = det->surface();  
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement( tsos, startPlane);
  return calculatePhiWindow( maxDistance, tsos, startPlane);
}


bool TECLayer::overlap( const GlobalPoint& gpos, const GeometricSearchDet& gsdet, float phiWin) const
{
  const TECPetal& petal = dynamic_cast<const TECPetal&>(gsdet);
  pair<float,float> phiRange(gpos.phi()-phiWin,gpos.phi()+phiWin);
  pair<float,float> petalPhiRange(petal.position().phi() - petal.specificSurface().phiExtension()/2.,
				  petal.position().phi() + petal.specificSurface().phiExtension()/2.);


  if ( rangesIntersect(phiRange, petalPhiRange, PhiLess())) {
//     edm::LogInfo(TkDetLayers) << " overlapInPhi:  Ranges intersect " ;
    return true;
  } else {
//     edm::LogInfo(TkDetLayers) << "  overlapInPhi: Ranges DO NOT intersect " ;
    return false;
  }
} 



BoundDisk*
TECLayer::computeDisk( vector<const GeometricSearchDet*>& petals) const
{
  // Attention: it is assumed that the petals do belong to one layer, and are all
  // of the same rmin/rmax extension !!  
  
  const TECPetal* frontPetal = dynamic_cast<const TECPetal*>(petals.front());

  float rmin = frontPetal->specificSurface().innerRadius();
  float rmax = frontPetal->specificSurface().outerRadius();
  
  float theZmax(petals.front()->position().z());
  float theZmin(theZmax);
  for ( vector<const GeometricSearchDet*>::const_iterator i = petals.begin(); i != petals.end(); i++ ) {
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
