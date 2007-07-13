#include "RecoTracker/TkDetLayers/interface/TIDRing.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"

#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

TIDRing::TIDRing(vector<const GeomDet*>& innerDets,
		 vector<const GeomDet*>& outerDets):
  theFrontDets(innerDets.begin(),innerDets.end()), 
  theBackDets(outerDets.begin(),outerDets.end())
{
  theDets.assign(theFrontDets.begin(),theFrontDets.end());
  theDets.insert(theDets.end(),theBackDets.begin(),theBackDets.end());


  // the dets should be already phi-ordered. TO BE CHECKED
  //sort( theFrontDets.begin(), theFrontDets.end(), DetLessPhi() );
  //sort( theBackDets.begin(), theBackDets.end(), DetLessPhi() );

  theDisk = ForwardRingDiskBuilderFromDet()( theDets );

  theFrontDisk = ForwardRingDiskBuilderFromDet()( theFrontDets );
  theBackDisk  = ForwardRingDiskBuilderFromDet()( theBackDets );

  theFrontBinFinder = BinFinderType( theFrontDets.front()->surface().position().phi(),
				     theFrontDets.size());
  theBackBinFinder  = BinFinderType( theBackDets.front()->surface().position().phi(),
				     theBackDets.size());  


  
  LogDebug("TkDetLayers") << "DEBUG INFO for TIDRing" ;
  for(vector<const GeomDet*>::const_iterator it=theFrontDets.begin(); 
      it!=theFrontDets.end(); it++){
    LogDebug("TkDetLayers") << "frontDet phi,z,r: " 
			    << (*it)->surface().position().phi()  << " , "
			    << (*it)->surface().position().z()    << " , "
			    << (*it)->surface().position().perp() ;
  }

  for(vector<const GeomDet*>::const_iterator it=theBackDets.begin(); 
      it!=theBackDets.end(); it++){
    LogDebug("TkDetLayers") << "backDet phi,z,r: " 
			    << (*it)->surface().position().phi() << " , "
			    << (*it)->surface().position().z()   << " , "
			    << (*it)->surface().position().perp() ;
  }


}

TIDRing::~TIDRing(){

} 

const vector<const GeometricSearchDet*>& 
TIDRing::components() const 
{
  throw DetLayerException("TIDRing doesn't have GeometricSearchDet components");
}

  
pair<bool, TrajectoryStateOnSurface>
TIDRing::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  edm::LogError("TkDetLayers") << "temporary dummy implementation of TIDRing::compatible()!!" ;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TIDRing::compatibleDets( const TrajectoryStateOnSurface& startingState,
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
TIDRing::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est) const
{
  vector<DetGroup> closestResult;
  SubLayerCrossings  crossings; 
  crossings = computeCrossings( tsos, prop.propagationDirection());
  if(! crossings.isValid()) return closestResult;

  addClosest( tsos, prop, est, crossings.closest(), closestResult); 
  if (closestResult.empty())     return closestResult;

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


SubLayerCrossings 
TIDRing::computeCrossings(const TrajectoryStateOnSurface& startingState,
			  PropagationDirection propDir) const
{
  double rho( startingState.transverseCurvature());
  
  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition() );
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum() );
  HelixForwardPlaneCrossing crossing(startPos,startDir,rho,propDir);

  pair<bool,double> frontPath = crossing.pathLength( *theFrontDisk);
  if (!frontPath.first) return SubLayerCrossings();

  GlobalPoint gFrontPoint(crossing.position(frontPath.second));

  int frontIndex = theFrontBinFinder.binIndex(gFrontPoint.phi());
  float frontDist = theFrontDets[frontIndex]->surface().phi()  - gFrontPoint.phi(); 
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);



  pair<bool,double> backPath = crossing.pathLength( *theBackDisk);
  if (!backPath.first) return SubLayerCrossings();

  GlobalPoint gBackPoint( crossing.position(backPath.second));
  int backIndex = theBackBinFinder.binIndex(gBackPoint.phi());
  float backDist = theBackDets[backIndex]->surface().phi()  - gBackPoint.phi(); 
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);

  
  // 0ss: frontDisk has index=0, backDisk has index=1
  frontDist *= PhiLess()( theFrontDets[frontIndex]->surface().phi(),gFrontPoint.phi()) ? -1. : 1.; 
  backDist  *= PhiLess()( theBackDets[backIndex]->surface().phi(),gBackPoint.phi()) ? -1. : 1.;
  if (frontDist < 0.) { frontDist += 2.*Geom::pi();}
  if ( backDist < 0.) { backDist  += 2.*Geom::pi();}

  if (frontDist < backDist) {
    return SubLayerCrossings( frontSLC, backSLC, 0);
  }
  else {
    return SubLayerCrossings( backSLC, frontSLC, 1);
  } 
}

bool TIDRing::addClosest( const TrajectoryStateOnSurface& tsos,
			  const Propagator& prop,
			  const MeasurementEstimator& est,
			  const SubLayerCrossing& crossing,
			  vector<DetGroup>& result) const
{
  const vector<const GeomDet*>& sub( subLayer( crossing.subLayerIndex()));
  const GeomDet* det(sub[crossing.closestDetIndex()]);
  return CompatibleDetToGroupAdder().add( *det, tsos, prop, est, result); 
}



float TIDRing::computeWindowSize( const GeomDet* det, 
				  const TrajectoryStateOnSurface& tsos, 
				  const MeasurementEstimator& est) const
{
  const BoundPlane& startPlane = det->surface() ;  
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement( tsos, startPlane);
  return calculatePhiWindow( maxDistance, tsos, startPlane);
}




void TIDRing::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     const SubLayerCrossing& crossing,
				     float window, 
				     vector<DetGroup>& result,
				     bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<const GeomDet*>& sLayer( subLayer( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if ( PhiLess()( gCrossingPos.phi(), sLayer[closestIndex]->surface().phi())) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  const BinFinderType& binFinder = (crossing.subLayerIndex()==0 ? theFrontBinFinder : theBackBinFinder);

  CompatibleDetToGroupAdder adder;
  int half = sLayer.size()/2;  // to check if dets are called twice....
  for (int idet=negStartIndex; idet >= negStartIndex - half; idet--) {
    const GeomDet* neighborDet = sLayer[binFinder.binIndex(idet)];
    if (!overlapInPhi( gCrossingPos, neighborDet, window)) break;
    if (!adder.add( *neighborDet, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + half; idet++) {
    const GeomDet* neighborDet = sLayer[binFinder.binIndex(idet)];
    if (!overlapInPhi( gCrossingPos, neighborDet, window)) break;
    if (!adder.add( *neighborDet, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}


bool 
TIDRing::overlapInPhi( const GlobalPoint& startPoint,const GeomDet* det, float phiWindow) const 
{  
  pair<float,float> phiRange(startPoint.phi()-phiWindow, startPoint.phi()+phiWindow);
  pair<float,float> detPhiRange = computeDetPhiRange( det->surface());
  if ( rangesIntersect( phiRange, detPhiRange, PhiLess())) { 
    return true;
  } 
  return false;
}

float 
TIDRing::calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
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


pair<float, float>
TIDRing::computeDetPhiRange( const BoundPlane& plane) const 
{

  const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  const RectangularPlaneBounds* rectangularBounds( dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));

  vector<GlobalPoint> corners;
  if (trapezoidalBounds) {  
    vector<float> parameters = (*trapezoidalBounds).parameters();
    if ( parameters[0] == 0 ) 
      edm::LogError("TkDetLayers") << "TkForwardRing: something weird going on with trapezoidal Plane Bounds!" ;
    
    float hbotedge = parameters[0];
    float htopedge = parameters[1];
    float hapothem = parameters[3];
    // float hthick   = parameters[2];   
    
    corners.push_back( plane.toGlobal( LocalPoint( -htopedge, hapothem, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint(  htopedge, hapothem, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint(  hbotedge, -hapothem, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( -hbotedge, -hapothem, 0.)));
    
  }else if(rectangularBounds){     
    float length = rectangularBounds->length();
    float width  = rectangularBounds->width();
    //cout << "in TIDRing, length and width: " << length << " , " << width << endl;
    
    corners.push_back( plane.toGlobal( LocalPoint( -width/2, -length/2, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( -width/2, +length/2, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( +width/2, -length/2, 0.)));
    corners.push_back( plane.toGlobal( LocalPoint( +width/2, +length/2, 0.)));
    
  } else{
    string errmsg="TkForwardRing: problems with dynamic cast to rectangular or trapezoidal bounds for Det";
    throw DetLayerException(errmsg);
    edm::LogError("TkDetLayers") << errmsg ;
  }
 
  float phimin = corners[0].phi();
  float phimax = phimin;
  for ( int i = 1; i < 4; i++ ) {
    float cPhi = corners[i].phi();
    if ( PhiLess()( cPhi, phimin)) { phimin = cPhi; }
      if ( PhiLess()( phimax, cPhi)) { phimax = cPhi; }
  }
  return make_pair( phimin, phimax);

}

