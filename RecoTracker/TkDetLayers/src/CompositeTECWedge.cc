#include "RecoTracker/TkDetLayers/interface/CompositeTECWedge.h"
#include "RecoTracker/TkDetLayers/interface/ForwardDiskSectorBuilderFromDet.h"

#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"

#include "Utilities/General/interface/CMSexception.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include "Geometry/Surface/interface/TrapezoidalPlaneBounds.h"

typedef GeometricSearchDet::DetWithState DetWithState;

CompositeTECWedge::CompositeTECWedge(vector<const GeomDet*>& innerDets,
				     vector<const GeomDet*>& outerDets):
  theFrontDets(innerDets.begin(),innerDets.end()), theBackDets(outerDets.begin(),outerDets.end())
{  
  theDets.assign(theFrontDets.begin(),theFrontDets.end());
  theDets.insert(theDets.end(),theBackDets.begin(),theBackDets.end());


  // We suppose that they are already phi orderd. To be checked!!
  //sort( theFrontDets.begin(), theFrontDets.end(), DetLessPhiWedge() );
  //sort( theBackDets.begin(),  theBackDets.end(),  DetLessPhiWedge() );
  
  theFrontSector = ForwardDiskSectorBuilderFromDet()( theFrontDets );
  theBackSector  = ForwardDiskSectorBuilderFromDet()( theBackDets );
  theDiskSector = ForwardDiskSectorBuilderFromDet()( theDets );
}

CompositeTECWedge::~CompositeTECWedge(){

} 

vector<const GeomDet*> 
CompositeTECWedge::basicComponents() const{
  return theDets;
}

vector<const GeometricSearchDet*> 
CompositeTECWedge::components() const{
  cout << "temporary dummy implementation of CompositeTECWedge::components()!!" << endl;
  return vector<const GeometricSearchDet*>();
}

  
pair<bool, TrajectoryStateOnSurface>
CompositeTECWedge::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
			       const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of CompositeTECWedge::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}



vector<DetGroup> 
CompositeTECWedge::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					  const MeasurementEstimator& est) const{
  vector<DetGroup> closestResult;
  SubLayerCrossings  crossings; 
  try{
    crossings = computeCrossings( tsos, prop.propagationDirection());  
  }
  //catch(DetLogicError& err){
  catch(Genexception& err){
    cout << "Aie, got a DetLogicError in CompositeTkPetalWedge::groupedCompatibleDets:" 
	 << err.what() << endl;
    return closestResult;
  }
  addClosest( tsos, prop, est, crossings.closest(), closestResult);
  if (closestResult.empty()) return closestResult;
  
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



SubLayerCrossings 
CompositeTECWedge::computeCrossings( const TrajectoryStateOnSurface& startingState,
				     PropagationDirection propDir) const
{
  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition() );
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum() );
  float rho( startingState.transverseCurvature());

  HelixForwardPlaneCrossing crossing( startPos, startDir, rho, propDir);

  pair<bool,double> frontPath = crossing.pathLength( *theFrontSector);
  if (!frontPath.first) {
    cout << "ERROR in CompositeTECWedge: front sector not crossed by track" << endl;
    throw Genexception("CompositeTECWedge: front sector not crossed by track");
  }
  GlobalPoint gFrontPoint( crossing.position(frontPath.second));
  int frontIndex = findClosestDet(gFrontPoint,0); 
  float frontDist = theFrontDets[frontIndex]->surface().position().phi()  - gFrontPoint.phi(); 
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);

  pair<bool,double> backPath = crossing.pathLength( *theBackSector);
  if (!backPath.first) {
    cout << "ERROR in CompositeTECWedge: back sector not crossed by track" << endl;
    throw Genexception("CompositeTECWedge: back sector not crossed by track");
  }
  GlobalPoint gBackPoint( crossing.position(backPath.second));
  int backIndex = findClosestDet(gBackPoint,1);
  float backDist = theBackDets[backIndex]->surface().position().phi()  - gBackPoint.phi(); 
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);

  frontDist *= PhiLess()( theFrontDets[frontIndex]->surface().position().phi(),gFrontPoint.phi()) ? -1. : 1.; 
  backDist  *= PhiLess()( theBackDets[backIndex]->surface().position().phi(),gBackPoint.phi()) ? -1. : 1.;
  if (frontDist < 0.) { frontDist += 2.*Geom::pi();}
  if ( backDist < 0.) { backDist  += 2.*Geom::pi();}

  
  if (frontDist < backDist) {
    return SubLayerCrossings( frontSLC, backSLC, 0);
  }
  else {
    return SubLayerCrossings( backSLC, frontSLC, 1);
  } 
}


bool CompositeTECWedge::addClosest( const TrajectoryStateOnSurface& tsos,
				    const Propagator& prop,
				    const MeasurementEstimator& est,
				    const SubLayerCrossing& crossing,
				    vector<DetGroup>& result) const
{
  const vector<const GeomDet*>& sWedge( subWedge( crossing.subLayerIndex()));
  return CompatibleDetToGroupAdder().add( *sWedge[crossing.closestDetIndex()], 
					  tsos, prop, est, result);
}


void CompositeTECWedge::searchNeighbors( const TrajectoryStateOnSurface& tsos,
					 const Propagator& prop,
					 const MeasurementEstimator& est,
					 const SubLayerCrossing& crossing,
					 float window, 
					 vector<DetGroup>& result,
					 bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<const GeomDet*>& sWedge( subWedge( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if ( PhiLess() (gCrossingPos.phi(), sWedge[closestIndex]->surface().position().phi()) ) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  CompatibleDetToGroupAdder adder;
  for (int idet=negStartIndex; idet >= 0; idet--) {
    //if(idet <0 || idet>=sWedge.size()) {cout << "==== warning! gone out vector bounds.idet: " << idet << endl;break;}
    if (!overlap( gCrossingPos, *sWedge[idet], window)) break;
    if (!adder.add( *sWedge[idet], tsos, prop, est, result)) break;
  }
  for (int idet=posStartIndex; idet < static_cast<int>(sWedge.size()); idet++) {
    //if(idet <0 || idet>=sWedge.size()) {cout << "==== warning! gone out vector bounds.idet: " << idet << endl;break;}
    if (!overlap( gCrossingPos, *sWedge[idet], window)) break;
    if (!adder.add( *sWedge[idet], tsos, prop, est, result)) break;
  }
}


bool CompositeTECWedge::overlap( const GlobalPoint& crossPoint, const GeomDet& det, float phiWindow) const
{
  const BoundPlane& plane( dynamic_cast<const BoundPlane&>( det.surface()));
  pair<float,float> phiRange(crossPoint.phi()-phiWindow, crossPoint.phi()+phiWindow);
  pair<float,float> detPhiRange = computeDetPhiRange( plane);
  if ( rangesIntersect( phiRange, detPhiRange, PhiLess())) { 
    return true;
  } 
  return false;
}
 
float CompositeTECWedge::computeWindowSize( const GeomDet* det, 
					    const TrajectoryStateOnSurface& tsos, 
					    const MeasurementEstimator& est) const
{
  const BoundPlane& plane( dynamic_cast<const BoundPlane&>(det->surface()) );
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement(tsos, plane);
  return calculatePhiWindow( maxDistance, tsos, plane);
}


float 
CompositeTECWedge::calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
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
CompositeTECWedge::computeDetPhiRange( const BoundPlane& plane) const 
{
  const TrapezoidalPlaneBounds* myBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  
  if (myBounds == 0) {
    string errmsg="CompositeTkPetalWedge: problems with dynamic cast to trapezoidal bounds for Det";
    throw Genexception(errmsg);
    cout << errmsg << endl;
  }
  vector<float> parameters = (*myBounds).parameters();
  if ( parameters[0] == 0 ) 
    cout << "CompositeTkPetalWedge: something weird going on with trapezoidal Plane Bounds!" << endl;
  // cout << " Parameters of DetUnit (L2/L1/T/H): " ;
  // for (int i = 0; i < 4; i++ ) { cout << "  " << 2.*parameters[i]; }
  // cout << endl;

  float hbotedge = parameters[0];
  float htopedge = parameters[1];
  float hapothem = parameters[3];


  vector<GlobalPoint> corners;

  corners.push_back( plane.toGlobal( LocalPoint( -htopedge, hapothem, 0.)));
  corners.push_back( plane.toGlobal( LocalPoint(  htopedge, hapothem, 0.)));
  corners.push_back( plane.toGlobal( LocalPoint(  hbotedge, -hapothem, 0.)));
  corners.push_back( plane.toGlobal( LocalPoint( -hbotedge, -hapothem, 0.)));

  float phimin = corners[0].phi();
  float phimax = phimin;
  for ( int i = 1; i < 4; i++ ) {
    float cPhi = corners[i].phi();
    if ( PhiLess()( cPhi, phimin)) { phimin = cPhi; }
    if ( PhiLess()( phimax, cPhi)) { phimax = cPhi; }
  }
  return make_pair( phimin, phimax);
}



int
CompositeTECWedge::findClosestDet( const GlobalPoint& startPos,int sectorId) const
{

  vector<const GeomDet*> myDets = sectorId==0 ? theFrontDets : theBackDets;
  
  int close = -1;
  float closeDist = 200.;
  for (unsigned int i = 0; i < myDets.size(); i++ ) {
    float dist = (myDets[i]->surface().toLocal(startPos)).x();
    if ( fabs(dist) < fabs(closeDist) ) {
      close = i;
      closeDist = dist;
    }
  }
  return close;
}


/*
int
CompositeTECWedge::findClosestDet( const GlobalPoint& startPos) const
{

  int close = -1;
  float closeDist = 200.;
  for (unsigned int i = 0; i < theDets.size(); i++ ) {
    float dist = (theDets[i]->surface().toLocal(startPos)).x();
    if ( close==-1 || fabs(dist)<fabs(closeDist) ) {
      close = i;
      closeDist = dist;
    }
  }
  return close;
}


*/
