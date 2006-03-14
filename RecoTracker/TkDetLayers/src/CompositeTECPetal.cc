#include "RecoTracker/TkDetLayers/interface/CompositeTECPetal.h"

#include "RecoTracker/TkDetLayers/interface/ForwardDiskSectorBuilderFromWedges.h"

#include "RecoTracker/TkDetLayers/interface/LayerCrossingSide.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"
#include "RecoTracker/TkDetLayers/interface/CompatibleDetToGroupAdder.h"

#include "Utilities/General/interface/CMSexception.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"


typedef GeometricSearchDet::DetWithState DetWithState;

CompositeTECPetal::CompositeTECPetal(vector<const TECWedge*>& innerWedges,
				     vector<const TECWedge*>& outerWedges) : 
  theFrontWedges(innerWedges.begin(),innerWedges.end()), 
  theBackWedges(outerWedges.begin(),outerWedges.end())
{
  theWedges.assign(theFrontWedges.begin(),theFrontWedges.end());
  theWedges.insert(theWedges.end(),theBackWedges.begin(),theBackWedges.end());

  for(vector<const TECWedge*>::const_iterator it=theWedges.begin();it!=theWedges.end();it++){
    theComponents.push_back(*it);
  }


  //the Wedge are already R ordered
  //sort( theWedges.begin(), theWedges.end(), DetLessR());
  //sort( theFrontWedges.begin(), theFrontWedges.end(), DetLessR() );
  //sort( theBackWedges.begin(), theBackWedges.end(), DetLessR() );

  theDiskSector  = ForwardDiskSectorBuilderFromWedges()( theWedges );
  theFrontSector = ForwardDiskSectorBuilderFromWedges()( theFrontWedges);
  theBackSector  = ForwardDiskSectorBuilderFromWedges()( theBackWedges);


}


CompositeTECPetal::~CompositeTECPetal(){
  vector<const TECWedge*>::const_iterator i;
  for (i=theWedges.begin(); i!=theWedges.end(); i++) {
    delete *i;
  }
} 



vector<const GeomDet*> 
CompositeTECPetal::basicComponents() const{
  cout << "temporary dummy implementation of CompositeTECPetal::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}

vector<const GeometricSearchDet*> 
CompositeTECPetal::components() const{
  return theComponents;
}

  
pair<bool, TrajectoryStateOnSurface>
CompositeTECPetal::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of CompositeTECPetal::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetGroup> 
CompositeTECPetal::groupedCompatibleDets( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					  const MeasurementEstimator& est) const
{
  vector<DetGroup> closestResult;
  SubLayerCrossings  crossings; 
  try{
    crossings = computeCrossings( tsos, prop.propagationDirection());  
  }
  catch(Genexception& err){
    cout << "Aie, got a Genexception in CompositeTECPetal::groupedCompatibleDets:" 
	 << err.what() << endl;
    return closestResult;
  } 
  addClosest( tsos, prop, est, crossings.closest(), closestResult); 

  if (closestResult.empty()){
    vector<DetGroup> nextResult;
    addClosest( tsos, prop, est, crossings.other(), nextResult); 
    if(nextResult.empty())    return nextResult;
    
    DetGroupElement nextGel( nextResult.front().front());  
    int crossingSide = LayerCrossingSide().endcapSide( nextGel.trajectoryState(), prop);
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

  int crossingSide = LayerCrossingSide().endcapSide( closestGel.trajectoryState(), prop);
  DetGroupMerger merger;
  return merger.orderAndMergeTwoLevels( closestResult, nextResult, 
					crossings.closestIndex(), crossingSide);
}


SubLayerCrossings 
CompositeTECPetal::computeCrossings(const TrajectoryStateOnSurface& startingState,
				   PropagationDirection propDir) const
{
  double rho( startingState.transverseCurvature());
  
  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition() );
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum() );
  HelixForwardPlaneCrossing crossing(startPos,startDir,rho,propDir);
  pair<bool,double> frontPath = crossing.pathLength( *theFrontSector);

  if (!frontPath.first) {
    cout << "ERROR in TkPetal: frontSector not crossed by track" << endl;
    throw Genexception("TkPetal: frontSector not crossed by track");
  }

  GlobalPoint gFrontPoint(crossing.position(frontPath.second));
  
  int frontIndex = findBin(gFrontPoint.perp(),0);
  float frontDist = fabs( findPosition(frontIndex,0).perp() - gFrontPoint.perp());
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);



  pair<bool,double> backPath = crossing.pathLength( *theBackSector);

  if (!backPath.first) {
    cout << "ERROR in TkPetal: backSector not crossed by track" << endl;
    throw Genexception("TkPetal: backSector not crossed by track");
  }
  

  GlobalPoint gBackPoint( crossing.position(backPath.second));
  int backIndex = findBin(gBackPoint.perp(),1);
  float backDist = fabs( findPosition(backIndex,1).perp() - gBackPoint.perp());
  
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);
  
  
  // 0ss: frontDisk has index=0, backDisk has index=1
  if (frontDist < backDist) {
    return SubLayerCrossings( frontSLC, backSLC, 0);
  }
  else {
    return SubLayerCrossings( backSLC, frontSLC, 1);
  } 
}

bool CompositeTECPetal::addClosest( const TrajectoryStateOnSurface& tsos,
				    const Propagator& prop,
				    const MeasurementEstimator& est,
				    const SubLayerCrossing& crossing,
				    vector<DetGroup>& result) const
{
  const vector<const TECWedge*>& sub( subLayer( crossing.subLayerIndex()));
  const GeometricSearchDet* det(sub[crossing.closestDetIndex()]);
  return CompatibleDetToGroupAdder().add( *det, tsos, prop, est, result);
}



void 
CompositeTECPetal::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				    const Propagator& prop,
				    const MeasurementEstimator& est,
				    const SubLayerCrossing& crossing,
				    float window, 
				    vector<DetGroup>& result,
				    bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();

  const vector<const TECWedge*>& sLayer( subLayer( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex(); 
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if ( gCrossingPos.perp() < sLayer[closestIndex]->position().perp() ) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }


  //const BinFinderType& binFinder = (crossing.subLayerIndex()==0 ? theFrontBinFinder : theBackBinFinder);
  int theSize = crossing.subLayerIndex()==0 ? theFrontWedges.size() : theBackWedges.size();
  
  CompatibleDetToGroupAdder adder;
  for (int idet=negStartIndex; idet >= 0; idet--) {
    //if(idet<0 || idet>= theSize) {cout << "===== error! gone out vector bounds.idet: " << idet << endl;exit;}
    const TECWedge* neighborWedge = sLayer[idet];
    if (!overlap( gCrossingPos, *neighborWedge, window)) break;  // --- to check
    if (!adder.add( *neighborWedge, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet <theSize; idet++) {
    //if(idet<0 || idet>= theSize) {cout << "===== error! gone out vector bounds.idet: " << idet << endl;exit;}
    const TECWedge* neighborWedge = sLayer[idet];
    if (!overlap( gCrossingPos, *neighborWedge, window)) break;  // ---- to check
    if (!adder.add( *neighborWedge, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}

bool 
CompositeTECPetal::overlap( const GlobalPoint& gpos, const TECWedge& wedge, float ymax) const
{
  // this method is just a duplication of overlapInR 
  // adapeted for groupedCompatibleDets() needs

  // assume "fixed theta window", i.e. margin in local y = r is changing linearly with z
  float tsRadius = gpos.perp();
  float thetamin = ( max((float)0.,tsRadius-ymax))/(fabs(gpos.z())+10.); // add 10 cm contingency 
  float thetamax = ( tsRadius + ymax)/(fabs(gpos.z())-10.);

  const BoundDiskSector& wedgeSector = wedge.specificSurface();                                           
  float wedgeMinZ = fabs( wedgeSector.position().z()) - wedgeSector.bounds().thickness()/2.;
  float wedgeMaxZ = fabs( wedgeSector.position().z()) + wedgeSector.bounds().thickness()/2.; 
  float thetaWedgeMin =  wedgeSector.innerRadius()/ wedgeMaxZ;
  float thetaWedgeMax =  wedgeSector.outerRadius()/ wedgeMinZ;
  
  // do the theta regions overlap ?

  if ( thetamin > thetaWedgeMax || thetaWedgeMin > thetamax) { return false;}
  
  return true;
} 





float CompositeTECPetal::computeWindowSize( const GeomDet* det, 
					    const TrajectoryStateOnSurface& tsos, 
					    const MeasurementEstimator& est) const
{
  double ymax = est.maximalLocalDisplacement(tsos, dynamic_cast<const BoundPlane&>(det->surface())).y();
  return ymax;
}


int CompositeTECPetal::findBin( float R,int diskSectorType) const 
{
  vector<const TECWedge*> localWedges = diskSectorType==0 ? theFrontWedges : theBackWedges;
  
  int theBin = -1;
  float rDiff = 200.;
  for (vector<const TECWedge*>::const_iterator i=localWedges.begin(); i !=localWedges.end(); i++){
    float testDiff = fabs( R - (**i).position().perp());
    if ( testDiff < rDiff) {
      rDiff = testDiff;
      theBin = i - localWedges.begin();
    }
  }
  return theBin;
}



GlobalPoint CompositeTECPetal::findPosition(int index,int diskSectorType) const 
{
  vector<const TECWedge*> diskSector = diskSectorType == 0 ? theFrontWedges : theBackWedges; 
  return (diskSector[index])->position();
}

