#include "TIDLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include<array>
#include "DetGroupMerger.h"


//#include "CommonDet/DetLayout/src/DetLessR.h"


using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

namespace {

// 2 0 1
/*
class TIDringLess  {
  // z-position ordering of TID rings indices
public:
  bool operator()( const pair<vector<DetGroup> const *,int> & a,const pair<vector<DetGroup>const *,int> & b) const {
    if(a.second==2) {return true;}
    else if(a.second==0) {
      if(b.second==2) return false;
      if(b.second==1) return true;
    }
    return false;    
  };
};
*/

  // groups in correct order: one may be empty
  inline
  void mergeOutward(std::array<vector<DetGroup>,3> & groups, 
		    std::vector<DetGroup> & result ) {
    typedef DetGroupMerger Merger;
    Merger::orderAndMergeTwoLevels(std::move(groups[0]),
				   std::move(groups[1]),result,1,1);
    if(!groups[2].empty()) {
      std::vector<DetGroup> tmp;
      tmp.swap(result);
      Merger::orderAndMergeTwoLevels(std::move(tmp),std::move(groups[2]),result,1,1);      
    }
    
  }
  
  inline
  void mergeInward(std::array<vector<DetGroup>,3> & groups,
		   std::vector<DetGroup> & result ) {
    typedef DetGroupMerger Merger;
    Merger::orderAndMergeTwoLevels(std::move(groups[2]),
				   std::move(groups[1]),result,1,1);
    if(!groups[0].empty()) {
      std::vector<DetGroup> tmp;
      tmp.swap(result);
      Merger::orderAndMergeTwoLevels(std::move(tmp),std::move(groups[0]),result,1,1);      
    }
    
  }
  
  
  void
  orderAndMergeLevels(const TrajectoryStateOnSurface& tsos,
		      const Propagator& prop,
		      std::array<vector<DetGroup>,3> & groups,
		      std::vector<DetGroup> & result ) {
    
    float zpos = tsos.globalPosition().z();
    if(tsos.globalMomentum().z()*zpos>0){ // momentum points outwards
      if(prop.propagationDirection() == alongMomentum)
	mergeOutward(groups,result);
      else
	mergeInward(groups,result);
    }
    else{ //  momentum points inwards
      if(prop.propagationDirection() == oppositeToMomentum)
	mergeOutward(groups,result);
      else
      mergeInward(groups,result);    
    }  
    
  }
  
}

//hopefully is never called!
const std::vector<const GeometricSearchDet*>& TIDLayer::components() const{
  static std::vector<const GeometricSearchDet*> crap;
  for ( auto c: theComps) crap.push_back(c);
  return crap;
 }


void
TIDLayer::fillRingPars(int i) {
  const BoundDisk& ringDisk = static_cast<const BoundDisk&>(theComps[i]->surface());
  float ringMinZ = fabs( ringDisk.position().z()) - ringDisk.bounds().thickness()/2.;
  float ringMaxZ = fabs( ringDisk.position().z()) + ringDisk.bounds().thickness()/2.; 
  ringPars[i].thetaRingMin =  ringDisk.innerRadius()/ ringMaxZ;
  ringPars[i].thetaRingMax =  ringDisk.outerRadius()/ ringMinZ;
  ringPars[i].theRingR=( ringDisk.innerRadius() +
			 ringDisk.outerRadius())/2.;

}


TIDLayer::TIDLayer(vector<const TIDRing*>& rings) {
  //They should be already R-ordered. TO BE CHECKED!!
  //sort( theRings.begin(), theRings.end(), DetLessR());

  if ( rings.size() != 3) throw DetLayerException("Number of rings in TID layer is not equal to 3 !!");
  setSurface( computeDisk( rings ) );

  for(int i=0; i!=3; ++i) {
    theComps[i]=rings[i];
    fillRingPars(i);
    theBasicComps.insert(theBasicComps.end(),	
			 (*rings[i]).basicComponents().begin(),
			 (*rings[i]).basicComponents().end());
  }

 
  LogDebug("TkDetLayers") << "==== DEBUG TIDLayer =====" ; 
  LogDebug("TkDetLayers") << "r,zed pos  , thickness, innerR, outerR: " 
			  << this->position().perp() << " , "
			  << this->position().z() << " , "
			  << this->specificSurface().bounds().thickness() << " , "
			  << this->specificSurface().innerRadius() << " , "
			  << this->specificSurface().outerRadius() ;
}


BoundDisk* 
TIDLayer::computeDisk( const vector<const TIDRing*>& rings) const
{
  float theRmin = rings.front()->specificSurface().innerRadius();
  float theRmax = rings.front()->specificSurface().outerRadius();
  float theZmin = rings.front()->position().z() -
    rings.front()->surface().bounds().thickness()/2;
  float theZmax = rings.front()->position().z() +
    rings.front()->surface().bounds().thickness()/2;
  
  for (vector<const TIDRing*>::const_iterator i = rings.begin(); i != rings.end(); i++) {
    float rmin = (**i).specificSurface().innerRadius();
    float rmax = (**i).specificSurface().outerRadius();
    float zmin = (**i).position().z() - (**i).surface().bounds().thickness()/2.;
    float zmax = (**i).position().z() + (**i).surface().bounds().thickness()/2.;
    theRmin = min( theRmin, rmin);
    theRmax = max( theRmax, rmax);
    theZmin = min( theZmin, zmin);
    theZmax = max( theZmax, zmax);
  }
  
  float zPos = (theZmax+theZmin)/2.;
  PositionType pos(0.,0.,zPos);
  RotationType rot;

  return new BoundDisk( pos, rot, new SimpleDiskBounds(theRmin, theRmax,    
				      theZmin-zPos, theZmax-zPos));

}


TIDLayer::~TIDLayer(){
  for (auto c : theComps) delete c;
} 

  

void
TIDLayer::groupedCompatibleDetsV( const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop,
				 const MeasurementEstimator& est,
				 std::vector<DetGroup> & result) const
{
  std::array<int,3> const & ringIndices = ringIndicesByCrossingProximity(startingState,prop);
  if ( ringIndices[0]==-1 ) {
    edm::LogError("TkDetLayers") << "TkRingedForwardLayer::groupedCompatibleDets : error in CrossingProximity";
    return;
  }

  std::array<vector<DetGroup>,3> groupsAtRingLevel;
  //order is ring3,ring1,ring2 i.e. 2 0 1
  //                                0 1 2  
  constexpr int ringOrder[3]{1,2,0};
  auto index = [&ringIndices,& ringOrder](int i) { return ringOrder[ringIndices[i]];};

  auto & closestResult =  groupsAtRingLevel[index(0)];
  theComps[ringIndices[0]]->groupedCompatibleDetsV( startingState, prop, est, closestResult);		
  if ( closestResult.empty() ){
    theComps[ringIndices[1]]->groupedCompatibleDetsV( startingState, prop, est, result); 
    return;
  }

  DetGroupElement closestGel( closestResult.front().front());  
  float rWindow = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est); 

  if(!overlapInR(closestGel.trajectoryState(),ringIndices[1],rWindow)) {
    result.swap(closestResult);
    return;
  }

 
  auto & nextResult =  groupsAtRingLevel[index(1)];
  theComps[ringIndices[1]]->groupedCompatibleDetsV( startingState, prop, est, nextResult);
  if(nextResult.empty()) {
    result.swap(closestResult);
    return;
  }
 
  if(!overlapInR(closestGel.trajectoryState(),ringIndices[2],rWindow) ) {
    //then merge 2 levels & return 
    orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,result);      
    return;
  }

  auto & nextNextResult =  groupsAtRingLevel[index(2)];
  theComps[ringIndices[2]]->groupedCompatibleDetsV( startingState, prop, est, nextNextResult);   
  if(nextNextResult.empty()) {
    // then merge 2 levels and return 
    orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,result); // 
    return;
  }

  // merge 3 level and return merged   
  orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel, result);  
}


std::array<int,3> 
TIDLayer::ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
					 const Propagator& prop ) const
{
  typedef HelixForwardPlaneCrossing Crossing; 
  typedef MeasurementEstimator::Local2DVector Local2DVector;

  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition());
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum());
  PropagationDirection propDir( prop.propagationDirection());
  float rho( startingState.transverseCurvature());

  // calculate the crossings with the ring surfaces
  // rings are assumed to be sorted in R !
  
  Crossing myXing(  startPos, startDir, rho, propDir );

  GlobalPoint   ringCrossings[3];
  // vector<GlobalVector>  ringXDirections;

  for (int i = 0; i < 3 ; i++ ) {
    const BoundDisk & theRing  = static_cast<const BoundDisk &>(theComps[i]->surface());
    pair<bool,double> pathlen = myXing.pathLength( theRing);
    if ( pathlen.first ) { 
      ringCrossings[i] = GlobalPoint( myXing.position(pathlen.second ));
      // ringXDirections.push_back( GlobalVector( myXing.direction(pathlen.second )));
    } else {
      // TO FIX.... perhaps there is something smarter to do
      //throw DetLayerException("trajectory doesn't cross TID rings");
      ringCrossings[i] = GlobalPoint( 0.,0.,0.);
      //  ringXDirections.push_back( GlobalVector( 0.,0.,0.));
    }
  }

  int closestIndex = findClosest(ringCrossings);
  int nextIndex    = findNextIndex(ringCrossings,closestIndex);
  if ( closestIndex<0 || nextIndex<0 )  return std::array<int,3>{{-1,-1,-1}};
  int nextNextIndex = -1;
  for(int i=0; i<3 ; i++){
    if(i!= closestIndex && i!=nextIndex) {
      nextNextIndex = i;
      break;
    }
  }
  
  std::array<int,3> indices{{closestIndex,nextIndex,nextNextIndex}};
  return indices;
}




float 
TIDLayer::computeWindowSize( const GeomDet* det, 
			     const TrajectoryStateOnSurface& tsos, 
			     const MeasurementEstimator& est) const
{
  const Plane& startPlane = det->surface();  
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement( tsos, startPlane);
  return maxDistance.y();
}


int
TIDLayer::findClosest(const GlobalPoint ringCrossing[3] ) const
{
  int theBin = 0;
  float initialR =  ringPars[0].theRingR;
  float rDiff = fabs( ringCrossing[0].perp() - initialR);
  for (int i = 1; i < 3 ; i++){
    float ringR =  ringPars[i].theRingR;
    float testDiff = fabs( ringCrossing[i].perp() - ringR);
    if ( testDiff<rDiff ) {
      rDiff = testDiff;
      theBin = i;
    }
  }
  return theBin;
}

int
TIDLayer::findNextIndex(const GlobalPoint ringCrossing[3], int closest ) const
{

  int firstIndexToCheck = (closest != 0)? 0 : 1; 
  float initialR =  ringPars[firstIndexToCheck].theRingR;	     
  float rDiff = fabs( ringCrossing[0].perp() - initialR);
  int theBin = firstIndexToCheck;
  for (int i = firstIndexToCheck+1; i < 3 ; i++){
    if ( i != closest) {
      float ringR =  ringPars[i].theRingR;
      float testDiff = fabs( ringCrossing[i].perp() - ringR);
      if ( testDiff<rDiff ) {
	rDiff = testDiff;
	theBin = i;
      }
    }
  }
  return theBin;
}



bool
TIDLayer::overlapInR( const TrajectoryStateOnSurface& tsos, int index, double ymax ) const 
{
  // assume "fixed theta window", i.e. margin in local y = r is changing linearly with z
  float tsRadius = tsos.globalPosition().perp();
  float thetamin = ( max(0.,tsRadius-ymax))/(fabs(tsos.globalPosition().z())+10.f); // add 10 cm contingency 
  float thetamax = ( tsRadius + ymax)/(fabs(tsos.globalPosition().z())-10.f);
  
  // do the theta regions overlap ?

  return !( thetamin > ringPars[index].thetaRingMax || ringPars[index].thetaRingMin > thetamax);
}


