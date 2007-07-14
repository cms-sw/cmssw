#include "RecoTracker/TkDetLayers/interface/TIDLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"

//#include "CommonDet/DetLayout/src/DetLessR.h"


using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;


class TIDringLess : public binary_function< int, int, bool> {
  // z-position ordering of TID rings indices
public:
  bool operator()( const pair<vector<DetGroup>,int> & a,const pair<vector<DetGroup>,int> & b) const {
    if(a.second==2) {return true;}
    else if(a.second==0) {
      if(b.second==2) return false;
      if(b.second==1) return true;
    }
    return false;    
  };
};


TIDLayer::TIDLayer(vector<const TIDRing*>& rings):
  theComps(rings.begin(),rings.end())
{
  //They should be already R-ordered. TO BE CHECKED!!
  //sort( theRings.begin(), theRings.end(), DetLessR());
  setSurface( computeDisk( rings ) );

  if ( theComps.size() != 3) throw DetLayerException("Number of rings in TID layer is not equal to 3 !!");

  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin();
      it!=theComps.end();it++){  
    theBasicComps.insert(theBasicComps.end(),	
			 (**it).basicComponents().begin(),
			 (**it).basicComponents().end());
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

  return new BoundDisk( pos, rot,SimpleDiskBounds(theRmin, theRmax,    
				 theZmin-zPos, theZmax-zPos));

}


TIDLayer::~TIDLayer(){
  vector<const GeometricSearchDet*>::const_iterator i;
  for (i=theComps.begin(); i!=theComps.end(); i++) {
    delete *i;
  }

} 

  

void
TIDLayer::groupedCompatibleDetsV( const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop,
				 const MeasurementEstimator& est,
				 std::vector<DetGroup> & result) const
{
  vector<int> ringIndices = ringIndicesByCrossingProximity(startingState,prop);
  if ( ringIndices.size()!=3 ) {
    edm::LogError("TkDetLayers") << "TkRingedForwardLayer::groupedCompatibleDets : ringIndices.size() = "
				 << ringIndices.size() << " and not =3!!" ;
    return;
  }

  vector<DetGroup> closestResult;
  vector<DetGroup> nextResult;
  vector<DetGroup> nextNextResult;
  vector<vector<DetGroup> > groupsAtRingLevel;

  theComps[ringIndices[0]]->groupedCompatibleDetsV( startingState, prop, est, closestResult);		
  if ( closestResult.empty() ){
    theComps[ringIndices[1]]->groupedCompatibleDetsV( startingState, prop, est, result); 
    return;
  }

  groupsAtRingLevel.push_back(closestResult);

  DetGroupElement closestGel( closestResult.front().front());  
  float rWindow = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est); 
  if(!overlapInR(closestGel.trajectoryState(),ringIndices[1],rWindow)) {
    result.swap(closestResult);
    return;
  };

  nextResult.clear();
  theComps[ringIndices[1]]->groupedCompatibleDetsV( startingState, prop, est, nextResult);
  if(nextResult.empty()) {
    result.swap(closestResult);
    return;
  }
  groupsAtRingLevel.push_back(nextResult);

  if(!overlapInR(closestGel.trajectoryState(),ringIndices[2],rWindow) ) 
    //then merge 2 levels & return 
    orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,ringIndices,result);      
    
  theComps[ringIndices[2]]->groupedCompatibleDetsV( startingState, prop, est, nextNextResult);   
  if(nextNextResult.empty()) 
    // then merge 2 levels and return 
    orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,ringIndices,result);
  
  groupsAtRingLevel.push_back(nextNextResult);
  // merge 3 level and return merged   
  orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,ringIndices, result);  
}


vector<int> 
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

  vector<GlobalPoint>   ringCrossings;
  vector<GlobalVector>  ringXDirections;

  for (int i = 0; i < 3 ; i++ ) {
    const TIDRing* theRing = dynamic_cast<const TIDRing*>(theComps[i]);
    pair<bool,double> pathlen = myXing.pathLength( theRing->specificSurface());
    if ( pathlen.first ) { 
      ringCrossings.push_back( GlobalPoint( myXing.position(pathlen.second )));
      ringXDirections.push_back( GlobalVector( myXing.direction(pathlen.second )));
    } else {
      // TO FIX.... perhaps there is something smarter to do
      //throw DetLayerException("trajectory doesn't cross TID rings");
      ringCrossings.push_back( GlobalPoint( 0.,0.,0.));
      ringXDirections.push_back( GlobalVector( 0.,0.,0.));
    }
  }
  if (ringCrossings.size() != 3 )  
    throw DetLayerException("TIDLayer::groupedCompDets => Problem with crossing size, not equal 3 !");
  

  int closestIndex = findClosest(ringCrossings);
  int nextIndex    = findNextIndex(ringCrossings,closestIndex);
  if ( closestIndex<0 || nextIndex<0 )  return vector<int>();
  int nextNextIndex = -1;
  for(int i=0; i<3 ; i++){
    if(i!= closestIndex && i!=nextIndex) {
      nextNextIndex = i;
      break;
    }
  }
  
  vector<int> indices;
  indices.push_back(closestIndex);
  indices.push_back(nextIndex);
  indices.push_back(nextNextIndex);
  return indices;
}




float 
TIDLayer::computeWindowSize( const GeomDet* det, 
			     const TrajectoryStateOnSurface& tsos, 
			     const MeasurementEstimator& est) const
{
  const BoundPlane& startPlane = det->surface();  
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement( tsos, startPlane);
  return maxDistance.y();
}

namespace {
  void mergeOutward(vector< pair<vector<DetGroup>,int> > const & groupPlusIndex, 
		    std::vector<DetGroup> & result ) {
    typedef DetGroupMerger Merger;
    Merger::orderAndMergeTwoLevels(groupPlusIndex[0].first,
				   groupPlusIndex[1].first,result,1,1);
    int size = groupPlusIndex.size();
    if(size==3) {
      std::vector<DetGroup> tmp;
      tmp.swap(result);
      Merger::orderAndMergeTwoLevels(tmp,groupPlusIndex[2].first,result,1,1);      
    }
    
  }
  
  void mergeInward(vector< pair<vector<DetGroup>,int> > const & groupPlusIndex, 
		   std::vector<DetGroup> & result ) {
    typedef DetGroupMerger Merger;
    int size = groupPlusIndex.size();
    
    if(size==2){
      Merger::orderAndMergeTwoLevels(groupPlusIndex[1].first,
				     groupPlusIndex[0].first,result,1,1);
    }else if(size==3){	
      std::vector<DetGroup> tmp;
      Merger::orderAndMergeTwoLevels(groupPlusIndex[2].first,
				     groupPlusIndex[1].first,tmp,1,1);
      Merger::orderAndMergeTwoLevels(tmp,groupPlusIndex[0].first,result,1,1);      
    }      
  }
}

void
TIDLayer::orderAndMergeLevels(const TrajectoryStateOnSurface& tsos,
			      const Propagator& prop,
			      const vector<vector<DetGroup> > groups,
			      const vector<int> indices,
			      std::vector<DetGroup> & result )
{
  vector< pair<vector<DetGroup>,int> > groupPlusIndex;
  for(unsigned int i=0;i<groups.size();i++){
    groupPlusIndex.push_back(pair<vector<DetGroup>,int>(groups[i],indices[i]) );
  }
  //order is ring3,ring1,ring2
  std::sort(groupPlusIndex.begin(),groupPlusIndex.end(),TIDringLess());
  

  float zpos = tsos.globalPosition().z();
  if(tsos.globalMomentum().z()*zpos>0){ // momentum points outwards
    if(prop.propagationDirection() == alongMomentum)
      mergeOutward(groupPlusIndex,result);
    else
      mergeInward(groupPlusIndex,result);
  }
  else{ //  momentum points inwards
    if(prop.propagationDirection() == oppositeToMomentum)
      mergeOutward(groupPlusIndex,result);
    else
      mergeInward(groupPlusIndex,result);    
  }  
  
}

int
TIDLayer::findClosest(const vector<GlobalPoint>& ringCrossing ) const
{
  int theBin = 0;
  const TIDRing* theFrontRing = dynamic_cast<const TIDRing*>(theComps[0]);
  //float initialR = ( theComps.front()->specificSurface().innerRadius() +
  //	     theComps.front()->specificSurface().outerRadius())/2.;
  float initialR = ( theFrontRing->specificSurface().innerRadius() +
		     theFrontRing->specificSurface().outerRadius())/2.;
  float rDiff = fabs( ringCrossing.front().perp() - initialR);
  for (int i = 0; i < 3 ; i++){
    const TIDRing* theRing = dynamic_cast<const TIDRing*>(theComps[i]);
    float ringR = ( theRing->specificSurface().innerRadius() + theRing->specificSurface().outerRadius())/2.;
    float testDiff = fabs( ringCrossing[i].perp() - ringR);
    if ( theBin<0 || testDiff<rDiff ) {
      rDiff = testDiff;
      theBin = i;
    }
  }
  return theBin;
}

int
TIDLayer::findNextIndex(const vector<GlobalPoint>& ringCrossing, int closest ) const
{

  int firstIndexToCheck = (closest != 0)? 0 : 1; 
  const TIDRing* theFrontRing = dynamic_cast<const TIDRing*>(theComps[firstIndexToCheck]);
  float initialR = ( theFrontRing->specificSurface().innerRadius() +
		     theFrontRing->specificSurface().outerRadius())/2.;	     

  float rDiff = fabs( ringCrossing.front().perp() - initialR);
  int theBin = firstIndexToCheck;
  for (int i = firstIndexToCheck+1; i < 3 ; i++){
    if ( i != closest) {
      const TIDRing* theRing = dynamic_cast<const TIDRing*>(theComps[i]);
      float ringR = ( theRing->specificSurface().innerRadius() + theRing->specificSurface().outerRadius())/2.;
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
  float thetamin = ( max(0.,tsRadius-ymax))/(fabs(tsos.globalPosition().z())+10.); // add 10 cm contingency 
  float thetamax = ( tsRadius + ymax)/(fabs(tsos.globalPosition().z())-10.);

  const TIDRing* theRing = dynamic_cast<const TIDRing*>(theComps[index]);
  const BoundDisk& ringDisk = theRing->specificSurface();
  float ringMinZ = fabs( ringDisk.position().z()) - ringDisk.bounds().thickness()/2.;
  float ringMaxZ = fabs( ringDisk.position().z()) + ringDisk.bounds().thickness()/2.; 
  float thetaRingMin =  ringDisk.innerRadius()/ ringMaxZ;
  float thetaRingMax =  ringDisk.outerRadius()/ ringMinZ;

  // do the theta regions overlap ?

  if ( thetamin > thetaRingMax || thetaRingMin > thetamax) { return false;}

  return true;
}


