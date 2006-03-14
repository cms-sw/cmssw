#include "RecoTracker/TkDetLayers/interface/TIDLayer.h"
#include "RecoTracker/TkDetLayers/interface/DetGroupMerger.h"

//#include "CommonDet/DetLayout/src/DetLessR.h"
#include "Geometry/Surface/interface/SimpleDiskBounds.h"
#include "Utilities/General/interface/CMSexception.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"


typedef GeometricSearchDet::DetWithState DetWithState;


class TIDringLess : public binary_function< int, int, bool> {
  // z-position ordering of TID rings indices
public:
  bool operator()( const pair<vector<DetGroup>,int>  a,const pair<vector<DetGroup>,int>  b) const {
    if(a.second==2) {return true;}
    else if(a.second==0) {
      if(b.second==2) return false;
      if(b.second==1) return true;
    }
    return false;    
  };
};


TIDLayer::TIDLayer(vector<const TIDRing*>& rings):
  theRings(rings.begin(),rings.end())
{
  //They should be already R-ordered. TO BE CHECKED!!
  //sort( theRings.begin(), theRings.end(), DetLessR());
  setSurface( computeDisk( theRings ) );

  if ( theRings.size() != 3) throw Genexception("Number of rings in TID layer is not equal to 3 !!");

}


BoundDisk* 
TIDLayer::computeDisk( const vector<const TIDRing*>& rings) const
{
  float theRmin( 1000.);
  float theRmax( 0.);
  float theZmin(1000.);
  float theZmax(-1000.);

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
  vector<const TIDRing*>::const_iterator i;
  for (i=theRings.begin(); i!=theRings.end(); i++) {
    delete *i;
  }

} 

vector<const GeomDet*> 
TIDLayer::basicComponents() const{
  cout << "temporary dummy implementation of TIDLayer::basicComponents()!!" << endl;
  return vector<const GeomDet*>();
}
  
pair<bool, TrajectoryStateOnSurface>
TIDLayer::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
		  const MeasurementEstimator&) const{
  cout << "temporary dummy implementation of TIDLayer::compatible()!!" << endl;
  return pair<bool,TrajectoryStateOnSurface>();
}


vector<DetWithState> 
TIDLayer::compatibleDets( const TrajectoryStateOnSurface& startingState,
			  const Propagator& prop, 
			  const MeasurementEstimator& est) const
{
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
TIDLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop,
				 const MeasurementEstimator& est) const
{
  vector<int> ringIndices = ringIndicesByCrossingProximity(startingState,prop);
  if ( ringIndices.size()!=3 ) {
    cout << "TkRingedForwardLayer::groupedCompatibleDets : ringIndices.size() = "
	 << ringIndices.size() << " and not =3!!" << endl;
    return vector<DetGroup>();
  }

  vector<DetGroup> closestResult;
  vector<DetGroup> nextResult;
  vector<DetGroup> nextNextResult;
  vector<vector<DetGroup> > groupsAtRingLevel;

  closestResult = theRings[ringIndices[0]]->groupedCompatibleDets( startingState, prop, est);		
  if ( closestResult.empty() ){
    nextResult = theRings[ringIndices[1]]->groupedCompatibleDets( startingState, prop, est); 
    return nextResult;
  }

  groupsAtRingLevel.push_back(closestResult);

  DetGroupElement closestGel( closestResult.front().front());  
  float rWindow = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est); 
  if(!overlapInR(closestGel.trajectoryState(),ringIndices[1],rWindow)) return closestResult;

  nextResult = theRings[ringIndices[1]]->groupedCompatibleDets( startingState, prop, est);
  if(nextResult.empty()) return closestResult;
  groupsAtRingLevel.push_back(nextResult);

  if(!overlapInR(closestGel.trajectoryState(),ringIndices[2],rWindow) ) 
    //then merge 2 levels & return 
    return orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,ringIndices);      
    
  nextNextResult = theRings[ringIndices[2]]->groupedCompatibleDets( startingState, prop, est);   
  if(nextNextResult.empty()) 
    // then merge 2 levels and return 
    return orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,ringIndices);
  
  groupsAtRingLevel.push_back(nextNextResult);
  // merge 3 level and return merged   
  return orderAndMergeLevels(closestGel.trajectoryState(),prop,groupsAtRingLevel,ringIndices);  
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
    pair<bool,double> pathlen = myXing.pathLength( theRings[i]->specificSurface());
    if ( pathlen.first ) { 
      ringCrossings.push_back( GlobalPoint( myXing.position(pathlen.second )));
      ringXDirections.push_back( GlobalVector( myXing.direction(pathlen.second )));
    } else {
      // TO FIX.... perhaps there is something smarter to do
      //throw Genexception("trajectory doesn't cross TID rings");
      ringCrossings.push_back( GlobalPoint( 0.,0.,0.));
      ringXDirections.push_back( GlobalVector( 0.,0.,0.));
    }
  }
  if (ringCrossings.size() != 3 )  
    throw Genexception("TIDLayer::groupedCompDets => Problem with crossing size, not equal 3 !");
  

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
  const BoundPlane& startPlane( dynamic_cast<const BoundPlane&>(det->surface()) );  
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement( tsos, startPlane);
  return maxDistance.y();
}



vector<DetGroup> 
TIDLayer::orderAndMergeLevels(const TrajectoryStateOnSurface& tsos,
			      const Propagator& prop,
			      const vector<vector<DetGroup> > groups,
			      const vector<int> indices ) const 
{
  vector< pair<vector<DetGroup>,int> > groupPlusIndex;
  for(unsigned int i=0;i<groups.size();i++){
    groupPlusIndex.push_back(pair<vector<DetGroup>,int>(groups[i],indices[i]) );
  }
  //order is ring3,ring1,ring2
  sort(groupPlusIndex.begin(),groupPlusIndex.end(),TIDringLess());
  

  DetGroupMerger merger;
  vector<DetGroup> result;
  
  float zpos = tsos.globalPosition().z();
  if(tsos.globalMomentum().z()*zpos>0){ // momentum points outwards
    if(prop.propagationDirection() == alongMomentum){
      result = merger.orderAndMergeTwoLevels(groupPlusIndex[0].first,
					     groupPlusIndex[1].first,1,1);
      if(groupPlusIndex.size()==3)
	result =  merger.orderAndMergeTwoLevels(result,groupPlusIndex[2].first,1,1);      
    }
    else{ 
      result = merger.orderAndMergeTwoLevels(groupPlusIndex[2].first,
					     groupPlusIndex[1].first,1,1);
      if(groupPlusIndex.size()==3)
	result =  merger.orderAndMergeTwoLevels(result,groupPlusIndex[0].first,1,1);      
    }      
  }
  else{ //  momentum points inwards
    // I don't know what to do 
    throw Genexception("TIDLayer::orderAndMergeLevels() inward implementation still to do");
  }
  
  return result;
}

int
TIDLayer::findClosest(const vector<GlobalPoint>& ringCrossing ) const
{
  int theBin = -1;
  float rDiff = 200.;
  for (int i = 0; i < 3 ; i++){
    float ringR = ( theRings[i]->specificSurface().innerRadius() + theRings[i]->specificSurface().outerRadius())/2.;
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
  int theBin = -1;
  float rDiff = 200.;
  for (int i = 0; i < 3 ; i++){
    if ( i != closest) {
      float ringR = ( theRings[i]->specificSurface().innerRadius() + theRings[i]->specificSurface().outerRadius())/2.;
      float testDiff = fabs( ringCrossing[i].perp() - ringR);
      if ( theBin<0 || testDiff<rDiff ) {
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

  const BoundDisk& ringDisk = theRings[index]->specificSurface();
  float ringMinZ = fabs( ringDisk.position().z()) - ringDisk.bounds().thickness()/2.;
  float ringMaxZ = fabs( ringDisk.position().z()) + ringDisk.bounds().thickness()/2.; 
  float thetaRingMin =  ringDisk.innerRadius()/ ringMaxZ;
  float thetaRingMax =  ringDisk.outerRadius()/ ringMinZ;

  // do the theta regions overlap ?

  if ( thetamin > thetaRingMax || thetaRingMin > thetamax) { return false;}

  return true;
}
