#include "RecoTracker/TkDetLayers/interface/TIDRing.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

#include "RecoTracker/TkDetLayers/interface/TkDetUtil.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include <boost/function.hpp>

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
TIDRing::compatible( const TrajectoryStateOnSurface&, const Propagator&, 
		  const MeasurementEstimator&) const{
  edm::LogError("TkDetLayers") << "temporary dummy implementation of TIDRing::compatible()!!" ;
  return pair<bool,TrajectoryStateOnSurface>();
}



void 
TIDRing::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
				 const Propagator& prop,
				 const MeasurementEstimator& est,
				 std::vector<DetGroup>& result) const
{
  SubLayerCrossings  crossings; 
  crossings = computeCrossings( tsos, prop.propagationDirection());
  if(! crossings.isValid()) return;

  std::vector<DetGroup> closestResult;
  addClosest( tsos, prop, est, crossings.closest(), closestResult); 
  if (closestResult.empty())     return;
  
  DetGroupElement closestGel( closestResult.front().front());  
  float phiWindow = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est); 
  searchNeighbors( tsos, prop, est, crossings.closest(), phiWindow,
		   closestResult, false); 

  vector<DetGroup> nextResult;
  searchNeighbors( tsos, prop, est, crossings.other(), phiWindow,
		   nextResult, true); 

  int crossingSide = LayerCrossingSide().endcapSide( closestGel.trajectoryState(), prop);
  DetGroupMerger::orderAndMergeTwoLevels( closestResult, nextResult, result,
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
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);



  pair<bool,double> backPath = crossing.pathLength( *theBackDisk);
  if (!backPath.first) return SubLayerCrossings();

  GlobalPoint gBackPoint( crossing.position(backPath.second));
  int backIndex = theBackBinFinder.binIndex(gBackPoint.phi());
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);

  
  // 0ss: frontDisk has index=0, backDisk has index=1
  float frontDist = std::abs(Geom::deltaPhi( double(gFrontPoint.barePhi()), 
					     double(theFrontDets[frontIndex]->surface().phi())));
  float backDist = std::abs(Geom::deltaPhi( double(gBackPoint.barePhi()), 
					    double(theBackDets[backIndex]->surface().phi())));


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
  return CompatibleDetToGroupAdder::add( *det, tsos, prop, est, result); 
}



float 
TIDRing::calculatePhiWindow( const MeasurementEstimator::Local2DVector& maxDistance, 
			     const TrajectoryStateOnSurface& ts, 
			     const BoundPlane& plane)
{
  LocalPoint start = ts.localPosition();
  float corners[] = { plane.toGlobal(LocalPoint( start.x()+maxDistance.x(), start.y()+maxDistance.y())).barePhi(),  
		      plane.toGlobal(LocalPoint( start.x()-maxDistance.x(), start.y()+maxDistance.y())).barePhi(),
		      plane.toGlobal(LocalPoint( start.x()-maxDistance.x(), start.y()-maxDistance.y())).barePhi(),
		      plane.toGlobal(LocalPoint( start.x()+maxDistance.x(), start.y()-maxDistance.y())).barePhi()
  };

  float phimin = corners[0];
  float phimax = phimin;
  for ( int i = 1; i<4; i++) {
    float cPhi = corners[i];
    if ( Geom::phiLess( cPhi, phimin)) { phimin = cPhi; }
    if ( Geom::phiLess( phimax, cPhi)) { phimax = cPhi; }
  }
  float phiWindow = phimax - phimin;
  if ( phiWindow < 0.) { phiWindow +=  2.*Geom::pi();}

  return phiWindow;
}



float TIDRing::computeWindowSize( const GeomDet* det, 
				  const TrajectoryStateOnSurface& tsos, 
				  const MeasurementEstimator& est)
{
  const BoundPlane& startPlane = det->surface() ;  
  MeasurementEstimator::Local2DVector maxDistance = 
    est.maximalLocalDisplacement( tsos, startPlane);
  return calculatePhiWindow( maxDistance, tsos, startPlane);
}



namespace {

  struct PhiLess {
    bool operator()(float a, float b) const {
      return Geom::phiLess(a,b);
    }
  };
  
  bool overlapInPhi( const GlobalPoint& crossPoint,const GeomDet & det, float phiWindow) 
  {
    float phi = crossPoint.barePhi();
    pair<float,float> phiRange(phi-phiWindow, phi+phiWindow);
    pair<float,float> detPhiRange = det.surface().phiSpan(); 
    //   return rangesIntersect( phiRange, detPhiRange, boost::function<bool(float,float)>(&Geom::phiLess));
    return rangesIntersect( phiRange, detPhiRange, PhiLess());
  }
  
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
    if ( Geom::phiLess( gCrossingPos.barePhi(), sLayer[closestIndex]->surface().phi())) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  const BinFinderType& binFinder = (crossing.subLayerIndex()==0 ? theFrontBinFinder : theBackBinFinder);

  typedef CompatibleDetToGroupAdder Adder;
  int half = sLayer.size()/2;  // to check if dets are called twice....
  for (int idet=negStartIndex; idet >= negStartIndex - half; idet--) {
    const GeomDet & neighborDet = *sLayer[binFinder.binIndex(idet)];
    if (!overlapInPhi( gCrossingPos, neighborDet, window)) break;
    if (!Adder::add( *neighborDet, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + half; idet++) {
    const GeomDet & neighborDet = *sLayer[binFinder.binIndex(idet)];
    if (!overlapInPhi( gCrossingPos, neighborDet, window)) break;
    if (!Adder::add( *neighborDet, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}
