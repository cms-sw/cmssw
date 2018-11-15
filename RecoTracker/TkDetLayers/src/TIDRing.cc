#include "TIDRing.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/DetLayers/interface/ForwardRingDiskBuilderFromDet.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TkDetUtil.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include <boost/function.hpp>

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

TIDRing::TIDRing(std::vector<const GeomDet*>& innerDets,
		 std::vector<const GeomDet*>& outerDets):
  GeometricSearchDet(true),
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
  float phiWindow =  tkDetUtil::computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est); 
  searchNeighbors( tsos, prop, est, crossings.closest(), phiWindow,
		   closestResult, false); 

  vector<DetGroup> nextResult;
  searchNeighbors( tsos, prop, est, crossings.other(), phiWindow,
		   nextResult, true); 

  int crossingSide = LayerCrossingSide().endcapSide( closestGel.trajectoryState(), prop);
  DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result,
					  crossings.closestIndex(), crossingSide);
}

// indentical in CompositeTECWedge
SubLayerCrossings 
TIDRing::computeCrossings(const TrajectoryStateOnSurface& startingState,
			  PropagationDirection propDir) const
{

  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition() );
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum() );
 
  auto rho = startingState.transverseCurvature();
  
  HelixForwardPlaneCrossing crossing(startPos,startDir,rho,propDir);
  
  pair<bool,double> frontPath = crossing.pathLength( *theFrontDisk);
  if (!frontPath.first) return SubLayerCrossings();

  pair<bool,double> backPath = crossing.pathLength( *theBackDisk);
  if (!backPath.first) return SubLayerCrossings();

  GlobalPoint gFrontPoint(crossing.position(frontPath.second));
  GlobalPoint gBackPoint( crossing.position(backPath.second));

  int frontIndex = theFrontBinFinder.binIndex(gFrontPoint.barePhi()); 
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);

  int backIndex = theBackBinFinder.binIndex(gBackPoint.barePhi());
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);

  
  // 0ss: frontDisk has index=0, backDisk has index=1
  float frontDist = std::abs(Geom::deltaPhi( gFrontPoint.barePhi(), 
					     theFrontDets[frontIndex]->surface().phi()));
  float backDist = std::abs(Geom::deltaPhi( gBackPoint.barePhi(), 
					    theBackDets[backIndex]->surface().phi()));


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



void TIDRing::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				     const Propagator& prop,
				     const MeasurementEstimator& est,
				     const SubLayerCrossing& crossing,
				     float window, 
				     vector<DetGroup>& result,
				     bool checkClosest) const
{
  const GlobalPoint& gCrossingPos = crossing.position();

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
    if (!tkDetUtil::overlapInPhi( gCrossingPos, neighborDet, window)) break;
    if (!Adder::add( neighborDet, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + half; idet++) {
    const GeomDet & neighborDet = *sLayer[binFinder.binIndex(idet)];
    if (!tkDetUtil::overlapInPhi( gCrossingPos, neighborDet, window)) break;
    if (!Adder::add( neighborDet, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}
