#include "CompositeTECWedge.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "ForwardDiskSectorBuilderFromDet.h"
#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"

#include "TrackingTools/DetLayers/interface/rangesIntersect.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"

#include "TkDetUtil.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include <boost/function.hpp>

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

// --------- Temporary solution. DetSorting.h has to be used.
namespace {
  class DetPhiLess {
  public:
    bool operator()(const GeomDet* a,const GeomDet* b) 
    {
      return Geom::phiLess(a->surface(), b->surface());
    } 
  };
}
// ---------------------

CompositeTECWedge::CompositeTECWedge(vector<const GeomDet*>& innerDets,
				     vector<const GeomDet*>& outerDets):
  theFrontDets(innerDets.begin(),innerDets.end()), theBackDets(outerDets.begin(),outerDets.end())
{  
  theDets.assign(theFrontDets.begin(),theFrontDets.end());
  theDets.insert(theDets.end(),theBackDets.begin(),theBackDets.end());


  // 
  std::sort( theFrontDets.begin(), theFrontDets.end(), DetPhiLess() );
  std::sort( theBackDets.begin(),  theBackDets.end(),  DetPhiLess() );
  
  theFrontSector = ForwardDiskSectorBuilderFromDet()( theFrontDets );
  theBackSector  = ForwardDiskSectorBuilderFromDet()( theBackDets );
  theDiskSector = ForwardDiskSectorBuilderFromDet()( theDets );

  //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "DEBUG INFO for CompositeTECWedge" << "\n"
			  << "TECWedge z, perp,innerRadius,outerR: " 
			  << this->position().z() << " , "
			  << this->position().perp() << " , "
			  << theDiskSector->innerRadius() << " , "
			  << theDiskSector->outerRadius() ;


  for(vector<const GeomDet*>::const_iterator it=theFrontDets.begin(); 
      it!=theFrontDets.end(); it++){
    LogDebug("TkDetLayers") << "frontDet phi,z,r: " 
			    << (*it)->surface().position().phi() << " , "
			    << (*it)->surface().position().z() <<   " , "
			    << (*it)->surface().position().perp();
  }

  for(vector<const GeomDet*>::const_iterator it=theBackDets.begin(); 
      it!=theBackDets.end(); it++){
    LogDebug("TkDetLayers") << "backDet phi,z,r: " 
			    << (*it)->surface().phi() << " , "
			    << (*it)->surface().position().z() <<   " , "
			    << (*it)->surface().position().perp() ;
  }
  //----------------------------------- 


}

CompositeTECWedge::~CompositeTECWedge(){

} 


const vector<const GeometricSearchDet*>& 
CompositeTECWedge::components() const{
  throw DetLayerException("CompositeTECWedge doesn't have GeometricSearchDet components");
}

  
pair<bool, TrajectoryStateOnSurface>
CompositeTECWedge::compatible( const TrajectoryStateOnSurface& ts, const Propagator&, 
			       const MeasurementEstimator&) const{
  edm::LogError("TkDetLayers") << "temporary dummy implementation of CompositeTECWedge::compatible()!!" ;
  return pair<bool,TrajectoryStateOnSurface>();
}


void
CompositeTECWedge::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					   const MeasurementEstimator& est,
					   std::vector<DetGroup> & result) const{
  SubLayerCrossings  crossings; 
  crossings = computeCrossings( tsos, prop.propagationDirection());
  if(! crossings.isValid()) return;

  std::vector<DetGroup> closestResult;
  addClosest( tsos, prop, est, crossings.closest(), closestResult);
  LogDebug("TkDetLayers") 
    << "in CompositeTECWedge::groupedCompatibleDets,closestResult.size(): "
    << closestResult.size() ;

  if (closestResult.empty()) return;
  
  DetGroupElement closestGel( closestResult.front().front());
  float window = tkDetUtil::computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);

  searchNeighbors( tsos, prop, est, crossings.closest(), window,
		   closestResult, false);

  vector<DetGroup> nextResult;
  searchNeighbors( tsos, prop, est, crossings.other(), window,
  		   nextResult, true);

  int crossingSide = LayerCrossingSide().endcapSide( closestGel.trajectoryState(), prop);
  DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result,
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
  if (!frontPath.first) return SubLayerCrossings();

  GlobalPoint gFrontPoint( crossing.position(frontPath.second));
  LogDebug("TkDetLayers") << "in TECWedge,front crossing r,z,phi: (" 
       << gFrontPoint.perp() << ","
       << gFrontPoint.z() << "," 
       << gFrontPoint.phi() << ")" ;


  int frontIndex = findClosestDet(gFrontPoint,0); 
  SubLayerCrossing frontSLC( 0, frontIndex, gFrontPoint);

  pair<bool,double> backPath = crossing.pathLength( *theBackSector);
  if (!backPath.first) return SubLayerCrossings();

  GlobalPoint gBackPoint( crossing.position(backPath.second));
  LogDebug("TkDetLayers") 
    << "in TECWedge,back crossing r,z,phi: (" 
    << gBackPoint.perp() << ","
    << gBackPoint.z() << "," 
    << gBackPoint.phi() << ")" << endl;
  
  int backIndex = findClosestDet(gBackPoint,1);
  SubLayerCrossing backSLC( 1, backIndex, gBackPoint);

  float frontDist = std::abs(Geom::deltaPhi( double(gFrontPoint.barePhi()), 
					     double(theFrontDets[frontIndex]->surface().phi())));
  /*
  float frontDist = theFrontDets[frontIndex]->surface().phi()  - gFrontPoint.phi(); 
  frontDist *= Geom::phiLess( theFrontDets[frontIndex]->surface().phi(),gFrontPoint.barePhi()) ? -1. : 1.; 
  if (frontDist < 0.) { frontDist += 2.*Geom::pi();}
  */
  float backDist = std::abs(Geom::deltaPhi( double(gBackPoint.barePhi()), 
					    double(theBackDets[backIndex]->surface().phi())));
  /*
  float backDist = theBackDets[backIndex]->surface().phi()  - gBackPoint.phi(); 
  backDist  *= Geom::phiLess( theBackDets[backIndex]->surface().phi(),gBackPoint.barePhi()) ? -1. : 1.;
  if ( backDist < 0.) { backDist  += 2.*Geom::pi();}
  */
  
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

  LogDebug("TkDetLayers")  
    << "in CompositeTECWedge,adding GeomDet at r,z,phi: (" 
    << sWedge[crossing.closestDetIndex()]->position().perp() << "," 
    << sWedge[crossing.closestDetIndex()]->position().z() << "," 
    << sWedge[crossing.closestDetIndex()]->position().phi() << ")" ;

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
    if ( Geom::phiLess(gCrossingPos.barePhi(), sWedge[closestIndex]->surface().phi()) ) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  typedef CompatibleDetToGroupAdder Adder;
  for (int idet=negStartIndex; idet >= 0; idet--) {
    //if(idet <0 || idet>=sWedge.size()) {edm::LogInfo(TkDetLayers) << "==== warning! gone out vector bounds.idet: " << idet ;break;}
    if (!tkDetUtil::overlapInPhi( gCrossingPos, *sWedge[idet], window)) break;
    if (!Adder::add( *sWedge[idet], tsos, prop, est, result)) break;
  }
  for (int idet=posStartIndex; idet < static_cast<int>(sWedge.size()); idet++) {
    //if(idet <0 || idet>=sWedge.size()) {edm::LogInfo(TkDetLayers) << "==== warning! gone out vector bounds.idet: " << idet ;break;}
    if (!tkDetUtil::overlapInPhi( gCrossingPos, *sWedge[idet], window)) break;
    if (!Adder::add( *sWedge[idet], tsos, prop, est, result)) break;
  }
}



int
CompositeTECWedge::findClosestDet( const GlobalPoint& startPos,int sectorId) const
{
  vector<const GeomDet*> const & myDets = sectorId==0 ? theFrontDets : theBackDets;

  int close = 0;
  float closeDist = fabs( (myDets.front()->toLocal(startPos)).x());
  for (unsigned int i = 0; i < myDets.size(); i++ ) {
    float dist = (myDets[i]->surface().toLocal(startPos)).x();
    if ( fabs(dist) < fabs(closeDist) ) {
      close = i;
      closeDist = dist;
    }
  }
  return close;
}
