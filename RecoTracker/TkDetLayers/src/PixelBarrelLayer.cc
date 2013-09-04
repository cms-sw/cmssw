#include "PixelBarrelLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/DetLayers/interface/rangesIntersect.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

PixelBarrelLayer::PixelBarrelLayer(vector<const PixelRod*>& innerRods,
				   vector<const PixelRod*>& outerRods) : 
  theInnerComps(innerRods.begin(),innerRods.end()), 
  theOuterComps(outerRods.begin(),outerRods.end())
{
  theComps.assign(theInnerComps.begin(),theInnerComps.end());
  theComps.insert(theComps.end(),theOuterComps.begin(),theOuterComps.end());

  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin();
      it!=theComps.end();it++){  
    theBasicComps.insert(theBasicComps.end(),	
			 (**it).basicComponents().begin(),
			 (**it).basicComponents().end());
  }

  theInnerCylinder = cylinder( theInnerComps);
  theOuterCylinder = cylinder( theOuterComps);

  theInnerBinFinder = BinFinderType(theInnerComps.front()->position().phi(),
				    theInnerComps.size());
  theOuterBinFinder = BinFinderType(theOuterComps.front()->position().phi(),
				    theOuterComps.size());

  
  BarrelDetLayer::initialize();


  //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "==== DEBUG PixelBarrelLayer =====" ; 
  LogDebug("TkDetLayers") << "PixelBarrelLayer innerCyl r,lenght: "
                          << theInnerCylinder->radius() << " , "
                          << theInnerCylinder->bounds().length();

  LogDebug("TkDetLayers") << "PixelBarrelLayer outerCyl r,lenght: " 
			  << theOuterCylinder->radius() << " , "
			  << theOuterCylinder->bounds().length();


    for (vector<const GeometricSearchDet*>::const_iterator i=theInnerComps.begin();
       i != theInnerComps.end(); i++){
    LogDebug("TkDetLayers") << "inner PixelRod pos z,perp,eta,phi: " 
			    << (**i).position().z()    << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta()  << " , " 
			    << (**i).position().phi()  ;
  }
  
  for (vector<const GeometricSearchDet*>::const_iterator i=theOuterComps.begin();
       i != theOuterComps.end(); i++){
    LogDebug("TkDetLayers") << "outer PixelRod pos z,perp,eta,phi: " 
			    << (**i).position().z()    << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta()  << " , " 
			    << (**i).position().phi()  ;
  }
  LogDebug("TkDetLayers") << "==== end DEBUG PixelBarrelLayer =====" ; 
  //----------------------------------- 
}

PixelBarrelLayer::~PixelBarrelLayer(){
  vector<const GeometricSearchDet*>::const_iterator i;
  for (i=theComps.begin(); i!=theComps.end(); i++) {
    delete *i;
  }
} 
  
namespace {

  bool groupSortByR(DetGroupElement i,DetGroupElement j) { return (i.det()->position().perp()<j.det()->position().perp()); }

}

void 
PixelBarrelLayer::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					   const MeasurementEstimator& est,
					   std::vector<DetGroup> & result) const {
  vector<DetGroup> closestResult;
  SubLayerCrossings  crossings;
  crossings = computeCrossings( tsos, prop.propagationDirection());
  if(! crossings.isValid()) return;  

  addClosest( tsos, prop, est, crossings.closest(), closestResult);
  if (closestResult.empty()) {
    addClosest( tsos, prop, est, crossings.other(), result);
    return;
  }

  //std::cout << "closest cross pos,r:" << crossings.closest().position()<<","<<crossings.closest().position().perp() 
  //	    << "  other cross pos,r:" << crossings.other().position()<<","<<crossings.other().position().perp()   << std::endl;

  DetGroupElement closestGel( closestResult.front().front());
  float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);

  //std::cout << "before searchNeighbors closestResult size:" << (closestResult.size()?closestResult.front().size():0) << std::endl;
  
  searchNeighbors( tsos, prop, est, crossings.closest(), window,
		   closestResult, false);
  
  vector<DetGroup> nextResult;
  searchNeighbors( tsos, prop, est, crossings.other(), window,
		   nextResult, true);

  //std::cout << "closestResult size:" << (closestResult.size()?closestResult.front().size():0)
  //	    << " nextResult size:" << (nextResult.size()?nextResult.front().size():0) << std::endl;
  
  int crossingSide = LayerCrossingSide().barrelSide( closestGel.trajectoryState(), prop);
  DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult), std::move(nextResult), result, 
					  crossings.closestIndex(), crossingSide);

  //std::cout << "PixelBarrelLayer::groupedCompatibleDetsV - result size=" << result.size() << std::endl;

  /*
  for (auto gr : result) {
    std::cout << "new group" << std::endl;
    for (auto dge : gr) {
      PixelBarrelNameUpgrade name(dge.det()->geographicalId());
      std::cout << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" name:"<<name.name()<<" isHalf:"<<name.isHalfModule()<<" tsos at:" <<dge.trajectoryState().globalPosition()<< std::endl;
    }
  }
  */

  if (theInnerCylinder->radius()>17.0) {
    //do splitting of groups for outer 'pixel' layer of phase 2 tracker
    //fixme: to be changed when moving to a new DetId schema with 'matched' hits
    std::vector<DetGroup> splitResult;
    for (auto gr : result) {
      if (gr.size()==1) {
	splitResult.push_back(gr);
	continue;
      }
      //sort according to R
      std::sort(gr.begin(),gr.end(),groupSortByR);
      DetGroup firstGroup; //this group contains the first dets of 2S/PS modules
      DetGroup secondGroup;//this group contains dets on the other side of 2S/PS modules
      for (auto dge : gr) {
	if (firstGroup.size()==0) {
	  firstGroup.push_back(dge);
	  continue;
	}
	bool foundInFirstGroup = false;
	for (auto dge_f : firstGroup) {
	  if (abs(int(dge.det()->geographicalId().rawId())-int(dge_f.det()->geographicalId().rawId()))==4 && 
	      fabs(dge.det()->position().perp()-dge_f.det()->position().perp())>0.15) {
	    //std::cout << "found dge for second group with id: " << dge.det()->geographicalId().rawId() << std::endl;
	    secondGroup.push_back(dge);
	    foundInFirstGroup = true;
	    break;
	  }
	}
	if (!foundInFirstGroup )firstGroup.push_back(dge);
      }
      splitResult.push_back(firstGroup);
      if (secondGroup.size()>0) splitResult.push_back(secondGroup);
    }
    splitResult.swap(result);
 
    /*
    std::cout << "AFTER SPLITTING" <<std::endl;
    for (auto gr : result) {
      std::cout << "new group" << std::endl;
      for (auto dge : gr) {
	PixelBarrelNameUpgrade name(dge.det()->geographicalId());
	std::cout << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" name:"<<name.name()<<" isHalf:"<<name.isHalfModule()<<" tsos at:" <<dge.trajectoryState().globalPosition()<< std::endl;
      }
    }
    */

  }//end of hack for phase 2 stacked layers

}


// private methods for the implementation of groupedCompatibleDets()

SubLayerCrossings PixelBarrelLayer::computeCrossings( const TrajectoryStateOnSurface& startingState,
						      PropagationDirection propDir) const
{
  GlobalPoint startPos( startingState.globalPosition());
  GlobalVector startDir( startingState.globalMomentum());
  double rho( startingState.transverseCurvature());
  
  HelixBarrelCylinderCrossing innerCrossing( startPos, startDir, rho,
					     propDir,*theInnerCylinder);

  if (!innerCrossing.hasSolution()) return SubLayerCrossings();
  //{
  //edm::LogInfo(TkDetLayers) << "ERROR in PixelBarrelLayer: inner cylinder not crossed by track" ;
  //throw DetLayerException("TkRodBarrelLayer: inner subRod not crossed by track");
  //}

  GlobalPoint gInnerPoint( innerCrossing.position());
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.phi());
  float innerDist = theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.phi();
  SubLayerCrossing innerSLC( 0, innerIndex, gInnerPoint);

  HelixBarrelCylinderCrossing outerCrossing( startPos, startDir, rho,
					     propDir,*theOuterCylinder);

  if (!outerCrossing.hasSolution()) return SubLayerCrossings();
  //if (!outerCrossing.hasSolution()) {
  //  throw DetLayerException("PixelBarrelLayer: inner cylinder not crossed by track");
  //}

  GlobalPoint gOuterPoint( outerCrossing.position());
  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.phi());
  float outerDist = theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.phi() ;
  SubLayerCrossing outerSLC( 1, outerIndex, gOuterPoint);
  
  innerDist *= PhiLess()( theInnerBinFinder.binPosition(innerIndex),gInnerPoint.phi()) ? -1. : 1.; 
  outerDist *= PhiLess()( theOuterBinFinder.binPosition(outerIndex),gOuterPoint.phi()) ? -1. : 1.; 
  if (innerDist < 0.) { innerDist += 2.*Geom::pi();}
  if (outerDist < 0.) { outerDist += 2.*Geom::pi();}
  

  if (innerDist < outerDist) {
    return SubLayerCrossings( innerSLC, outerSLC, 0);
  }
  else {
    return SubLayerCrossings( outerSLC, innerSLC, 1);
  } 
}

bool PixelBarrelLayer::addClosest( const TrajectoryStateOnSurface& tsos,
				   const Propagator& prop,
				   const MeasurementEstimator& est,
				   const SubLayerCrossing& crossing,
				   vector<DetGroup>& result) const
{
  const vector<const GeometricSearchDet*>& sub( subLayer( crossing.subLayerIndex()));
  const GeometricSearchDet* det(sub[crossing.closestDetIndex()]);
  return CompatibleDetToGroupAdder::add( *det, tsos, prop, est, result);
}

float PixelBarrelLayer::computeWindowSize( const GeomDet* det, 
					   const TrajectoryStateOnSurface& tsos, 
					   const MeasurementEstimator& est) const
{
  double xmax = 
    est.maximalLocalDisplacement(tsos, det->surface()).x();
  return calculatePhiWindow( xmax, *det, tsos);
}


double PixelBarrelLayer::calculatePhiWindow( double Xmax, const GeomDet& det,
					     const TrajectoryStateOnSurface& state) const
{

  LocalPoint startPoint = state.localPosition();
  LocalVector shift( Xmax , 0. , 0.);
  LocalPoint shift1 = startPoint + shift;
  LocalPoint shift2 = startPoint + (-shift); 
  //LocalPoint shift2( startPoint); //original code;
  //shift2 -= shift;

  double phi1 = det.surface().toGlobal(shift1).phi();
  double phi2 = det.surface().toGlobal(shift2).phi();
  double phiStart = state.globalPosition().phi();
  double phiWin = min(fabs(phiStart-phi1),fabs(phiStart-phi2));

  return phiWin;
}


void PixelBarrelLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
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

  const BinFinderType& binFinder = (crossing.subLayerIndex()==0 ? theInnerBinFinder : theOuterBinFinder);

  CompatibleDetToGroupAdder adder;
  int quarter = sLayer.size()/4;
  for (int idet=negStartIndex; idet >= negStartIndex - quarter; idet--) {
    const GeometricSearchDet* neighborRod = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborRod, window)) break;
    if (!adder.add( *neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + quarter; idet++) {
    const GeometricSearchDet* neighborRod = sLayer[binFinder.binIndex(idet)];
    if (!overlap( gCrossingPos, *neighborRod, window)) break;
    if (!adder.add( *neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}

bool PixelBarrelLayer::overlap( const GlobalPoint& gpos, const GeometricSearchDet& gsdet, float phiWin) const
{
  GlobalPoint crossPoint(gpos);

  // introduce offset (extrapolated point and true propagated point differ by 0.0003 - 0.00033, 
  // due to thickness of Rod of 1 cm) 
  const float phiOffset = 0.00034;  //...TOBE CHECKED LATER...
  phiWin += phiOffset;

  // detector phi range
  
  pair<float,float> phiRange(crossPoint.phi()-phiWin, crossPoint.phi()+phiWin);

  return rangesIntersect(phiRange, gsdet.surface().phiSpan(), PhiLess());

} 


BoundCylinder* PixelBarrelLayer::cylinder( const vector<const GeometricSearchDet*>& rods) const 
{
  vector<const GeomDet*> tmp;
  for (vector<const GeometricSearchDet*>::const_iterator it=rods.begin(); it!=rods.end(); it++) {    
    tmp.insert(tmp.end(),(*it)->basicComponents().begin(),(*it)->basicComponents().end());
  }
  return CylinderBuilderFromDet()( tmp.begin(), tmp.end());
}

