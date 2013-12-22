#include "TOBLayer.h"

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

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

TOBLayer::TOBLayer(vector<const TOBRod*>& innerRods,
		   vector<const TOBRod*>& outerRods) : 
  TBLayer(innerRods,outerRods, GeomDetEnumerators::TOB)

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

  if (theInnerComps.size())
    theInnerBinFinder = BinFinderType(theInnerComps.front()->position().phi(),
				      theInnerComps.size());

  if (theOuterComps.size())
    theOuterBinFinder = BinFinderType(theOuterComps.front()->position().phi(),
				      theOuterComps.size());
  
  BarrelDetLayer::initialize();

   //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "==== DEBUG TOBLayer =====" ; 
  LogDebug("TkDetLayers") << "innerCyl radius, thickness, lenght: " 
			  << theInnerCylinder->radius() << " , "
			  << theInnerCylinder->bounds().thickness() << " , "
			  << theInnerCylinder->bounds().length() ;
 
  LogDebug("TkDetLayers") << "outerCyl radius, thickness, lenght: " 
			  << theOuterCylinder->radius() << " , "
			  << theOuterCylinder->bounds().thickness() << " , "
			  << theOuterCylinder->bounds().length() ;

  LogDebug("TkDetLayers") << "Cyl radius, thickness, lenght: " 
			  << specificSurface().radius() << " , "
			  << specificSurface().bounds().thickness() << " , "
			  << specificSurface().bounds().length() ;

  for (vector<const GeometricSearchDet*>::const_iterator i=theInnerComps.begin();
       i != theInnerComps.end(); i++){
     LogDebug("TkDetLayers") << "inner TOBRod pos z,perp,eta,phi: " 
			     << (**i).position().z() << " , " 
			     << (**i).position().perp() << " , " 
			     << (**i).position().eta() << " , " 
			     << (**i).position().phi() ;
  }
  
  for (vector<const GeometricSearchDet*>::const_iterator i=theOuterComps.begin();
       i != theOuterComps.end(); i++){
    LogDebug("TkDetLayers") << "outer TOBRod pos z,perp,eta,phi: " 
			    << (**i).position().z() << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta() << " , " 
			    << (**i).position().phi() ;
  }
  LogDebug("TkDetLayers") << "==== end DEBUG TOBLayer =====" ;   
}


TOBLayer::~TOBLayer(){} 

BoundCylinder* TOBLayer::cylinder( const vector<const GeometricSearchDet*>& rods) const 
{
  vector<const GeomDet*> tmp;
  for (vector<const GeometricSearchDet*>::const_iterator it=rods.begin(); it!=rods.end(); it++) {    
    tmp.insert(tmp.end(),(*it)->basicComponents().begin(),(*it)->basicComponents().end());
  }
  return CylinderBuilderFromDet()( tmp.begin(), tmp.end());
}


// private methods for the implementation of groupedCompatibleDets()

std::tuple<bool,int,int>  TOBLayer::computeIndexes(GlobalPoint gInnerPoint, GlobalPoint gOuterPoint) const {
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.barePhi());
  float innerDist = theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.barePhi();

  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.barePhi());
  float outerDist = theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.barePhi() ;

  
  innerDist *= PhiLess()( theInnerBinFinder.binPosition(innerIndex),gInnerPoint.barePhi()) ? -1.f : 1.f; 
  outerDist *= PhiLess()( theOuterBinFinder.binPosition(outerIndex),gOuterPoint.barePhi()) ? -1.f : 1.f; 
  if (innerDist < 0.) { innerDist += Geom::ftwoPi();}
  if (outerDist < 0.) { outerDist += Geom::ftwoPi();}
 
  return std::make_tuple(innerDist < outerDist,innerIndex, outerIndex);
}



float TOBLayer::computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est) const
{
  double xmax = 
    est.maximalLocalDisplacement(tsos, det->surface()).x();
  return calculatePhiWindow( xmax, *det, tsos);
}


double TOBLayer::calculatePhiWindow( double Xmax, const GeomDet& det,
				     const TrajectoryStateOnSurface& state) const
{

  LocalPoint startPoint = state.localPosition();
  LocalVector shift( Xmax , 0. , 0.);
  LocalPoint shift1 = startPoint + shift;
  LocalPoint shift2 = startPoint + (-shift); 
  //LocalPoint shift2( startPoint); //original code;
  //shift2 -= shift;

  auto phi1 = det.surface().toGlobal(shift1).barePhi();
  auto phi2 = det.surface().toGlobal(shift2).barePhi();
  auto phiStart = state.globalPosition().barePhi();
  auto phiWin = std::min(std::abs(phiStart-phi1),std::abs(phiStart-phi2));

  return phiWin;
}


namespace {

  inline
  bool overlap(float phi, const GeometricSearchDet& gsdet, float phiWin) {
    
    
    // introduce offset (extrapolated point and true propagated point differ by 0.0003 - 0.00033, 
    // due to thickness of Rod of 1 cm) 
    constexpr float phiOffset = 0.00034;  //...TOBE CHECKED LATER...
    phiWin += phiOffset;
    
    // detector phi range
    std::pair<float,float> phiRange(phi-phiWin, phi+phiWin);
    
    //   // debug
    //   edm::LogInfo(TkDetLayers) ;
    //   edm::LogInfo(TkDetLayers) << " overlapInPhi: position, det phi range " 
    //        << "("<< rod.position().perp() << ", " << rod.position().phi() << ")  "
    //        << rodRange.phiRange().first << " " << rodRange.phiRange().second ;
    //   edm::LogInfo(TkDetLayers) << " overlapInPhi: cross point phi, window " << crossPoint.phi() << " " << phiWin ;
    //   edm::LogInfo(TkDetLayers) << " overlapInPhi: search window: " << crossPoint.phi()-phiWin << "  " << crossPoint.phi()+phiWin ;
    
    return rangesIntersect(phiRange, gsdet.surface().phiSpan(), PhiLess());
  } 



}


void TOBLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est,
				const SubLayerCrossing& crossing,
				float window, 
				vector<DetGroup>& result,
				bool checkClosest) const
{
  GlobalPoint gCrossingPos = crossing.position();
  auto gphi = gCrossingPos.barePhi();  

  const vector<const GeometricSearchDet*>& sLayer( subLayer( crossing.subLayerIndex()));
 
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if ( PhiLess()( gphi, sLayer[closestIndex]->surface().phi())) {
      posStartIndex = closestIndex;
    }
    else {
      negStartIndex = closestIndex;
    }
  }

  const BinFinderType& binFinder = (crossing.subLayerIndex()==0 ? theInnerBinFinder : theOuterBinFinder);

  typedef CompatibleDetToGroupAdder Adder;
  int quarter = sLayer.size()/4;
  for (int idet=negStartIndex; idet >= negStartIndex - quarter; idet--) {
    const GeometricSearchDet & neighborRod = *sLayer[binFinder.binIndex(idet)];
    if (!overlap( gphi, neighborRod, window)) break;
    if (!Adder::add( neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
  for (int idet=posStartIndex; idet < posStartIndex + quarter; idet++) {
    const GeometricSearchDet & neighborRod = *sLayer[binFinder.binIndex(idet)];
    if (!overlap( gphi, neighborRod, window)) break;
    if (!Adder::add( neighborRod, tsos, prop, est, result)) break;
    // maybe also add shallow crossing angle test here???
  }
}


