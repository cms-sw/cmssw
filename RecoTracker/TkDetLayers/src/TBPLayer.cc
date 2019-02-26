#include "TBPLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"
#include "BarrelUtil.h"


#include "TrackingTools/DetLayers/interface/DetLayerException.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/HelixBarrelCylinderCrossing.h"
#include "TrackingTools/DetLayers/interface/CylinderBuilderFromDet.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"


using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

 
void TBPLayer::construct() {
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

  if (!theInnerComps.empty())
    theInnerBinFinder = BinFinderType(theInnerComps.front()->position().phi(),
				      theInnerComps.size());

  if (!theOuterComps.empty())
    theOuterBinFinder = BinFinderType(theOuterComps.front()->position().phi(),
				      theOuterComps.size());
  
  BarrelDetLayer::initialize();

   //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "==== DEBUG TBPLayer =====" ; 
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
     LogDebug("TkDetLayers") << "inner Rod pos z,perp,eta,phi: " 
			     << (**i).position().z() << " , " 
			     << (**i).position().perp() << " , " 
			     << (**i).position().eta() << " , " 
			     << (**i).position().phi() ;
  }
  
  for (vector<const GeometricSearchDet*>::const_iterator i=theOuterComps.begin();
       i != theOuterComps.end(); i++){
    LogDebug("TkDetLayers") << "outer Rod pos z,perp,eta,phi: " 
			    << (**i).position().z() << " , " 
			    << (**i).position().perp() << " , " 
			    << (**i).position().eta() << " , " 
			    << (**i).position().phi() ;
  }
  LogDebug("TkDetLayers") << "==== end DEBUG TBPLayer =====" ;   
}


TBPLayer::~TBPLayer(){} 

BoundCylinder* TBPLayer::cylinder( const vector<const GeometricSearchDet*>& rods) const 
{
  vector<const GeomDet*> tmp;
  for (vector<const GeometricSearchDet*>::const_iterator it=rods.begin(); it!=rods.end(); it++) {    
    tmp.insert(tmp.end(),(*it)->basicComponents().begin(),(*it)->basicComponents().end());
  }
  return CylinderBuilderFromDet()( tmp.begin(), tmp.end());
}


// private methods for the implementation of groupedCompatibleDets()

std::tuple<bool,int,int>  TBPLayer::computeIndexes(GlobalPoint gInnerPoint, GlobalPoint gOuterPoint) const {
  int innerIndex = theInnerBinFinder.binIndex(gInnerPoint.barePhi());
  float innerDist = theInnerBinFinder.binPosition(innerIndex) - gInnerPoint.barePhi();

  int outerIndex = theOuterBinFinder.binIndex(gOuterPoint.barePhi());
  float outerDist = theOuterBinFinder.binPosition(outerIndex) - gOuterPoint.barePhi() ;

  
  innerDist *= Geom::phiLess( theInnerBinFinder.binPosition(innerIndex),gInnerPoint.barePhi()) ? -1.f : 1.f; 
  outerDist *= Geom::phiLess( theOuterBinFinder.binPosition(outerIndex),gOuterPoint.barePhi()) ? -1.f : 1.f; 
  if (innerDist < 0.f) { innerDist += Geom::ftwoPi();}
  if (outerDist < 0.f) { outerDist += Geom::ftwoPi();}
 
  return std::make_tuple(innerDist < outerDist,innerIndex, outerIndex);
}



float TBPLayer::computeWindowSize( const GeomDet* det, 
				   const TrajectoryStateOnSurface& tsos, 
				   const MeasurementEstimator& est) const
{

  return barrelUtil::computeWindowSize(det,tsos,est);
}


void TBPLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				const Propagator& prop,
				const MeasurementEstimator& est,
				const SubLayerCrossing& crossing,
				float window, 
				vector<DetGroup>& result,
				bool checkClosest) const {
  using barrelUtil::overlap;
  
  const GlobalPoint& gCrossingPos = crossing.position();
  auto gphi = gCrossingPos.barePhi();  
  
  const vector<const GeometricSearchDet*>& sLayer( subLayer( crossing.subLayerIndex()));
  
  int closestIndex = crossing.closestDetIndex();
  int negStartIndex = closestIndex-1;
  int posStartIndex = closestIndex+1;

  if (checkClosest) { // must decide if the closest is on the neg or pos side
    if ( Geom::phiLess( gphi, sLayer[closestIndex]->surface().phi())) {
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


