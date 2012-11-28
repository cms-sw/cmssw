#include "RecoTracker/TkDetLayers/src/PixelForwardLayerPhase1.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/BoundingBox.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

#include "TrackingTools/DetLayers/interface/simple_stat.h"
#include "TrackingTools/DetLayers/interface/PhiLess.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing2Order.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"


#include "LayerCrossingSide.h"
#include "DetGroupMerger.h"
#include "CompatibleDetToGroupAdder.h"

#include <algorithm>

using namespace std;

typedef GeometricSearchDet::DetWithState DetWithState;

PixelForwardLayerPhase1::PixelForwardLayerPhase1(vector<const PixelBlade*>& blades):
  PixelForwardLayer(blades)
{
  // Assumes blades are ordered inner first then outer; and within each by phi
  // where we go 0 -> pi, and then -pi -> 0
  // we also need the number of inner blades for the offset in theComps vector
  //
  // this->specificSurface() not yet available so need to calculate average R
  // we need some way to flag if FPIX is made of an inner and outer disk
  // or probably need to change the way this is done, e.g. a smarter binFinder
  float theRmin = (*(theComps.begin()))->surface().position().perp();
  float theRmax = theRmin;
  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin(); 
      it!=theComps.end(); it++){
      theRmin = std::min( theRmin, (*it)->surface().position().perp());
      theRmax = std::max( theRmax, (*it)->surface().position().perp());
  }
  float split_inner_outer_radius = 0.5 * (theRmin + theRmax);
  _num_innerpanels = 0;
  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin();
      it!=theComps.end();it++){  
    if((**it).surface().position().perp() <= split_inner_outer_radius) ++_num_innerpanels;
  }
  _num_outerpanels = theComps.size() - _num_innerpanels;
  //std::cout << " Rmin, Rmax, R_average = " << theRmin << ", " << theRmax << ", "
  //          << split_inner_outer_radius << std::endl;
  //std::cout << " num inner, outer disks = " << _num_innerpanels << ", " << _num_outerpanels << std::endl;

  theBinFinder_inner = BinFinderType( theComps.front()->surface().position().phi(),
                                      _num_innerpanels);
  theBinFinder_outer = BinFinderType( theComps[_num_innerpanels]->surface().position().phi(),
                                      _num_outerpanels);
    
}

PixelForwardLayerPhase1::~PixelForwardLayerPhase1(){
} 

void
PixelForwardLayerPhase1::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop,
					   const MeasurementEstimator& est,
					   std::vector<DetGroup> & result) const {
  vector<DetGroup> closestResult_inner;
  vector<DetGroup> closestResult_outer;
  vector<DetGroup> nextResult_inner;
  vector<DetGroup> nextResult_outer;
  vector<DetGroup> result_inner;
  vector<DetGroup> result_outer;
  int frontindex_inner = 0;
  int frontindex_outer = 0;
  SubTurbineCrossings  crossings_inner; 
  SubTurbineCrossings  crossings_outer; 

  crossings_inner = computeCrossings( tsos, prop.propagationDirection(), true);
  crossings_outer = computeCrossings( tsos, prop.propagationDirection(), false);
  if (!crossings_inner.isValid){
    //edm::LogInfo("TkDetLayers") << "inner computeCrossings returns invalid in PixelForwardLayerPhase1::groupedCompatibleDets:";
    return;
  }
  if (!crossings_outer.isValid){
    //edm::LogInfo("TkDetLayers") << "outer computeCrossings returns invalid in PixelForwardLayerPhase1::groupedCompatibleDets:";
    return;
  }

  typedef CompatibleDetToGroupAdder Adder;
  Adder::add( *theComps[theBinFinder_inner.binIndex(crossings_inner.closestIndex)], 
	     tsos, prop, est, closestResult_inner);

  if(closestResult_inner.empty()){
    Adder::add( *theComps[theBinFinder_inner.binIndex(crossings_inner.nextIndex)], 
	       tsos, prop, est, result_inner);
    frontindex_inner = crossings_inner.nextIndex;
  } else {
    if (Adder::add( *theComps[theBinFinder_inner.binIndex(crossings_inner.nextIndex)], 
  		  tsos, prop, est, nextResult_inner)) {
      int crossingSide = LayerCrossingSide().endcapSide( tsos, prop);
      int theHelicity = computeHelicity(theComps[theBinFinder_inner.binIndex(crossings_inner.closestIndex)],
  					theComps[theBinFinder_inner.binIndex(crossings_inner.nextIndex)] );
      DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult_inner), std::move(nextResult_inner), result_inner, 
  					    theHelicity, crossingSide);
      if (theHelicity == crossingSide) frontindex_inner = crossings_inner.closestIndex;
      else                             frontindex_inner = crossings_inner.nextIndex;
    } else {
      result_inner.swap(closestResult_inner);
      frontindex_inner = crossings_inner.closestIndex;
    }
  }
  if(!closestResult_inner.empty()){
    DetGroupElement closestGel( closestResult_inner.front().front());
    float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);
    searchNeighbors( tsos, prop, est, crossings_inner, window, result_inner, true);
  }

  //DetGroupElement closestGel( closestResult.front().front());
  //float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);
  //float detWidth = closestGel.det()->surface().bounds().width();
  //if (crossings.nextDistance < detWidth + window) {

  Adder::add( *theComps[(theBinFinder_outer.binIndex(crossings_outer.closestIndex)) + _num_innerpanels], 
	     tsos, prop, est, closestResult_outer);

  if(closestResult_outer.empty()){
    Adder::add( *theComps[theBinFinder_outer.binIndex(crossings_outer.nextIndex) + _num_innerpanels], 
	       tsos, prop, est, result_outer);
    frontindex_outer = crossings_outer.nextIndex;
  } else {
    if (Adder::add( *theComps[theBinFinder_outer.binIndex(crossings_outer.nextIndex) + _num_innerpanels], 
  		  tsos, prop, est, nextResult_outer)) {
      int crossingSide = LayerCrossingSide().endcapSide( tsos, prop);
      int theHelicity = computeHelicity(theComps[theBinFinder_outer.binIndex(crossings_outer.closestIndex) + _num_innerpanels],
  					theComps[theBinFinder_outer.binIndex(crossings_outer.nextIndex) + _num_innerpanels] );
      DetGroupMerger::orderAndMergeTwoLevels( std::move(closestResult_outer), std::move(nextResult_outer), result_outer, 
  					    theHelicity, crossingSide);
      if (theHelicity == crossingSide) frontindex_outer = crossings_outer.closestIndex;
      else                             frontindex_outer = crossings_outer.nextIndex;
    } else {
      result_outer.swap(closestResult_outer);
      frontindex_outer = crossings_outer.closestIndex;
    }
  }
  if(!closestResult_outer.empty()){
    DetGroupElement closestGel( closestResult_outer.front().front());
    float window = computeWindowSize( closestGel.det(), closestGel.trajectoryState(), est);
    searchNeighbors( tsos, prop, est, crossings_inner, window, result_outer, false);
  }

  if(result_inner.empty() && result_outer.empty() ) return;
  if(result_inner.empty()) result.swap(result_outer);
  else if(result_outer.empty()) result.swap(result_inner);
  else {
    int crossingSide = LayerCrossingSide().endcapSide( tsos, prop);
    int theHelicity = computeHelicity(theComps[theBinFinder_inner.binIndex(frontindex_inner)],
  					theComps[theBinFinder_outer.binIndex(frontindex_outer) + _num_innerpanels] );
    DetGroupMerger::orderAndMergeTwoLevels( std::move(result_inner), std::move(result_outer), result, 
  					    theHelicity, crossingSide);
  }
}



void 
PixelForwardLayerPhase1::searchNeighbors( const TrajectoryStateOnSurface& tsos,
				    const Propagator& prop,
				    const MeasurementEstimator& est,
				    const SubTurbineCrossings& crossings,
				    float window, 
				    vector<DetGroup>& result,
                                    bool innerDisk) const
{
  typedef CompatibleDetToGroupAdder Adder;
  int crossingSide = LayerCrossingSide().endcapSide( tsos, prop);
  typedef DetGroupMerger Merger;

  int negStart = min( crossings.closestIndex, crossings.nextIndex) - 1;
  int posStart = max( crossings.closestIndex, crossings.nextIndex) + 1;

  int quarter = theComps.size()/4;
  if(innerDisk) quarter = _num_innerpanels/4;
  else quarter = _num_outerpanels/4;
 
  vector<DetGroup> tmp;
  vector<DetGroup> newResult;
  for (int idet=negStart; idet >= negStart - quarter+1; idet--) {
    tmp.clear();
    newResult.clear();
    if(innerDisk) {
      const GeometricSearchDet* neighbor = theComps[theBinFinder_inner.binIndex(idet)];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp)) break;
      int theHelicity = computeHelicity(theComps[theBinFinder_inner.binIndex(idet)],
  				      theComps[theBinFinder_inner.binIndex(idet+1)] );
      Merger::orderAndMergeTwoLevels( std::move(tmp), std::move(result), newResult, theHelicity, crossingSide);
    } else {
      const GeometricSearchDet* neighbor = theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp)) break;
      int theHelicity = computeHelicity(theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels],
  				      theComps[(theBinFinder_outer.binIndex(idet+1)) + _num_innerpanels] );
      Merger::orderAndMergeTwoLevels( std::move(tmp), std::move(result), newResult, theHelicity, crossingSide);
    }
    result.swap(newResult);
  }
  for (int idet=posStart; idet < posStart + quarter-1; idet++) {
    tmp.clear();
    newResult.clear();
    if(innerDisk) {
      const GeometricSearchDet* neighbor = theComps[theBinFinder_inner.binIndex(idet)];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp)) break;
      int theHelicity = computeHelicity(theComps[theBinFinder_inner.binIndex(idet-1)],
  				      theComps[theBinFinder_inner.binIndex(idet)] );
      Merger::orderAndMergeTwoLevels( std::move(result), std::move(tmp), newResult, theHelicity, crossingSide);
    } else {
      const GeometricSearchDet* neighbor = theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp)) break;
      int theHelicity = computeHelicity(theComps[(theBinFinder_outer.binIndex(idet-1)) + _num_innerpanels],
  				      theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels] );
      Merger::orderAndMergeTwoLevels( std::move(result), std::move(tmp), newResult, theHelicity, crossingSide);
    }
    result.swap(newResult);
  }
}


PixelForwardLayerPhase1::SubTurbineCrossings 
PixelForwardLayerPhase1::computeCrossings( const TrajectoryStateOnSurface& startingState,
				     PropagationDirection propDir, bool innerDisk) const
{  
  typedef MeasurementEstimator::Local2DVector Local2DVector;

  HelixPlaneCrossing::PositionType startPos( startingState.globalPosition());
  HelixPlaneCrossing::DirectionType startDir( startingState.globalMomentum());
  float rho( startingState.transverseCurvature());

  HelixArbitraryPlaneCrossing turbineCrossing( startPos, startDir, rho,
					       propDir);

  pair<bool,double> thePath = turbineCrossing.pathLength( specificSurface() );
  
  if (!thePath.first) {
    //edm::LogInfo("TkDetLayers") << "ERROR in PixelForwardLayerPhase1: disk not crossed by track" ;
    return SubTurbineCrossings();
  }

  HelixPlaneCrossing::PositionType  turbinePoint( turbineCrossing.position(thePath.second));
  HelixPlaneCrossing::DirectionType turbineDir( turbineCrossing.direction(thePath.second));
  int closestIndex = 0;
  if(innerDisk)
    closestIndex = theBinFinder_inner.binIndex(turbinePoint.phi());
  else
    closestIndex = theBinFinder_outer.binIndex(turbinePoint.phi());

  HelixArbitraryPlaneCrossing2Order theBladeCrossing(turbinePoint, turbineDir, rho);

  float closestDist = 0;
  int nextIndex = 0;
  if(innerDisk) {
    const BoundPlane& closestPlane( static_cast<const BoundPlane&>( 
      theComps[closestIndex]->surface()));

    pair<bool,double> theClosestBladePath = theBladeCrossing.pathLength( closestPlane );
    LocalPoint closestPos = closestPlane.toLocal(GlobalPoint(theBladeCrossing.position(theClosestBladePath.second)) );
    
    closestDist = closestPos.x(); // use fact that local X perp to global Y

    nextIndex = PhiLess()( closestPlane.position().phi(), turbinePoint.phi()) ? 
      closestIndex+1 : closestIndex-1;
  } else {
    const BoundPlane& closestPlane( static_cast<const BoundPlane&>( 
      theComps[closestIndex + _num_innerpanels]->surface()));

    pair<bool,double> theClosestBladePath = theBladeCrossing.pathLength( closestPlane );
    LocalPoint closestPos = closestPlane.toLocal(GlobalPoint(theBladeCrossing.position(theClosestBladePath.second)) );
    
    closestDist = closestPos.x(); // use fact that local X perp to global Y

    nextIndex = PhiLess()( closestPlane.position().phi(), turbinePoint.phi()) ? 
      closestIndex+1 : closestIndex-1;
  }

  float nextDist = 0;
  if(innerDisk) {
    const BoundPlane& nextPlane( static_cast<const BoundPlane&>( 
      theComps[ theBinFinder_inner.binIndex(nextIndex)]->surface()));
    pair<bool,double> theNextBladePath    = theBladeCrossing.pathLength( nextPlane );
    LocalPoint nextPos = nextPlane.toLocal(GlobalPoint(theBladeCrossing.position(theNextBladePath.second)) );
    nextDist = nextPos.x();
  } else {
    const BoundPlane& nextPlane( static_cast<const BoundPlane&>( 
      theComps[ theBinFinder_outer.binIndex(nextIndex) + _num_innerpanels]->surface()));
    pair<bool,double> theNextBladePath    = theBladeCrossing.pathLength( nextPlane );
    LocalPoint nextPos = nextPlane.toLocal(GlobalPoint(theBladeCrossing.position(theNextBladePath.second)) );
    nextDist = nextPos.x();
  }

  if (fabs(closestDist) < fabs(nextDist)) {
    return SubTurbineCrossings( closestIndex, nextIndex, nextDist);
  }
  else {
    return SubTurbineCrossings( nextIndex, closestIndex, closestDist);
  }
}



