#include "PixelForwardLayer.h"

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

PixelForwardLayer::PixelForwardLayer(vector<const PixelBlade*>& blades):
  theComps(blades.begin(),blades.end())
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

  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin();
      it!=theComps.end();it++){  
    theBasicComps.insert(theBasicComps.end(),	
			 (**it).basicComponents().begin(),
			 (**it).basicComponents().end());
  }

  //They should be already phi-ordered. TO BE CHECKED!!
  //sort( theBlades.begin(), theBlades.end(), PhiLess());
  setSurface( computeSurface() );
  
  theBinFinder_inner = BinFinderType( theComps.front()->surface().position().phi(),
                                      _num_innerpanels);
  theBinFinder_outer = BinFinderType( theComps[_num_innerpanels]->surface().position().phi(),
                                      _num_outerpanels);

  //--------- DEBUG INFO --------------
  LogDebug("TkDetLayers") << "DEBUG INFO for PixelForwardLayer" << "\n"
                          << "Num of inner and outer panels = " << _num_innerpanels << ", " << _num_outerpanels << "\n"
			  << "PixelForwardLayer.surfcace.phi(): " 
			  << this->surface().position().phi() << "\n"
			  << "PixelForwardLayer.surfcace.z(): " 
			  << this->surface().position().z() << "\n"
			  << "PixelForwardLayer.surfcace.innerR(): " 
			  << this->specificSurface().innerRadius() << "\n"
			  << "PixelForwardLayer.surfcace.outerR(): " 
			  << this->specificSurface().outerRadius() ;

  for(vector<const GeometricSearchDet*>::const_iterator it=theComps.begin(); 
      it!=theComps.end(); it++){
    LogDebug("TkDetLayers") << "blades phi,z,r: "
                            << (*it)->surface().position().phi() << " , "
                            << (*it)->surface().position().z() <<   " , "
                            << (*it)->surface().position().perp();
    //for(vector<const GeomDet*>::const_iterator iu=(**it).basicComponents().begin();
    //    iu!=(**it).basicComponents().end();iu++){  
    //  std::cout << "   basic component rawId = " << hex << (**iu).geographicalId().rawId() << dec <<std::endl;
    //}
  }
  //-----------------------------------

    
}

PixelForwardLayer::~PixelForwardLayer(){
  vector<const GeometricSearchDet*>::const_iterator i;
  for (i=theComps.begin(); i!=theComps.end(); i++) {
    delete *i;
  }
} 

namespace {

  bool groupSortByZ(DetGroupElement i,DetGroupElement j) { return (fabs(i.det()->position().z())<fabs(j.det()->position().z())); }

}

void
PixelForwardLayer::groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
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
    //edm::LogInfo("TkDetLayers") << "inner computeCrossings returns invalid in PixelForwardLayer::groupedCompatibleDets:";
    return;
  }
  if (!crossings_outer.isValid){
    //edm::LogInfo("TkDetLayers") << "outer computeCrossings returns invalid in PixelForwardLayer::groupedCompatibleDets:";
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
      vector<DetGroup> tmp99 = closestResult_inner;
      DetGroupMerger::orderAndMergeTwoLevels( std::move(tmp99), std::move(nextResult_inner), result_inner,
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
      vector<DetGroup> tmp99 = closestResult_outer;
      DetGroupMerger::orderAndMergeTwoLevels( std::move(tmp99), std::move(nextResult_outer), result_outer, 
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

  /*
  for (auto gr : result) {
    std::cout << "new group" << std::endl;
    for (auto dge : gr) {
      PixelBarrelNameUpgrade name(dge.det()->geographicalId());
      std::cout << "new det with geom det at r:"<<dge.det()->position().perp()<<" id:"<<dge.det()->geographicalId().rawId()<<" name:"<<name.name()<<" isHalf:"<<name.isHalfModule()<<" tsos at:" <<dge.trajectoryState().globalPosition()<< std::endl;
    }
  }
  */

  if (this->specificSurface().innerRadius()>17.0) {
    //do splitting of groups for outer 'pixel' layer of phase 2 tracker
    //fixme: to be changed when moving to a new DetId schema with 'matched' hits
    std::vector<DetGroup> splitResult;
    for (auto gr : result) {
      if (gr.size()==1) {
	splitResult.push_back(gr);
	continue;
      }
      //sort according to Z
      std::sort(gr.begin(),gr.end(),groupSortByZ);
      DetGroup firstGroup; //this group contains the innermost dets of 2S/PS modules
      DetGroup secondGroup;//this group contains the outermost dets of 2S/PS modules
      for (auto dge : gr) {
	if (firstGroup.size()==0) {
	  firstGroup.push_back(dge);
	  continue;
	}
	bool foundInFirstGroup = false;
	for (auto dge_f : firstGroup) {
	  if (abs(int(dge.det()->geographicalId().rawId())-int(dge_f.det()->geographicalId().rawId()))==4 &&
	      fabs(dge.det()->position().z()-dge_f.det()->position().z())>0.15 ) {
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



void 
PixelForwardLayer::searchNeighbors( const TrajectoryStateOnSurface& tsos,
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
 
  for (int idet=negStart; idet >= negStart - quarter+1; idet--) {
    vector<DetGroup> tmp1;
    vector<DetGroup> newResult;
    if(innerDisk) {
      const GeometricSearchDet* neighbor = theComps[theBinFinder_inner.binIndex(idet)];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp1)) break;
      int theHelicity = computeHelicity(theComps[theBinFinder_inner.binIndex(idet)],
  				      theComps[theBinFinder_inner.binIndex(idet+1)] );
      vector<DetGroup> tmp2; tmp2.swap(result);
      Merger::orderAndMergeTwoLevels( std::move(tmp1), std::move(tmp2), newResult, theHelicity, crossingSide);
    } else {
      const GeometricSearchDet* neighbor = theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp1)) break;
      int theHelicity = computeHelicity(theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels],
  				      theComps[(theBinFinder_outer.binIndex(idet+1)) + _num_innerpanels] );
      vector<DetGroup> tmp2; tmp2.swap(result);
      Merger::orderAndMergeTwoLevels( std::move(tmp1), std::move(tmp2), newResult, theHelicity, crossingSide);
    }
    result.swap(newResult);
  }
  for (int idet=posStart; idet < posStart + quarter-1; idet++) {
    vector<DetGroup> tmp1;
    vector<DetGroup> newResult;
    if(innerDisk) {
      const GeometricSearchDet* neighbor = theComps[theBinFinder_inner.binIndex(idet)];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp1)) break;
      int theHelicity = computeHelicity(theComps[theBinFinder_inner.binIndex(idet-1)],
  				      theComps[theBinFinder_inner.binIndex(idet)] );
      vector<DetGroup> tmp2; tmp2.swap(result);
      Merger::orderAndMergeTwoLevels(std::move(tmp2), std::move(tmp1), newResult, theHelicity, crossingSide);
    } else {
      const GeometricSearchDet* neighbor = theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels];
      // if (!overlap( gCrossingPos, *neighbor, window)) break; // mybe not needed?
      // maybe also add shallow crossing angle test here???
      if (!Adder::add( *neighbor, tsos, prop, est, tmp1)) break;
      int theHelicity = computeHelicity(theComps[(theBinFinder_outer.binIndex(idet-1)) + _num_innerpanels],
  				      theComps[(theBinFinder_outer.binIndex(idet)) + _num_innerpanels] );
      vector<DetGroup> tmp2; tmp2.swap(result);
      Merger::orderAndMergeTwoLevels(std::move(tmp2), std::move(tmp1), newResult, theHelicity, crossingSide);
    }
    result.swap(newResult);
  }
}

int 
PixelForwardLayer::computeHelicity(const GeometricSearchDet* firstBlade,const GeometricSearchDet* secondBlade) const
{  
  if( fabs(firstBlade->position().z()) < fabs(secondBlade->position().z()) ) return 0;
  return 1;
}

PixelForwardLayer::SubTurbineCrossings 
PixelForwardLayer::computeCrossings( const TrajectoryStateOnSurface& startingState,
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
    //edm::LogInfo("TkDetLayers") << "ERROR in PixelForwardLayer: disk not crossed by track" ;
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

float 
PixelForwardLayer::computeWindowSize( const GeomDet* det, 
				      const TrajectoryStateOnSurface& tsos, 
				      const MeasurementEstimator& est) const
{
  return est.maximalLocalDisplacement(tsos, det->surface()).x();
}


