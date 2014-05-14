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
#include <climits>

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
  edm::LogInfo("TkDetLayers") << " Rmin, Rmax, R_average = " << theRmin << ", " << theRmax << ", "
                              << split_inner_outer_radius << std::endl
                              << " num inner, outer disks = "
                              << _num_innerpanels << ", " << _num_outerpanels
                              << std::endl;

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
                          << "Based on phi separation for inner: " << theComps.front()->surface().position().phi()
                          << " and on phi separation for outer: "  << theComps[_num_innerpanels]->surface().position().phi()
			  << "PixelForwardLayer.surfcace.phi(): "  << std::endl
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
    edm::LogInfo("TkDetLayers") << "inner computeCrossings returns invalid in PixelForwardLayer::groupedCompatibleDets:";
    return;
  }
  if (!crossings_outer.isValid){
    edm::LogInfo("TkDetLayers") << "outer computeCrossings returns invalid in PixelForwardLayer::groupedCompatibleDets:";
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
    return SubTurbineCrossings();
  }

  HelixPlaneCrossing::PositionType  turbinePoint( turbineCrossing.position(thePath.second));
  HelixPlaneCrossing::DirectionType turbineDir( turbineCrossing.direction(thePath.second));
  int closestIndex = 0;
  int nextIndex = 0;
  // The next if is needed in order to properly treat the PhaseII
  // geometry in the forward region using a blade-like geometry on a
  // real ring-based one. As ugly as it could be. 20 [cm] is the
  // separation in R between true forward pixel blades (inner
  // w.r.t. 20) and the outer tracker rings (outer w.r.t. 20).
  if (turbinePoint.perp() < 20) {
    if(innerDisk)
      closestIndex = theBinFinder_inner.binIndex(turbinePoint.phi());
    else
      closestIndex = theBinFinder_outer.binIndex(turbinePoint.phi());

    HelixArbitraryPlaneCrossing2Order theBladeCrossing(turbinePoint, turbineDir, rho);

    float closestDist = 0;
    if(innerDisk) {
      const BoundPlane& closestPlane( static_cast<const BoundPlane&>(
          theComps[closestIndex]->surface()));

      pair<bool,double> theClosestBladePath = theBladeCrossing.pathLength( closestPlane );
      LocalPoint closestPos = closestPlane.toLocal(GlobalPoint(theBladeCrossing.position(theClosestBladePath.second)) );
      closestDist = closestPos.x();
      // use fact that local X perp to global Y
      nextIndex = PhiLess()( closestPlane.position().phi(), turbinePoint.phi()) ?
          closestIndex+1 : closestIndex-1;
    } else {
      const BoundPlane& closestPlane( static_cast<const BoundPlane&>(
          theComps[closestIndex + _num_innerpanels]->surface()));

      pair<bool,double> theClosestBladePath = theBladeCrossing.pathLength( closestPlane );
      LocalPoint closestPos = closestPlane.toLocal(GlobalPoint(theBladeCrossing.position(theClosestBladePath.second)) );
      closestDist = closestPos.x();
      // use fact that local X perp to global Y
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
  } else {
    // Special treatment is needed in the PhaseII geometry to properly
    // handle the forward region of the outer tracker pretending to
    // use a blade-like geometry on a ring-based one. The main
    // difference here is that the check it not done on phi first, and
    // then on R, but it is done directly on R, phi being derived
    // later on in cascade. In order to do that, the PixelBlade has
    // been extended to include 2 pair<float, float>, each
    // representing the range of its front and back component. To be
    // noted also that the front and back component of a "pixel blade"
    // in the case of the outer Tracker has no concept of front and
    // back, but are simply rings at different radii, hence the need
    // to have 2 pairs into PixelBlade. In particular rings are joint
    // into a single blade following the counter-intuitive logic 0-7,
    // 1-8, 2-9, etc..., i.e. using the innermost ring together w/ the
    // middle one, and so on (see PixelForwardLayerBuilder for further
    // insight). To get things even more complex, we have to manually
    // take care of assignign the indices 'closestIndex' and
    // 'nextIndex' in such a way that the logic behind them did not
    // change for the rest of the code. The search is now done
    // everywhere everytime, guessing a-posteriori if the index
    // belongs to the inner or outer part, and hence adjusting the
    // indices accordingly (offsetting them of _num_innerpanels).
    closestIndex = 0;
    nextIndex = 0;
    float target_radius = turbinePoint.perp();
    bool found = false;
    bool foundNext = false;
    // The check is done every time on both sides of the blade, front
    // and back, assigning a posteriori the correct index to the
    // respective category, either innerDisk or outerDisk. Both checks
    // have to be performed at the same time since not all indices are
    // accessible from the different regions (innerdisk has access to
    // 0-_num_innerpanels, while outer to the remainings).
    for (auto blade : theComps) {
      ++closestIndex;
      // Another horrible hack: once we land on the last ring/blade,
      // we need to look only in one region(outer) and not the
      // other(inner) to avoid double counting of hits.
      if (closestIndex < (int)theComps.size()) {
        if (dynamic_cast<const PixelBlade*>(blade)->inRange(target_radius, innerDisk) ||
            dynamic_cast<const PixelBlade*>(blade)->inRange(target_radius, !innerDisk)) {
          found = true;
          break;
        }
      } else {
        assert(closestIndex == (int)theComps.size());
        if (!innerDisk) {
          if (dynamic_cast<const PixelBlade*>(blade)->inRange(target_radius, innerDisk)) {
            found = true;
            break;
          }
        }
      }
    }
    int counter = 0;
    nextIndex = closestIndex;
    // Do not comment this second loop since potentially it could find
    // overlapping regions on different rings. In principle, to be
    // verified in practice.
    if (found) {
      for (auto blade : theComps) {
        // do not test what has been already checked.
        if (counter < closestIndex) {
          ++counter;
          continue ;
        }
        ++nextIndex;
        if (nextIndex < (int)theComps.size()) {
          if (dynamic_cast<const PixelBlade*>(blade)->inRange(target_radius, innerDisk) ||
              dynamic_cast<const PixelBlade*>(blade)->inRange(target_radius, !innerDisk)) {
            foundNext = true;
            break;
          }
        } else {
          assert(nextIndex == (int)theComps.size());
          if (!innerDisk) {
            if (dynamic_cast<const PixelBlade*>(blade)->inRange(target_radius, innerDisk)) {
              foundNext = true;
              break;
            }
          }
        }
      }
    }

    --closestIndex;
    --nextIndex;
    if (!found && !foundNext) {
      closestIndex = nextIndex = 0;
    }
    if (found && !foundNext) {
      if (closestIndex >= (int)_num_innerpanels) {
        closestIndex = innerDisk ? 0 : closestIndex - _num_innerpanels;
      }
      // Carefully avoid having closestIndex == nextIndex, since this
      // will trigger a likely double counting of hits.
      nextIndex = closestIndex + 1;
    }
    if (found && foundNext) {
      if (closestIndex >= (int)_num_innerpanels) {
        closestIndex = innerDisk ? 0 : closestIndex - _num_innerpanels;
      }
      if (nextIndex >= (int)_num_innerpanels) {
        nextIndex = innerDisk ? 0 : nextIndex - _num_innerpanels;
      }
    }
    return SubTurbineCrossings( closestIndex, nextIndex, 1.);
  }
}

float
PixelForwardLayer::computeWindowSize( const GeomDet* det,
				      const TrajectoryStateOnSurface& tsos,
				      const MeasurementEstimator& est) const
{
  return est.maximalLocalDisplacement(tsos, det->surface()).x();
}


