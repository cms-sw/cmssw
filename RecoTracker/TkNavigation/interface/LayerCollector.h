#ifndef TkNavigation_LayerCollector_H_
#define TkNavigation_LayerCollector_H_
/**
 *   \class LayerCollector
 *   Class collecting all layers of the tracker.  
 *    $Date: 2006/06/01 15:04:40 $
 *    $Revision: 1.3 $
 *   
 *   
 */

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "RecoTracker/TkNavigation/interface/StartingLayerFinder.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"




class LayerCollector {

private:

  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;
  typedef std::pair<float, float> Range;

public:

  LayerCollector(const Propagator* aPropagator,
		 const StartingLayerFinder* aFinder,
		 float dr, 
		 float dz) : 
    thePropagator(aPropagator),
    theStartingLayerFinder(aFinder),
    theDeltaR(dr),
    theDeltaZ(dz) { }

  ~LayerCollector() {}

  std::vector<const DetLayer*> allLayers(const FTS& aFts) const;
  std::vector<const BarrelDetLayer*> barrelLayers(const FTS& aFts) const;
  std::vector<const ForwardDetLayer*> forwardLayers(const FTS& aFts) const;

  const Propagator* propagator() const {return thePropagator;}
  const StartingLayerFinder* finder() const {return theStartingLayerFinder;}
  float deltaR() const {return theDeltaR;}
  float deltaZ() const {return theDeltaZ;}
  
private:

  const Propagator* thePropagator;
  const StartingLayerFinder* theStartingLayerFinder;
  float theDeltaR;
  float theDeltaZ;



  inline bool rangesIntersect( const Range& a, const Range& b) const {
    if ( a.first > b.second || b.first > a.second) return false;
    else return true;
  }


};

#endif //TR_LayerCollector_H_
