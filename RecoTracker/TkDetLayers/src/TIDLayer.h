#ifndef TkDetLayers_TIDLayer_h
#define TkDetLayers_TIDLayer_h


#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "TIDRing.h"
#include<array>


/** A concrete implementation for TID layer 
 *  built out of TIDRings
 */

#pragma GCC visibility push(hidden)
class TIDLayer GCC11_FINAL : public RingedForwardLayer, public GeometricSearchDetWithGroups {
 public:
  TIDLayer(std::vector<const TIDRing*>& rings);
  ~TIDLayer();
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const;

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const;

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::TID;}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  BoundDisk* computeDisk( const std::vector<const TIDRing*>& rings) const;

  std::array<int,3> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
						   const Propagator& prop ) const;

  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  int findClosest( const GlobalPoint[3] ) const;
  
  int findNextIndex( const GlobalPoint[3] , int ) const;
  
  bool overlapInR( const TrajectoryStateOnSurface& tsos, int i, double ymax) const;
  
  
  float computeWindowSize( const GeomDet* det, 
  			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const;
  
  void fillRingPars(int i);

 private:
  std::vector<GeomDet const*> theBasicComps;
  const TIDRing* theComps[3];
  struct RingPar { float theRingR, thetaRingMin, thetaRingMax;};
  RingPar ringPars[3];

};


#pragma GCC visibility pop
#endif 
