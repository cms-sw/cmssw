#ifndef TkDetLayers_TIDLayer_h
#define TkDetLayers_TIDLayer_h


#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "TIDRing.h"
#include <array>
#include <atomic>

/** A concrete implementation for TID layer 
 *  built out of TIDRings
 */

#pragma GCC visibility push(hidden)
class TIDLayer GCC11_FINAL : public RingedForwardLayer {
 public:
  TIDLayer(std::vector<const TIDRing*>& rings)  __attribute__ ((cold));
  ~TIDLayer()  __attribute__ ((cold));

  //default implementations would not manage memory correctly
  TIDLayer(const TIDLayer&) = delete;
  TIDLayer& operator=(const TIDLayer&) = delete;
  
  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold));

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::TID];}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  BoundDisk* computeDisk( const std::vector<const TIDRing*>& rings) const  __attribute__ ((cold));

  std::array<int,3> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
						   const Propagator& prop ) const;

  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  int findClosest( const GlobalPoint[3] ) const __attribute__ ((hot));
  
  int findNextIndex( const GlobalPoint[3] , int ) const __attribute__ ((hot));
  
  bool overlapInR( const TrajectoryStateOnSurface& tsos, int i, double ymax) const __attribute__ ((hot));
  
  
  float computeWindowSize( const GeomDet* det, 
  			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const __attribute__ ((hot));
  
  void fillRingPars(int i)  __attribute__ ((cold));

 private:
  std::vector<GeomDet const*> theBasicComps;
  mutable std::atomic<std::vector<const GeometricSearchDet*>*> theComponents;
  const TIDRing* theComps[3];
  struct RingPar { float theRingR, thetaRingMin, thetaRingMax;};
  RingPar ringPars[3];

};


#pragma GCC visibility pop
#endif 
