#ifndef TkDetLayers_Phase2EndcapLayer_h
#define TkDetLayers_Phase2EndcapLayer_h

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Phase2EndcapRing.h"
#include <array>
#include <atomic>

/** A concrete implementation for Phase 2 Endcap/Forward layer 
 *  built out of Phase2EndcapRings
 *  this classs is used for both OT and Pixel detector
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapLayer final : public RingedForwardLayer {
 public:
  Phase2EndcapLayer(std::vector<const Phase2EndcapRing*>& rings, const bool isOT)  __attribute__ ((cold));
  ~Phase2EndcapLayer()  __attribute__ ((cold));

  // Default implementations would not properly manage memory
  Phase2EndcapLayer( const Phase2EndcapLayer& ) = delete;
  Phase2EndcapLayer& operator=( const Phase2EndcapLayer&) = delete;

  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold));

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));

  // DetLayer interface
  virtual SubDetector subDetector() const { if(isOuterTracker) return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P2OTEC];
                                            else return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P2PXEC];}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  BoundDisk* computeDisk( const std::vector<const Phase2EndcapRing*>& rings) const __attribute__ ((cold));

  std::array<int,3> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
						   const Propagator& prop ) const;

  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  std::array<int,3> findThreeClosest( std::vector<GlobalPoint> ) const __attribute__ ((hot));
  
  bool overlapInR( const TrajectoryStateOnSurface& tsos, int i, double ymax) const __attribute__ ((hot));
  
  
  float computeWindowSize( const GeomDet* det, 
  			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const __attribute__ ((hot));
  
  void fillRingPars(int i) __attribute__ ((cold));

 private:
  std::vector<GeomDet const*> theBasicComps;
  const bool isOuterTracker;
  mutable std::atomic<std::vector<const GeometricSearchDet*>*> theComponents;
  std::vector<const Phase2EndcapRing*> theComps;
  struct RingPar { float theRingR, thetaRingMin, thetaRingMax;};
  std::vector<RingPar> ringPars;
  int theRingSize;
};


#pragma GCC visibility pop
#endif 
