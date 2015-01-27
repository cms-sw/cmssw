#ifndef TkDetLayers_Phase2OTECRingedLayer_h
#define TkDetLayers_Phase2OTECRingedLayer_h

#define NOTECRINGS 15   // FIXME: for sure to be fixed. Hopefully a better algorithm would not require looking for compatible hits in all the layers !!

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Phase2OTECRing.h"
#include <array>
#include <atomic>

/** A concrete implementation for Phase 2 OT EC layer 
 *  built out of Phase2OTECRings
 */

#pragma GCC visibility push(hidden)
class Phase2OTECRingedLayer GCC11_FINAL : public RingedForwardLayer {
 public:
  Phase2OTECRingedLayer(std::vector<const Phase2OTECRing*>& rings)  __attribute__ ((cold));
  ~Phase2OTECRingedLayer()  __attribute__ ((cold));

  // Default implementations would not properly manage memory
  Phase2OTECRingedLayer( const Phase2OTECRingedLayer& ) = delete;
  Phase2OTECRingedLayer& operator=( const Phase2OTECRingedLayer&) = delete;

  // GeometricSearchDet interface
  
  virtual const std::vector<const GeomDet*>& basicComponents() const {return theBasicComps;}
  
  virtual const std::vector<const GeometricSearchDet*>& components() const __attribute__ ((cold));

  void groupedCompatibleDetsV( const TrajectoryStateOnSurface& tsos,
			       const Propagator& prop,
			       const MeasurementEstimator& est,
			       std::vector<DetGroup> & result) const __attribute__ ((hot));

  // DetLayer interface
  virtual SubDetector subDetector() const {return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P2OTEC];}


 private:
  // private methods for the implementation of groupedCompatibleDets()
  BoundDisk* computeDisk( const std::vector<const Phase2OTECRing*>& rings) const __attribute__ ((cold));

  std::array<int,3> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
						   const Propagator& prop ) const;

  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  std::array<int,3> findThreeClosest( const GlobalPoint[NOTECRINGS] ) const __attribute__ ((hot));
  
  bool overlapInR( const TrajectoryStateOnSurface& tsos, int i, double ymax) const __attribute__ ((hot));
  
  
  float computeWindowSize( const GeomDet* det, 
  			   const TrajectoryStateOnSurface& tsos, 
			   const MeasurementEstimator& est) const __attribute__ ((hot));
  
  void fillRingPars(int i) __attribute__ ((cold));

 private:
  std::vector<GeomDet const*> theBasicComps;
  mutable std::atomic<std::vector<const GeometricSearchDet*>*> theComponents;
  const Phase2OTECRing* theComps[NOTECRINGS];
  struct RingPar { float theRingR, thetaRingMin, thetaRingMax;};
  RingPar ringPars[NOTECRINGS];

};


#pragma GCC visibility pop
#endif 
