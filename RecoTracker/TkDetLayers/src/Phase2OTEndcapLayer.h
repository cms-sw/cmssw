#ifndef TkDetLayers_Phase2OTEndcapLayer_h
#define TkDetLayers_Phase2OTEndcapLayer_h

#define NOTECRINGS 15   // FIXME: for sure to be fixed. Hopefully a better algorithm would not require looking for compatible hits in all the layers !!

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Phase2OTEndcapRing.h"
#include <array>
#include <atomic>

/** A concrete implementation for Phase 2 OT EC layer 
 *  built out of Phase2OTEndcapRings
 */

#pragma GCC visibility push(hidden)
class Phase2OTEndcapLayer final : public RingedForwardLayer {
 public:
  Phase2OTEndcapLayer(std::vector<const Phase2OTEndcapRing*>& rings)  __attribute__ ((cold));
  ~Phase2OTEndcapLayer()  __attribute__ ((cold));

  // Default implementations would not properly manage memory
  Phase2OTEndcapLayer( const Phase2OTEndcapLayer& ) = delete;
  Phase2OTEndcapLayer& operator=( const Phase2OTEndcapLayer&) = delete;

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
  BoundDisk* computeDisk( const std::vector<const Phase2OTEndcapRing*>& rings) const __attribute__ ((cold));

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
  const Phase2OTEndcapRing* theComps[NOTECRINGS];
  struct RingPar { float theRingR, thetaRingMin, thetaRingMax;};
  RingPar ringPars[NOTECRINGS];

};


#pragma GCC visibility pop
#endif 
