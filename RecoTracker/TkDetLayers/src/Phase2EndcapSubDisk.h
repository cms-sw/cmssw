#ifndef TkDetLayers_Phase2EndcapSubDisk_h
#define TkDetLayers_Phase2EndcapSubDisk_h

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Phase2EndcapSingleRing.h"
#include "TkDetUtil.h"
#include <array>
#include <atomic>

/** A concrete implementation for Phase 2 Endcap/Forward layer 
 *  built out of Phase2EndcapSingleRings
 *  this classs is used for both OT and Pixel detector
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapSubDisk final : public RingedForwardLayer {
public:
  Phase2EndcapSubDisk(std::vector<const Phase2EndcapSingleRing*>& rings) __attribute__((cold));
  ~Phase2EndcapSubDisk() override __attribute__((cold));

  // Default implementations would not properly manage memory
  Phase2EndcapSubDisk(const Phase2EndcapSubDisk&) = delete;
  Phase2EndcapSubDisk& operator=(const Phase2EndcapSubDisk&) = delete;

  // GeometricSearchDet interface

  const std::vector<const GeomDet*>& basicComponents() const override { return theBasicComps; }

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__((cold));

  void groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                              const Propagator& prop,
                              const MeasurementEstimator& est,
                              std::vector<DetGroup>& result) const override __attribute__((hot));

  // DetLayer interface
  SubDetector subDetector() const override { return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P2PXEC]; }

private:
  // private methods for the implementation of groupedCompatibleDets()
  BoundDisk* computeDisk(const std::vector<const Phase2EndcapSingleRing*>& rings) const __attribute__((cold));

  std::array<int, 3> ringIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
                                                    const Propagator& prop) const;

  //  bool isCompatible( const TrajectoryStateOnSurface& ms,
  //	     const MeasurementEstimator& est) const;

  std::array<int, 3> findThreeClosest(std::vector<tkDetUtil::RingPar> ringParams,
                                      std::vector<GlobalPoint> ringCrossing,
                                      int ringSize) const __attribute__((hot));

  bool overlapInR(const TrajectoryStateOnSurface& tsos,
                  int i,
                  double ymax,
                  std::vector<tkDetUtil::RingPar> ringParams) const __attribute__((hot));

  float computeWindowSize(const GeomDet* det,
                          const TrajectoryStateOnSurface& tsos,
                          const MeasurementEstimator& est) const __attribute__((hot));

  void fillRingPars(int i) __attribute__((cold));

private:
  std::vector<GeomDet const*> theBasicComps;
  mutable std::atomic<std::vector<const GeometricSearchDet*>*> theComponents;
  std::vector<const Phase2EndcapSingleRing*> theComps;
  std::vector<tkDetUtil::RingPar> ringPars;
  int theRingSize;
};

#pragma GCC visibility pop
#endif
