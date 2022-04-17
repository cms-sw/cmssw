#ifndef TkDetLayers_Phase2EndcapSubDisk_h
#define TkDetLayers_Phase2EndcapSubDisk_h

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Phase2EndcapSingleRing.h"
#include "TkDetUtil.h"
#include <array>
#include <atomic>

/** A concrete implementation for Phase 2 Endcap/Forward layer 
 *  built out of Phase2EndcapSingleRings
 *  this class is used for the inner tracker
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapSubDisk final : public RingedForwardLayer {
public:
  Phase2EndcapSubDisk(std::vector<const Phase2EndcapSingleRing*>& rings);
  ~Phase2EndcapSubDisk() override;

  // Default implementations would not properly manage memory
  Phase2EndcapSubDisk(const Phase2EndcapSubDisk&) = delete;
  Phase2EndcapSubDisk& operator=(const Phase2EndcapSubDisk&) = delete;

  // GeometricSearchDet interface

  const std::vector<const GeomDet*>& basicComponents() const override { return theBasicComps; }

  const std::vector<const GeometricSearchDet*>& components() const override;

  void groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                              const Propagator& prop,
                              const MeasurementEstimator& est,
                              std::vector<DetGroup>& result) const override;

  // DetLayer interface
  SubDetector subDetector() const override { return GeomDetEnumerators::subDetGeom[GeomDetEnumerators::P2PXEC]; }

private:
  // private methods for the implementation of groupedCompatibleDets()
  BoundDisk* computeDisk(const std::vector<const Phase2EndcapSingleRing*>& rings) const;

  bool overlapInR(const TrajectoryStateOnSurface& tsos,
                  int i,
                  double ymax,
                  std::vector<tkDetUtil::RingPar> ringParams) const;

  float computeWindowSize(const GeomDet* det,
                          const TrajectoryStateOnSurface& tsos,
                          const MeasurementEstimator& est) const;

  void fillRingPars(int i);

  std::vector<GeomDet const*> theBasicComps;
  mutable std::atomic<std::vector<const GeometricSearchDet*>*> theComponents;
  std::vector<const Phase2EndcapSingleRing*> theComps;
  std::vector<tkDetUtil::RingPar> ringPars;
  int theRingSize;
};

#pragma GCC visibility pop
#endif
