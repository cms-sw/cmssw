#ifndef TkDetLayers_Phase2EndcapLayerDoubleDisk_h
#define TkDetLayers_Phase2EndcapLayerDoubleDisk_h

#include "TrackingTools/DetLayers/interface/RingedForwardLayer.h"
#include "Phase2EndcapSubDisk.h"
#include <array>
#include <atomic>

/** A concrete implementation for Phase 2 Endcap/Forward layer 
 *  built out of Phase2EndcapRings
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapLayerDoubleDisk final : public RingedForwardLayer {
public:
  Phase2EndcapLayerDoubleDisk(std::vector<const Phase2EndcapSubDisk*>& subDisks);
  ~Phase2EndcapLayerDoubleDisk() override;

  // Default implementations would not properly manage memory
  Phase2EndcapLayerDoubleDisk(const Phase2EndcapLayerDoubleDisk&) = delete;
  Phase2EndcapLayerDoubleDisk& operator=(const Phase2EndcapLayerDoubleDisk&) = delete;

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
  BoundDisk* computeDisk(const std::vector<const Phase2EndcapSubDisk*>& subDisks) const;

  std::array<int, 2> subDiskIndicesByCrossingProximity(const TrajectoryStateOnSurface& startingState,
                                                       const Propagator& prop) const;

  std::array<int, 2> findTwoClosest(std::vector<GlobalPoint>) const;

  void fillSubDiskPars(int i);

private:
  std::vector<GeomDet const*> theBasicComps;
  mutable std::atomic<std::vector<const GeometricSearchDet*>*> theComponents;
  std::vector<const Phase2EndcapSubDisk*> theComps;
  struct SubDiskPar {
    float theSubDiskZ;
  };
  std::vector<SubDiskPar> subDiskPars;
  int theSubDisksSize;
};

#pragma GCC visibility pop
#endif
