#ifndef TkDetLayers_Phase2EndcapSingleRing_h
#define TkDetLayers_Phase2EndcapSingleRing_h

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"
#include "SubLayerCrossings.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"

/** A concrete implementation for Phase2 SubDisk rings 
 */

#pragma GCC visibility push(hidden)
class Phase2EndcapSingleRing final : public GeometricSearchDet {
public:
  Phase2EndcapSingleRing(std::vector<const GeomDet*>& allDets);
  ~Phase2EndcapSingleRing() override;

  // GeometricSearchDet interface
  const BoundSurface& surface() const override { return *theDisk; }

  const std::vector<const GeomDet*>& basicComponents() const override { return theDets; }

  const std::vector<const GeometricSearchDet*>& components() const override;

  std::pair<bool, TrajectoryStateOnSurface> compatible(const TrajectoryStateOnSurface&,
                                                       const Propagator&,
                                                       const MeasurementEstimator&) const override;

  void groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                              const Propagator& prop,
                              const MeasurementEstimator& est,
                              std::vector<DetGroup>& result) const override;

  //Extension of interface
  virtual const BoundDisk& specificSurface() const { return *theDisk; }

private:
  // private methods for the implementation of groupedCompatibleDets()

  SubLayerCrossing computeCrossing(const TrajectoryStateOnSurface& tsos, PropagationDirection propDir) const;

  bool addClosest(const TrajectoryStateOnSurface& tsos,
                  const Propagator& prop,
                  const MeasurementEstimator& est,
                  const SubLayerCrossing& crossing,
                  std::vector<DetGroup>& result) const;

  void searchNeighbors(const TrajectoryStateOnSurface& tsos,
                       const Propagator& prop,
                       const MeasurementEstimator& est,
                       const SubLayerCrossing& crossing,
                       float window,
                       std::vector<DetGroup>& result,
                       bool checkClosest) const;

  const std::vector<const GeomDet*>& subLayer(int ind) const { return theDets; }

private:
  std::vector<const GeomDet*> theDets;

  ReferenceCountingPointer<BoundDisk> theDisk;

  typedef PeriodicBinFinderInPhi<float> BinFinderType;

  BinFinderType theBinFinder;
};

#pragma GCC visibility pop
#endif
