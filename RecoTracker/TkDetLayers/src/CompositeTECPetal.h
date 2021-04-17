#ifndef TkDetLayers_CompositeTECPetal_h
#define TkDetLayers_CompositeTECPetal_h

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"

#include "TECWedge.h"

#include "DataFormats/GeometrySurface/interface/BoundDiskSector.h"

#include "SubLayerCrossings.h"

#include "FWCore/Utilities/interface/Visibility.h"

/** A concrete implementation for TEC petals
 */

#pragma GCC visibility push(hidden)
class CompositeTECPetal final : public GeometricSearchDet {
public:
  struct WedgePar {
    float theR, thetaMin, thetaMax;
  };

  CompositeTECPetal(std::vector<const TECWedge*>& innerWedges, std::vector<const TECWedge*>& outerWedges)
      __attribute__((cold));

  ~CompositeTECPetal() override __attribute__((cold));

  // GeometricSearchDet interface
  const BoundSurface& surface() const final { return *theDiskSector; }
  //Extension of the interface
  virtual const BoundDiskSector& specificSurface() const final { return *theDiskSector; }

  // GeometricSearchDet interface
  const std::vector<const GeomDet*>& basicComponents() const override { return theBasicComps; }

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__((cold)) { return theComps; }

  std::pair<bool, TrajectoryStateOnSurface> compatible(const TrajectoryStateOnSurface& ts,
                                                       const Propagator&,
                                                       const MeasurementEstimator&) const override
      __attribute__((cold));

  void groupedCompatibleDetsV(const TrajectoryStateOnSurface& startingState,
                              const Propagator& prop,
                              const MeasurementEstimator& est,
                              std::vector<DetGroup>& result) const override __attribute__((hot));

private:
  ReferenceCountingPointer<BoundDiskSector> theDiskSector;

  // private methods for the implementation of groupedCompatibleDets()
  SubLayerCrossings computeCrossings(const TrajectoryStateOnSurface& tsos, PropagationDirection propDir) const
      __attribute__((hot)) dso_internal;

  bool addClosest(const TrajectoryStateOnSurface& tsos,
                  const Propagator& prop,
                  const MeasurementEstimator& est,
                  const SubLayerCrossing& crossing,
                  std::vector<DetGroup>& result) const __attribute__((hot)) dso_internal;

  void searchNeighbors(const TrajectoryStateOnSurface& tsos,
                       const Propagator& prop,
                       const MeasurementEstimator& est,
                       const SubLayerCrossing& crossing,
                       float window,
                       std::vector<DetGroup>& result,
                       bool checkClosest) const __attribute__((hot)) dso_internal;

  static float computeWindowSize(const GeomDet* det,
                                 const TrajectoryStateOnSurface& tsos,
                                 const MeasurementEstimator& est) __attribute__((hot)) dso_internal;

  int findBin(float R, int layer) const dso_internal;

  WedgePar const& findPar(int index, int diskSectorType) const dso_internal {
    return (diskSectorType == 0) ? theFrontPars[index] : theBackPars[index];
  }

  const std::vector<const TECWedge*>& subLayer(int ind) const dso_internal {
    return (ind == 0 ? theFrontComps : theBackComps);
  }

private:
  std::vector<const GeomDet*> theBasicComps;
  std::vector<const GeometricSearchDet*> theComps;

  std::vector<const TECWedge*> theFrontComps;
  std::vector<const TECWedge*> theBackComps;

  std::vector<float> theFrontBoundaries;
  std::vector<float> theBackBoundaries;
  std::vector<WedgePar> theFrontPars;
  std::vector<WedgePar> theBackPars;

  ReferenceCountingPointer<BoundDiskSector> theFrontSector;
  ReferenceCountingPointer<BoundDiskSector> theBackSector;
};

#pragma GCC visibility pop
#endif
