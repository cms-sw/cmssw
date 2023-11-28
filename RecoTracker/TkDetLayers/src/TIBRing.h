#ifndef TkDetLayers_TIBRing_h
#define TkDetLayers_TIBRing_h

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

#include <optional>

/** A concrete implementation for TIB rings 
 */

#pragma GCC visibility push(hidden)
class TIBRing final : public GeometricSearchDet {
public:
  TIBRing(std::vector<const GeomDet*>& theGeomDets) __attribute__((cold));
  ~TIBRing() override __attribute__((cold));

  // GeometricSearchDet interface
  const BoundSurface& surface() const override { return *theCylinder; }

  const std::vector<const GeomDet*>& basicComponents() const override { return theDets; }

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__((cold));

  std::pair<bool, TrajectoryStateOnSurface> compatible(const TrajectoryStateOnSurface& ts,
                                                       const Propagator&,
                                                       const MeasurementEstimator&) const override
      __attribute__((cold));

  void groupedCompatibleDetsV(const TrajectoryStateOnSurface& startingState,
                              const Propagator& prop,
                              const MeasurementEstimator& est,
                              std::vector<DetGroup>& result) const override __attribute__((hot));

  //--- Extension of the interface

  /// Return the ring surface as a
  virtual const BoundCylinder& specificSurface() const { return *theCylinder; }

private:
  //general private methods

  void checkPeriodicity(std::vector<const GeomDet*>::const_iterator first,
                        std::vector<const GeomDet*>::const_iterator last) __attribute__((cold));

  void checkRadius(std::vector<const GeomDet*>::const_iterator first, std::vector<const GeomDet*>::const_iterator last)
      __attribute__((cold));

  void computeHelicity() __attribute__((cold));

  // methods for groupedCompatibleDets implementation
  struct SubRingCrossings {
    SubRingCrossings(int ci, int ni, float nd) : closestIndex(ci), nextIndex(ni), nextDistance(nd) {}

    int closestIndex;
    int nextIndex;
    float nextDistance;
  };

  void searchNeighbors(const TrajectoryStateOnSurface& tsos,
                       const Propagator& prop,
                       const MeasurementEstimator& est,
                       const SubRingCrossings& crossings,
                       float window,
                       std::vector<DetGroup>& result) const __attribute__((hot));

  std::optional<SubRingCrossings> computeCrossings(const TrajectoryStateOnSurface& startingState,
                                                   PropagationDirection propDir) const __attribute__((hot));

  float computeWindowSize(const GeomDet* det,
                          const TrajectoryStateOnSurface& tsos,
                          const MeasurementEstimator& est) const __attribute__((hot));

private:
  typedef PeriodicBinFinderInPhi<float> BinFinderType;
  BinFinderType theBinFinder;

  std::vector<const GeomDet*> theDets;
  ReferenceCountingPointer<BoundCylinder> theCylinder;
  int theHelicity;
};

#pragma GCC visibility pop
#endif
