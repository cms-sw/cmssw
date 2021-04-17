#ifndef TkDetLayers_Phase2OTBarrelRod_h
#define TkDetLayers_Phase2OTBarrelRod_h

#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"
#include "TrackingTools/DetLayers/interface/DetRod.h"
#include "Utilities/BinningTools/interface/GenericBinFinderInZ.h"
#include "SubLayerCrossings.h"

/** A concrete implementation for TOB Rod 
 *  
 */

#pragma GCC visibility push(hidden)
class Phase2OTBarrelRod final : public DetRod {
public:
  typedef GenericBinFinderInZ<float, GeomDet> BinFinderType;

  Phase2OTBarrelRod(std::vector<const GeomDet*>& innerDets,
                    std::vector<const GeomDet*>& outerDets,
                    const std::vector<const GeomDet*>& innerDetBrothers = std::vector<const GeomDet*>(),
                    const std::vector<const GeomDet*>& outerDetBrothers = std::vector<const GeomDet*>())
      __attribute__((cold));
  ~Phase2OTBarrelRod() override __attribute__((cold));

  // GeometricSearchDet interface

  const std::vector<const GeomDet*>& basicComponents() const override { return theDets; }

  const std::vector<const GeometricSearchDet*>& components() const override __attribute__((cold));

  std::pair<bool, TrajectoryStateOnSurface> compatible(const TrajectoryStateOnSurface& ts,
                                                       const Propagator&,
                                                       const MeasurementEstimator&) const override
      __attribute__((cold));

  void groupedCompatibleDetsV(const TrajectoryStateOnSurface& tsos,
                              const Propagator& prop,
                              const MeasurementEstimator& est,
                              std::vector<DetGroup>& result) const override __attribute__((hot));

private:
  // private methods for the implementation of groupedCompatibleDets()

  SubLayerCrossings computeCrossings(const TrajectoryStateOnSurface& tsos, PropagationDirection propDir) const
      __attribute__((hot));

  bool addClosest(const TrajectoryStateOnSurface& tsos,
                  const Propagator& prop,
                  const MeasurementEstimator& est,
                  const SubLayerCrossing& crossing,
                  std::vector<DetGroup>& result,
                  std::vector<DetGroup>& brotherresult) const __attribute__((hot));

  float computeWindowSize(const GeomDet* det,
                          const TrajectoryStateOnSurface& tsos,
                          const MeasurementEstimator& est) const __attribute__((hot));

  void searchNeighbors(const TrajectoryStateOnSurface& tsos,
                       const Propagator& prop,
                       const MeasurementEstimator& est,
                       const SubLayerCrossing& crossing,
                       float window,
                       std::vector<DetGroup>& result,
                       std::vector<DetGroup>& brotherresult,
                       bool checkClosest) const __attribute__((hot));

  const std::vector<const GeomDet*>& subRod(int ind) const { return (ind == 0 ? theInnerDets : theOuterDets); }

  const std::vector<const GeomDet*>& subRodBrothers(int ind) const {
    return (ind == 0 ? theInnerDetBrothers : theOuterDetBrothers);
  }

private:
  std::vector<const GeomDet*> theDets;
  std::vector<const GeomDet*> theInnerDets;
  std::vector<const GeomDet*> theOuterDets;
  std::vector<const GeomDet*> theInnerDetBrothers;
  std::vector<const GeomDet*> theOuterDetBrothers;

  ReferenceCountingPointer<Plane> theInnerPlane;
  ReferenceCountingPointer<Plane> theOuterPlane;

  BinFinderType theInnerBinFinder;
  BinFinderType theOuterBinFinder;
};

#pragma GCC visibility pop
#endif
