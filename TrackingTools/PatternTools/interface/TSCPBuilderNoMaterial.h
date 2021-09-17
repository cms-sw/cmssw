#ifndef TSCPBuilderNoMaterial_H
#define TSCPBuilderNoMaterial_H

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToPointBuilder.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"
#include "FWCore/Utilities/interface/Visibility.h"

/**
 * This class builds a TrajectoryStateClosestToPoint given an original 
 * TrajectoryStateOnSurface or FreeTrajectoryState. This new state is then 
 * defined at the point of closest approach to the reference point.
 * In case the propagation was not successful, this state can be invalid.
 */

class TSCPBuilderNoMaterial final : public TrajectoryStateClosestToPointBuilder {
public:
  ~TSCPBuilderNoMaterial() override {}

  TrajectoryStateClosestToPoint operator()(const FTS& originalFTS, const GlobalPoint& referencePoint) const override;

  TrajectoryStateClosestToPoint operator()(const TSOS& originalTSOS, const GlobalPoint& referencePoint) const override;

private:
  typedef Point3DBase<double, GlobalTag> GlobalPointDouble;
  typedef Vector3DBase<double, GlobalTag> GlobalVectorDouble;
  typedef std::pair<bool, FreeTrajectoryState> PairBoolFTS;

  PairBoolFTS createFTSatTransverseImpactPoint(const FTS& originalFTS,
                                               const GlobalPoint& referencePoint) const dso_internal;

  PairBoolFTS createFTSatTransverseImpactPointCharged(const FTS& originalFTS,
                                                      const GlobalPoint& referencePoint) const dso_internal;

  PairBoolFTS createFTSatTransverseImpactPointNeutral(const FTS& originalFTS,
                                                      const GlobalPoint& referencePoint) const dso_internal;
};
#endif
