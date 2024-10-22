#include "TrackingTools/GsfTracking/interface/TsosGaussianStateConversions.h"

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"
#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"

using namespace SurfaceSideDefinition;

namespace GaussianStateConversions {

  MultiGaussianState<5> multiGaussianStateFromTSOS(const TrajectoryStateOnSurface& tsos) {
    if (!tsos.isValid())
      return MultiGaussianState<5>();

    using SingleStatePtr = std::shared_ptr<SingleGaussianState<5>>;
    auto const& components = tsos.components();
    MultiGaussianState<5>::SingleStateContainer singleStates;
    singleStates.reserve(components.size());
    for (auto const& ic : components) {
      if (ic.isValid()) {
        auto sgs = std::make_shared<SingleGaussianState<5>>(
            ic.localParameters().vector(), ic.localError().matrix(), ic.weight());
        singleStates.push_back(sgs);
      }
    }
    return MultiGaussianState<5>(singleStates);
  }

  TrajectoryStateOnSurface tsosFromMultiGaussianState(const MultiGaussianState<5>& multiState,
                                                      const TrajectoryStateOnSurface& refTsos) {
    if (multiState.components().empty())
      return TrajectoryStateOnSurface();
    const Surface& surface = refTsos.surface();
    SurfaceSide side = refTsos.surfaceSide();
    const MagneticField* field = refTsos.magneticField();
    auto const& refTsos1 = refTsos.components().front();
    auto pzSign = refTsos1.localParameters().pzSign();
    bool charged = refTsos1.charge() != 0;

    auto const& singleStates = multiState.components();
    std::vector<TrajectoryStateOnSurface> components;
    components.reserve(singleStates.size());
    for (auto const& ic : singleStates) {
      //require states to be positive-definite
      if (double det = 0; (*ic).covariance().Det2(det) && det > 0) {
        components.emplace_back((*ic).weight(),
                                LocalTrajectoryParameters((*ic).mean(), pzSign, charged),
                                LocalTrajectoryError((*ic).covariance()),
                                surface,
                                field,
                                side);
      }
    }
    return components.empty()
               ? TrajectoryStateOnSurface()
               : TrajectoryStateOnSurface((BasicTrajectoryState*)new BasicMultiTrajectoryState(components));
  }
}  // namespace GaussianStateConversions
