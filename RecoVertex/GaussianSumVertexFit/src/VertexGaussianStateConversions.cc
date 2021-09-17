#include "RecoVertex/GaussianSumVertexFit/interface/VertexGaussianStateConversions.h"

#include "RecoVertex/GaussianSumVertexFit/interface/BasicMultiVertexState.h"
#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"

namespace GaussianStateConversions {

  MultiGaussianState<3> multiGaussianStateFromVertex(const VertexState aState) {
    typedef std::shared_ptr<SingleGaussianState<3> > SingleStatePtr;
    const std::vector<VertexState> components = aState.components();
    MultiGaussianState<3>::SingleStateContainer singleStates;
    singleStates.reserve(components.size());
    for (std::vector<VertexState>::const_iterator ic = components.begin(); ic != components.end(); ic++) {
      if (ic->isValid()) {
        GlobalPoint pos(ic->position());
        AlgebraicVector3 parameters;
        parameters(0) = pos.x();
        parameters(1) = pos.y();
        parameters(2) = pos.z();
        SingleStatePtr sgs(new SingleGaussianState<3>(parameters, ic->error().matrix(), ic->weightInMixture()));
        singleStates.push_back(sgs);
      }
    }
    return MultiGaussianState<3>(singleStates);
  }

  VertexState vertexFromMultiGaussianState(const MultiGaussianState<3>& multiState) {
    if (multiState.components().empty())
      return VertexState();

    const MultiGaussianState<3>::SingleStateContainer& singleStates = multiState.components();
    std::vector<VertexState> components;
    components.reserve(singleStates.size());
    for (MultiGaussianState<3>::SingleStateContainer::const_iterator ic = singleStates.begin();
         ic != singleStates.end();
         ic++) {
      const AlgebraicVector3& par = (**ic).mean();
      GlobalPoint position(par(0), par(1), par(2));
      const AlgebraicSymMatrix33& cov = (**ic).covariance();
      GlobalError error(cov(0, 0), cov(1, 0), cov(2, 0), cov(1, 1), cov(2, 1), cov(2, 2));
      components.push_back(VertexState(position, error, (**ic).weight()));
    }
    return VertexState(new BasicMultiVertexState(components));
  }
}  // namespace GaussianStateConversions
