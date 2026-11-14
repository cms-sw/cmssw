#ifndef RecoVertex_PrimaryVertexProducer_GNNClusterizerFromAlpaka_h
#define RecoVertex_PrimaryVertexProducer_GNNClusterizerFromAlpaka_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <vector>

namespace vertexgnn {

  class GNNClusterizerFromAlpaka {
  public:
    GNNClusterizerFromAlpaka(const edm::ParameterSet& conf);

    std::vector<TransientVertex> vertices(const std::vector<reco::TransientTrack>& tracks,
                                          const std::vector<int>& trackSlot,
                                          const std::vector<float>& trackMaxProb,
                                          const std::vector<float>& z_hat,
                                          const std::vector<float>& p) const;

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

  private:
    const double existenceThreshold_;
    const double trackAssignmentThreshold_;
    const bool verbose_;
  };

}  // namespace vertexgnn

#endif
