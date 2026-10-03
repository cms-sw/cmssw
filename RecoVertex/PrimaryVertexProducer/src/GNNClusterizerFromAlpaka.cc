#include "RecoVertex/PrimaryVertexProducer/interface/GNNClusterizerFromAlpaka.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace vertexgnn {

  GNNClusterizerFromAlpaka::GNNClusterizerFromAlpaka(const edm::ParameterSet& conf)
      : existenceThreshold_(conf.getParameter<double>("existenceThreshold")),
        trackAssignmentThreshold_(conf.getParameter<double>("trackAssignmentThreshold")),
        verbose_(conf.getUntrackedParameter<bool>("verbose", false)) {}

  std::vector<TransientVertex> GNNClusterizerFromAlpaka::vertices(const std::vector<reco::TransientTrack>& tracks,
                                                                  const std::vector<int>& trackSlot,
                                                                  const std::vector<float>& trackMaxProb,
                                                                  const std::vector<float>& z_hat,
                                                                  const std::vector<float>& p) const {
    std::vector<TransientVertex> clusters;
    const int N = tracks.size();
    const int K = z_hat.size();
    if (N == 0 || K == 0) {
      return clusters;
    }

    std::vector<int> slotTrackCount(K, 0);
    for (int i = 0; i < N; ++i) {
      if (trackSlot[i] >= 0 && trackSlot[i] < K) {
        slotTrackCount[trackSlot[i]]++;
      }
    }

    const GlobalError dummyError(0.01, 0, 0.01, 0., 0., 0.01);
    for (int k = 0; k < K; ++k) {
      if (!(p[k] > existenceThreshold_ && slotTrackCount[k] >= 1)) {
        continue;
      }
      std::vector<reco::TransientTrack> clusterTracks;
      std::vector<float> clusterWeights;
      for (int i = 0; i < N; ++i) {
        if (trackSlot[i] == k && trackMaxProb[i] >= trackAssignmentThreshold_) {
          clusterTracks.push_back(tracks[i]);
          clusterWeights.push_back(trackMaxProb[i]);
        }
      }
      if (clusterTracks.empty()) {
        continue;
      }
      TransientVertex vertex(GlobalPoint(0, 0, z_hat[k]), dummyError, clusterTracks, 0);
      TransientVertex::TransientTrackToFloatMap weightMap;
      for (size_t i = 0; i < clusterTracks.size(); ++i) {
        weightMap[clusterTracks[i]] = clusterWeights[i];
      }
      vertex.weightMap(weightMap);
      clusters.push_back(vertex);
    }

    if (verbose_) {
      edm::LogPrint("GNNClusterizerFromAlpaka") << "vertex-slot GNN: " << clusters.size() << " vertex candidates from "
                                                << N << " tracks and " << K << " slots";
    }
    return clusters;
  }

  void GNNClusterizerFromAlpaka::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<double>("existenceThreshold", 0.5)->setComment("minimum slot existence probability");
    desc.add<double>("trackAssignmentThreshold", 0.5)->setComment("minimum track assignment probability");
    desc.addUntracked<bool>("verbose", false);
  }

}  // namespace vertexgnn
