// Original Author: Leonardo Cristella

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace edm {
  class EDProductGetter;
}

namespace hgcal {
  struct detIdInfoInCluster {
    bool operator==(const detIdInfoInCluster &o) const { return clusterId == o.clusterId; };
    long unsigned int clusterId;
    float fraction;
    detIdInfoInCluster(long unsigned int cId, float fr) {
      clusterId = cId;
      fraction = fr;
    }
  };

  struct detIdInfoInMultiCluster {
    bool operator==(const detIdInfoInMultiCluster &o) const { return multiclusterId == o.multiclusterId; };
    unsigned int multiclusterId;
    long unsigned int clusterId;
    float fraction;
  };

  struct simClusterOnLayer {
    unsigned int simClusterId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<int, std::pair<float, float>> layerClusterIdToEnergyAndScore;
  };

  typedef std::vector<std::vector<std::pair<unsigned int, float>>> layerClusterToSimCluster;
  typedef std::vector<std::vector<hgcal::simClusterOnLayer>> simClusterToLayerCluster;
  typedef std::tuple<layerClusterToSimCluster, simClusterToLayerCluster> association;
}  // namespace hgcal

class LCToSCAssociatorByEnergyScoreImpl : public hgcal::LayerClusterToSimClusterAssociatorBaseImpl {
public:
  explicit LCToSCAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                             bool,
                                             std::shared_ptr<hgcal::RecHitTools>,
                                             const std::unordered_map<DetId, const HGCRecHit *> *);

  hgcal::RecoToSimCollectionWithSimClusters associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCH,
      const edm::Handle<SimClusterCollection> &sCCH) const override;

  hgcal::SimToRecoCollectionWithSimClusters associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCH,
      const edm::Handle<SimClusterCollection> &sCCH) const override;

private:
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> recHitTools_;
  const std::unordered_map<DetId, const HGCRecHit *> *hitMap_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
  hgcal::association makeConnections(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                     const edm::Handle<SimClusterCollection> &sCCH) const;
};
