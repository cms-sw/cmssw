
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace edm {
  class EDProductGetter;
}

namespace ticl {
  struct detIdInfoInCluster {
    bool operator==(const detIdInfoInCluster &o) const { return clusterId == o.clusterId; };
    long unsigned int clusterId;
    float fraction;
    detIdInfoInCluster(long unsigned int cId, float fr) {
      clusterId = cId;
      fraction = fr;
    }
  };

  struct simClusterOnCLayer {
    unsigned int simClusterId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<int, std::pair<float, float>> layerClusterIdToEnergyAndScore;
  };

  typedef std::vector<std::vector<std::pair<unsigned int, float>>> layerClusterToSimCluster;
  typedef std::vector<std::vector<ticl::simClusterOnCLayer>> simClusterToLayerCluster;
  typedef std::tuple<layerClusterToSimCluster, simClusterToLayerCluster> association;
}  // namespace ticl

class BarrelLCToSCAssociatorByEnergyScoreImpl : public ticl::LayerClusterToSimClusterAssociatorBaseImpl {
public:
  explicit BarrelLCToSCAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                                   bool,
                                                   const std::unordered_map<DetId, const reco::PFRecHit *> *);

  ticl::RecoToSimCollectionWithSimClusters associateRecoToSim(
      const edm::Handle<reco::CaloClusterCollection> &cCH,
      const edm::Handle<SimClusterCollection> &sCCH) const override;

  ticl::SimToRecoCollectionWithSimClusters associateSimToReco(
      const edm::Handle<reco::CaloClusterCollection> &cCH,
      const edm::Handle<SimClusterCollection> &sCCH) const override;

private:
  const bool hardScatterOnly_;
  const std::unordered_map<DetId, const reco::PFRecHit *> *hitMap_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
  ticl::association makeConnections(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                    const edm::Handle<SimClusterCollection> &sCCH) const;
};
