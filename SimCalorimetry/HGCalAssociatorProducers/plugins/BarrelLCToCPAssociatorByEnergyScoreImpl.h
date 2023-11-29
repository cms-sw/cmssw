#include <vector>
#include <map>
#include <unordered_map>
#include <memory>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"

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

  struct caloParticleOnLayer {
    unsigned int caloParticleId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<int, std::pair<float, float>> layerClusterIdToEnergyAndScore;
  };

  typedef std::vector<std::vector<std::pair<unsigned int, float>>> layerClusterToCaloParticle;
  typedef std::vector<std::vector<ticl::caloParticleOnLayer>> caloParticleToLayerCluster;
  typedef std::tuple<layerClusterToCaloParticle, caloParticleToLayerCluster> association;
}  //namespace ticl

class BarrelLCToCPAssociatorByEnergyScoreImpl : public ticl::LayerClusterToCaloParticleAssociatorBaseImpl {
public:
  explicit BarrelLCToCPAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                                   bool,
                                                   const std::unordered_map<DetId, const reco::PFRecHit *> *);

  ticl::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                               const edm::Handle<CaloParticleCollection> &cPCH) const override;

  ticl::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                               const edm::Handle<CaloParticleCollection> &cPCH) const override;

private:
  const bool hardScatterOnly_;
  const std::unordered_map<DetId, const reco::PFRecHit *> *hitMap_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
  ticl::association makeConnections(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                    const edm::Handle<CaloParticleCollection> &cPCH) const;
};
