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
  // This structure is used both for LayerClusters and SimClusters storing their id and the fraction of a hit
  // that belongs to the LayerCluster or SimCluster. The meaning of the operator is extremely important since
  // this struct will be used inside maps and other containers and when searching for one particular occurence
  // only the clusterId member will be used in the check, skipping the fraction part.
  struct detIdInfoInCluster {
    bool operator==(const detIdInfoInCluster &o) const { return clusterId == o.clusterId; };
    long unsigned int clusterId;
    float fraction;
    detIdInfoInCluster(long unsigned int cId, float fr) {
      clusterId = cId;
      fraction = fr;
    }
  };

  // This introduces a simCluster on layer concept. For a simCluster it stores:
  // 1. Its id: simClusterId.
  // 2. The energy that the simCluster deposited in a specific layer and it was reconstructed.
  // 3. The hits_and_fractions that contributed to that deposition. SimHits that aren't reconstructed
  //    and doesn't have any matched rechits are disregarded.
  // 4. A map to save the LayerClusters ids (id is the key) that reconstructed at least one SimHit of the simCluster under study
  //    together with the energy that the Layercluster reconstructed from the SimClusters and the score. The energy
  //    is not the energy of the LayerCluster, but the energy of the LayerCluster coming from the SimCluster.
  //    So, there will be energy of the LayerCluster that is disregarded here, since there may be LayerCluster's
  //    cells that the SimCluster didn't contribute.
  struct simClusterOnCLayer {
    unsigned int simClusterId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<int, std::pair<float, float>> layerClusterIdToEnergyAndScore;
  };

  // This object connects a LayerCluster, identified through its id (lcId), with a vector of pairs containing all the SimClusters
  // (via their ids (scIds)) that share at least one cell with the LayerCluster. In that pair it
  // stores the score (lcId->(scId,score)).
  typedef std::vector<std::vector<std::pair<unsigned int, float>>> layerClusterToSimCluster;
  // This is used to save the simClusterOnLayer structure for all simClusters in each layer.
  // It is not exactly what is returned outside, but out of its entries, the output object is build.
  typedef std::vector<std::vector<hgcal::simClusterOnCLayer>> simClusterToLayerCluster;
  //This is the output of the makeConnections function that contain all the work with SC2LC and LC2SC
  //association. It will be read by the relevant associateSimToReco and associateRecoToSim functions to
  //provide the final product.
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
