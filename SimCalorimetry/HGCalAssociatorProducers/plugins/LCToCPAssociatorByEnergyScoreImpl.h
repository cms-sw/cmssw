// Original Author: Marco Rovere

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace edm {
  class EDProductGetter;
}

namespace hgcal {
  // This structure is used both for LayerClusters and CaloParticles storing their id and the fraction of a hit
  // that belongs to the LayerCluster or CaloParticle. The meaning of the operator is extremely important since
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

  // This introduces a CaloParticle on layer concept. For a CaloParticle it stores:
  // 1. Its id: caloParticleId.
  // 2. The energy that the CaloParticle deposited in a specific layer and it was reconstructed.
  // 3. The hits_and_fractions that contributed to that deposition. SimHits that aren't reconstructed
  //    and doesn't have any matched rechits are disregarded. Keep in mind that since a CaloParticle
  //    should most probably have more than one SimCluster, all different contributions from the same CaloParticle
  //    to a single hit are merged into a single entry, with the fractions properly summed.
  // 4. A map to save the LayerClusters ids (id is the key) that reconstructed at least one SimHit of the CaloParticle under study
  //    together with the energy that the LayerCluster reconstructed from the CaloParticle and the score. The energy
  //    is not the energy of the LayerCluster, but the energy of the LayerCluster coming from the CaloParticle.
  //    So, there will be energy of the LayerCluster that is disregarded here, since there may be LayerCluster's
  //    cells that the CaloParticle didn't contribute.
  struct caloParticleOnLayer {
    unsigned int caloParticleId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<int, std::pair<float, float>> layerClusterIdToEnergyAndScore;
  };

  // This object connects a LayerCluster, identified through its id (lcId), with a vector of pairs containing all the CaloParticles
  // (via their ids (cpIds)) that share at least one cell with the LayerCluster. In that pair it
  // stores the score (lcId->(cpId,score)). Keep in mind that the association is not unique, since there could be several instances
  // of the same CaloParticle from several related SimClusters that each contributed to the same LayerCluster.
  typedef std::vector<std::vector<std::pair<unsigned int, float>>> layerClusterToCaloParticle;
  // This is used to save the caloParticleOnLayer structure for all CaloParticles in each layer.
  // It is not exactly what is returned outside, but out of its entries, the output object is build.
  typedef std::vector<std::vector<hgcal::caloParticleOnLayer>> caloParticleToLayerCluster;
  //This is the output of the makeConnections function that contain all the work with CP2LC and LC2CP
  //association. It will be read by the relevant associateSimToReco and associateRecoToSim functions to
  //provide the final product.
  typedef std::tuple<layerClusterToCaloParticle, caloParticleToLayerCluster> association;
}  // namespace hgcal

class LCToCPAssociatorByEnergyScoreImpl : public hgcal::LayerClusterToCaloParticleAssociatorBaseImpl {
public:
  explicit LCToCPAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                             bool,
                                             std::shared_ptr<hgcal::RecHitTools>,
                                             const std::unordered_map<DetId, const HGCRecHit *> *);

  hgcal::RecoToSimCollection associateRecoToSim(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<CaloParticleCollection> &cPCH) const override;

  hgcal::SimToRecoCollection associateSimToReco(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                                const edm::Handle<CaloParticleCollection> &cPCH) const override;

private:
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> recHitTools_;
  const std::unordered_map<DetId, const HGCRecHit *> *hitMap_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
  hgcal::association makeConnections(const edm::Handle<reco::CaloClusterCollection> &cCH,
                                     const edm::Handle<CaloParticleCollection> &cPCH) const;
};
