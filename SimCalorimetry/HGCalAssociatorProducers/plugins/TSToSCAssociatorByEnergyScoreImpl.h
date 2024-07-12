// Original Author: Leonardo Cristella

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociator.h"
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

  struct simClusterOnBLayer {
    unsigned int simClusterId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<int, std::pair<float, float>> tracksterIdToEnergyAndScore;
  };

  typedef std::vector<std::vector<std::pair<unsigned int, float>>> tracksterToSimCluster;
  typedef std::vector<ticl::simClusterOnBLayer> simClusterToTrackster;
  typedef std::tuple<tracksterToSimCluster, simClusterToTrackster> association;
}  // namespace ticl

class TSToSCAssociatorByEnergyScoreImpl : public ticl::TracksterToSimClusterAssociatorBaseImpl {
public:
  explicit TSToSCAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                             bool,
                                             std::shared_ptr<hgcal::RecHitTools>,
                                             const std::unordered_map<DetId, const unsigned int> *,
                                             std::vector<const HGCRecHit *> &hits);

  ticl::RecoToSimCollectionTracksters associateRecoToSim(const edm::Handle<ticl::TracksterCollection> &tCH,
                                                         const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                                         const edm::Handle<SimClusterCollection> &sCCH) const override;

  ticl::SimToRecoCollectionTracksters associateSimToReco(const edm::Handle<ticl::TracksterCollection> &tCH,
                                                         const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                                         const edm::Handle<SimClusterCollection> &sCCH) const override;

private:
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> recHitTools_;
  const std::unordered_map<DetId, const unsigned int> *hitMap_;
  std::vector<const HGCRecHit *> hits_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
  ticl::association makeConnections(const edm::Handle<ticl::TracksterCollection> &tCH,
                                    const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                    const edm::Handle<SimClusterCollection> &sCCH) const;
};
