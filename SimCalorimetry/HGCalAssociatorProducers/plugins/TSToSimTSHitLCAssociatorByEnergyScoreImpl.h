// Original Author: Leonardo Cristella

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace edm {
  class EDProductGetter;
}

namespace hgcal {

  struct detIdInfoInCluster {
    bool operator==(const detIdInfoInCluster &o) const { return clusterId == o.clusterId; };
    long unsigned int clusterId;
    float fraction;
  };

  struct detIdInfoInTrackster {
    bool operator==(const detIdInfoInTrackster &o) const { return tracksterId == o.tracksterId; };
    unsigned int tracksterId;
    long unsigned int clusterId;
    float fraction;
  };

  struct caloParticleOnLayer {
    unsigned int caloParticleId;
    float energy = 0;
    std::vector<std::pair<DetId, float>> hits_and_fractions;
    std::unordered_map<unsigned int, std::pair<float, float>> layerClusterIdToEnergyAndScore;
  };

  // This object connects a Trackster, identified through its id (tsId), with a vector of pairs containing all
  // the SimTracksters (via their ids (stIds)) that share at least one LayerCluster. In that pair
  // it stores the score (tsId->(stId,score)). Keep in mind that the association is not unique, since there could be
  // several instances of the same SimTrackster from several related SimClusters that each contributed to the same Trackster.
}  // namespace hgcal

class TSToSimTSHitLCAssociatorByEnergyScoreImpl : public hgcal::TracksterToSimTracksterHitLCAssociatorBaseImpl {
public:
  explicit TSToSimTSHitLCAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                                     bool,
                                                     std::shared_ptr<hgcal::RecHitTools>,
                                                     const std::unordered_map<DetId, const HGCRecHit *> *);

  hgcal::association_t makeConnections(const edm::Handle<ticl::TracksterCollection> &tCH,
                                       const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                       const edm::Handle<SimClusterCollection> &sCCH,
                                       const edm::Handle<CaloParticleCollection> &cPCH,
                                       const edm::Handle<ticl::TracksterCollection> &sTCH) const;

  hgcal::RecoToSimCollectionSimTracksters associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const override;

  hgcal::SimToRecoCollectionSimTracksters associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<SimClusterCollection> &sCCH,
      const edm::Handle<CaloParticleCollection> &cPCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const override;

private:
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> recHitTools_;
  const std::unordered_map<DetId, const HGCRecHit *> *hitMap_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
};
