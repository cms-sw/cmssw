// Original Author: Leonardo Cristella

#include <vector>
#include <map>
#include <unordered_map>
#include <memory>  // shared_ptr

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

namespace edm {
  class EDProductGetter;
}

namespace hgcal {
  struct lcInfoInTrackster {
    bool operator==(const lcInfoInTrackster &o) const { return clusterId == o.clusterId; };
    long unsigned int clusterId;
    float fraction;
    lcInfoInTrackster(long unsigned int cId, float fr) {
      clusterId = cId;
      fraction = fr;
    }
  };

  struct simTracksterOnLayer {
    unsigned int simTracksterId;
    float energy = 0;
    std::vector<std::pair<unsigned int, float>> lcs_and_fractions;
    std::unordered_map<int, std::pair<float, float>> tracksterIdToEnergyAndScore;
  };

  typedef std::vector<std::vector<std::pair<unsigned int, float>>> tracksterToSimTrackster;
  typedef std::vector<hgcal::simTracksterOnLayer> simTracksterToTrackster;
  typedef std::tuple<tracksterToSimTrackster, simTracksterToTrackster> association;
}  // namespace hgcal

class TSToSimTSAssociatorByEnergyScoreImpl : public hgcal::TracksterToSimTracksterAssociatorBaseImpl {
public:
  explicit TSToSimTSAssociatorByEnergyScoreImpl(edm::EDProductGetter const &,
                                                bool,
                                                std::shared_ptr<hgcal::RecHitTools>,
                                                const std::unordered_map<DetId, const HGCRecHit *> *);

  hgcal::RecoToSimCollectionSimTracksters associateRecoToSim(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const override;

  hgcal::SimToRecoCollectionSimTracksters associateSimToReco(
      const edm::Handle<ticl::TracksterCollection> &tCH,
      const edm::Handle<reco::CaloClusterCollection> &lCCH,
      const edm::Handle<ticl::TracksterCollection> &sTCH) const override;

private:
  const bool hardScatterOnly_;
  std::shared_ptr<hgcal::RecHitTools> recHitTools_;
  const std::unordered_map<DetId, const HGCRecHit *> *hitMap_;
  unsigned layers_;
  edm::EDProductGetter const *productGetter_;
  hgcal::association makeConnections(const edm::Handle<ticl::TracksterCollection> &tCH,
                                     const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                     const edm::Handle<ticl::TracksterCollection> &sTCH) const;
};
