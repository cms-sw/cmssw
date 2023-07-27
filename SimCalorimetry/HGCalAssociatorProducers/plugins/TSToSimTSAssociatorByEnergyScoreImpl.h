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
  // This structure is used for SimTracksters storing its id and the energy fraction of
  // a LayerCluster that is related to that SimTrackster. Be careful not to be confused by the fact that
  // similar structs are used in other HGCAL associators where the fraction is a hit's fraction.
  // The meaning of the operator is extremely important since this struct will be used inside maps and
  // other containers and when searching for one particular occurence only the clusterId member will be
  // used in the check, skipping the fraction part.
  struct lcInfoInTrackster {
    bool operator==(const lcInfoInTrackster &o) const { return clusterId == o.clusterId; };
    long unsigned int clusterId;
    float fraction;
    lcInfoInTrackster(long unsigned int cId, float fr) {
      clusterId = cId;
      fraction = fr;
    }
  };

  // In this structure, although it contains LayerClusters and per layer information through them,
  // most of the information is 3D information. For a simTrackster it stores:
  // 1. Its id: simTracksterId.
  // 2. The energy related to the SimTrackster. It is the sum of the LayerClusters energy in which a SimCluster
  //    contributed. Therefore, there will be energy from each LayerCluster that is disregarded.
  // 3. lcs_and_fractions: This is a vector of pairs. The pair is build from the LayerCluster id and the energy
  //    fraction of that LayerCluster which contributed to SimTrackster under study. So, be careful this is not the
  //    fraction of the hits. This is the fraction of the LayerCluster's energy in which the SimCluster contributed.
  //    This quantifies the statement above in 2 about the disregarded energy, by exactly storing the ratio of the
  //    reconstructed from SimCluster energy over the total LayerCluster energy.
  // 4. A map to save the tracksters id (id is the key) that have at least one LayerCluster in common with the
  //    SimTrackster under study together with the energy and score. Energy in this case is defined as the sum of all
  //    LayerClusters (shared between the SimTrackster and the trackster) energy (coming from SimCluster of the SimTrackster)
  //    times the LayerCluster's fraction in trackster.
  struct simTracksterOnLayer {
    unsigned int simTracksterId;
    float energy = 0;
    std::vector<std::pair<unsigned int, float>> lcs_and_fractions;
    std::unordered_map<int, std::pair<float, float>> tracksterIdToEnergyAndScore;
  };

  // This object connects a Trackster, identified through its id (tsId), with a vector of pairs containing all
  // the SimTracksters (via their ids (stIds)) that share at least one LayerCluster. In that pair
  // it stores the score (tsId->(stId,score)). Keep in mind that the association is not unique, since there could be
  // several instances of the same SimTrackster from several related SimClusters that each contributed to the same Trackster.
  typedef std::vector<std::vector<std::pair<unsigned int, std::pair<float, float>>>> tracksterToSimTrackster;
  // This is used to save the simTracksterOnLayer structure for all simTracksters.
  // It is not exactly what is returned outside, but out of its entries, the output object is build.
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
