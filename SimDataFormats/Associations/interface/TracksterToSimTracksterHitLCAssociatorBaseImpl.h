#ifndef SimDataFormats_Associations_TracksterToSimTracksterHitLCAssociatorBaseImpl_h
#define SimDataFormats_Associations_TracksterToSimTracksterHitLCAssociatorBaseImpl_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
typedef std::vector<SimCluster> SimClusterCollection;
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"

namespace hgcal {

  enum validationType { Linking = 0, PatternRecognition, PatternRecognition_CP };

  typedef std::vector<std::vector<std::pair<float, float>>> sharedEnergyAndScore_t;
  // This is used to save the simTracksterOnLayer structure for all simTracksters.
  // It is not exactly what is returned outside, but out of its entries, the output object is build.
  typedef std::tuple<sharedEnergyAndScore_t, sharedEnergyAndScore_t> association_t;

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<ticl::TracksterCollection, ticl::TracksterCollection, std::pair<float, float>>>
      SimToRecoCollectionSimTracksters;
  typedef SimToRecoCollectionSimTracksters RecoToSimCollectionSimTracksters;

  class TracksterToSimTracksterHitLCAssociatorBaseImpl {
  public:
    /// Constructor
    TracksterToSimTracksterHitLCAssociatorBaseImpl();
    /// Destructor
    virtual ~TracksterToSimTracksterHitLCAssociatorBaseImpl();

    hgcal::association_t makeConnections(const edm::Handle<ticl::TracksterCollection> &tCH,
                                         const edm::Handle<reco::CaloClusterCollection> &lCCH,
                                         const edm::Handle<SimClusterCollection> &sCCH,
                                         const edm::Handle<CaloParticleCollection> &cPCH,
                                         const edm::Handle<ticl::TracksterCollection> &sTCH) const;

    /// Associate a Trackster to SimClusters
    virtual hgcal::RecoToSimCollectionSimTracksters associateRecoToSim(
        const edm::Handle<ticl::TracksterCollection> &tCH,
        const edm::Handle<reco::CaloClusterCollection> &lCCH,
        const edm::Handle<SimClusterCollection> &sCCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH) const;

    /// Associate a SimCluster to Tracksters
    virtual hgcal::SimToRecoCollectionSimTracksters associateSimToReco(
        const edm::Handle<ticl::TracksterCollection> &tCH,
        const edm::Handle<reco::CaloClusterCollection> &lCCH,
        const edm::Handle<SimClusterCollection> &sCCH,
        const edm::Handle<CaloParticleCollection> &cPCH,
        const edm::Handle<ticl::TracksterCollection> &sTCH) const;
  };
}  // namespace hgcal

#endif
