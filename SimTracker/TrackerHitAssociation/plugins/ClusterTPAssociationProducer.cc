#include <memory>
#include <vector>
#include <utility>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Track/interface/UniqueSimTrackId.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

namespace {

  template <typename T>
  void getSimTrackId(std::vector<UniqueSimTrackId>& simTrkId,
                     const edm::Handle<edm::DetSetVector<T> >& simLinks,
                     const DetId& detId,
                     uint32_t channel) {
    auto isearch = simLinks->find(detId);
    if (isearch != simLinks->end()) {
      // Loop over DigiSimLink in this det unit
      edm::DetSet<T> link_detset = (*isearch);
      for (typename edm::DetSet<T>::const_iterator it = link_detset.data.begin(); it != link_detset.data.end(); ++it) {
        if (channel == it->channel()) {
          simTrkId.emplace_back(it->SimTrackId(), it->eventId());
        }
      }
    }
  }

}  // namespace

class ClusterTPAssociationProducer final : public edm::global::EDProducer<> {
public:
  using OmniClusterCollection = std::vector<OmniClusterRef>;

  explicit ClusterTPAssociationProducer(const edm::ParameterSet&);
  ~ClusterTPAssociationProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > sipixelSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > sistripSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > siphase2OTSimLinksToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClustersToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D> > phase2OTClustersToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;
  bool throwOnMissingCollections_;
};

ClusterTPAssociationProducer::ClusterTPAssociationProducer(const edm::ParameterSet& cfg)
    : sipixelSimLinksToken_(
          consumes<edm::DetSetVector<PixelDigiSimLink> >(cfg.getParameter<edm::InputTag>("pixelSimLinkSrc"))),
      sistripSimLinksToken_(
          consumes<edm::DetSetVector<StripDigiSimLink> >(cfg.getParameter<edm::InputTag>("stripSimLinkSrc"))),
      siphase2OTSimLinksToken_(
          consumes<edm::DetSetVector<PixelDigiSimLink> >(cfg.getParameter<edm::InputTag>("phase2OTSimLinkSrc"))),
      pixelClustersToken_(
          consumes<edmNew::DetSetVector<SiPixelCluster> >(cfg.getParameter<edm::InputTag>("pixelClusterSrc"))),
      stripClustersToken_(
          consumes<edmNew::DetSetVector<SiStripCluster> >(cfg.getParameter<edm::InputTag>("stripClusterSrc"))),
      phase2OTClustersToken_(consumes<edmNew::DetSetVector<Phase2TrackerCluster1D> >(
          cfg.getParameter<edm::InputTag>("phase2OTClusterSrc"))),
      trackingParticleToken_(
          consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticleSrc"))),
      throwOnMissingCollections_(cfg.getParameter<bool>("throwOnMissingCollections")) {
  produces<ClusterTPAssociation>();
}

void ClusterTPAssociationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simTrackSrc", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis"));
  desc.add<edm::InputTag>("stripSimLinkSrc", edm::InputTag("simSiStripDigis"));
  desc.add<edm::InputTag>("phase2OTSimLinkSrc", edm::InputTag("simSiPixelDigis", "Tracker"));
  desc.add<edm::InputTag>("pixelClusterSrc", edm::InputTag("siPixelClusters"));
  desc.add<edm::InputTag>("stripClusterSrc", edm::InputTag("siStripClusters"));
  desc.add<edm::InputTag>("phase2OTClusterSrc", edm::InputTag("siPhase2Clusters"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<bool>("throwOnMissingCollections", true);
  descriptions.add("tpClusterProducerDefault", desc);
}

void ClusterTPAssociationProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& es) const {
  // Pixel Cluster
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
  bool foundPixelClusters = iEvent.getByToken(pixelClustersToken_, pixelClusters);

  // Strip Cluster
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
  bool foundStripClusters = iEvent.getByToken(stripClustersToken_, stripClusters);

  // Phase2 Cluster
  edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D> > phase2OTClusters;
  bool foundPhase2OTClusters = iEvent.getByToken(phase2OTClustersToken_, phase2OTClusters);

  // Pixel DigiSimLink
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > sipixelSimLinks;
  auto pixelSimLinksFound = iEvent.getByToken(sipixelSimLinksToken_, sipixelSimLinks);
  if (not throwOnMissingCollections_ and foundPixelClusters and not pixelSimLinksFound) {
    auto clusterTPList = std::make_unique<ClusterTPAssociation>();
    iEvent.put(std::move(clusterTPList));
    return;
  }

  // SiStrip DigiSimLink
  edm::Handle<edm::DetSetVector<StripDigiSimLink> > sistripSimLinks;
  auto stripSimLinksFound = iEvent.getByToken(sistripSimLinksToken_, sistripSimLinks);
  if (not throwOnMissingCollections_ and foundStripClusters and not stripSimLinksFound) {
    auto clusterTPList = std::make_unique<ClusterTPAssociation>();
    iEvent.put(std::move(clusterTPList));
    return;
  }

  // Phase2 OT DigiSimLink
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > siphase2OTSimLinks;
  auto phase2OTSimLinksFound = iEvent.getByToken(siphase2OTSimLinksToken_, siphase2OTSimLinks);
  if (not throwOnMissingCollections_ and foundPhase2OTClusters and not phase2OTSimLinksFound) {
    auto clusterTPList = std::make_unique<ClusterTPAssociation>();
    iEvent.put(std::move(clusterTPList));
    return;
  }

  // TrackingParticle
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  auto tpFound = iEvent.getByToken(trackingParticleToken_, TPCollectionH);
  if (not throwOnMissingCollections_ and not tpFound) {
    auto clusterTPList = std::make_unique<ClusterTPAssociation>();
    iEvent.put(std::move(clusterTPList));
    return;
  }

  auto clusterTPList = std::make_unique<ClusterTPAssociation>(TPCollectionH);

  // prepare temporary map between SimTrackId and TrackingParticle index
  std::unordered_map<UniqueSimTrackId, TrackingParticleRef, UniqueSimTrackIdHash> mapping;
  auto const& tpColl = *TPCollectionH.product();
  for (TrackingParticleCollection::size_type itp = 0; itp < tpColl.size(); ++itp) {
    TrackingParticleRef trackingParticleRef(TPCollectionH, itp);
    auto const& trackingParticle = tpColl[itp];
    // SimTracks inside TrackingParticle
    EncodedEventId eid(trackingParticle.eventId());
    //size_t index = 0;
    for (auto const& trk : trackingParticle.g4Tracks()) {
      UniqueSimTrackId trkid(trk.trackId(), eid);
      //std::cout << "creating map for id: " << trkid.first << " with tp: " << trackingParticle.key() << std::endl;
      mapping.insert(std::make_pair(trkid, trackingParticleRef));
    }
  }

  std::unordered_set<UniqueSimTrackId, UniqueSimTrackIdHash> simTkIds;
  std::vector<UniqueSimTrackId> trkid;
  if (foundPixelClusters) {
    // Pixel Clusters
    clusterTPList->addKeyID(pixelClusters.id());
    for (edmNew::DetSetVector<SiPixelCluster>::const_iterator iter = pixelClusters->begin();
         iter != pixelClusters->end();
         ++iter) {
      uint32_t detid = iter->id();
      DetId detId(detid);
      edmNew::DetSet<SiPixelCluster> link_pixel = (*iter);
      for (edmNew::DetSet<SiPixelCluster>::const_iterator di = link_pixel.begin(); di != link_pixel.end(); ++di) {
        const SiPixelCluster& cluster = (*di);
        edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> c_ref = edmNew::makeRefTo(pixelClusters, di);

        simTkIds.clear();
        for (int irow = cluster.minPixelRow(); irow <= cluster.maxPixelRow(); ++irow) {
          for (int icol = cluster.minPixelCol(); icol <= cluster.maxPixelCol(); ++icol) {
            uint32_t channel = PixelChannelIdentifier::pixelToChannel(irow, icol);
            trkid.clear();
            getSimTrackId<PixelDigiSimLink>(trkid, sipixelSimLinks, detId, channel);
            simTkIds.insert(trkid.begin(), trkid.end());
          }
        }
        for (auto iset = simTkIds.begin(); iset != simTkIds.end(); iset++) {
          auto ipos = mapping.find(*iset);
          if (ipos != mapping.end()) {
            //std::cout << "cluster in detid: " << detid << " from tp: " << ipos->second.key() << " " << iset->first << std::endl;
            clusterTPList->emplace_back(OmniClusterRef(c_ref), ipos->second);
          }
        }
      }
    }
  }

  if (foundStripClusters) {
    // Strip Clusters
    clusterTPList->addKeyID(stripClusters.id());
    for (edmNew::DetSetVector<SiStripCluster>::const_iterator iter = stripClusters->begin(false),
                                                              eter = stripClusters->end(false);
         iter != eter;
         ++iter) {
      if (!(*iter).isValid())
        continue;
      uint32_t detid = iter->id();
      DetId detId(detid);
      edmNew::DetSet<SiStripCluster> link_strip = (*iter);
      for (edmNew::DetSet<SiStripCluster>::const_iterator di = link_strip.begin(); di != link_strip.end(); di++) {
        const SiStripCluster& cluster = (*di);
        edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> c_ref = edmNew::makeRefTo(stripClusters, di);

        simTkIds.clear();
        int first = cluster.firstStrip();
        int last = first + cluster.amplitudes().size();

        for (int istr = first; istr < last; ++istr) {
          trkid.clear();
          getSimTrackId<StripDigiSimLink>(trkid, sistripSimLinks, detId, istr);
          simTkIds.insert(trkid.begin(), trkid.end());
        }
        for (auto iset = simTkIds.begin(); iset != simTkIds.end(); iset++) {
          auto ipos = mapping.find(*iset);
          if (ipos != mapping.end()) {
            //std::cout << "cluster in detid: " << detid << " from tp: " << ipos->second.key() << " " << iset->first << std::endl;
            clusterTPList->emplace_back(OmniClusterRef(c_ref), ipos->second);
          }
        }
      }
    }
  }

  if (foundPhase2OTClusters) {
    // Phase2 Clusters
    clusterTPList->addKeyID(phase2OTClusters.id());
    if (phase2OTClusters.isValid()) {
      for (edmNew::DetSetVector<Phase2TrackerCluster1D>::const_iterator iter = phase2OTClusters->begin(false),
                                                                        eter = phase2OTClusters->end(false);
           iter != eter;
           ++iter) {
        if (!(*iter).isValid())
          continue;
        uint32_t detid = iter->id();
        DetId detId(detid);
        edmNew::DetSet<Phase2TrackerCluster1D> link_phase2 = (*iter);
        for (edmNew::DetSet<Phase2TrackerCluster1D>::const_iterator di = link_phase2.begin(); di != link_phase2.end();
             di++) {
          const Phase2TrackerCluster1D& cluster = (*di);
          edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> c_ref =
              edmNew::makeRefTo(phase2OTClusters, di);

          simTkIds.clear();

          for (unsigned int istr(0); istr < cluster.size(); ++istr) {
            uint32_t channel = Phase2TrackerDigi::pixelToChannel(cluster.firstRow() + istr, cluster.column());
            trkid.clear();
            getSimTrackId<PixelDigiSimLink>(trkid, siphase2OTSimLinks, detId, channel);
            simTkIds.insert(trkid.begin(), trkid.end());
          }

          for (auto iset = simTkIds.begin(); iset != simTkIds.end(); iset++) {
            auto ipos = mapping.find(*iset);
            if (ipos != mapping.end()) {
              clusterTPList->emplace_back(OmniClusterRef(c_ref), ipos->second);
            }
          }
        }
      }
    }
  }
  clusterTPList->sortAndUnique();
  iEvent.put(std::move(clusterTPList));
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ClusterTPAssociationProducer);
