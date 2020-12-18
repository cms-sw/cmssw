#include <memory>
#include <vector>
#include <utility>

#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"

#include "ClusterSLOnGPU.h"

class ClusterTPAssociationProducerCUDA : public edm::global::EDProducer<> {
public:
  typedef std::vector<OmniClusterRef> OmniClusterCollection;

  using ClusterSLGPU = trackerHitAssociationHeterogeneous::ClusterSLView;
  using Clus2TP = ClusterSLGPU::Clus2TP;
  using ProductCUDA = trackerHitAssociationHeterogeneous::ProductCUDA;

  explicit ClusterTPAssociationProducerCUDA(const edm::ParameterSet &);
  ~ClusterTPAssociationProducerCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID streamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const override;

  std::map<std::pair<size_t, EncodedEventId>, TrackingParticleRef> makeMap(const edm::Event &iEvent) const;

  template <typename T>
  std::vector<std::pair<uint32_t, EncodedEventId>> getSimTrackId(const edm::Handle<edm::DetSetVector<T>> &simLinks,
                                                                 const DetId &detId,
                                                                 uint32_t channel) const;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> sipixelSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink>> sistripSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> siphase2OTSimLinksToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClustersToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripClustersToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D>> phase2OTClustersToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;

  edm::EDGetTokenT<cms::cuda::Product<SiPixelDigisCUDA>> tGpuDigis;
  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DCUDA>> tGpuHits;

  edm::EDPutTokenT<cms::cuda::Product<ProductCUDA>> tokenGPUProd_;

  clusterSLOnGPU::Kernel m_gpuAlgo;
};

ClusterTPAssociationProducerCUDA::ClusterTPAssociationProducerCUDA(const edm::ParameterSet &cfg)
    : sipixelSimLinksToken_(
          consumes<edm::DetSetVector<PixelDigiSimLink>>(cfg.getParameter<edm::InputTag>("pixelSimLinkSrc"))),
      sistripSimLinksToken_(
          consumes<edm::DetSetVector<StripDigiSimLink>>(cfg.getParameter<edm::InputTag>("stripSimLinkSrc"))),
      siphase2OTSimLinksToken_(
          consumes<edm::DetSetVector<PixelDigiSimLink>>(cfg.getParameter<edm::InputTag>("phase2OTSimLinkSrc"))),
      pixelClustersToken_(
          consumes<edmNew::DetSetVector<SiPixelCluster>>(cfg.getParameter<edm::InputTag>("pixelClusterSrc"))),
      stripClustersToken_(
          consumes<edmNew::DetSetVector<SiStripCluster>>(cfg.getParameter<edm::InputTag>("stripClusterSrc"))),
      phase2OTClustersToken_(consumes<edmNew::DetSetVector<Phase2TrackerCluster1D>>(
          cfg.getParameter<edm::InputTag>("phase2OTClusterSrc"))),
      trackingParticleToken_(
          consumes<TrackingParticleCollection>(cfg.getParameter<edm::InputTag>("trackingParticleSrc"))),
      tGpuDigis(consumes<cms::cuda::Product<SiPixelDigisCUDA>>(
          cfg.getParameter<edm::InputTag>("heterogeneousPixelDigiClusterSrc"))),
      tGpuHits(consumes<cms::cuda::Product<TrackingRecHit2DCUDA>>(
          cfg.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"))),
      m_gpuAlgo(cfg.getParameter<bool>("dumpCSV")) {
  tokenGPUProd_ = produces<cms::cuda::Product<ProductCUDA>>();
}

void ClusterTPAssociationProducerCUDA::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("simTrackSrc", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("pixelSimLinkSrc", edm::InputTag("simSiPixelDigis"));
  desc.add<edm::InputTag>("stripSimLinkSrc", edm::InputTag("simSiStripDigis"));
  desc.add<edm::InputTag>("phase2OTSimLinkSrc", edm::InputTag("simSiPixelDigis", "Tracker"));
  desc.add<edm::InputTag>("pixelClusterSrc", edm::InputTag("siPixelClusters"));
  desc.add<edm::InputTag>("stripClusterSrc", edm::InputTag("siStripClusters"));
  desc.add<edm::InputTag>("phase2OTClusterSrc", edm::InputTag("siPhase2Clusters"));
  desc.add<edm::InputTag>("trackingParticleSrc", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("heterogeneousPixelDigiClusterSrc", edm::InputTag("siPixelClustersPreSplittingCUDA"));
  desc.add<edm::InputTag>("heterogeneousPixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingCUDA"));

  desc.add<bool>("dumpCSV", false);

  descriptions.add("tpClusterProducerCUDADefault", desc);
}

std::map<std::pair<size_t, EncodedEventId>, TrackingParticleRef> ClusterTPAssociationProducerCUDA::makeMap(
    const edm::Event &iEvent) const {
  // TrackingParticle
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleToken_, TPCollectionH);

  // prepare temporary map between SimTrackId and TrackingParticle index
  std::map<std::pair<size_t, EncodedEventId>, TrackingParticleRef> mapping;
  for (TrackingParticleCollection::size_type itp = 0; itp < TPCollectionH.product()->size(); ++itp) {
    TrackingParticleRef trackingParticle(TPCollectionH, itp);

    // SimTracks inside TrackingParticle
    EncodedEventId eid(trackingParticle->eventId());
    for (auto itrk = trackingParticle->g4Track_begin(); itrk != trackingParticle->g4Track_end(); ++itrk) {
      std::pair<uint32_t, EncodedEventId> trkid(itrk->trackId(), eid);
      //std::cout << "creating map for id: " << trkid.first << " with tp: " << trackingParticle.key() << std::endl;
      mapping.insert(std::make_pair(trkid, trackingParticle));
    }
  }
  return mapping;
}

void ClusterTPAssociationProducerCUDA::produce(edm::StreamID streamID,
                                               edm::Event &iEvent,
                                               const edm::EventSetup &iSetup) const {
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom);

  // Pixel DigiSimLink
  edm::Handle<edm::DetSetVector<PixelDigiSimLink>> sipixelSimLinks;
  //  iEvent.getByLabel(_pixelSimLinkSrc, sipixelSimLinks);
  iEvent.getByToken(sipixelSimLinksToken_, sipixelSimLinks);

  // TrackingParticle
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleToken_, TPCollectionH);

  auto mapping = makeMap(iEvent);

  edm::Handle<cms::cuda::Product<SiPixelDigisCUDA>> gd;
  iEvent.getByToken(tGpuDigis, gd);
  edm::Handle<cms::cuda::Product<TrackingRecHit2DCUDA>> gh;
  iEvent.getByToken(tGpuHits, gh);

  cms::cuda::ScopedContextProduce ctx{*gd};
  auto const &gDigis = ctx.get(*gd);
  auto const &gHits = ctx.get(*gh);
  auto ndigis = gDigis.nDigis();
  auto nhits = gHits.nHits();

  std::vector<Clus2TP> digi2tp;
  digi2tp.push_back({{0, 0, 0, 0, 0, 0, 0}});  // put at 0 0
  for (auto const &links : *sipixelSimLinks) {
    DetId detId(links.detId());
    const GeomDetUnit *genericDet = geom->idToDetUnit(detId);
    uint32_t gind = genericDet->index();
    for (auto const &link : links) {
      if (link.fraction() < 0.5f) {
        continue;
      }
      auto tkid = std::make_pair(link.SimTrackId(), link.eventId());
      auto ipos = mapping.find(tkid);
      if (ipos != mapping.end()) {
        uint32_t pt = 1000 * (*ipos).second->pt();
        uint32_t eta = 10000 * (*ipos).second->eta();
        uint32_t z0 = 10000 * (*ipos).second->vz();  // in um
        uint32_t r0 = 10000 * std::sqrt((*ipos).second->vx() * (*ipos).second->vx() +
                                        (*ipos).second->vy() * (*ipos).second->vy());  // in um
        digi2tp.push_back({{gind, uint32_t(link.channel()), (*ipos).second.key(), pt, eta, z0, r0}});
      }
    }
  }

  std::sort(digi2tp.begin(), digi2tp.end());

  ctx.emplace(iEvent,
              tokenGPUProd_,
              m_gpuAlgo.makeAsync(gDigis, ndigis, gHits, digi2tp.data(), nhits, digi2tp.size(), ctx.stream()));
}

template <typename T>
std::vector<std::pair<uint32_t, EncodedEventId>>
//std::pair<uint32_t, EncodedEventId>
ClusterTPAssociationProducerCUDA::getSimTrackId(const edm::Handle<edm::DetSetVector<T>> &simLinks,
                                                const DetId &detId,
                                                uint32_t channel) const {
  //std::pair<uint32_t, EncodedEventId> simTrkId;
  std::vector<std::pair<uint32_t, EncodedEventId>> simTrkId;
  auto isearch = simLinks->find(detId);
  if (isearch != simLinks->end()) {
    // Loop over DigiSimLink in this det unit
    edm::DetSet<T> link_detset = (*isearch);
    for (typename edm::DetSet<T>::const_iterator it = link_detset.data.begin(); it != link_detset.data.end(); ++it) {
      if (channel == it->channel()) {
        simTrkId.push_back(std::make_pair(it->SimTrackId(), it->eventId()));
      }
    }
  }
  return simTrkId;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ClusterTPAssociationProducerCUDA);
