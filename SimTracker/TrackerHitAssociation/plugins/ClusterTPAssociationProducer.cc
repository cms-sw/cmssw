#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <utility>

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociationProducer.h"

ClusterTPAssociationProducer::ClusterTPAssociationProducer(const edm::ParameterSet & cfg) 
  : _verbose(cfg.getParameter<bool>("verbose")),
    _pixelSimLinkSrc(cfg.getParameter<edm::InputTag>("pixelSimLinkSrc")),
    _stripSimLinkSrc(cfg.getParameter<edm::InputTag>("stripSimLinkSrc")),
    _pixelClusterSrc(cfg.getParameter<edm::InputTag>("pixelClusterSrc")),
    _stripClusterSrc(cfg.getParameter<edm::InputTag>("stripClusterSrc")),
    _trackingParticleSrc(cfg.getParameter<edm::InputTag>("trackingParticleSrc"))
{
  produces<ClusterTPAssociationList>();
}

ClusterTPAssociationProducer::~ClusterTPAssociationProducer() {
}
		
void ClusterTPAssociationProducer::produce(edm::Event& iEvent, const edm::EventSetup& es) {
  std::auto_ptr<ClusterTPAssociationList> clusterTPList(new ClusterTPAssociationList);
 
  // Pixel DigiSimLink
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > sipixelSimLinks;
  iEvent.getByLabel(_pixelSimLinkSrc, sipixelSimLinks);

  // SiStrip DigiSimLink
  edm::Handle<edm::DetSetVector<StripDigiSimLink> > sistripSimLinks;
  iEvent.getByLabel(_stripSimLinkSrc, sistripSimLinks);

  // Pixel Cluster
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > pixelClusters;
  iEvent.getByLabel(_pixelClusterSrc, pixelClusters);

  // Strip Cluster
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > stripClusters;
  iEvent.getByLabel(_stripClusterSrc, stripClusters);

  // TrackingParticle
  edm::Handle<TrackingParticleCollection>  TPCollectionH;
  iEvent.getByLabel(_trackingParticleSrc,  TPCollectionH);

  // prepare temporary map between SimTrackId and TrackingParticle index
  std::map<std::pair<size_t, EncodedEventId>, TrackingParticleRef> mapping;
  for (TrackingParticleCollection::size_type itp = 0;
                                             itp < TPCollectionH.product()->size(); ++itp) {
    TrackingParticleRef trackingParticle(TPCollectionH, itp);

    // SimTracks inside TrackingParticle
    EncodedEventId eid(trackingParticle->eventId());
    //size_t index = 0;
    for (std::vector<SimTrack>::const_iterator itrk  = trackingParticle->g4Track_begin(); 
                                               itrk != trackingParticle->g4Track_end(); ++itrk) {
      std::pair<uint32_t, EncodedEventId> trkid(itrk->trackId(), eid);
      //std::cout << "creating map for id: " << trkid.first << " with tp: " << trackingParticle.key() << std::endl;
      mapping.insert(std::make_pair(trkid, trackingParticle));
    }
  }

  // Pixel Clusters 
  for (edmNew::DetSetVector<SiPixelCluster>::const_iterator iter  = pixelClusters->begin(); 
                                                            iter != pixelClusters->end(); ++iter) 
  {
    uint32_t detid = iter->id(); 
    DetId detId(detid);
    edmNew::DetSet<SiPixelCluster> link_pixel = (*iter);
    for (edmNew::DetSet<SiPixelCluster>::const_iterator di  = link_pixel.begin(); 
	                                                di != link_pixel.end(); ++di) 
    {
      const SiPixelCluster& cluster = (*di);
      edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> c_ref = 
          edmNew::makeRefTo(pixelClusters, di);

      std::set<std::pair<uint32_t, EncodedEventId> > simTkIds; 
      for (int irow = cluster.minPixelRow(); irow <= cluster.maxPixelRow(); ++irow) {
	for (int icol = cluster.minPixelCol(); icol <= cluster.maxPixelCol(); ++icol) {
	  uint32_t channel = PixelChannelIdentifier::pixelToChannel(irow, icol);
	  std::vector<std::pair<uint32_t, EncodedEventId> > trkid(getSimTrackId<PixelDigiSimLink>(sipixelSimLinks, detId, channel));
	  if (trkid.size()==0) continue; 
	  simTkIds.insert(trkid.begin(),trkid.end());
	}
      }
      for (std::set<std::pair<uint32_t, EncodedEventId> >::const_iterator iset  = simTkIds.begin(); 
                                                                          iset != simTkIds.end(); iset++) {
	auto ipos = mapping.find(*iset);
        if (ipos != mapping.end()) {
	  //std::cout << "cluster in detid: " << detid << " from tp: " << ipos->second.key() << " " << iset->first << std::endl;
          clusterTPList->push_back(std::make_pair(OmniClusterRef(c_ref), ipos->second));
        }
      }
    } 
  }

  // Strip Clusters
  for (edmNew::DetSetVector<SiStripCluster>::const_iterator iter  = stripClusters->begin(false); 
                                                            iter != stripClusters->end(false); ++iter) {
    if (!(*iter).isValid()) continue;
    uint32_t detid = iter->id();  
    DetId detId(detid);
    edmNew::DetSet<SiStripCluster> link_strip = (*iter);
    for (edmNew::DetSet<SiStripCluster>::const_iterator di  = link_strip.begin(); 
	                                                di != link_strip.end(); di++) {
      const SiStripCluster& cluster = (*di);
      edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> c_ref = 
        edmNew::makeRefTo(stripClusters, di);

      std::set<std::pair<uint32_t, EncodedEventId> > simTkIds; 
      int first  = cluster.firstStrip();     
      int last   = first + cluster.amplitudes().size();
   
      for (int istr = first; istr < last; ++istr) {
	std::vector<std::pair<uint32_t, EncodedEventId> > trkid(getSimTrackId<StripDigiSimLink>(sistripSimLinks, detId, istr));
        if (trkid.size()==0) continue; 
        simTkIds.insert(trkid.begin(),trkid.end());
      }
      for (std::set<std::pair<uint32_t, EncodedEventId> >::const_iterator iset  = simTkIds.begin(); 
                                                                          iset != simTkIds.end(); iset++) {
	auto ipos = mapping.find(*iset);
        if (ipos != mapping.end()) {
	  //std::cout << "cluster in detid: " << detid << " from tp: " << ipos->second.key() << " " << iset->first << std::endl;
          clusterTPList->push_back(std::make_pair(OmniClusterRef(c_ref), ipos->second));
        } 
      }
    } 
  }
  iEvent.put(clusterTPList);
}
template <typename T>
std::vector<std::pair<uint32_t, EncodedEventId> >
//std::pair<uint32_t, EncodedEventId>
ClusterTPAssociationProducer::getSimTrackId(const edm::Handle<edm::DetSetVector<T> >& simLinks,
                                            const DetId& detId, uint32_t channel) const 
{
  //std::pair<uint32_t, EncodedEventId> simTrkId;
  std::vector<std::pair<uint32_t, EncodedEventId> > simTrkId;
  auto isearch = simLinks->find(detId);
  if (isearch != simLinks->end()) {
    // Loop over DigiSimLink in this det unit
    edm::DetSet<T> link_detset = (*isearch);
    for (typename edm::DetSet<T>::const_iterator it  = link_detset.data.begin(); 
                                                 it != link_detset.data.end(); ++it) {
      if (channel == it->channel()) {
        simTrkId.push_back(std::make_pair(it->SimTrackId(), it->eventId()));
      } 
    }
  }
  return simTrkId;
}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ClusterTPAssociationProducer);
