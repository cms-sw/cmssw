#ifndef SimTracker_TrackerHitAssociation_ClusterTPAssociationProducer_h
#define SimTracker_TrackerHitAssociation_ClusterTPAssociationProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

class EncodedEventId;

class ClusterTPAssociationProducer : public edm::stream::EDProducer<> 
{
public:
  //typedef std::pair<uint32_t, EncodedEventId> SimTrackIdentifier;
  typedef std::vector<OmniClusterRef> OmniClusterCollection;

  explicit ClusterTPAssociationProducer(const edm::ParameterSet&);
  ~ClusterTPAssociationProducer();

private:
  virtual void beginJob() {}
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() {}

  template <typename T>
  std::vector<std::pair<uint32_t, EncodedEventId> >
  getSimTrackId(const edm::Handle<edm::DetSetVector<T> >& simLinks, const DetId& detId, uint32_t channel) const;

  bool _verbose;
  edm::InputTag _pixelSimLinkSrc;
  edm::InputTag _stripSimLinkSrc;
  edm::InputTag _pixelClusterSrc;
  edm::InputTag _stripClusterSrc;
  edm::InputTag _trackingParticleSrc;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > sipixelSimLinksToken_;
  edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > sistripSimLinksToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClustersToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > stripClustersToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;

};
#endif
