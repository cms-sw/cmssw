#ifndef SimTracker_TrackerHitAssociation_ClusterTPAssociationProducer_h
#define SimTracker_TrackerHitAssociation_ClusterTPAssociationProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
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

class EncodedEventId;

class ClusterTPAssociationProducer : public edm::EDProducer 
{
public:
  //typedef std::pair<uint32_t, EncodedEventId> SimTrackIdentifier;
  typedef std::vector<std::pair<OmniClusterRef, TrackingParticleRef> > ClusterTPAssociationList;
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
};
#endif
