#ifndef SimGeneral_TrackingAnalysis_SimHitTPAssociationProducer_h
#define SimGeneral_TrackingAnalysis_SimHitTPAssociationProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class SimHitTPAssociationProducer : public edm::EDProducer 
{
public:

  typedef std::pair<TrackingParticleRef, TrackPSimHitRef> SimHitTPPair;
  typedef std::vector<SimHitTPPair> SimHitTPAssociationList;

  explicit SimHitTPAssociationProducer(const edm::ParameterSet&);
  ~SimHitTPAssociationProducer();

  static bool simHitTPAssociationListGreater(SimHitTPPair i,SimHitTPPair j) { return (i.first.key()>j.first.key()); }

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<edm::InputTag> _simHitSrc;
  edm::InputTag _trackingParticleSrc;
};
#endif
