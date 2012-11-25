#ifndef TauAnalysis_MCEmbeddingTools_MixedGenMEtProducer_h
#define TauAnalysis_MCEmbeddingTools_MixedGenMEtProducer_h

/** \class MixedGenMEtProducer
 *
 * Produce generator level missing transverse energy
 * for "hybrid" event consisting of of gen. Z --> mu+ mu- event plus embedded simulated tau decay products
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.3 $
 *
 * $Id: MixedGenMEtProducer.h,v 1.3 2012/11/07 17:27:30 aburgmei Exp $
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class MixedGenMEtProducer : public edm::EDProducer 
{
 public:
  explicit MixedGenMEtProducer(const edm::ParameterSet&);
  ~MixedGenMEtProducer() {}

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcGenParticles1_;
  edm::InputTag srcGenParticles2_;
  edm::InputTag srcGenRemovedMuons_;
};

#endif

