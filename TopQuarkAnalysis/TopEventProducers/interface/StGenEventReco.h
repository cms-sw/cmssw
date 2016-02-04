#include <memory>
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class StGenEventReco : public edm::EDProducer {
 public:

  explicit StGenEventReco(const edm::ParameterSet&);
  ~StGenEventReco();
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  
  void fillInitialPartons(const reco::GenParticle*, std::vector<const reco::GenParticle*>&);
  
 private:
  
  edm::InputTag src_, init_; 
};
