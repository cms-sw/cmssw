#include <memory>
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

class TtGenEventReco : public edm::EDProducer {
 public:
  explicit TtGenEventReco(const edm::ParameterSet&);
  ~TtGenEventReco();
  virtual void produce(edm::Event&, const edm::EventSetup&);
 private:
  edm::InputTag src_;  
};
