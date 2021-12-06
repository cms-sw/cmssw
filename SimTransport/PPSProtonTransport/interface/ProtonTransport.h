#ifndef PROTONTRANSPORT
#define PROTONTRANSPORT
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepMC/GenEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"

#include <memory>
#include <vector>

class ProtonTransport {
public:
  ProtonTransport(const edm::ParameterSet& iConfig, edm::ConsumesCollector iC);
  ~ProtonTransport() = default;

  std::vector<LHCTransportLink>& getCorrespondenceMap() { return instance_->getCorrespondenceMap(); }
  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) {
    instance_->process(ev, es, engine);
  }
  void addPartToHepMC(const HepMC::GenEvent* iev, HepMC::GenEvent* ev) { instance_->addPartToHepMC(iev, ev); }

private:
  std::unique_ptr<BaseProtonTransport> instance_;
};
#endif
