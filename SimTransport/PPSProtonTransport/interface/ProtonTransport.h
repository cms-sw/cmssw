#ifndef PROTONTRANSPORT
#define PROTONTRANSPORT
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepMC/GenEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"

#include <vector>

class ProtonTransport {
public:
  ProtonTransport(const edm::ParameterSet& iConfig);
  ~ProtonTransport() {
    if (instance_)
      delete instance_;
  };

  std::vector<LHCTransportLink>& getCorrespondenceMap() { return instance_->getCorrespondenceMap(); }
  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) {
    instance_->process(ev, es, engine);
    instance_->addPartToHepMC(const_cast<HepMC::GenEvent*>(ev));
  }

private:
  BaseProtonTransport* instance_;
};
#endif
