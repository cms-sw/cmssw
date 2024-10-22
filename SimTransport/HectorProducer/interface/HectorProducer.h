#ifndef SimTransport_HectorProducer_H
#define SimTransport_HectorProducer_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class HepMCProduct;
}  // namespace edm

class HectorManager;

class Hector;

namespace HepMC {
  class GenEvent;
}
class HectorProducer : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit HectorProducer(edm::ParameterSet const &p);
  ~HectorProducer() override;
  void produce(edm::Event &e, const edm::EventSetup &c) override;

private:
  std::unique_ptr<Hector> m_Hector;

  edm::ESGetToken<HepPDT::ParticleDataTable, PDTRecord> tok_pdt_;
  edm::EDGetTokenT<edm::HepMCProduct> m_HepMC;
  bool m_verbosity;
  bool m_FP420Transport;
  bool m_ZDCTransport;
  int m_evtAnalysed;  //!< just to count events that have been analysed
};

#endif
