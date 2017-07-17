#ifndef SimTransport_HectorProducer_H
#define SimTransport_HectorProducer_H
 
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class HepMCProduct;
}

class HectorManager;

class Hector;

namespace HepMC {
  class GenEvent;
}
class HectorProducer : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns>
{
 public:
  explicit HectorProducer(edm::ParameterSet const & p);  
  virtual ~HectorProducer();  
  virtual void beginRun(const edm::Run & r,const edm::EventSetup& c) override;
  virtual void endRun(const edm::Run & r,const edm::EventSetup& c) override;
  virtual void produce(edm::Event & e, const edm::EventSetup& c) override;

 private:

  HepMC::GenEvent * evt_;
  Hector* m_Hector;
  
  edm::EDGetTokenT<edm::HepMCProduct> m_HepMC;
  bool m_verbosity;
  bool m_FP420Transport;
  bool m_ZDCTransport;
  int  m_evtAnalysed; //!< just to count events that have been analysed
};

#endif

