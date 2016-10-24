#ifndef SimTransport_HectorProducerForCTPPS_H
#define SimTransport_HectorProducerForCTPPS_H
 
#include "FWCore/Framework/interface/EDProducer.h"
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

class HectorForCTPPS;

namespace HepMC {
  class GenEvent;
}
class HectorProducerForCTPPS : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns>
{
    public:
        explicit HectorProducerForCTPPS(edm::ParameterSet const & p);    //!< default constructor
        virtual ~HectorProducerForCTPPS();   //!< default destructor
        virtual void beginRun(const edm::Run & r,const edm::EventSetup& c) override;
        virtual void endRun(const edm::Run & r,const edm::EventSetup& c) override;
        virtual void produce(edm::Event & e, const edm::EventSetup & c)  override;
    private:
        HepMC::GenEvent * evt_;
        HectorForCTPPS * hector_ctpps;
  
        std::string m_InTag;
        edm::EDGetTokenT<edm::HepMCProduct> m_InTagToken;

        bool m_verbosity;
        bool m_CTPPSTransport;
        int  eventsAnalysed; //!< just to count events that have been analysed
};

#endif

