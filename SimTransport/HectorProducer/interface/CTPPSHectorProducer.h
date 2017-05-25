#ifndef SimTransport_CTPPSHectorProducer_H
#define SimTransport_CTPPSHectorProducer_H
 
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

class CTPPSHector;

namespace HepMC {
  class GenEvent;
}
class CTPPSHectorProducer : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns>
{
    public:
        explicit CTPPSHectorProducer(edm::ParameterSet const & p);    //!< default constructor
        virtual ~CTPPSHectorProducer();   //!< default destructor
        virtual void beginRun(const edm::Run & r,const edm::EventSetup& c) override;
        virtual void endRun(const edm::Run & r,const edm::EventSetup& c) override;
        virtual void produce(edm::Event & e, const edm::EventSetup & c)  override;
    private:
        //HepMC::GenEvent * evt_;
        CTPPSHector * hector_ctpps;
  
        std::string m_InTag;
        edm::EDGetTokenT<edm::HepMCProduct> m_InTagToken;

        bool m_verbosity;
        bool m_CTPPSTransport;
        int  eventsAnalysed; //!< just to count events that have been analysed
};

#endif

