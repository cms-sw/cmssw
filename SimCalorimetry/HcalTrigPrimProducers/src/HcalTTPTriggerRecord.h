#ifndef HcalTrigPrimProducers_HcalTTPTriggerRecord_h
#define HcalTrigPrimProducers_HcalTTPTriggerRecord_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class HcalTTPTriggerRecord : public edm::stream::EDProducer<>
{
public:

    explicit HcalTTPTriggerRecord(const edm::ParameterSet& ps);
    virtual ~HcalTTPTriggerRecord();
    
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
     
    edm::EDGetTokenT<HcalTTPDigiCollection> tok_ttp_; 
    std::vector<unsigned int> ttpBits_ ;
    std::vector<std::string> names_ ; 

};

#endif


