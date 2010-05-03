#ifndef HcalTrigPrimProducers_HcalTTPDigiProducer_h
#define HcalTrigPrimProducers_HcalTTPDigiProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalTTPDigiProducer : public edm::EDProducer
{
public:

  explicit HcalTTPDigiProducer(const edm::ParameterSet& ps);
  virtual ~HcalTTPDigiProducer();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:

    bool isMasked(HcalDetId id) ; 
    bool decision(int nP, int nM, int bit) ; 
    
    edm::InputTag hfDigis_ ;
    std::vector<unsigned int> maskedChannels_ ; 
    std::string bit_[4] ;
    int calc_[4] ; 
    int nHits_[4], nHFp_[4], nHFm_[4] ;
    char pReq_[4], mReq_[4], pmLogic_[4] ; 
    int samples_, presamples_ ;
    int fwAlgo_ ; 
    int iEtaMin_, iEtaMax_ ;
    unsigned int threshold_ ;

    int SoI_ ;
    
    static const int inputs_[] ;
};

#endif


