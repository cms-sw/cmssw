//
// $Id: TopLeptonCountFilter.h,v 1.1.4.1 2007/10/30 01:17:33 lowette Exp $
//

#ifndef TopObjectProducers_TopLeptonCountFilter_h
#define TopObjectProducers_TopLeptonCountFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


class TopLeptonCountFilter : public edm::EDFilter {

  public:

    explicit TopLeptonCountFilter(const edm::ParameterSet & iConfig);
    virtual ~TopLeptonCountFilter();

  private:

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    edm::InputTag electronSource_;
    edm::InputTag muonSource_;
    edm::InputTag tauSource_;
    bool          countElectrons_;
    bool          countMuons_;
    bool          countTaus_;
    unsigned int  minNumber_;
    unsigned int  maxNumber_;

};


#endif
