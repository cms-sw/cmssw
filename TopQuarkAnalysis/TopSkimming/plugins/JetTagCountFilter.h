#ifndef JetTagCountFilter_h
#define JetTagCountFilter_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class JetTagCountFilter : public edm::EDFilter {
public:
    JetTagCountFilter(const edm::ParameterSet&);
    ~JetTagCountFilter();
private:
    virtual bool filter(edm::Event&, const edm::EventSetup&);
    edm::InputTag src_;
    double minDiscriminator_;
    double minJetEt_;
    double maxJetEta_;
    unsigned int minNumber_;
};

#endif
