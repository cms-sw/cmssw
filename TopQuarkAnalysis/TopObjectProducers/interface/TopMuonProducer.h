// -*- C++ -*-
//
// Package:    TopMuonProducer
// Class:      TopMuonProducer
// 
/**\class TopMuonProducer TopMuonProducer.cc Top/TopEventProducers/src/TopMuonProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TopMuonProducer.h,v 1.2 2007/05/30 21:50:46 lowette Exp $
//
//


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"

#include <vector>
#include <string>


class TopLeptonLRCalc;
class TopObjectResolutionCalc;


class TopMuonProducer : public edm::EDProducer {

  public:

    explicit TopMuonProducer(const edm::ParameterSet & iConfig);
    ~TopMuonProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    // configurables
    edm::InputTag  muonSrc_;
    double         muonPTcut_;
    double         muonEtacut_;
    double         muonLRcut_;
    bool           doGenMatch_;
    bool           addResolutions_;  
    bool           addLRValues_;
    edm::InputTag  genPartSrc_;
    std::string    muonResoFile_;
    std::string    muonLRFile_;
    // tools
    TopObjectResolutionCalc *    theResoCalc_;
    TopLeptonLRCalc *            theLeptonLRCalc_;
    PtInverseComparator<TopMuon> pTMuonComparator_;

};
