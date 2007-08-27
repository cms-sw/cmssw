//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopMuonProducer.h,v 1.6 2007/08/06 14:37:41 tsirig Exp $
//

#ifndef TopObjectProducers_TopMuonProducer_h
#define TopObjectProducers_TopMuonProducer_h

/**
  \class    TopMuonProducer TopMuonProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopMuonProducer.h"
  \brief    Produces TopMuon's

   TopMuonProducer produces TopMuon's starting from a MuonType collection,
   with possible matching to generator level, adding of resolutions and
   calculation of a lepton likelihood ratio

  \author   Jan Heyninck, Steven Lowette
  \version  $Id: TopMuonProducer.h,v 1.6 2007/08/06 14:37:41 tsirig Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"

#include <string>


class TopObjectResolutionCalc;
class TopLeptonLRCalc;


class TopMuonProducer : public edm::EDProducer {

  public:

    explicit TopMuonProducer(const edm::ParameterSet & iConfig);
    ~TopMuonProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    // configurables
    edm::InputTag  muonSrc_;
    bool           doGenMatch_;
    bool           addResolutions_;  
    bool           addLRValues_;
    edm::InputTag  genPartSrc_;
    edm::InputTag tracksTag_;
    std::string    muonResoFile_;
    std::string    muonLRFile_;
    // tools
    TopObjectResolutionCalc *    theResoCalc_;
    TopLeptonLRCalc *            theLeptonLRCalc_;
    //    PtInverseComparator<TopMuon> pTMuonComparator_;
    GreaterByPt<TopMuon> pTMuonComparator_;
};


#endif
