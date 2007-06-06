// -*- C++ -*-
//
// Package:    TopMETProducer
// Class:      TopMETProducer
// 
/**\class TopMETProducer TopMETProducer.cc Top/TopEventProducers/src/TopMETProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TopMETProducer.h,v 1.1 2007/05/22 17:01:43 heyninck Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"


//
// class decleration
//

class TopMETProducer : public edm::EDProducer {

  public:

    explicit TopMETProducer(const edm::ParameterSet & iConfig);
    ~TopMETProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    // configurables
    edm::InputTag metSrc_;
    double        metCut_;
    bool          addResolutions_;
    bool          addMuonCorr_;
    std::string   metResoFile_;
    edm::InputTag muonSrc_;
    // tools
    TopObjectResolutionCalc *   metResoCalc_;
    EtInverseComparator<TopMET> eTComparator_;

};
