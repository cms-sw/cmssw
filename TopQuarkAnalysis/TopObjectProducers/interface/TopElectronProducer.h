// -*- C++ -*-
//
// Package:    TopElectronProducer
// Class:      TopElectronProducer
// 
/**\class TopElectronProducer TopElectronProducer.cc Top/TopEventProducers/src/TopElectronProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TopElectronProducer.h,v 1.1 2007/05/22 17:01:43 heyninck Exp $
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


class TopElectronProducer : public edm::EDProducer {

  public:

    explicit TopElectronProducer(const edm::ParameterSet & iConfig);
    ~TopElectronProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    PtInverseComparator<TopElectron> pTElectronComparator;
    double electronPTcut_;
    double electronEtacut_;
    double electronLRcut_;
    bool doGenMatch_;
    bool addResolutions_;
    bool addLRValues_;
    std::string electronLRFile_, electronResoFile_;
    edm::InputTag genPartSrc_;
    TopLeptonLRCalc * theLeptonLRCalc_;
    TopObjectResolutionCalc * theResCalc_;

};
