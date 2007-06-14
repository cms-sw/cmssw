//
// Author:  Jan Heyninck, Steven Lowette
// Created: Tue Apr  10 12:01:49 CEST 2007
//
// $Id: TopElectronProducer.h,v 1.4 2007/06/07 05:49:16 lowette Exp $
//

#ifndef TopElectronProducer_h
#define TopElectronProducer_h

/**
  \class    TopElectronProducer TopElectronProducer.h "TopQuarkAnalysis/TopObjectProducers/interface/TopElectronProducer.h"
  \brief    Produces TopElectron's

   TopElectronProducer produces TopElectron's starting from an ElectronType
   collection, with possible matching to generator level, adding of resolutions
   and calculation of a lepton likelihood ratio

  \author   Jan Heyninck, Steven Lowette
  \version  $Id: TopElectronProducer.h,v 1.4 2007/06/07 05:49:16 lowette Exp $
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


class TopElectronProducer : public edm::EDProducer {

  public:

    explicit TopElectronProducer(const edm::ParameterSet & iConfig);
    ~TopElectronProducer();

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    std::vector<ElectronType> removeEleDupes(const std::vector<ElectronType> &electrons);
    
    // configurables
    edm::InputTag  electronSrc_;
    bool           doGenMatch_;
    bool           addResolutions_;
    bool           addLRValues_;
    edm::InputTag  genPartSrc_;
    std::string    electronResoFile_;
    std::string    electronLRFile_;
    // tools
    TopObjectResolutionCalc *        theResoCalc_;
    TopLeptonLRCalc *                theLeptonLRCalc_;
    PtInverseComparator<TopElectron> pTElectronComparator_;

};


#endif
