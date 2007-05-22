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
// $Id: TopElectronProducer.h,v 1.3 2007/05/08 14:01:20 heyninck Exp $
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

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"
#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include <vector>
#include <string>


class TopLeptonLRCalc;


using namespace std;

//
// class decleration
//

class TopElectronProducer : public edm::EDProducer {
   public:
      explicit TopElectronProducer(const edm::ParameterSet&);
      ~TopElectronProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
     double electronPTcut_;
     double electronEtacut_;
     double electronLRcut_;
     bool   addResolutions_;
     bool   addLRValues_;
     string electronLRFile_, electronResoFile_;
     TopLeptonLRCalc 	     * theLeptonLRCalc_;
     PtInverseComparator<TopElectron>     pTElectronComparator;
     TopObjectResolutionCalc *elResCalc;

};
