// -*- C++ -*-
//
// Package:    TopElectronObjectProducer
// Class:      TopElectronObjectProducer
// 
/**\class TopElectronObjectProducer TopElectronObjectProducer.cc Top/TopEventProducers/src/TopElectronObjectProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TopElectronObjectProducer.h,v 1.1 2007/05/02 15:10:51 lowette Exp $
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

#include "AnalysisDataFormats/TopObjects/interface/TopElectronObject.h"

#include <vector>
#include <string>


class TopLeptonLRCalc;


using namespace std;
using namespace edm;
using namespace reco;

//
// class decleration
//

class TopElectronObjectProducer : public edm::EDProducer {
   public:
      explicit TopElectronObjectProducer(const edm::ParameterSet&);
      ~TopElectronObjectProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
     double electronPTcut_;
     double electronEtacut_;
     double electronLRcut_;
     bool   addResolutions_;
     bool   addLRValues_;
     string electronLRFile_;

     TopLeptonLRCalc * theLeptonLRCalc_;
     
     struct ComparePtElectron {
       bool operator()( TopElectronObject e1, TopElectronObject e2 ) const {
         return e1.getRecElectron().pt() > e2.getRecElectron().pt();
       }
     };
     ComparePtElectron pTElectronComparator;

};
