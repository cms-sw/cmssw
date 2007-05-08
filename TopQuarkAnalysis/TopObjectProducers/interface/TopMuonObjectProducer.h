// -*- C++ -*-
//
// Package:    TopMuonObjectProducer
// Class:      TopMuonObjectProducer
// 
/**\class TopMuonObjectProducer TopMuonObjectProducer.cc Top/TopEventProducers/src/TopMuonObjectProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TopMuonObjectProducer.h,v 1.2 2007/05/04 01:04:16 lowette Exp $
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

#include "AnalysisDataFormats/TopObjects/interface/TopMuonObject.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

#include <vector>
#include <string>


class TopLeptonLRCalc;


using namespace std;
using namespace edm;
using namespace reco;

//
// class decleration
//

class TopMuonObjectProducer : public edm::EDProducer {
   public:
      explicit TopMuonObjectProducer(const edm::ParameterSet&);
      ~TopMuonObjectProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
     double muonPTcut_;
     double muonEtacut_;
     double muonLRcut_;
     bool   addResolutions_;  
     bool   addLRValues_;
     string muonLRFile_, muonResoFile_;

     TopLeptonLRCalc * theLeptonLRCalc_;
        
     struct ComparePtMuon {
       bool operator()( TopMuonObject m1, TopMuonObject m2 ) const {
         return m1.getRecMuon().pt() > m2.getRecMuon().pt();
       }
     };
     ComparePtMuon     pTMuonComparator;
     TopObjectResolutionCalc *muResCalc;

};
