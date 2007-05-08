// -*- C++ -*-
//
// Package:    TopJetObjectProducer
// Class:      TopJetObjectProducer
// 
/**\class TopJetObjectProducer TopJetObjectProducer.cc Top/TopEventProducers/src/TopJetObjectProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TopJetObjectProducer.h,v 1.1 2007/05/02 15:10:51 lowette Exp $
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

#include "AnalysisDataFormats/TopObjects/interface/TopJetObject.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"


#include <vector>
#include <Math/VectorUtil.h>

using namespace std;
using namespace edm;

//
// class decleration
//

class TopJetObjectProducer : public edm::EDProducer {
   public:
      explicit TopJetObjectProducer(const edm::ParameterSet&);
      ~TopJetObjectProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
     string jetTagsLabel_;
     string recJetsLabel_;
     string lcaliJetsLabel_;
     string bcaliJetsLabel_;
     string lcaliJetResoFile_;
     string bcaliJetResoFile_;
     double recJetETcut_;
     double calJetETcut_;
     double jetEtaCut_;
     int minNrConstis_;
     bool addResolutions_;  
     
     // compare two jets in ET
     struct CompareET {
       bool operator()( TopJetObject j1, TopJetObject j2 ) const {
         return j1.getRecJet().et() > j2.getRecJet().et();
       }
     };
     CompareET eTComparator;
     TopObjectResolutionCalc *lJetsResCalc, *bJetsResCalc;

};
