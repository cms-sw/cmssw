// -*- C++ -*-
//
// Package:    TopMETObjectProducer
// Class:      TopMETObjectProducer
// 
/**\class TopMETObjectProducer TopMETObjectProducer.cc Top/TopEventProducers/src/TopMETObjectProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TopMETObjectProducer.h,v 1.1 2007/05/02 15:10:51 lowette Exp $
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

#include "AnalysisDataFormats/TopObjects/interface/TopMETObject.h"
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"


#include <vector>

using namespace std;
using namespace edm;

//
// class decleration
//

class TopMETObjectProducer : public edm::EDProducer {
   public:
      explicit TopMETObjectProducer(const edm::ParameterSet&);
      ~TopMETObjectProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
     string METLabel_, metResoFile_;
     double METcut_;
     bool addResolutions_;
     
     // compare two MET in ET
     struct CompareET {
       bool operator()( TopMETObject m1, TopMETObject m2 ) const {
         return m1.getRecMET().et() > m2.getRecMET().et();
       }
     };
     CompareET eTComparator;
     TopObjectResolutionCalc *metResCalc;
};
