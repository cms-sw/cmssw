// -*- C++ -*-
//
// Package:    TtGenEventReco
// Class:      TtGenEventReco
// 
/**\class TtGenEventReco TtGenEventReco.cc AnalysisDataFormats/TopObjectsProducers/src/TtGenEventReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: TtGenEventReco.h,v 1.1 2007/05/02 15:10:44 lowette Exp $
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

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include <vector>

using namespace std;

//
// class decleration
//

class TtGenEventReco : public edm::EDProducer {
   public:
      explicit TtGenEventReco(const edm::ParameterSet&);
      ~TtGenEventReco();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
};
