// -*- C++ -*-
//
// Package:    StGenEventReco
// Class:      StGenEventReco
// 
/**\class StGenEventReco StGenEventReco.cc AnalysisDataFormats/TopObjectsProducers/src/StGenEventReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jan Heyninck
//         Created:  Tue Apr  10 12:01:49 CEST 2007
// $Id: StGenEventReco.h,v 1.1 2007/05/11 15:33:49 giamman Exp $
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

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include <vector>

using namespace std;

//
// class decleration
//

class StGenEventReco : public edm::EDProducer {
   public:
      explicit StGenEventReco(const edm::ParameterSet&);
      ~StGenEventReco();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
};
