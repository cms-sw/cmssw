// -*- C++ -*-
//
// Package:    TestMixedSource
// Class:      TestMixedSource
// 
/**\class TestMixedSource TestMixedSource.cc TestMixed/TestMixedSource/src/TestMixedSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Emilia Lubenova Becheva
//         Created:  Wed May 20 16:46:58 CEST 2009
// $Id: TestMixedSource.h,v 1.3 2013/03/01 00:13:36 wmtan Exp $
//
//

#ifndef TestMixedSource_h
#define TestMixedSource_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH1I.h"
#include "TFile.h"

#include <iostream>
#include <fstream>

//
// class decleration
//
namespace edm
{
class TestMixedSource : public edm::EDAnalyzer {
   public:
      explicit TestMixedSource(const edm::ParameterSet&);
      ~TestMixedSource();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      // ----------member data ---------------------------
      ofstream outputFile;
      std::string fileName_;
      int bunchcr_;
      int minbunch_;
      int maxbunch_;
      TH1I * histTrack_bunchPileups_;
      TH1I * histTrack_bunchSignal_;
      TH1I * histVertex_bunch_;
      TH1I * histPCaloHit_bunch_;
      TH1I * histPSimHit_bunchSignal_TrackerHitsTECHighTof_;
      TH1I * histPSimHit_bunchPileups_TrackerHitsTECHighTof_;
      TH1I * tofhist_;
      TH1I * tofhist_sig_;
      TH1I * histPSimHit_bunchSignal_MuonCSCHits_;
      TH1I * histPSimHit_bunchPileups_MuonCSCHits_;
      TH1I * histHepMCProduct_bunch_;
      TFile *histFile_;
       

};
}//edm
#endif
