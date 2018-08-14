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
//
//

#ifndef TestMixedSource_h
#define TestMixedSource_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH1I.h"
#include "TFile.h"

#include <iostream>
#include <fstream>

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class decleration
//
namespace edm
{
  class TestMixedSource : public edm::one::EDAnalyzer<> {
   public:
      explicit TestMixedSource(const edm::ParameterSet&);
      ~TestMixedSource() override;

   private:
      void beginJob() override ;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;

      // ----------member data ---------------------------
      std::ofstream outputFile;
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
       
    edm::EDGetTokenT<CrossingFrame<PSimHit>> TrackerToken0_;
    edm::EDGetTokenT<CrossingFrame<PSimHit>> MuonToken_;

    edm::EDGetTokenT<CrossingFrame<PCaloHit>> CaloToken1_;

    edm::EDGetTokenT<CrossingFrame<SimTrack>> SimTrackToken_;
    edm::EDGetTokenT<CrossingFrame<SimVertex>> SimVertexToken_;
    edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct>> HepMCToken_;

};
}//edm
#endif
