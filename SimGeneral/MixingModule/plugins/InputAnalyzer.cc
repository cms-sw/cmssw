// -*- C++ -*-
//
// Package:    InputAnalyzer
// Class:      InputAnalyzer
//
/**\class InputAnalyzer InputAnalyzer.cc Analyzer/InputAnalyzer/src/InputAnalyzer.cc

 Description: Get the data from the source file using getByLabel method.

 Implementation:
    
*/
//
// Original Author:  Emilia Lubenova Becheva
//         Created:  Mon Apr 20 13:43:06 CEST 2009
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "InputAnalyzer.h"

//
// constructors and destructor
//
namespace edm {

  InputAnalyzer::InputAnalyzer(const edm::ParameterSet& iConfig) {
    dataStep2_ = iConfig.getParameter<bool>("dataStep2");

    labelPCF_ = consumes<PCrossingFrame<SimTrack>>(iConfig.getParameter<edm::InputTag>("collPCF"));

    //will only be needed if not Step2:
    labelSimTr_ = consumes<SimTrackContainer>(iConfig.getParameter<edm::InputTag>("collSimTrack"));
  }

  InputAnalyzer::~InputAnalyzer() {}

  //
  // member functions
  //

  // ------------ method called to for each event  ------------
  void InputAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    std::cout << " dataStep2_ = " << dataStep2_ << std::endl;

    if (!dataStep2_) {
      // Get the SimTrack collection

      //double simPt=0;

      // Get the SimTrack collection from the event
      edm::Handle<SimTrackContainer> simTracks;
      bool gotTracks = iEvent.getByToken(labelSimTr_, simTracks);

      if (!gotTracks) {
        std::cout << "-> Could not read SimTracks !!!!" << std::endl;
      } else {
        std::cout << "-> Could read SimTracks !!!!" << std::endl;
      }
    } else {
      // Get the PCrossingFrame collection given as signal

      edm::Handle<PCrossingFrame<SimTrack>> cf_simtrack;
      bool gotTracks = iEvent.getByToken(labelPCF_, cf_simtrack);

      if (!gotTracks) {
        std::cout << "-> Could not read PCrossingFrame<SimTracks> !!!!" << std::endl;
      } else
        std::cout << "-> Could read PCrossingFrame<SimTracks> !!!!" << std::endl;
    }
  }

  // ------------ method called once each job just before starting event loop  ------------
  void InputAnalyzer::beginJob() {}

  // ------------ method called once each job just after ending the event loop  ------------
  void InputAnalyzer::endJob() {}

}  // namespace edm
