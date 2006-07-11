// -*- C++ -*-
//
// Package:    PrimaryVertexProducer
// Class:      PrimaryVertexAnalyzer
// 
/**\class PrimaryVertexAnalyzer PrimaryVertexAnalyzer.cc RecoVertex/PrimaryVertexProducer/src/PrimaryVertexAnalyzer.cc

 Description: simple primary vertex analyzer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Wolfram Erdmann
//         Created:  Fri Jun  2 10:54:05 CEST 2006
// $Id$
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// vertex stuff
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// Root
#include <TH1.h>
#include <TFile.h>


// class declaration
//

class PrimaryVertexAnalyzer : public edm::EDAnalyzer {
   public:
      explicit PrimaryVertexAnalyzer(const edm::ParameterSet&);
      ~PrimaryVertexAnalyzer();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();

   private:
      // ----------member data ---------------------------
      // root file to store histograms
      std::string outputFile_; // output file
      TFile*  rootFile_;
      TH1*   h1_pullx_; 
      TH1*   h1_pully_;
      TH1*   h1_pullz_;
      TH1*   h1_chi2_;
};

