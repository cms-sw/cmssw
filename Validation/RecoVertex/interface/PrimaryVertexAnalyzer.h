// -*- C++ -*-
//
// Package:    PrimaryVertexAnalyzer
// Class:      PrimaryVertexAnalyzer
// 
/**\class PrimaryVertexAnalyzer PrimaryVertexAnalyzer.cc Validation/RecoVertex/src/PrimaryVertexAnalyzer.cc

 Description: simple primary vertex analyzer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Wolfram Erdmann
//         Created:  Fri Jun  2 10:54:05 CEST 2006
// $Id: PrimaryVertexAnalyzer.h,v 1.3 2006/09/15 15:42:04 vanlaer Exp $
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
  //      TH1*   h1_nbtks_in_event_; 
  //      TH1*   h1_tks_chi2_;
  //      TH1*   h1_tks_ndf_;
      TH1*   h1_nbvtx_in_event_; 
      TH1*   h1_nbtks_in_vtx_; 
      TH1*   h1_resx_; 
      TH1*   h1_resy_;
      TH1*   h1_resz_;
      TH1*   h1_pullx_; 
      TH1*   h1_pully_;
      TH1*   h1_pullz_;
      TH1*   h1_vtx_chi2_;
      TH1*   h1_vtx_ndf_;
      TH1*   h1_tklinks_;
      TH1*   h1_nans_;
};

