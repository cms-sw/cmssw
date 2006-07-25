// -*- C++ -*-
//
// Package:    TrackParameterAnalyzer
// Class:      TrackParameterAnalyzer
// 
/**\class TrackParameterAnalyzer TrackParameterAnalyzer.cc RecoVertex/PrimaryVertexProducer/src/TrackParameterAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Wolfram Erdmann
//         Created:  Fri Jun  2 10:54:05 CEST 2006
// $Id: TrackParameterAnalyzer.h,v 1.1 2006/07/11 11:40:32 werdmann Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Hep MC stuff from CLHEP, add  <use name=clhep> to the buildfile
#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/HepMC/GenVertex.h"
#include "CLHEP/HepMC/GenParticle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include <CLHEP/Vector/LorentzVector.h>

// simulated vertices,..., add <use name=SimDataFormats/Vertex> and <../Track>
#include <SimDataFormats/Vertex/interface/SimVertex.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>
#include <SimDataFormats/Track/interface/SimTrack.h>
#include <SimDataFormats/Track/interface/SimTrackContainer.h>
// pdg stuff , gives me SyntaxErrors when I use it, no clue why
//#include "SimGeneral/HepPDT/interface/HepPDT.h"

// vertex stuff
#include <DataFormats/VertexReco/interface/Vertex.h>
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
// perigee 
#include <TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h>
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h>

// Root
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>


// class declaration
//

class TrackParameterAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrackParameterAnalyzer(const edm::ParameterSet&);
      ~TrackParameterAnalyzer();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();

   private:
       bool match(const PerigeeTrajectoryParameters::ParameterVector  *a, const PerigeeTrajectoryParameters::ParameterVector  *b);
      // ----------member data ---------------------------
      std::string recoTrackProducer_;
      // root file to store histograms
      std::string outputFile_; // output file
      TFile*  rootFile_;
      TH1*   h1_pull0_; 
      TH1*   h1_pull1_;
      TH1*   h1_pull2_;
      TH1*   h1_pull3_;
      TH1*   h1_pull4_;
      TH1*   h1_Beff_;
      TH2*   h2_dvsphi_;

     // pdg
     //HepPDT* pdg;
};
