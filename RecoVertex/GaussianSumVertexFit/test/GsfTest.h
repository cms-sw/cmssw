// -*- C++ -*-
//
// Package:    GsfTest
// Class:      GsfTest
// 
/**\class GsfTest GsfTest.cc RecoVertex/GsfTest/src/GsfTest.cc

 Description: steers tracker primary vertex reconstruction and storage

 Implementation:
     <Notes on implementation>
*/


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "RecoVertex/KalmanVertexFit/interface/SimpleVertexTree.h"
#include <TFile.h>

  /**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class GsfTest : public edm::EDAnalyzer {
public:
  explicit GsfTest(const edm::ParameterSet&);
  ~GsfTest();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginJob();
  virtual void endJob();

private:

  TrackingVertex getSimVertex(const edm::Event& iEvent) const;

  edm::ParameterSet theConfig;
  edm::ParameterSet gsfPSet;

  std::unique_ptr<SimpleVertexTree> tree;
  TFile*  rootFile_;

  std::string outputFile_; // output file
  edm::EDGetTokenT<reco::TrackCollection> token_tracks; 
//   edm::EDGetTokenT<TrackingParticleCollection> token_TrackTruth;
  edm::EDGetTokenT<TrackingVertexCollection> token_VertexTruth;
};
