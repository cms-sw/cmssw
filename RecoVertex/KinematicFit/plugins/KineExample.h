// -*- C++ -*-
//
// Package:    KineExample
// Class:      KineExample
// 
/**\class KineExample KineExample.cc RecoVertex/KineExample/src/KineExample.cc

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
// #include "RecoVertex/KalmanVertexFit/test/SimpleVertexTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
// #include "RecoVertex/KinematicFitPrimitives/interface/"
#include <TFile.h>

  /**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class KineExample : public edm::EDAnalyzer {
public:
  explicit KineExample(const edm::ParameterSet&);
  ~KineExample();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endJob();

private:

  void printout(const RefCountedKinematicVertex& myVertex) const;
  void printout(const RefCountedKinematicParticle& myParticle) const;
  void printout(const RefCountedKinematicTree& myTree) const;

  TrackingVertex getSimVertex(const edm::Event& iEvent) const;

  edm::ParameterSet theConfig;
  edm::ParameterSet kvfPSet;
//   std::unique_ptr<SimpleVertexTree> tree;
//   TFile*  rootFile_;

  std::string outputFile_; // output file
  edm::EDGetTokenT<reco::TrackCollection> token_tracks; 
//   edm::EDGetTokenT<TrackingParticleCollection> token_TrackTruth;
  edm::EDGetTokenT<TrackingVertexCollection> token_VertexTruth;
};
