// -*- C++ -*-
//
// Package:    KVFTest
// Class:      KVFTest
//
/**\class KVFTest KVFTest.cc RecoVertex/KVFTest/src/KVFTest.cc

 Description: steers tracker primary vertex reconstruction and storage

 Implementation:
     <Notes on implementation>
*/

// system include files
#include <memory>

// user include files
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/SimpleVertexTree.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include <TFile.h>

/**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class KVFTest : public edm::one::EDAnalyzer<> {
public:
  explicit KVFTest(const edm::ParameterSet&);
  ~KVFTest() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void beginJob() override;
  void endJob() override;

private:
  TrackingVertex getSimVertex(const edm::Event& iEvent) const;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> estoken_MF;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> estoken_TTB;

  edm::ParameterSet theConfig;
  edm::ParameterSet kvfPSet;
  std::unique_ptr<SimpleVertexTree> tree;
  TFile* rootFile_;

  std::string outputFile_;  // output file
  edm::EDGetTokenT<reco::TrackCollection> token_tracks;
  edm::EDGetTokenT<TrackingParticleCollection> token_TrackTruth;
  edm::EDGetTokenT<TrackingVertexCollection> token_VertexTruth;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> token_associatorForParamAtPca;
};
