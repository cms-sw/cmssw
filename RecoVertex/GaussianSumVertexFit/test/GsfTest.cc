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
#include <iostream>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/SimpleVertexTree.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include <TFile.h>

/**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class GsfTest : public edm::one::EDAnalyzer<> {
public:
  explicit GsfTest(const edm::ParameterSet&);
  ~GsfTest();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  TrackingVertex getSimVertex(const edm::Event& iEvent) const;

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> estoken_ttk;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> estoken_mf;

  edm::ParameterSet gsfPSet;

  std::unique_ptr<SimpleVertexTree> tree;
  TFile* rootFile_;

  std::string outputFile_;  // output file
  edm::EDGetTokenT<reco::TrackCollection> token_tracks;
  //   edm::EDGetTokenT<TrackingParticleCollection> token_TrackTruth;
  edm::EDGetTokenT<TrackingVertexCollection> token_VertexTruth;
};

using namespace reco;
using namespace edm;
using namespace std;

GsfTest::GsfTest(const edm::ParameterSet& iConfig)
    : estoken_ttk(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))), estoken_mf(esConsumes()) {
  token_tracks = consumes<TrackCollection>(iConfig.getParameter<string>("TrackLabel"));
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile");
  gsfPSet = iConfig.getParameter<edm::ParameterSet>("GsfParameters");
  rootFile_ = TFile::Open(outputFile_.c_str(), "RECREATE");
  edm::LogInfo("RecoVertex/GsfTest") << "Initializing KVF TEST analyser  - Output file: " << outputFile_ << "\n";
  //   token_TrackTruth = consumes<TrackingParticleCollection>(edm::InputTag("trackingtruth", "TrackTruth"));
  token_VertexTruth = consumes<TrackingVertexCollection>(edm::InputTag("trackingtruth", "VertexTruth"));
}

GsfTest::~GsfTest() { delete rootFile_; }

//
// member functions
//

void GsfTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (not tree) {
    const MagneticField* magField = &iSetup.getData(estoken_mf);
    tree.reset(new SimpleVertexTree("VertexFitter", magField));
  }

  try {
    edm::LogInfo("RecoVertex/GsfTest") << "Reconstructing event number: " << iEvent.id() << "\n";

    // get RECO tracks from the event
    // `tks` can be used as a ptr to a reco::TrackCollection
    edm::Handle<edm::View<reco::Track> > tks;
    iEvent.getByToken(token_tracks, tks);

    edm::LogInfo("RecoVertex/GsfTest") << "Found: " << (*tks).size() << " reconstructed tracks"
                                       << "\n";

    // Transform Track to TransientTrack

    //get the builder:
    edm::ESHandle<TransientTrackBuilder> theB = iSetup.getHandle(estoken_ttk);
    //do the conversion:
    vector<TransientTrack> t_tks = (*theB).build(tks);

    edm::LogInfo("RecoVertex/GsfTest") << "Found: " << t_tks.size() << " reconstructed tracks"
                                       << "\n";

    // Call the KalmanVertexFitter if more than 1 track
    if (t_tks.size() > 1) {
      //      KalmanVertexFitter kvf(kvfPSet);
      // For the analysis: compare to your SimVertex
      //       TrackingVertex sv = getSimVertex(iEvent);
      //       edm::LogPrint("GsfTest") << "SimV Position: " << Vertex::Point(sv.position()) << "\n";

      KalmanVertexFitter kvf;
      TransientVertex tv2 = kvf.vertex(t_tks);
      edm::LogPrint("GsfTest") << "KVF Position:  " << Vertex::Point(tv2.position()) << tv2.normalisedChiSquared()
                               << " " << tv2.degreesOfFreedom() << "\n";

      GsfVertexFitter gsf(gsfPSet);
      TransientVertex tv = gsf.vertex(t_tks);
      edm::LogPrint("GsfTest") << "Position: " << Vertex::Point(tv.position()) << "\n";

      //   edm::Handle<TrackingParticleCollection>  TPCollectionH ;
      //   iEvent.getByToken(token_TrackTruth, TPCollectionH);
      //   const TrackingParticleCollection tPC = *(TPCollectionH.product());
      //       reco::RecoToSimCollection recSimColl=associatorForParamAtPca->associateRecoToSim(tks,
      // 									      TPCollectionH,
      // 									      &iEvent);

      //       tree->fill(tv, &sv, 0, 0.0);
    }

  }

  catch (cms::Exception& err) {
    edm::LogError("GsfTest") << "Exception during event number: " << iEvent.id() << "\n" << err.what() << "\n";
  }
}

//Returns the first vertex in the list.

TrackingVertex GsfTest::getSimVertex(const edm::Event& iEvent) const {
  // get the simulated vertices
  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(token_VertexTruth, TVCollectionH);
  const TrackingVertexCollection tPC = *(TVCollectionH.product());

  //    Handle<edm::SimVertexContainer> simVtcs;
  //    iEvent.getByLabel("g4SimHits", simVtcs);
  //    edm::LogPrint("GsfTest") << "SimVertex " << simVtcs->size() << std::endl;
  //    for(edm::SimVertexContainer::const_iterator v=simVtcs->begin();
  //        v!=simVtcs->end(); ++v){
  //      edm::LogPrint("GsfTest") << "simvtx "
  // 	       << v->position().x() << " "
  // 	       << v->position().y() << " "
  // 	       << v->position().z() << " "
  // 	       << v->parentIndex() << " "
  // 	       << v->noParent() << " "
  //               << std::endl;
  //    }
  return *(tPC.begin());
}

DEFINE_FWK_MODULE(GsfTest);
