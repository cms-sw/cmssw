#include "Validation/RecoVertex/interface/PrimaryVertexAnalyzer4PUSlimmed.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// reco track and vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

// associator
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

// DQM
#include "DQMServices/Core/interface/MonitorElement.h"

//
// constructors and destructor
//
PrimaryVertexAnalyzer4PUSlimmed::PrimaryVertexAnalyzer4PUSlimmed(
    const edm::ParameterSet& iConfig)
    : verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      use_TP_associator_(
          iConfig.getUntrackedParameter<bool>("use_TP_associator", false)),
      sigma_z_match_(
          iConfig.getUntrackedParameter<double>("sigma_z_match", 3.0)),
      recoTrackProducer_(
          iConfig.getUntrackedParameter<std::string>("recoTrackProducer")),
      vecPileupSummaryInfoToken_(consumes<std::vector<PileupSummaryInfo> >(
          edm::InputTag(std::string("addPileupInfo")))),
      recoTrackCollectionToken_(consumes<reco::TrackCollection>(edm::InputTag(
          iConfig.getUntrackedParameter<std::string>("recoTrackProducer")))),
      edmView_recoTrack_Token_(consumes<edm::View<reco::Track> >(edm::InputTag(
          iConfig.getUntrackedParameter<std::string>("recoTrackProducer")))),
      trackingParticleCollectionToken_(consumes<TrackingParticleCollection>(
          edm::InputTag(std::string("mix"), std::string("MergedTrackTruth")))),
      trackingVertexCollectionToken_(consumes<TrackingVertexCollection>(
          edm::InputTag(std::string("mix"), std::string("MergedTrackTruth")))) {
  reco_vertex_collections_ = iConfig.getParameter<std::vector<edm::InputTag> >(
      "vertexRecoCollections");
  for (auto const& l : reco_vertex_collections_) {
    reco_vertex_collection_tokens_.push_back(
        edm::EDGetTokenT<reco::VertexCollection>(
            consumes<reco::VertexCollection>(l)));
  }
}

PrimaryVertexAnalyzer4PUSlimmed::~PrimaryVertexAnalyzer4PUSlimmed() {}

//
// member functions
//
void PrimaryVertexAnalyzer4PUSlimmed::bookHistograms(
    DQMStore::IBooker& i, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // TODO(rovere) make this booking method shorter and smarter,
  // factorizing similar histograms with different prefix in a single
  // method call.
  float log_bins[9] = {0.0, 0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.};
  std::string root_folder = "Validation/Vertices/";
  i.setCurrentFolder(root_folder);
  mes_["root_folder"]["GenVtx_vs_BX"] =
      i.book2D("GenVtx_vs_BX", "GenVtx_vs_BX", 20, -12., 3., 200, 0., 200.);
  // Generated Primary Vertex Plots
  mes_["root_folder"]["GenPV_X"] =
      i.book1D("GenPV_X", "GeneratedPV_X", 120, -0.6, 0.6);
  mes_["root_folder"]["GenPV_Y"] =
      i.book1D("GenPV_Y", "GeneratedPV_Y", 120, -0.6, 0.6);
  mes_["root_folder"]["GenPV_Z"] =
      i.book1D("GenPV_Z", "GeneratedPV_Z", 120, -60., 60.);
  mes_["root_folder"]["GenPV_R"] =
      i.book1D("GenPV_R", "GeneratedPV_R", 120, 0, 0.6);
  mes_["root_folder"]["GenPV_Pt2"] =
      i.book1D("GenPV_Pt2", "GeneratedPV_Sum-pt2", 8, &log_bins[0]);
  mes_["root_folder"]["GenPV_NumTracks"] =
      i.book1D("GenPV_NumTracks", "GeneratedPV_NumTracks", 200, 0., 200.);
  mes_["root_folder"]["GenPV_ClosestDistanceZ"] =
      i.book1D("GenPV_ClosestDistanceZ", "GeneratedPV_ClosestDistanceZ", 8,
               &log_bins[0]);

  // All Generated Vertices, used for efficiency plots
  mes_["root_folder"]["GenAllV_NumVertices"] = i.book1D(
      "GenAllV_NumVertices", "GeneratedAllV_NumVertices", 200, 0., 200.);
  mes_["root_folder"]["GenAllV_X"] =
      i.book1D("GenAllV_X", "GeneratedAllV_X", 120, -0.6, 0.6);
  mes_["root_folder"]["GenAllV_Y"] =
      i.book1D("GenAllV_Y", "GeneratedAllV_Y", 120, -0.6, 0.6);
  mes_["root_folder"]["GenAllV_Z"] =
      i.book1D("GenAllV_Z", "GeneratedAllV_Z", 120, -60, 60);
  mes_["root_folder"]["GenAllV_R"] =
      i.book1D("GenAllV_R", "GeneratedAllV_R", 120, 0, 0.6);
  mes_["root_folder"]["GenAllV_Pt2"] =
      i.book1D("GenAllV_Pt2", "GeneratedAllV_Sum-pt2", 8, &log_bins[0]);
  mes_["root_folder"]["GenAllV_NumTracks"] =
      i.book1D("GenAllV_NumTracks", "GeneratedAllV_NumTracks", 200, 0., 200.);
  mes_["root_folder"]["GenAllV_ClosestDistanceZ"] =
      i.book1D("GenAllV_ClosestDistanceZ", "GeneratedAllV_ClosestDistanceZ", 8,
               &log_bins[0]);

  for (auto const& l : reco_vertex_collections_) {
    std::string label = l.label();
    std::string current_folder = root_folder + label;
    i.setCurrentFolder(current_folder);

    mes_[label]["RecoVtx_vs_GenVtx"] = i.book2D(
        "RecoVtx_vs_GenVtx", "RecoVtx_vs_GenVtx", 200, 0., 200., 200, 0., 200.);
    mes_[label]["MatchedRecoVtx_vs_GenVtx"] =
        i.book2D("MatchedRecoVtx_vs_GenVtx", "MatchedRecoVtx_vs_GenVtx", 200,
                 0., 200., 200, 0., 200.);
    mes_[label]["MisTagRate"] =
        i.book1D("MisTagRate", "MisTagRate", 2, -0.5, 1.5);
    mes_[label]["TruePVLocationIndex"] =
        i.book1D("TruePVLocationIndex",
                 "TruePVLocationIndexInRecoVertexCollection", 12, -1.5, 10.5);
    // All Generated Vertices. Used for Efficiency plots We kind of
    // duplicate plots here in case we want to perform more detailed
    // studies on a selection of generated vertices, not on all of them.
    mes_[label]["GenAllAssoc2Reco_NumVertices"] =
        i.book1D("GenAllAssoc2Reco_NumVertices",
                 "GeneratedAllAssoc2Reco_NumVertices", 200, 0., 200.);
    mes_[label]["GenAllAssoc2Reco_X"] = i.book1D(
        "GenAllAssoc2Reco_X", "GeneratedAllAssoc2Reco_X", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2Reco_Y"] = i.book1D(
        "GenAllAssoc2Reco_Y", "GeneratedAllAssoc2Reco_Y", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2Reco_Z"] = i.book1D(
        "GenAllAssoc2Reco_Z", "GeneratedAllAssoc2Reco_Z", 120, -60, 60);
    mes_[label]["GenAllAssoc2Reco_R"] =
        i.book1D("GenAllAssoc2Reco_R", "GeneratedAllAssoc2Reco_R", 120, 0, 0.6);
    mes_[label]["GenAllAssoc2Reco_Pt2"] =
        i.book1D("GenAllAssoc2Reco_Pt2", "GeneratedAllAssoc2Reco_Sum-pt2", 8,
                 &log_bins[0]);
    mes_[label]["GenAllAssoc2Reco_NumTracks"] =
        i.book1D("GenAllAssoc2Reco_NumTracks",
                 "GeneratedAllAssoc2Reco_NumTracks", 200, 0., 200.);
    mes_[label]["GenAllAssoc2Reco_ClosestDistanceZ"] =
        i.book1D("GenAllAssoc2Reco_ClosestDistanceZ",
                 "GeneratedAllAssoc2Reco_ClosestDistanceZ", 8, &log_bins[0]);

    // All Generated Vertices Matched to a Reconstructed vertex. Used
    // for Efficiency plots
    mes_[label]["GenAllAssoc2RecoMatched_NumVertices"] =
        i.book1D("GenAllAssoc2RecoMatched_NumVertices",
                 "GeneratedAllAssoc2RecoMatched_NumVertices", 200, 0., 200.);
    mes_[label]["GenAllAssoc2RecoMatched_X"] =
        i.book1D("GenAllAssoc2RecoMatched_X", "GeneratedAllAssoc2RecoMatched_X",
                 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMatched_Y"] =
        i.book1D("GenAllAssoc2RecoMatched_Y", "GeneratedAllAssoc2RecoMatched_Y",
                 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMatched_Z"] =
        i.book1D("GenAllAssoc2RecoMatched_Z", "GeneratedAllAssoc2RecoMatched_Z",
                 120, -60, 60);
    mes_[label]["GenAllAssoc2RecoMatched_R"] =
        i.book1D("GenAllAssoc2RecoMatched_R", "GeneratedAllAssoc2RecoMatched_R",
                 120, 0, 0.6);
    mes_[label]["GenAllAssoc2RecoMatched_Pt2"] =
        i.book1D("GenAllAssoc2RecoMatched_Pt2",
                 "GeneratedAllAssoc2RecoMatched_Sum-pt2", 8, &log_bins[0]);
    mes_[label]["GenAllAssoc2RecoMatched_NumTracks"] =
        i.book1D("GenAllAssoc2RecoMatched_NumTracks",
                 "GeneratedAllAssoc2RecoMatched_NumTracks", 200, 0., 200.);
    mes_[label]["GenAllAssoc2RecoMatched_ClosestDistanceZ"] = i.book1D(
        "GenAllAssoc2RecoMatched_ClosestDistanceZ",
        "GeneratedAllAssoc2RecoMatched_ClosestDistanceZ", 8, &log_bins[0]);

    // All Generated Vertices Multi-Matched to a Reconstructed vertex. Used
    // for Merged plots
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumVertices"] = i.book1D(
        "GenAllAssoc2RecoMultiMatched_NumVertices",
        "GeneratedAllAssoc2RecoMultiMatched_NumVertices", 200, 0., 200.);
    mes_[label]["GenAllAssoc2RecoMultiMatched_X"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_X",
                 "GeneratedAllAssoc2RecoMultiMatched_X", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Y"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_Y",
                 "GeneratedAllAssoc2RecoMultiMatched_Y", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Z"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_Z",
                 "GeneratedAllAssoc2RecoMultiMatched_Z", 120, -60, 60);
    mes_[label]["GenAllAssoc2RecoMultiMatched_R"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_R",
                 "GeneratedAllAssoc2RecoMultiMatched_R", 120, 0, 0.6);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Pt2"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_Pt2",
                 "GeneratedAllAssoc2RecoMultiMatched_Sum-pt2", 8, &log_bins[0]);
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumTracks"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_NumTracks",
                 "GeneratedAllAssoc2RecoMultiMatched_NumTracks", 200, 0., 200.);
    mes_[label]["GenAllAssoc2RecoMultiMatched_ClosestDistanceZ"] = i.book1D(
        "GenAllAssoc2RecoMultiMatched_ClosestDistanceZ",
        "GeneratedAllAssoc2RecoMultiMatched_ClosestDistanceZ", 8, &log_bins[0]);

    // All Reco Vertices. Used for {Fake,Duplicate}-Rate plots
    mes_[label]["RecoAllAssoc2Gen_NumVertices"] =
        i.book1D("RecoAllAssoc2Gen_NumVertices",
                 "ReconstructedAllAssoc2Gen_NumVertices", 200, 0., 200.);
    mes_[label]["RecoAllAssoc2Gen_X"] = i.book1D(
        "RecoAllAssoc2Gen_X", "ReconstructedAllAssoc2Gen_X", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2Gen_Y"] = i.book1D(
        "RecoAllAssoc2Gen_Y", "ReconstructedAllAssoc2Gen_Y", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2Gen_Z"] = i.book1D(
        "RecoAllAssoc2Gen_Z", "ReconstructedAllAssoc2Gen_Z", 120, -60, 60);
    mes_[label]["RecoAllAssoc2Gen_R"] = i.book1D(
        "RecoAllAssoc2Gen_R", "ReconstructedAllAssoc2Gen_R", 120, 0, 0.6);
    mes_[label]["RecoAllAssoc2Gen_Pt2"] =
        i.book1D("RecoAllAssoc2Gen_Pt2", "ReconstructedAllAssoc2Gen_Sum-pt2", 8,
                 &log_bins[0]);
    mes_[label]["RecoAllAssoc2Gen_NumTracks"] =
        i.book1D("RecoAllAssoc2Gen_NumTracks",
                 "ReconstructedAllAssoc2Gen_NumTracks", 200, 0., 200.);
    mes_[label]["RecoAllAssoc2Gen_ClosestDistanceZ"] =
        i.book1D("RecoAllAssoc2Gen_ClosestDistanceZ",
                 "ReconstructedAllAssoc2Gen_ClosestDistanceZ", 8, &log_bins[0]);

    // All Reconstructed Vertices Matched to a Generated vertex. Used
    // for Fake-Rate plots
    mes_[label]["RecoAllAssoc2GenMatched_NumVertices"] =
        i.book1D("RecoAllAssoc2GenMatched_NumVertices",
                 "ReconstructedAllAssoc2GenMatched_NumVertices", 200, 0., 200.);
    mes_[label]["RecoAllAssoc2GenMatched_X"] =
        i.book1D("RecoAllAssoc2GenMatched_X",
                 "ReconstructedAllAssoc2GenMatched_X", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMatched_Y"] =
        i.book1D("RecoAllAssoc2GenMatched_Y",
                 "ReconstructedAllAssoc2GenMatched_Y", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMatched_Z"] =
        i.book1D("RecoAllAssoc2GenMatched_Z",
                 "ReconstructedAllAssoc2GenMatched_Z", 120, -60, 60);
    mes_[label]["RecoAllAssoc2GenMatched_R"] =
        i.book1D("RecoAllAssoc2GenMatched_R",
                 "ReconstructedAllAssoc2GenMatched_R", 120, 0, 0.6);
    mes_[label]["RecoAllAssoc2GenMatched_Pt2"] =
        i.book1D("RecoAllAssoc2GenMatched_Pt2",
                 "ReconstructedAllAssoc2GenMatched_Sum-pt2", 8, &log_bins[0]);
    mes_[label]["RecoAllAssoc2GenMatched_NumTracks"] =
        i.book1D("RecoAllAssoc2GenMatched_NumTracks",
                 "ReconstructedAllAssoc2GenMatched_NumTracks", 200, 0., 200.);
    mes_[label]["RecoAllAssoc2GenMatched_ClosestDistanceZ"] = i.book1D(
        "RecoAllAssoc2GenMatched_ClosestDistanceZ",
        "ReconstructedAllAssoc2GenMatched_ClosestDistanceZ", 8, &log_bins[0]);

    // All Reconstructed Vertices  Multi-Matched to a Generated vertex. Used
    // for Duplicate-Rate plots
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumVertices"] = i.book1D(
        "RecoAllAssoc2GenMultiMatched_NumVertices",
        "ReconstructedAllAssoc2GenMultiMatched_NumVertices", 200, 0., 200.);
    mes_[label]["RecoAllAssoc2GenMultiMatched_X"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_X",
                 "ReconstructedAllAssoc2GenMultiMatched_X", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Y"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_Y",
                 "ReconstructedAllAssoc2GenMultiMatched_Y", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Z"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_Z",
                 "ReconstructedAllAssoc2GenMultiMatched_Z", 120, -60, 60);
    mes_[label]["RecoAllAssoc2GenMultiMatched_R"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_R",
                 "ReconstructedAllAssoc2GenMultiMatched_R", 120, 0, 0.6);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Pt2"] = i.book1D(
        "RecoAllAssoc2GenMultiMatched_Pt2",
        "ReconstructedAllAssoc2GenMultiMatched_Sum-pt2", 8, &log_bins[0]);
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumTracks"] = i.book1D(
        "RecoAllAssoc2GenMultiMatched_NumTracks",
        "ReconstructedAllAssoc2GenMultiMatched_NumTracks", 200, 0., 200.);
    mes_[label]["RecoAllAssoc2GenMultiMatched_ClosestDistanceZ"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_ClosestDistanceZ",
                 "ReconstructedAllAssoc2GenMultiMatched_ClosestDistanceZ", 8,
                 &log_bins[0]);
  }
}

void PrimaryVertexAnalyzer4PUSlimmed::fillGenericGenVertexHistograms(
    const simPrimaryVertex& v) {
  if (v.eventId.event() == 0) {
    mes_["root_folder"]["GenPV_X"]->Fill(v.x);
    mes_["root_folder"]["GenPV_Y"]->Fill(v.y);
    mes_["root_folder"]["GenPV_Z"]->Fill(v.z);
    mes_["root_folder"]["GenPV_R"]->Fill(v.r);
    mes_["root_folder"]["GenPV_Pt2"]->Fill(v.ptsq);
    mes_["root_folder"]["GenPV_NumTracks"]->Fill(v.nGenTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_["root_folder"]["GenPV_ClosestDistanceZ"]
          ->Fill(v.closest_vertex_distance_z);
  }
  mes_["root_folder"]["GenAllV_X"]->Fill(v.x);
  mes_["root_folder"]["GenAllV_Y"]->Fill(v.y);
  mes_["root_folder"]["GenAllV_Z"]->Fill(v.z);
  mes_["root_folder"]["GenAllV_R"]->Fill(v.r);
  mes_["root_folder"]["GenAllV_Pt2"]->Fill(v.ptsq);
  mes_["root_folder"]["GenAllV_NumTracks"]->Fill(v.nGenTrk);
  if (v.closest_vertex_distance_z > 0.)
    mes_["root_folder"]["GenAllV_ClosestDistanceZ"]
        ->Fill(v.closest_vertex_distance_z);
}

void PrimaryVertexAnalyzer4PUSlimmed::fillRecoAssociatedGenVertexHistograms(
    const std::string& label,
    const PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex& v) {
  mes_[label]["GenAllAssoc2Reco_X"]->Fill(v.x);
  mes_[label]["GenAllAssoc2Reco_Y"]->Fill(v.y);
  mes_[label]["GenAllAssoc2Reco_Z"]->Fill(v.z);
  mes_[label]["GenAllAssoc2Reco_R"]->Fill(v.r);
  mes_[label]["GenAllAssoc2Reco_Pt2"]->Fill(v.ptsq);
  mes_[label]["GenAllAssoc2Reco_NumTracks"]->Fill(v.nGenTrk);
  if (v.closest_vertex_distance_z > 0.)
    mes_[label]["GenAllAssoc2Reco_ClosestDistanceZ"]
        ->Fill(v.closest_vertex_distance_z);
  if (v.rec_vertices.size()) {
    mes_[label]["GenAllAssoc2RecoMatched_X"]->Fill(v.x);
    mes_[label]["GenAllAssoc2RecoMatched_Y"]->Fill(v.y);
    mes_[label]["GenAllAssoc2RecoMatched_Z"]->Fill(v.z);
    mes_[label]["GenAllAssoc2RecoMatched_R"]->Fill(v.r);
    mes_[label]["GenAllAssoc2RecoMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["GenAllAssoc2RecoMatched_NumTracks"]->Fill(v.nGenTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_[label]["GenAllAssoc2RecoMatched_ClosestDistanceZ"]
          ->Fill(v.closest_vertex_distance_z);
  }
  if (v.rec_vertices.size() > 1) {
    mes_[label]["GenAllAssoc2RecoMultiMatched_X"]->Fill(v.x);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Y"]->Fill(v.y);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Z"]->Fill(v.z);
    mes_[label]["GenAllAssoc2RecoMultiMatched_R"]->Fill(v.r);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumTracks"]->Fill(v.nGenTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_[label]["GenAllAssoc2RecoMultiMatched_ClosestDistanceZ"]
          ->Fill(v.closest_vertex_distance_z);
  }
}

void PrimaryVertexAnalyzer4PUSlimmed::fillGenAssociatedRecoVertexHistograms(
    const std::string& label,
    const PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex& v) {
  mes_[label]["RecoAllAssoc2Gen_X"]->Fill(v.x);
  mes_[label]["RecoAllAssoc2Gen_Y"]->Fill(v.y);
  mes_[label]["RecoAllAssoc2Gen_Z"]->Fill(v.z);
  mes_[label]["RecoAllAssoc2Gen_R"]->Fill(v.r);
  mes_[label]["RecoAllAssoc2Gen_Pt2"]->Fill(v.ptsq);
  mes_[label]["RecoAllAssoc2Gen_NumTracks"]->Fill(v.nRecoTrk);
  if (v.closest_vertex_distance_z > 0.)
    mes_[label]["RecoAllAssoc2Gen_ClosestDistanceZ"]
        ->Fill(v.closest_vertex_distance_z);
  if (v.sim_vertices.size()) {
    mes_[label]["RecoAllAssoc2GenMatched_X"]->Fill(v.x);
    mes_[label]["RecoAllAssoc2GenMatched_Y"]->Fill(v.y);
    mes_[label]["RecoAllAssoc2GenMatched_Z"]->Fill(v.z);
    mes_[label]["RecoAllAssoc2GenMatched_R"]->Fill(v.r);
    mes_[label]["RecoAllAssoc2GenMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["RecoAllAssoc2GenMatched_NumTracks"]->Fill(v.nRecoTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_[label]["RecoAllAssoc2GenMatched_ClosestDistanceZ"]
          ->Fill(v.closest_vertex_distance_z);
  }
  if (v.sim_vertices.size() > 1) {
    mes_[label]["RecoAllAssoc2GenMultiMatched_X"]->Fill(v.x);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Y"]->Fill(v.y);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Z"]->Fill(v.z);
    mes_[label]["RecoAllAssoc2GenMultiMatched_R"]->Fill(v.r);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumTracks"]->Fill(v.nRecoTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_[label]["RecoAllAssoc2GenMultiMatched_ClosestDistanceZ"]
          ->Fill(v.closest_vertex_distance_z);
  }
}

/* Extract information form TrackingParticles/TrackingVertex and fill
 * the helper class simPrimaryVertex with proper generation-level
 * information */
std::vector<PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex>
PrimaryVertexAnalyzer4PUSlimmed::getSimPVs(
    const edm::Handle<TrackingVertexCollection> tVC) {
  std::vector<PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex> simpv;
  int current_event = -1;

  if (verbose_) {
    std::cout << "getSimPVs TrackingVertexCollection " << std::endl;
  }

  for (TrackingVertexCollection::const_iterator v = tVC->begin();
       v != tVC->end(); ++v) {
    if (verbose_) {
      std::cout << "BunchX.EventId: " << v->eventId().bunchCrossing() << "."
                << (v->eventId()).event() << " Position: " << v->position()
                << " G4/HepMC Vertices: " << v->g4Vertices().size() << "/"
                << v->genVertices().size()
                << "   t = " << v->position().t() * 1.e12
                << "    == 0:" << (v->position().t() > 0) << std::endl;
      for (TrackingVertex::g4v_iterator gv = v->g4Vertices_begin();
           gv != v->g4Vertices_end(); gv++) {
        std::cout << *gv << std::endl;
      }
      std::cout << "----------" << std::endl;
    }  // end of verbose_ session

    // I'd rather change this and select only vertices that come from
    // BX=0.  We should keep only the first vertex from all the events
    // at BX=0.
    if (v->eventId().bunchCrossing() != 0) continue;
    if (v->eventId().event() != current_event) {
      current_event = v->eventId().event();
    } else {
      continue;
    }
    // TODO(rovere) is this really necessary?
    if (fabs(v->position().z()) > 1000) continue;  // skip funny junk vertices

    // could be a new vertex, check  all primaries found so far to avoid
    // multiple entries
    simPrimaryVertex sv(v->position().x(), v->position().y(),
                        v->position().z());
    sv.eventId = v->eventId();
    sv.sim_vertex = &(*v);

    for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin();
         iTrack != v->daughterTracks_end(); ++iTrack) {
      // TODO(rovere) isn't it always the case? Is it really worth
      // checking this out?
      // sv.eventId = (**iTrack).eventId();
      assert((**iTrack).eventId().bunchCrossing() == 0);
    }
    // TODO(rovere) maybe get rid of this old logic completely ... ?
    simPrimaryVertex* vp = NULL;  // will become non-NULL if a vertex
                                  // is found and then point to it
    for (std::vector<simPrimaryVertex>::iterator v0 = simpv.begin();
         v0 != simpv.end(); v0++) {
      if ((sv.eventId == v0->eventId) && (fabs(sv.x - v0->x) < 1e-5) &&
          (fabs(sv.y - v0->y) < 1e-5) && (fabs(sv.z - v0->z) < 1e-5)) {
        vp = &(*v0);
        break;
      }
    }
    if (!vp) {
      // this is a new vertex, add it to the list of sim-vertices
      simpv.push_back(sv);
      vp = &simpv.back();
      if (verbose_) {
        std::cout << "this is a new vertex " << sv.eventId.event() << "   "
                  << sv.x << " " << sv.y << " " << sv.z << std::endl;
      }
    } else {
      if (verbose_) {
        std::cout << "this is not a new vertex" << sv.x << " " << sv.y << " "
                  << sv.z << std::endl;
      }
    }

    // Loop over daughter track(s) as Tracking Particles
    for (TrackingVertex::tp_iterator iTP = v->daughterTracks_begin();
         iTP != v->daughterTracks_end(); ++iTP) {
      auto momentum = (*(*iTP)).momentum();
      const reco::Track* matched_best_reco_track = nullptr;
      double match_quality = -1;
      if (s2r_.find(*iTP) != s2r_.end()) {
        matched_best_reco_track = s2r_[*iTP][0].first.get();
        match_quality = s2r_[*iTP][0].second;
      }
      if (verbose_) {
        std::cout << "  Daughter momentum:      " << momentum;
        std::cout << "  Daughter type     " << (*(*iTP)).pdgId();
        std::cout << "  matched: " << (matched_best_reco_track != nullptr);
        std::cout << "  match-quality: " << match_quality;
        std::cout << std::endl;
      }
      vp->ptot.setPx(vp->ptot.x() + momentum.x());
      vp->ptot.setPy(vp->ptot.y() + momentum.y());
      vp->ptot.setPz(vp->ptot.z() + momentum.z());
      vp->ptot.setE(vp->ptot.e() + (**iTP).energy());
      vp->ptsq += ((**iTP).pt() * (**iTP).pt());
      // TODO(rovere) only select charged sim-particles? If so, maybe
      // put it as a configuration parameter?
      if (matched_best_reco_track) {
        vp->num_matched_reco_tracks++;
        vp->average_match_quality += match_quality;
      }
      // TODO(rovere) get rid of cuts on sim-tracks
      // TODO(rovere) be consistent between simulated tracks and
      // reconstructed tracks selection
      // count relevant particles
      if (((**iTP).pt() > 0.2) && (fabs((**iTP).eta()) < 2.5) &&
          (**iTP).charge() != 0) {
        vp->nGenTrk++;
      }
    }  // End of for loop on daugthers sim-particles
    if (vp->num_matched_reco_tracks)
      vp->average_match_quality /=
          static_cast<float>(vp->num_matched_reco_tracks);
    if (verbose_) {
      std::cout << "average number of associated tracks: "
                << vp->num_matched_reco_tracks / static_cast<float>(vp->nGenTrk)
                << " with average quality: " << vp->average_match_quality
                << std::endl;
    }
  }  // End of for loop on tracking vertices

  if (verbose_) {
    std::cout << "------- PrimaryVertexAnalyzer4PUSlimmed simPVs from "
                 "TrackingVertices "
                 "-------" << std::endl;
    for (std::vector<simPrimaryVertex>::iterator v0 = simpv.begin();
         v0 != simpv.end(); v0++) {
      std::cout << "z=" << v0->z << "  event=" << v0->eventId.event()
                << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;
  }  // End of for summary on discovered simulated primary vertices.

  // Now compute the closest distance in z between all simulated vertex
  for (std::vector<simPrimaryVertex>::iterator vsim = simpv.begin();
       vsim != simpv.end(); vsim++) {
    std::vector<simPrimaryVertex>::iterator vsim2 = vsim;
    vsim2++;
    for (; vsim2 != simpv.end(); vsim2++) {
      double distance_z = fabs(vsim->z - vsim2->z);
      // Initialize with the next-sibling in the vector: minimization
      // is performed by the next if.
      if (vsim->closest_vertex_distance_z < 0) {
        vsim->closest_vertex_distance_z = distance_z;
        continue;
      }
      if (distance_z < vsim->closest_vertex_distance_z)
        vsim->closest_vertex_distance_z = distance_z;
    }
  }
  return simpv;
}

/* Extract information form recoVertex and fill the helper class
 * recoPrimaryVertex with proper reco-level information */
std::vector<PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex>
PrimaryVertexAnalyzer4PUSlimmed::getRecoPVs(
    const edm::Handle<reco::VertexCollection> tVC) {
  std::vector<PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex> recopv;

  if (verbose_) {
    std::cout << "getRecoPVs TrackingVertexCollection " << std::endl;
  }

  for (std::vector<reco::Vertex>::const_iterator v = tVC->begin();
       v != tVC->end(); ++v) {
    if (verbose_) {
      std::cout << " Position: " << v->position() << std::endl;
    }

    // Skip junk vertices
    if (fabs(v->z()) > 1000) continue;
    if (v->isFake() || !v->isValid()) continue;

    recoPrimaryVertex sv(v->position().x(), v->position().y(),
                         v->position().z());
    sv.recVtx = &(*v);
    // this is a new vertex, add it to the list of sim-vertices
    recopv.push_back(sv);
    PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex* vp = &recopv.back();

    // Loop over daughter track(s)
    for (auto iTrack = v->tracks_begin(); iTrack != v->tracks_end(); ++iTrack) {
      auto momentum = (*(*iTrack)).innerMomentum();
      if (verbose_) {
        std::cout << "  Daughter momentum:      " << momentum;
        std::cout << std::endl;
      }
      vp->ptsq += (momentum.perp2());
      vp->nRecoTrk++;
    }  // End of for loop on daugthers reconstructed tracks
  }    // End of for loop on tracking vertices

  if (verbose_) {
    std::cout << "------- PrimaryVertexAnalyzer4PUSlimmed recoPVs from "
                 "VertexCollection "
                 "-------" << std::endl;
    for (std::vector<recoPrimaryVertex>::iterator v0 = recopv.begin();
         v0 != recopv.end(); v0++) {
      std::cout << "z=" << v0->z << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;
  }  // End of for summary on reconstructed primary vertices.

  // Now compute the closest distance in z between all reconstructed vertex
  for (std::vector<recoPrimaryVertex>::iterator vreco = recopv.begin();
       vreco != recopv.end(); vreco++) {
    std::vector<recoPrimaryVertex>::iterator vreco2 = vreco;
    vreco2++;
    for (; vreco2 != recopv.end(); vreco2++) {
      double distance_z = fabs(vreco->z - vreco2->z);
      // Initialize with the next-sibling in the vector: minimization
      // is performed by the next if.
      if (vreco->closest_vertex_distance_z < 0) {
        vreco->closest_vertex_distance_z = distance_z;
        continue;
      }
      if (distance_z < vreco->closest_vertex_distance_z)
        vreco->closest_vertex_distance_z = distance_z;
    }
  }
  return recopv;
}

// ------------ method called to produce the data  ------------
void PrimaryVertexAnalyzer4PUSlimmed::matchSim2RecoVertices(
    std::vector<simPrimaryVertex>& simpv,
    const reco::VertexCollection& reco_vertices) {
  if (verbose_) {
    std::cout << "PrimaryVertexAnalyzer4PUSlimmed::matchSim2RecoVertices "
              << reco_vertices.size() << std::endl;
  }
  for (std::vector<simPrimaryVertex>::iterator vsim = simpv.begin();
       vsim != simpv.end(); vsim++) {
    for (std::vector<reco::Vertex>::const_iterator vrec = reco_vertices.begin();
         vrec != reco_vertices.end(); vrec++) {
      if (vrec->isFake() || vrec->ndof() < 0.) {
        continue;  // skip fake vertices
      }
      if (verbose_) {
        std::cout << "Considering reconstructed vertex at Z:" << vrec->z()
                  << std::endl;
      }
      if ((fabs(vrec->z() - vsim->z) / vrec->zError()) < sigma_z_match_) {
        vsim->rec_vertices.push_back(&(*vrec));
        if (verbose_) {
          std::cout << "Trying a matching vertex for " << vsim->z << " at "
                    << vrec->z() << " with sign: "
                    << fabs(vrec->z() - vsim->z) / vrec->zError() << std::endl;
        }
      }
    }  // end of loop on reconstructed vertices
    if (verbose_) {
      if (vsim->rec_vertices.size()) {
        for (auto const& v : vsim->rec_vertices) {
          std::cout << "Found a matching vertex for " << vsim->z << " at "
                    << v->z()
                    << " with sign: " << fabs(v->z() - vsim->z) / v->zError()
                    << std::endl;
        }
      } else {
        std::cout << "No matching vertex for " << vsim->z << std::endl;
      }
    }
  }  // end for loop on simulated vertices
}

void PrimaryVertexAnalyzer4PUSlimmed::matchReco2SimVertices(
    std::vector<recoPrimaryVertex>& recopv,
    const TrackingVertexCollection& gen_vertices) {
  for (std::vector<recoPrimaryVertex>::iterator vrec = recopv.begin();
       vrec != recopv.end(); vrec++) {
    int current_event = -1;
    for (std::vector<TrackingVertex>::const_iterator vsim =
             gen_vertices.begin();
         vsim != gen_vertices.end(); vsim++) {
      // Keep only signal events
      if (vsim->eventId().bunchCrossing() != 0) continue;

      // Keep only the primary vertex for each PU event
      if (vsim->eventId().event() != current_event) {
        current_event = vsim->eventId().event();
      } else {
        continue;
      }

      // if the matching critera are fulfilled, accept all the gen-vertices
      // that are close in z, in unit of sigma_z of the reconstructed
      // vertex, at least of sigma_z_match_.
      if ((fabs(vrec->z - vsim->position().z()) / vrec->recVtx->zError()) <
          sigma_z_match_) {
        vrec->sim_vertices.push_back(&(*vsim));
        if (verbose_) {
          std::cout << "Matching Reco vertex for " << vrec->z
                    << " at Sim Vertex " << vsim->position().z()
                    << " with sign: " << fabs(vsim->position().z() - vrec->z) /
                                             vrec->recVtx->zError()
                    << std::endl;
        }
      }
    }  // end of loop on simulated vertices
    if (verbose_) {
      for (auto v : vrec->sim_vertices) {
        std::cout << "Found a matching vertex for reco: " << vrec->z
                  << " at gen:" << v->position().z() << " with sign: "
                  << fabs(vrec->z - v->position().z()) / vrec->recVtx->zError()
                  << std::endl;
      }
    }
  }  // end for loop on reconstructed vertices
}

void PrimaryVertexAnalyzer4PUSlimmed::analyze(const edm::Event& iEvent,
                                              const edm::EventSetup& iSetup) {
  using std::vector;
  using std::cout;
  using std::endl;
  using edm::Handle;
  using edm::View;
  using edm::LogInfo;
  using namespace reco;

  std::vector<float> pileUpInfo_z;

  // get the pileup information
  edm::Handle<std::vector<PileupSummaryInfo> > puinfoH;
  if (iEvent.getByToken(vecPileupSummaryInfoToken_, puinfoH)) {
    for (auto const& pu_info : *puinfoH.product()) {
      mes_["root_folder"]["GenVtx_vs_BX"]
          ->Fill(pu_info.getBunchCrossing(), pu_info.getPU_NumInteractions());
      if (pu_info.getBunchCrossing() == 0) {
        pileUpInfo_z = pu_info.getPU_zpositions();
        if (verbose_) {
          for (auto const& p : pileUpInfo_z) {
            std::cout << "PileUpInfo on Z vertex: " << p << std::endl;
          }
        }
        break;
      }
    }
  }

  Handle<reco::TrackCollection> recTrks;
  iEvent.getByToken(recoTrackCollectionToken_, recTrks);

  // for the associator
  Handle<View<Track> > trackCollectionH;
  iEvent.getByToken(edmView_recoTrack_Token_, trackCollectionH);

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  edm::Handle<TrackingVertexCollection> TVCollectionH;
  bool gotTP =
      iEvent.getByToken(trackingParticleCollectionToken_, TPCollectionH);
  bool gotTV = iEvent.getByToken(trackingVertexCollectionToken_, TVCollectionH);

  // TODO(rovere) the idea is to put in case a trackselector in fron
  // of this module and then use its label to get the selected tracks
  // out of the event instead of making an hard-coded selection in the
  // code.

  if (gotTP) {
    // TODO(rovere) fetch an already existing collection from the
    // event instead of making another association on the fly???
    if (use_TP_associator_) {
      edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
      iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",
                                              theHitsAssociator);
      associatorByHits_ = reinterpret_cast<const TrackAssociatorBase*>(
          theHitsAssociator.product());
      r2s_ = associatorByHits_->associateRecoToSim(
          trackCollectionH, TPCollectionH, &iEvent, &iSetup);
      s2r_ = associatorByHits_->associateSimToReco(
          trackCollectionH, TPCollectionH, &iEvent, &iSetup);
    }
  }

  int label_index = -1;
  for (auto const& vertex_token : reco_vertex_collection_tokens_) {
    std::vector<simPrimaryVertex> simpv;  // a list of simulatedprimary
                                          // MC vertices

    std::vector<recoPrimaryVertex> recopv;  // a list of recontructed
                                            // primary MC vertices
    std::string label = reco_vertex_collections_[++label_index].label();
    Handle<reco::VertexCollection> recVtxs;
    if (!iEvent.getByToken(vertex_token, recVtxs)) {
      LogInfo("PrimaryVertexAnalyzer4PUSlimmed")
          << "Skipping vertex collection: " << label << " since it is missing."
          << std::endl;
      continue;
    }
    if (gotTV) {
      // TODO(rovere) use move semantic?
      simpv = getSimPVs(TVCollectionH);
      matchSim2RecoVertices(simpv, *recVtxs.product());
      recopv = getRecoPVs(recVtxs);
      matchReco2SimVertices(recopv, *TVCollectionH.product());
    }

    int num_total_gen_vertices_assoc2reco = 0;
    int num_total_reco_vertices_assoc2gen = 0;
    int num_total_gen_vertices_multiassoc2reco = 0;
    int num_total_reco_vertices_multiassoc2gen = 0;
    for (auto const& v : simpv) {
      // TODO(rovere) put selectors here in front of fill* methods.
      if (v.eventId.event() == 0) {
        if (std::find(v.rec_vertices.begin(), v.rec_vertices.end(),
                      &((*recVtxs.product())[0])) != v.rec_vertices.end()) {
          mes_[label]["MisTagRate"]->Fill(1.);
        } else {
          mes_[label]["MisTagRate"]->Fill(0.0);
          // Now check at which location the Simulated PV has been
          // reconstructed in the primvary vertex collection
          // at-hand. Mark it with fake index -1 if it was not
          // recontructed at all.
          auto iv = (*recVtxs.product()).begin();
          int pv_position_in_reco_collection = 0;
          while (++iv != (*recVtxs.product()).end()) {
            pv_position_in_reco_collection++;
            if (std::find(v.rec_vertices.begin(), v.rec_vertices.end(),
                          &(*iv)) != v.rec_vertices.end()) {
              mes_[label]["TruePVLocationIndex"]
                  ->Fill(pv_position_in_reco_collection);
              break;
            }
          }
          // If we reached the end, it means that the Simulated PV has not
          // been associated to any recontructed vertex: mark it as
          // missing in the reconstructed vertex collection using the fake
          // index -1.
          if (iv == (*recVtxs.product()).end())
            mes_[label]["TruePVLocationIndex"]->Fill(-1.);
        }
      }
      if (v.rec_vertices.size()) num_total_gen_vertices_assoc2reco++;
      if (v.rec_vertices.size() > 1) num_total_gen_vertices_multiassoc2reco++;
      // No need to N-tplicate the Gen-related cumulative histograms:
      // fill them only at the first iteration
      if (label_index == 0) fillGenericGenVertexHistograms(v);
      fillRecoAssociatedGenVertexHistograms(label, v);
    }
    mes_[label]["GenAllAssoc2Reco_NumVertices"]
        ->Fill(simpv.size(), simpv.size());
    mes_[label]["GenAllAssoc2RecoMatched_NumVertices"]
        ->Fill(simpv.size(), num_total_gen_vertices_assoc2reco);
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumVertices"]
        ->Fill(simpv.size(), num_total_gen_vertices_multiassoc2reco);
    for (auto const& v : recopv) {
      fillGenAssociatedRecoVertexHistograms(label, v);
      if (v.sim_vertices.size()) num_total_reco_vertices_assoc2gen++;
      if (v.sim_vertices.size() > 1) num_total_reco_vertices_multiassoc2gen++;
    }
    mes_[label]["RecoAllAssoc2Gen_NumVertices"]
        ->Fill(recopv.size(), recopv.size());
    mes_[label]["RecoAllAssoc2GenMatched_NumVertices"]
        ->Fill(recopv.size(), num_total_reco_vertices_assoc2gen);
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumVertices"]
        ->Fill(recopv.size(), num_total_reco_vertices_multiassoc2gen);
    mes_[label]["RecoVtx_vs_GenVtx"]->Fill(simpv.size(), recopv.size());
    mes_[label]["MatchedRecoVtx_vs_GenVtx"]
        ->Fill(simpv.size(), num_total_reco_vertices_assoc2gen);
  }
}  // end of analyze
