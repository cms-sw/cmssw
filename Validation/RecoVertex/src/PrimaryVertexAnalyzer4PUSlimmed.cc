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
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"

#include <numeric>

namespace {
  template <typename T, size_t N>
  std::array<T, N + 1> makeLogBins(const double min, const double max) {
    const double minLog10 = std::log10(min);
    const double maxLog10 = std::log10(max);
    const double width = (maxLog10 - minLog10) / N;
    std::array<T, N + 1> ret;
    ret[0] = std::pow(10, minLog10);
    const double mult = std::pow(10, width);
    for (size_t i = 1; i <= N; ++i) {
      ret[i] = ret[i - 1] * mult;
    }
    return ret;
  }
}  // namespace

//
// constructors and destructor
//
PrimaryVertexAnalyzer4PUSlimmed::PrimaryVertexAnalyzer4PUSlimmed(const edm::ParameterSet& iConfig)
    : verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      use_only_charged_tracks_(iConfig.getUntrackedParameter<bool>("use_only_charged_tracks", true)),
      do_generic_sim_plots_(iConfig.getUntrackedParameter<bool>("do_generic_sim_plots")),
      root_folder_(iConfig.getUntrackedParameter<std::string>("root_folder", "Validation/Vertices")),
      vecPileupSummaryInfoToken_(consumes<std::vector<PileupSummaryInfo>>(edm::InputTag(std::string("addPileupInfo")))),
      trackingParticleCollectionToken_(consumes<TrackingParticleCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("trackingParticleCollection"))),
      trackingVertexCollectionToken_(
          consumes<TrackingVertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackingVertexCollection"))),
      simToRecoAssociationToken_(
          consumes<reco::SimToRecoCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociatorMap"))),
      recoToSimAssociationToken_(
          consumes<reco::RecoToSimCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociatorMap"))),
      vertexAssociatorToken_(consumes<reco::VertexToTrackingVertexAssociator>(
          iConfig.getUntrackedParameter<edm::InputTag>("vertexAssociator"))),
      nPUbins_(iConfig.getParameter<unsigned int>("nPUbins")) {
  reco_vertex_collections_ = iConfig.getParameter<std::vector<edm::InputTag>>("vertexRecoCollections");
  for (auto const& l : reco_vertex_collections_) {
    reco_vertex_collection_tokens_.push_back(
        edm::EDGetTokenT<edm::View<reco::Vertex>>(consumes<edm::View<reco::Vertex>>(l)));
  }
  errorPrintedForColl_.resize(reco_vertex_collections_.size(), false);
}

PrimaryVertexAnalyzer4PUSlimmed::~PrimaryVertexAnalyzer4PUSlimmed() {}

//
// member functions
//
void PrimaryVertexAnalyzer4PUSlimmed::bookHistograms(DQMStore::IBooker& i,
                                                     edm::Run const& iRun,
                                                     edm::EventSetup const& iSetup) {
  // TODO(rovere) make this booking method shorter and smarter,
  // factorizing similar histograms with different prefix in a single
  // method call.
  float log_bins[31] = {0.0,  0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01,
                        0.02, 0.04,   0.06,   0.08,   0.1,    0.2,   0.4,   0.6,   0.8,   1.0,   2.0,
                        4.0,  6.0,    8.0,    10.0,   20.0,   40.0,  60.0,  80.0,  100.0};
  auto const log_mergez_bins = makeLogBins<float, 16>(1e-4, 1);  // 4 bins / 10x

  auto const log_pt_bins = makeLogBins<float, 100>(0.1, 1e4);
  float log_pt2_bins[16] = {
      0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0};
  float log_ntrk_bins[25] = {0.,   2.0,  4.0,  6.0,  8.0,  10.,  12.0, 14.0, 16.0, 18.0,  22.0,  26.0, 30.0,
                             35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0};

  // TODO(rovere) Possibly change or add the main DQMStore booking
  // interface to allow booking a TProfile with variable bin-width
  // using an array of floats, as done for the TH1F case, not of
  // doubles.
  double log_pt2_bins_double[16] = {
      0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0};

  i.setCurrentFolder(root_folder_);
  if (do_generic_sim_plots_) {
    mes_["root_folder"]["GenVtx_vs_BX"] =
        i.book2D("GenVtx_vs_BX", "GenVtx_vs_BX", 16, -12.5, 3.5, nPUbins_, 0., double(nPUbins_));
    // Generated Primary Vertex Plots
    mes_["root_folder"]["GenPV_X"] = i.book1D("GenPV_X", "GeneratedPV_X", 120, -0.6, 0.6);
    mes_["root_folder"]["GenPV_Y"] = i.book1D("GenPV_Y", "GeneratedPV_Y", 120, -0.6, 0.6);
    mes_["root_folder"]["GenPV_Z"] = i.book1D("GenPV_Z", "GeneratedPV_Z", 120, -60., 60.);
    mes_["root_folder"]["GenPV_R"] = i.book1D("GenPV_R", "GeneratedPV_R", 120, 0, 0.6);
    mes_["root_folder"]["GenPV_Pt2"] = i.book1D("GenPV_Pt2", "GeneratedPV_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_["root_folder"]["GenPV_NumTracks"] =
        i.book1D("GenPV_NumTracks", "GeneratedPV_NumTracks", 24, &log_ntrk_bins[0]);
    mes_["root_folder"]["GenPV_ClosestDistanceZ"] =
        i.book1D("GenPV_ClosestDistanceZ", "GeneratedPV_ClosestDistanceZ", 30, &log_bins[0]);

    // All Generated Vertices, used for efficiency plots
    mes_["root_folder"]["GenAllV_NumVertices"] =
        i.book1D("GenAllV_NumVertices", "GeneratedAllV_NumVertices", int(nPUbins_ / 2), 0., double(nPUbins_));
    mes_["root_folder"]["GenAllV_X"] = i.book1D("GenAllV_X", "GeneratedAllV_X", 120, -0.6, 0.6);
    mes_["root_folder"]["GenAllV_Y"] = i.book1D("GenAllV_Y", "GeneratedAllV_Y", 120, -0.6, 0.6);
    mes_["root_folder"]["GenAllV_Z"] = i.book1D("GenAllV_Z", "GeneratedAllV_Z", 120, -60, 60);
    mes_["root_folder"]["GenAllV_R"] = i.book1D("GenAllV_R", "GeneratedAllV_R", 120, 0, 0.6);
    mes_["root_folder"]["GenAllV_Pt2"] = i.book1D("GenAllV_Pt2", "GeneratedAllV_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_["root_folder"]["GenAllV_NumTracks"] =
        i.book1D("GenAllV_NumTracks", "GeneratedAllV_NumTracks", 24, &log_ntrk_bins[0]);
    mes_["root_folder"]["GenAllV_ClosestDistanceZ"] =
        i.book1D("GenAllV_ClosestDistanceZ", "GeneratedAllV_ClosestDistanceZ", 30, &log_bins[0]);
    mes_["root_folder"]["GenAllV_PairDistanceZ"] =
        i.book1D("GenAllV_PairDistanceZ", "GeneratedAllV_PairDistanceZ", 1000, 0, 20);
    mes_["root_folder"]["SignalIsHighestPt2"] = i.book1D("SignalIsHighestPt2", "SignalIsHighestPt2", 2, -0.5, 1.5);
  }

  for (auto const& l : reco_vertex_collections_) {
    std::string label = l.label();
    std::string current_folder = root_folder_ + "/" + label;
    i.setCurrentFolder(current_folder);

    auto book1d = [&](const char* name, int bins, double min, double max) {
      mes_[label][name] = i.book1D(name, name, bins, min, max);
    };
    auto book1dlogx = [&](const char* name, int bins, const float* xbinedges) {
      mes_[label][name] = i.book1D(name, name, bins, xbinedges);
    };
    auto book2d = [&](const char* name, int xbins, double xmin, double xmax, int ybins, double ymin, double ymax) {
      mes_[label][name] = i.book2D(name, name, xbins, xmin, xmax, ybins, ymin, ymax);
    };
    auto book2dlogx = [&](const char* name, int xbins, const float* xbinedges, int ybins, double ymin, double ymax) {
      auto me = i.book2D(name, name, xbins, xbinedges[0], xbinedges[xbins], ybins, ymin, ymax);
      me->getTH2F()->GetXaxis()->Set(xbins, xbinedges);
      mes_[label][name] = me;
    };

    mes_[label]["RecoVtx_vs_GenVtx"] = i.bookProfile(
        "RecoVtx_vs_GenVtx", "RecoVtx_vs_GenVtx", nPUbins_, 0., double(nPUbins_), nPUbins_, 0., double(nPUbins_));
    mes_[label]["MatchedRecoVtx_vs_GenVtx"] = i.bookProfile("MatchedRecoVtx_vs_GenVtx",
                                                            "MatchedRecoVtx_vs_GenVtx",
                                                            int(nPUbins_ / 2),
                                                            0.,
                                                            double(nPUbins_),
                                                            nPUbins_,
                                                            0.,
                                                            double(nPUbins_));
    mes_[label]["KindOfSignalPV"] = i.book1D("KindOfSignalPV", "KindOfSignalPV", 9, -0.5, 8.5);
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(1, "!Highest!Assoc2Any");
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(2, "Highest!Assoc2Any");
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(3, "!HighestAssoc2First");
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(4, "HighestAssoc2First");
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(5, "!HighestAssoc2!First");
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(6, "HighestAssoc2!First");
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(7, "!HighestAssoc2First");
    mes_[label]["KindOfSignalPV"]->getTH1()->GetXaxis()->SetBinLabel(8, "HighestAssoc2First");
    mes_[label]["MisTagRate"] = i.book1D("MisTagRate", "MisTagRate", 2, -0.5, 1.5);
    mes_[label]["MisTagRate_vs_PU"] =
        i.bookProfile("MisTagRate_vs_PU", "MisTagRate_vs_PU", int(nPUbins_ / 2), 0., double(nPUbins_), 2, 0., 1.);
    mes_[label]["MisTagRate_vs_sum-pt2"] =
        i.bookProfile("MisTagRate_vs_sum-pt2", "MisTagRate_vs_sum-pt2", 15, &log_pt2_bins_double[0], 2, 0., 1.);
    mes_[label]["MisTagRate_vs_Z"] = i.bookProfile("MisTagRate_vs_Z", "MisTagRate_vs_Z", 120, -60., 60., 2, 0., 1.);
    mes_[label]["MisTagRate_vs_R"] = i.bookProfile("MisTagRate_vs_R", "MisTagRate_vs_R", 120, 0., 0.6, 2, 0., 1.);
    mes_[label]["MisTagRate_vs_NumTracks"] = i.bookProfile(
        "MisTagRate_vs_NumTracks", "MisTagRate_vs_NumTracks", int(nPUbins_ / 2), 0., double(nPUbins_), 2, 0., 1.);
    mes_[label]["MisTagRateSignalIsHighest"] =
        i.book1D("MisTagRateSignalIsHighest", "MisTagRateSignalIsHighest", 2, -0.5, 1.5);
    mes_[label]["MisTagRateSignalIsHighest_vs_PU"] = i.bookProfile(
        "MisTagRateSignalIsHighest_vs_PU", "MisTagRateSignalIsHighest_vs_PU", nPUbins_, 0., double(nPUbins_), 2, 0., 1.);
    mes_[label]["MisTagRateSignalIsHighest_vs_sum-pt2"] = i.bookProfile("MisTagRateSignalIsHighest_vs_sum-pt2",
                                                                        "MisTagRateSignalIsHighest_vs_sum-pt2",
                                                                        15,
                                                                        &log_pt2_bins_double[0],
                                                                        2,
                                                                        0.,
                                                                        1.);
    mes_[label]["MisTagRateSignalIsHighest_vs_Z"] =
        i.bookProfile("MisTagRateSignalIsHighest_vs_Z", "MisTagRateSignalIsHighest_vs_Z", 120, -60., 60., 2, 0., 1.);
    mes_[label]["MisTagRateSignalIsHighest_vs_R"] =
        i.bookProfile("MisTagRateSignalIsHighest_vs_R", "MisTagRateSignalIsHighest_vs_R", 120, 0., 0.6, 2, 0., 1.);
    mes_[label]["MisTagRateSignalIsHighest_vs_NumTracks"] = i.bookProfile("MisTagRateSignalIsHighest_vs_NumTracks",
                                                                          "MisTagRateSignalIsHighest_vs_NumTracks",
                                                                          int(nPUbins_ / 2),
                                                                          0.,
                                                                          double(nPUbins_),
                                                                          2,
                                                                          0.,
                                                                          1.);
    mes_[label]["MisTagRateSignalIsNotHighest"] =
        i.book1D("MisTagRateSignalIsNotHighest", "MisTagRateSignalIsNotHighest", 2, -0.5, 1.5);
    mes_[label]["MisTagRateSignalIsNotHighest_vs_PU"] = i.bookProfile("MisTagRateSignalIsNotHighest_vs_PU",
                                                                      "MisTagRateSignalIsNotHighest_vs_PU",
                                                                      int(nPUbins_ / 2),
                                                                      0.,
                                                                      double(nPUbins_),
                                                                      2,
                                                                      0.,
                                                                      1.);
    mes_[label]["MisTagRateSignalIsNotHighest_vs_sum-pt2"] = i.bookProfile("MisTagRateSignalIsNotHighest_vs_sum-pt2",
                                                                           "MisTagRateSignalIsNotHighest_vs_sum-pt2",
                                                                           15,
                                                                           &log_pt2_bins_double[0],
                                                                           2,
                                                                           0.,
                                                                           1.);
    mes_[label]["MisTagRateSignalIsNotHighest_vs_Z"] = i.bookProfile(
        "MisTagRateSignalIsNotHighest_vs_Z", "MisTagRateSignalIsNotHighest_vs_Z", 120, -60., 60., 2, 0., 1.);
    mes_[label]["MisTagRateSignalIsNotHighest_vs_R"] = i.bookProfile(
        "MisTagRateSignalIsNotHighest_vs_R", "MisTagRateSignalIsNotHighest_vs_R", 120, 0., 0.6, 2, 0., 1.);
    mes_[label]["MisTagRateSignalIsNotHighest_vs_NumTracks"] =
        i.bookProfile("MisTagRateSignalIsNotHighest_vs_NumTracks",
                      "MisTagRateSignalIsNotHighest_vs_NumTracks",
                      int(nPUbins_ / 2),
                      0.,
                      double(nPUbins_),
                      2,
                      0.,
                      1.);
    mes_[label]["TruePVLocationIndex"] =
        i.book1D("TruePVLocationIndex", "TruePVLocationIndexInRecoVertexCollection", 12, -1.5, 10.5);
    mes_[label]["TruePVLocationIndexCumulative"] =
        i.book1D("TruePVLocationIndexCumulative", "TruePVLocationIndexInRecoVertexCollectionCumulative", 3, -1.5, 1.5);
    mes_[label]["TruePVLocationIndexSignalIsHighest"] =
        i.book1D("TruePVLocationIndexSignalIsHighest",
                 "TruePVLocationIndexSignalIsHighestInRecoVertexCollection",
                 12,
                 -1.5,
                 10.5);
    mes_[label]["TruePVLocationIndexSignalIsNotHighest"] =
        i.book1D("TruePVLocationIndexSignalIsNotHighest",
                 "TruePVLocationIndexSignalIsNotHighestInRecoVertexCollection",
                 12,
                 -1.5,
                 10.5);
    // All Generated Vertices. Used for Efficiency plots We kind of
    // duplicate plots here in case we want to perform more detailed
    // studies on a selection of generated vertices, not on all of them.
    mes_[label]["GenAllAssoc2Reco_NumVertices"] = i.book1D(
        "GenAllAssoc2Reco_NumVertices", "GeneratedAllAssoc2Reco_NumVertices", int(nPUbins_ / 2), 0., double(nPUbins_));
    mes_[label]["GenAllAssoc2Reco_X"] = i.book1D("GenAllAssoc2Reco_X", "GeneratedAllAssoc2Reco_X", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2Reco_Y"] = i.book1D("GenAllAssoc2Reco_Y", "GeneratedAllAssoc2Reco_Y", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2Reco_Z"] = i.book1D("GenAllAssoc2Reco_Z", "GeneratedAllAssoc2Reco_Z", 120, -60, 60);
    mes_[label]["GenAllAssoc2Reco_R"] = i.book1D("GenAllAssoc2Reco_R", "GeneratedAllAssoc2Reco_R", 120, 0, 0.6);
    mes_[label]["GenAllAssoc2Reco_Pt2"] =
        i.book1D("GenAllAssoc2Reco_Pt2", "GeneratedAllAssoc2Reco_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_[label]["GenAllAssoc2Reco_NumTracks"] =
        i.book1D("GenAllAssoc2Reco_NumTracks", "GeneratedAllAssoc2Reco_NumTracks", 24, &log_ntrk_bins[0]);
    mes_[label]["GenAllAssoc2Reco_ClosestDistanceZ"] =
        i.book1D("GenAllAssoc2Reco_ClosestDistanceZ", "GeneratedAllAssoc2Reco_ClosestDistanceZ", 30, &log_bins[0]);
    book1d("GenPVAssoc2RecoPV_X", 120, -0.6, 0.6);
    book1d("GenPVAssoc2RecoPV_Y", 120, -0.6, 0.6);
    book1d("GenPVAssoc2RecoPV_Z", 120, -60, 60);

    // All Generated Vertices Matched to a Reconstructed vertex. Used
    // for Efficiency plots
    mes_[label]["GenAllAssoc2RecoMatched_NumVertices"] = i.book1D("GenAllAssoc2RecoMatched_NumVertices",
                                                                  "GeneratedAllAssoc2RecoMatched_NumVertices",
                                                                  int(nPUbins_ / 2),
                                                                  0.,
                                                                  double(nPUbins_));
    mes_[label]["GenAllAssoc2RecoMatched_X"] =
        i.book1D("GenAllAssoc2RecoMatched_X", "GeneratedAllAssoc2RecoMatched_X", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMatched_Y"] =
        i.book1D("GenAllAssoc2RecoMatched_Y", "GeneratedAllAssoc2RecoMatched_Y", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMatched_Z"] =
        i.book1D("GenAllAssoc2RecoMatched_Z", "GeneratedAllAssoc2RecoMatched_Z", 120, -60, 60);
    mes_[label]["GenAllAssoc2RecoMatched_R"] =
        i.book1D("GenAllAssoc2RecoMatched_R", "GeneratedAllAssoc2RecoMatched_R", 120, 0, 0.6);
    mes_[label]["GenAllAssoc2RecoMatched_Pt2"] =
        i.book1D("GenAllAssoc2RecoMatched_Pt2", "GeneratedAllAssoc2RecoMatched_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_[label]["GenAllAssoc2RecoMatched_NumTracks"] =
        i.book1D("GenAllAssoc2RecoMatched_NumTracks", "GeneratedAllAssoc2RecoMatched_NumTracks", 24, &log_ntrk_bins[0]);
    mes_[label]["GenAllAssoc2RecoMatched_ClosestDistanceZ"] = i.book1D(
        "GenAllAssoc2RecoMatched_ClosestDistanceZ", "GeneratedAllAssoc2RecoMatched_ClosestDistanceZ", 30, &log_bins[0]);
    book1d("GenPVAssoc2RecoPVMatched_X", 120, -0.6, 0.6);
    book1d("GenPVAssoc2RecoPVMatched_Y", 120, -0.6, 0.6);
    book1d("GenPVAssoc2RecoPVMatched_Z", 120, -60, 60);

    // All Generated Vertices Multi-Matched to a Reconstructed vertex. Used
    // for Duplicate rate plots
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumVertices"] = i.book1D("GenAllAssoc2RecoMultiMatched_NumVertices",
                                                                       "GeneratedAllAssoc2RecoMultiMatched_NumVertices",
                                                                       int(nPUbins_ / 2),
                                                                       0.,
                                                                       double(nPUbins_));
    mes_[label]["GenAllAssoc2RecoMultiMatched_X"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_X", "GeneratedAllAssoc2RecoMultiMatched_X", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Y"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_Y", "GeneratedAllAssoc2RecoMultiMatched_Y", 120, -0.6, 0.6);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Z"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_Z", "GeneratedAllAssoc2RecoMultiMatched_Z", 120, -60, 60);
    mes_[label]["GenAllAssoc2RecoMultiMatched_R"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_R", "GeneratedAllAssoc2RecoMultiMatched_R", 120, 0, 0.6);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Pt2"] = i.book1D(
        "GenAllAssoc2RecoMultiMatched_Pt2", "GeneratedAllAssoc2RecoMultiMatched_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumTracks"] = i.book1D("GenAllAssoc2RecoMultiMatched_NumTracks",
                                                                     "GeneratedAllAssoc2RecoMultiMatched_NumTracks",
                                                                     24,
                                                                     &log_ntrk_bins[0]);
    mes_[label]["GenAllAssoc2RecoMultiMatched_ClosestDistanceZ"] =
        i.book1D("GenAllAssoc2RecoMultiMatched_ClosestDistanceZ",
                 "GeneratedAllAssoc2RecoMultiMatched_ClosestDistanceZ",
                 30,
                 &log_bins[0]);

    // All Reco Vertices. Used for {Fake,Duplicate}-Rate plots
    mes_[label]["RecoAllAssoc2Gen_NumVertices"] = i.book1D("RecoAllAssoc2Gen_NumVertices",
                                                           "ReconstructedAllAssoc2Gen_NumVertices",
                                                           int(nPUbins_ / 2),
                                                           0.,
                                                           double(nPUbins_));
    mes_[label]["RecoAllAssoc2Gen_X"] = i.book1D("RecoAllAssoc2Gen_X", "ReconstructedAllAssoc2Gen_X", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2Gen_Y"] = i.book1D("RecoAllAssoc2Gen_Y", "ReconstructedAllAssoc2Gen_Y", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2Gen_Z"] = i.book1D("RecoAllAssoc2Gen_Z", "ReconstructedAllAssoc2Gen_Z", 120, -60, 60);
    mes_[label]["RecoAllAssoc2Gen_R"] = i.book1D("RecoAllAssoc2Gen_R", "ReconstructedAllAssoc2Gen_R", 120, 0, 0.6);
    mes_[label]["RecoAllAssoc2Gen_Pt2"] =
        i.book1D("RecoAllAssoc2Gen_Pt2", "ReconstructedAllAssoc2Gen_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_[label]["RecoAllAssoc2Gen_Ndof"] =
        i.book1D("RecoAllAssoc2Gen_Ndof", "ReconstructedAllAssoc2Gen_Ndof", 120, 0., 240.);
    mes_[label]["RecoAllAssoc2Gen_NumTracks"] =
        i.book1D("RecoAllAssoc2Gen_NumTracks", "ReconstructedAllAssoc2Gen_NumTracks", 24, &log_ntrk_bins[0]);
    mes_[label]["RecoAllAssoc2Gen_PU"] =
        i.book1D("RecoAllAssoc2Gen_PU", "ReconstructedAllAssoc2Gen_PU", int(nPUbins_ / 2), 0., double(nPUbins_));
    mes_[label]["RecoAllAssoc2Gen_ClosestDistanceZ"] =
        i.book1D("RecoAllAssoc2Gen_ClosestDistanceZ", "ReconstructedAllAssoc2Gen_ClosestDistanceZ", 30, &log_bins[0]);
    mes_[label]["RecoAllAssoc2GenProperties"] =
        i.book1D("RecoAllAssoc2GenProperties", "ReconstructedAllAssoc2Gen_Properties", 8, -0.5, 7.5);
    mes_[label]["RecoAllAssoc2Gen_PairDistanceZ"] =
        i.book1D("RecoAllAssoc2Gen_PairDistanceZ", "RecoAllAssoc2Gen_PairDistanceZ", 1000, 0, 20);

    // All Reconstructed Vertices Matched to a Generated vertex. Used
    // for Fake-Rate plots
    mes_[label]["RecoAllAssoc2GenMatched_NumVertices"] = i.book1D("RecoAllAssoc2GenMatched_NumVertices",
                                                                  "ReconstructedAllAssoc2GenMatched_NumVertices",
                                                                  int(nPUbins_ / 2),
                                                                  0.,
                                                                  double(nPUbins_));
    mes_[label]["RecoAllAssoc2GenMatched_X"] =
        i.book1D("RecoAllAssoc2GenMatched_X", "ReconstructedAllAssoc2GenMatched_X", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMatched_Y"] =
        i.book1D("RecoAllAssoc2GenMatched_Y", "ReconstructedAllAssoc2GenMatched_Y", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMatched_Z"] =
        i.book1D("RecoAllAssoc2GenMatched_Z", "ReconstructedAllAssoc2GenMatched_Z", 120, -60, 60);
    mes_[label]["RecoAllAssoc2GenMatched_R"] =
        i.book1D("RecoAllAssoc2GenMatched_R", "ReconstructedAllAssoc2GenMatched_R", 120, 0, 0.6);
    mes_[label]["RecoAllAssoc2GenMatched_Pt2"] =
        i.book1D("RecoAllAssoc2GenMatched_Pt2", "ReconstructedAllAssoc2GenMatched_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_[label]["RecoAllAssoc2GenMatched_Ndof"] =
        i.book1D("RecoAllAssoc2GenMatched_Ndof", "ReconstructedAllAssoc2GenMatched_Ndof", 120, 0., 240.);
    mes_[label]["RecoAllAssoc2GenMatched_NumTracks"] = i.book1D(
        "RecoAllAssoc2GenMatched_NumTracks", "ReconstructedAllAssoc2GenMatched_NumTracks", 24, &log_ntrk_bins[0]);
    mes_[label]["RecoAllAssoc2GenMatched_PU"] = i.book1D(
        "RecoAllAssoc2GenMatched_PU", "ReconstructedAllAssoc2GenMatched_PU", int(nPUbins_ / 2), 0., double(nPUbins_));
    mes_[label]["RecoAllAssoc2GenMatched_ClosestDistanceZ"] =
        i.book1D("RecoAllAssoc2GenMatched_ClosestDistanceZ",
                 "ReconstructedAllAssoc2GenMatched_ClosestDistanceZ",
                 30,
                 &log_bins[0]);

    // All Reconstructed Vertices  Multi-Matched to a Generated vertex. Used
    // for Merge-Rate plots
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumVertices"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_NumVertices",
                 "ReconstructedAllAssoc2GenMultiMatched_NumVertices",
                 int(nPUbins_ / 2),
                 0.,
                 double(nPUbins_));
    mes_[label]["RecoAllAssoc2GenMultiMatched_X"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_X", "ReconstructedAllAssoc2GenMultiMatched_X", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Y"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_Y", "ReconstructedAllAssoc2GenMultiMatched_Y", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Z"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_Z", "ReconstructedAllAssoc2GenMultiMatched_Z", 120, -60, 60);
    mes_[label]["RecoAllAssoc2GenMultiMatched_R"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_R", "ReconstructedAllAssoc2GenMultiMatched_R", 120, 0, 0.6);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Pt2"] = i.book1D(
        "RecoAllAssoc2GenMultiMatched_Pt2", "ReconstructedAllAssoc2GenMultiMatched_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumTracks"] = i.book1D("RecoAllAssoc2GenMultiMatched_NumTracks",
                                                                     "ReconstructedAllAssoc2GenMultiMatched_NumTracks",
                                                                     24,
                                                                     &log_ntrk_bins[0]);
    mes_[label]["RecoAllAssoc2GenMultiMatched_PU"] = i.book1D("RecoAllAssoc2GenMultiMatched_PU",
                                                              "ReconstructedAllAssoc2GenMultiMatched_PU",
                                                              int(nPUbins_ / 2),
                                                              0.,
                                                              double(nPUbins_));
    mes_[label]["RecoAllAssoc2GenMultiMatched_ClosestDistanceZ"] =
        i.book1D("RecoAllAssoc2GenMultiMatched_ClosestDistanceZ",
                 "ReconstructedAllAssoc2GenMultiMatched_ClosestDistanceZ",
                 log_mergez_bins.size() - 1,
                 &log_mergez_bins[0]);

    // All Reconstructed Vertices Matched to a Multi-Matched Gen
    // Vertex. Used for Duplicate rate plots done w.r.t. Reco
    // Quantities. We basically want to ask how many times a RecoVTX
    // has been reconstructed and associated to a SimulatedVTX that
    // has been linked to at least another RecoVTX. In this sense this
    // RecoVTX is a duplicate of the same, real GenVTX.
    mes_[label]["RecoAllAssoc2MultiMatchedGen_NumVertices"] = i.book1D("RecoAllAssoc2MultiMatchedGen_NumVertices",
                                                                       "RecoAllAssoc2MultiMatchedGen_NumVertices",
                                                                       int(nPUbins_ / 2),
                                                                       0.,
                                                                       double(nPUbins_));
    mes_[label]["RecoAllAssoc2MultiMatchedGen_X"] =
        i.book1D("RecoAllAssoc2MultiMatchedGen_X", "RecoAllAssoc2MultiMatchedGen_X", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2MultiMatchedGen_Y"] =
        i.book1D("RecoAllAssoc2MultiMatchedGen_Y", "RecoAllAssoc2MultiMatchedGen_Y", 120, -0.6, 0.6);
    mes_[label]["RecoAllAssoc2MultiMatchedGen_Z"] =
        i.book1D("RecoAllAssoc2MultiMatchedGen_Z", "RecoAllAssoc2MultiMatchedGen_Z", 120, -60, 60);
    mes_[label]["RecoAllAssoc2MultiMatchedGen_R"] =
        i.book1D("RecoAllAssoc2MultiMatchedGen_R", "RecoAllAssoc2MultiMatchedGen_R", 120, 0, 0.6);
    mes_[label]["RecoAllAssoc2MultiMatchedGen_Pt2"] =
        i.book1D("RecoAllAssoc2MultiMatchedGen_Pt2", "RecoAllAssoc2MultiMatchedGen_Sum-pt2", 15, &log_pt2_bins[0]);
    mes_[label]["RecoAllAssoc2MultiMatchedGen_NumTracks"] = i.book1D(
        "RecoAllAssoc2MultiMatchedGen_NumTracks", "RecoAllAssoc2MultiMatchedGen_NumTracks", 24, &log_ntrk_bins[0]);
    mes_[label]["RecoAllAssoc2MultiMatchedGen_PU"] = i.book1D(
        "RecoAllAssoc2MultiMatchedGen_PU", "RecoAllAssoc2MultiMatchedGen_PU", int(nPUbins_ / 2), 0., double(nPUbins_));
    mes_[label]["RecoAllAssoc2MultiMatchedGen_ClosestDistanceZ"] =
        i.book1D("RecoAllAssoc2MultiMatchedGen_ClosestDistanceZ",
                 "RecoAllAssoc2MultiMatchedGen_ClosestDistanceZ",
                 30,
                 &log_bins[0]);
    mes_[label]["RecoAllAssoc2GenSimForMerge_ClosestDistanceZ"] =
        i.book1D("RecoAllAssoc2GenSimForMerge_ClosestDistanceZ",
                 "RecoAllAssoc2GenSimForMerge_ClosestDistanceZ",
                 log_mergez_bins.size() - 1,
                 &log_mergez_bins[0]);

    // Resolution and pull histograms
    const double resolpt2 = 10;

    const double minPull = -10;
    const double maxPull = 10;
    const double nPull = 100;

    auto bookResolPull = [&](const std::string& prefix, const double resolx, const double resoly, const double resolz) {
      book1d((prefix + "_ResolX").c_str(), 100, -resolx, resolx);
      book1d((prefix + "_ResolY").c_str(), 100, -resoly, resoly);
      book1d((prefix + "_ResolZ").c_str(), 100, -resolz, resolz);
      book1d((prefix + "_ResolPt2").c_str(), 100, -resolpt2, resolpt2);
      book1d((prefix + "_PullX").c_str(), 250, -25, 25);
      book1d((prefix + "_PullY").c_str(), 250, -25, 25);
      book1d((prefix + "_PullZ").c_str(), 250, -25, 25);

      book2d((prefix + "_ResolX_vs_PU").c_str(), int(nPUbins_ / 2), 0., nPUbins_, 100, -resolx, resolx);
      book2d((prefix + "_ResolY_vs_PU").c_str(), int(nPUbins_ / 2), 0., double(nPUbins_), 100, -resoly, resoly);
      book2d((prefix + "_ResolZ_vs_PU").c_str(), int(nPUbins_ / 2), 0., double(nPUbins_), 100, -resolz, resolz);
      book2d((prefix + "_ResolPt2_vs_PU").c_str(), int(nPUbins_ / 2), 0., double(nPUbins_), 100, -resolpt2, resolpt2);
      book2d((prefix + "_PullX_vs_PU").c_str(), int(nPUbins_ / 2), 0., double(nPUbins_), nPull, minPull, maxPull);
      book2d((prefix + "_PullY_vs_PU").c_str(), int(nPUbins_ / 2), 0., double(nPUbins_), nPull, minPull, maxPull);
      book2d((prefix + "_PullZ_vs_PU").c_str(), int(nPUbins_ / 2), 0., double(nPUbins_), nPull, minPull, maxPull);

      book2dlogx((prefix + "_ResolX_vs_NumTracks").c_str(), 24, &log_ntrk_bins[0], 100, -resolx, resolx);
      book2dlogx((prefix + "_ResolY_vs_NumTracks").c_str(), 24, &log_ntrk_bins[0], 100, -resoly, resoly);
      book2dlogx((prefix + "_ResolZ_vs_NumTracks").c_str(), 24, &log_ntrk_bins[0], 100, -resolz, resolz);
      book2dlogx((prefix + "_ResolPt2_vs_NumTracks").c_str(), 24, &log_ntrk_bins[0], 100, -resolpt2, resolpt2);
      book2dlogx((prefix + "_PullX_vs_NumTracks").c_str(), 24, &log_ntrk_bins[0], nPull, minPull, maxPull);
      book2dlogx((prefix + "_PullY_vs_NumTracks").c_str(), 24, &log_ntrk_bins[0], nPull, minPull, maxPull);
      book2dlogx((prefix + "_PullZ_vs_NumTracks").c_str(), 24, &log_ntrk_bins[0], nPull, minPull, maxPull);

      book2d((prefix + "_ResolX_vs_Z").c_str(), 120, -60, 60, 100, -resolx, resolx);
      book2d((prefix + "_ResolY_vs_Z").c_str(), 120, -60, 60, 100, -resoly, resoly);
      book2d((prefix + "_ResolZ_vs_Z").c_str(), 120, -60, 60, 100, -resolz, resolz);
      book2d((prefix + "_ResolPt2_vs_Z").c_str(), 120, -60, 60, 100, -resolpt2, resolpt2);
      book2d((prefix + "_PullX_vs_Z").c_str(), 120, -60, 60, nPull, minPull, maxPull);
      book2d((prefix + "_PullY_vs_Z").c_str(), 120, -60, 60, nPull, minPull, maxPull);
      book2d((prefix + "_PullZ_vs_Z").c_str(), 120, -60, 60, nPull, minPull, maxPull);

      book2dlogx((prefix + "_ResolX_vs_Pt").c_str(), log_pt_bins.size() - 1, &log_pt_bins[0], 100, -resolx, resolx);
      book2dlogx((prefix + "_ResolY_vs_Pt").c_str(), log_pt_bins.size() - 1, &log_pt_bins[0], 100, -resoly, resoly);
      book2dlogx((prefix + "_ResolZ_vs_Pt").c_str(), log_pt_bins.size() - 1, &log_pt_bins[0], 100, -resolz, resolz);
      book2dlogx(
          (prefix + "_ResolPt2_vs_Pt").c_str(), log_pt_bins.size() - 1, &log_pt_bins[0], 100, -resolpt2, resolpt2);
      book2dlogx((prefix + "_PullX_vs_Pt").c_str(), log_pt_bins.size() - 1, &log_pt_bins[0], nPull, minPull, maxPull);
      book2dlogx((prefix + "_PullY_vs_Pt").c_str(), log_pt_bins.size() - 1, &log_pt_bins[0], nPull, minPull, maxPull);
      book2dlogx((prefix + "_PullZ_vs_Pt").c_str(), log_pt_bins.size() - 1, &log_pt_bins[0], nPull, minPull, maxPull);
    };

    bookResolPull("RecoAllAssoc2GenMatched", 0.1, 0.1, 0.1);        // Non-merged vertices
    bookResolPull("RecoAllAssoc2GenMatchedMerged", 0.1, 0.1, 0.1);  // Merged vertices
    bookResolPull(
        "RecoPVAssoc2GenPVMatched", 0.01, 0.01, 0.01);  // PV, when correctly matched, regardless if merged or not

    // Purity histograms
    // Reco PV (vtx0) matched to hard-scatter gen vertex
    book1d("RecoPVAssoc2GenPVMatched_Purity", 50, 0, 1);
    book1d("RecoPVAssoc2GenPVMatched_Missing", 50, 0, 1);
    book2d("RecoPVAssoc2GenPVMatched_Purity_vs_Index", 100, 0, 100, 50, 0, 1);

    // RECO PV (vtx0) not matched to hard-scatter gen vertex
    book1d("RecoPVAssoc2GenPVNotMatched_Purity", 50, 0, 1);
    book1d("RecoPVAssoc2GenPVNotMatched_Missing", 50, 0, 1);
    book2d("RecoPVAssoc2GenPVNotMatched_Purity_vs_Index", 100, 0, 100, 50, 0, 1);

    // Purity vs. fake rate
    book1d("RecoAllAssoc2Gen_Purity", 50, 0, 1);         // denominator
    book1d("RecoAllAssoc2GenMatched_Purity", 50, 0, 1);  // 1-numerator

    // Vertex sum(pt2)
    // The first two are orthogonal (i.e. their sum includes all reco vertices)
    book1dlogx("RecoAssoc2GenPVMatched_Pt2", 15, &log_pt2_bins[0]);
    book1dlogx("RecoAssoc2GenPVNotMatched_Pt2", 15, &log_pt2_bins[0]);

    book1dlogx("RecoAssoc2GenPVMatchedNotHighest_Pt2", 15, &log_pt2_bins[0]);
    book1dlogx("RecoAssoc2GenPVNotMatched_GenPVTracksRemoved_Pt2", 15, &log_pt2_bins[0]);

    // Shared tracks
    book1d("RecoAllAssoc2GenSingleMatched_SharedTrackFractionReco", 50, 0, 1);
    book1d("RecoAllAssoc2GenMultiMatched_SharedTrackFractionReco", 50, 0, 1);
    book1d("RecoAllAssoc2GenSingleMatched_SharedTrackFractionRecoMatched", 50, 0, 1);
    book1d("RecoAllAssoc2GenMultiMatched_SharedTrackFractionRecoMatched", 50, 0, 1);

    book1d("RecoAllAssoc2GenSingleMatched_SharedTrackFractionSim", 50, 0, 1);
    book1d("RecoAllAssoc2GenMultiMatched_SharedTrackFractionSim", 50, 0, 1);
    book1d("RecoAllAssoc2GenSingleMatched_SharedTrackFractionSimMatched", 50, 0, 1);
    book1d("RecoAllAssoc2GenMultiMatched_SharedTrackFractionSimMatched", 50, 0, 1);
  }
}

void PrimaryVertexAnalyzer4PUSlimmed::fillGenericGenVertexHistograms(const simPrimaryVertex& v) {
  if (v.eventId.event() == 0) {
    mes_["root_folder"]["GenPV_X"]->Fill(v.x);
    mes_["root_folder"]["GenPV_Y"]->Fill(v.y);
    mes_["root_folder"]["GenPV_Z"]->Fill(v.z);
    mes_["root_folder"]["GenPV_R"]->Fill(v.r);
    mes_["root_folder"]["GenPV_Pt2"]->Fill(v.ptsq);
    mes_["root_folder"]["GenPV_NumTracks"]->Fill(v.nGenTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_["root_folder"]["GenPV_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);
  }
  mes_["root_folder"]["GenAllV_X"]->Fill(v.x);
  mes_["root_folder"]["GenAllV_Y"]->Fill(v.y);
  mes_["root_folder"]["GenAllV_Z"]->Fill(v.z);
  mes_["root_folder"]["GenAllV_R"]->Fill(v.r);
  mes_["root_folder"]["GenAllV_Pt2"]->Fill(v.ptsq);
  mes_["root_folder"]["GenAllV_NumTracks"]->Fill(v.nGenTrk);
  if (v.closest_vertex_distance_z > 0.)
    mes_["root_folder"]["GenAllV_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);
}

void PrimaryVertexAnalyzer4PUSlimmed::fillRecoAssociatedGenVertexHistograms(
    const std::string& label, const PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex& v) {
  mes_[label]["GenAllAssoc2Reco_X"]->Fill(v.x);
  mes_[label]["GenAllAssoc2Reco_Y"]->Fill(v.y);
  mes_[label]["GenAllAssoc2Reco_Z"]->Fill(v.z);
  mes_[label]["GenAllAssoc2Reco_R"]->Fill(v.r);
  mes_[label]["GenAllAssoc2Reco_Pt2"]->Fill(v.ptsq);
  mes_[label]["GenAllAssoc2Reco_NumTracks"]->Fill(v.nGenTrk);
  if (v.closest_vertex_distance_z > 0.)
    mes_[label]["GenAllAssoc2Reco_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);
  if (!v.rec_vertices.empty()) {
    mes_[label]["GenAllAssoc2RecoMatched_X"]->Fill(v.x);
    mes_[label]["GenAllAssoc2RecoMatched_Y"]->Fill(v.y);
    mes_[label]["GenAllAssoc2RecoMatched_Z"]->Fill(v.z);
    mes_[label]["GenAllAssoc2RecoMatched_R"]->Fill(v.r);
    mes_[label]["GenAllAssoc2RecoMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["GenAllAssoc2RecoMatched_NumTracks"]->Fill(v.nGenTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_[label]["GenAllAssoc2RecoMatched_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);
  }
  if (v.rec_vertices.size() > 1) {
    mes_[label]["GenAllAssoc2RecoMultiMatched_X"]->Fill(v.x);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Y"]->Fill(v.y);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Z"]->Fill(v.z);
    mes_[label]["GenAllAssoc2RecoMultiMatched_R"]->Fill(v.r);
    mes_[label]["GenAllAssoc2RecoMultiMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumTracks"]->Fill(v.nGenTrk);
    if (v.closest_vertex_distance_z > 0.)
      mes_[label]["GenAllAssoc2RecoMultiMatched_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);
  }
}

void PrimaryVertexAnalyzer4PUSlimmed::fillRecoAssociatedGenPVHistograms(
    const std::string& label, const PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex& v, bool genPVMatchedToRecoPV) {
  mes_[label]["GenPVAssoc2RecoPV_X"]->Fill(v.x);
  mes_[label]["GenPVAssoc2RecoPV_Y"]->Fill(v.y);
  mes_[label]["GenPVAssoc2RecoPV_Z"]->Fill(v.z);
  if (genPVMatchedToRecoPV) {
    mes_[label]["GenPVAssoc2RecoPVMatched_X"]->Fill(v.x);
    mes_[label]["GenPVAssoc2RecoPVMatched_Y"]->Fill(v.y);
    mes_[label]["GenPVAssoc2RecoPVMatched_Z"]->Fill(v.z);
  }
}

void PrimaryVertexAnalyzer4PUSlimmed::fillGenAssociatedRecoVertexHistograms(
    const std::string& label, int num_pileup_vertices, PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex& v) {
  mes_[label]["RecoAllAssoc2Gen_X"]->Fill(v.x);
  mes_[label]["RecoAllAssoc2Gen_Y"]->Fill(v.y);
  mes_[label]["RecoAllAssoc2Gen_Z"]->Fill(v.z);
  mes_[label]["RecoAllAssoc2Gen_R"]->Fill(v.r);
  mes_[label]["RecoAllAssoc2Gen_Pt2"]->Fill(v.ptsq);
  mes_[label]["RecoAllAssoc2Gen_Ndof"]->Fill(v.recVtx->ndof());
  mes_[label]["RecoAllAssoc2Gen_NumTracks"]->Fill(v.nRecoTrk);
  mes_[label]["RecoAllAssoc2Gen_PU"]->Fill(num_pileup_vertices);
  mes_[label]["RecoAllAssoc2Gen_Purity"]->Fill(v.purity);
  if (v.closest_vertex_distance_z > 0.)
    mes_[label]["RecoAllAssoc2Gen_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);
  if (!v.sim_vertices.empty()) {
    v.kind_of_vertex |= recoPrimaryVertex::MATCHED;
    mes_[label]["RecoAllAssoc2GenMatched_X"]->Fill(v.x);
    mes_[label]["RecoAllAssoc2GenMatched_Y"]->Fill(v.y);
    mes_[label]["RecoAllAssoc2GenMatched_Z"]->Fill(v.z);
    mes_[label]["RecoAllAssoc2GenMatched_R"]->Fill(v.r);
    mes_[label]["RecoAllAssoc2GenMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["RecoAllAssoc2GenMatched_Ndof"]->Fill(v.recVtx->ndof());
    mes_[label]["RecoAllAssoc2GenMatched_NumTracks"]->Fill(v.nRecoTrk);
    mes_[label]["RecoAllAssoc2GenMatched_PU"]->Fill(num_pileup_vertices);
    mes_[label]["RecoAllAssoc2GenMatched_Purity"]->Fill(v.purity);
    if (v.closest_vertex_distance_z > 0.)
      mes_[label]["RecoAllAssoc2GenMatched_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);

    // Fill resolution and pull plots here (as in MultiTrackValidator)
    fillResolutionAndPullHistograms(label, num_pileup_vertices, v, false);

    // Now keep track of all RecoVTX associated to a SimVTX that
    // itself is associated to more than one RecoVTX, for
    // duplicate-rate plots on reco quantities.
    if (v.sim_vertices_internal[0]->rec_vertices.size() > 1) {
      v.kind_of_vertex |= recoPrimaryVertex::DUPLICATE;
      mes_[label]["RecoAllAssoc2MultiMatchedGen_X"]->Fill(v.x);
      mes_[label]["RecoAllAssoc2MultiMatchedGen_Y"]->Fill(v.y);
      mes_[label]["RecoAllAssoc2MultiMatchedGen_Z"]->Fill(v.z);
      mes_[label]["RecoAllAssoc2MultiMatchedGen_R"]->Fill(v.r);
      mes_[label]["RecoAllAssoc2MultiMatchedGen_Pt2"]->Fill(v.ptsq);
      mes_[label]["RecoAllAssoc2MultiMatchedGen_NumTracks"]->Fill(v.nRecoTrk);
      mes_[label]["RecoAllAssoc2MultiMatchedGen_PU"]->Fill(num_pileup_vertices);
      if (v.closest_vertex_distance_z > 0.)
        mes_[label]["RecoAllAssoc2MultiMatchedGen_ClosestDistanceZ"]->Fill(v.closest_vertex_distance_z);
    }
    // This is meant to be used as "denominator" for the merge-rate
    // plots produced starting from reco quantities. We   enter here
    // only if the reco vertex has been associated, since we need info
    // from the SimVTX associated to it. In this regard, the final
    // merge-rate plot coming from reco is not to be intended as a
    // pure efficiency-like plot, since the normalization is biased.
    if (v.sim_vertices_internal[0]->closest_vertex_distance_z > 0.)
      mes_[label]["RecoAllAssoc2GenSimForMerge_ClosestDistanceZ"]->Fill(
          v.sim_vertices_internal[0]->closest_vertex_distance_z);
  }
  // this plots are meant to be used to compute the merge rate
  if (v.sim_vertices.size() > 1) {
    v.kind_of_vertex |= recoPrimaryVertex::MERGED;
    mes_[label]["RecoAllAssoc2GenMultiMatched_X"]->Fill(v.x);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Y"]->Fill(v.y);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Z"]->Fill(v.z);
    mes_[label]["RecoAllAssoc2GenMultiMatched_R"]->Fill(v.r);
    mes_[label]["RecoAllAssoc2GenMultiMatched_Pt2"]->Fill(v.ptsq);
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumTracks"]->Fill(v.nRecoTrk);
    mes_[label]["RecoAllAssoc2GenMultiMatched_PU"]->Fill(num_pileup_vertices);
    if (v.sim_vertices_internal[0]->closest_vertex_distance_z > 0.)
      mes_[label]["RecoAllAssoc2GenMultiMatched_ClosestDistanceZ"]->Fill(
          v.sim_vertices_internal[0]->closest_vertex_distance_z);
  }
  mes_[label]["RecoAllAssoc2GenProperties"]->Fill(v.kind_of_vertex);

  std::string prefix;
  if (v.sim_vertices.size() == 1) {
    prefix = "RecoAllAssoc2GenSingleMatched_SharedTrackFraction";
  } else if (v.sim_vertices.size() > 1) {
    prefix = "RecoAllAssoc2GenMultiMatched_SharedTrackFraction";
  }

  for (size_t i = 0; i < v.sim_vertices.size(); ++i) {
    const double sharedTracks = v.sim_vertices_num_shared_tracks[i];
    const simPrimaryVertex* simV = v.sim_vertices_internal[i];
    mes_[label][prefix + "Reco"]->Fill(sharedTracks / v.nRecoTrk);
    mes_[label][prefix + "RecoMatched"]->Fill(sharedTracks / v.num_matched_sim_tracks);
    mes_[label][prefix + "Sim"]->Fill(sharedTracks / simV->nGenTrk);
    mes_[label][prefix + "SimMatched"]->Fill(sharedTracks / simV->num_matched_reco_tracks);
  }
}

void PrimaryVertexAnalyzer4PUSlimmed::fillResolutionAndPullHistograms(
    const std::string& label,
    int num_pileup_vertices,
    PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex& v,
    bool isPV) {
  std::string prefix = "RecoAllAssoc2GenMatched";
  if (v.sim_vertices_internal.size() > 1) {
    prefix += "Merged";
  }
  if (isPV) {
    prefix = "RecoPVAssoc2GenPVMatched";
  }

  // Use the best match as defined by the vertex truth associator
  // reco-tracks as the best match
  const simPrimaryVertex& bestMatch = *(v.sim_vertices_internal[0]);
  const double xres = v.x - bestMatch.x;
  const double yres = v.y - bestMatch.y;
  const double zres = v.z - bestMatch.z;
  const double pt2res = v.ptsq - bestMatch.ptsq;

  const double xresol = xres;
  const double yresol = yres;
  const double zresol = zres;
  const double pt2resol = pt2res / v.ptsq;
  const double xpull = xres / v.recVtx->xError();
  const double ypull = yres / v.recVtx->yError();
  const double zpull = zres / v.recVtx->zError();

  mes_[label][prefix + "_ResolX"]->Fill(xresol);
  mes_[label][prefix + "_ResolY"]->Fill(yresol);
  mes_[label][prefix + "_ResolZ"]->Fill(zresol);
  mes_[label][prefix + "_ResolPt2"]->Fill(pt2resol);
  mes_[label][prefix + "_PullX"]->Fill(xpull);
  mes_[label][prefix + "_PullY"]->Fill(ypull);
  mes_[label][prefix + "_PullZ"]->Fill(zpull);

  mes_[label][prefix + "_ResolX_vs_PU"]->Fill(num_pileup_vertices, xresol);
  mes_[label][prefix + "_ResolY_vs_PU"]->Fill(num_pileup_vertices, yresol);
  mes_[label][prefix + "_ResolZ_vs_PU"]->Fill(num_pileup_vertices, zresol);
  mes_[label][prefix + "_ResolPt2_vs_PU"]->Fill(num_pileup_vertices, pt2resol);
  mes_[label][prefix + "_PullX_vs_PU"]->Fill(num_pileup_vertices, xpull);
  mes_[label][prefix + "_PullY_vs_PU"]->Fill(num_pileup_vertices, ypull);
  mes_[label][prefix + "_PullZ_vs_PU"]->Fill(num_pileup_vertices, zpull);

  mes_[label][prefix + "_ResolX_vs_NumTracks"]->Fill(v.nRecoTrk, xresol);
  mes_[label][prefix + "_ResolY_vs_NumTracks"]->Fill(v.nRecoTrk, yresol);
  mes_[label][prefix + "_ResolZ_vs_NumTracks"]->Fill(v.nRecoTrk, zresol);
  mes_[label][prefix + "_ResolPt2_vs_NumTracks"]->Fill(v.nRecoTrk, pt2resol);
  mes_[label][prefix + "_PullX_vs_NumTracks"]->Fill(v.nRecoTrk, xpull);
  mes_[label][prefix + "_PullY_vs_NumTracks"]->Fill(v.nRecoTrk, ypull);
  mes_[label][prefix + "_PullZ_vs_NumTracks"]->Fill(v.nRecoTrk, zpull);

  mes_[label][prefix + "_ResolX_vs_Z"]->Fill(v.z, xresol);
  mes_[label][prefix + "_ResolY_vs_Z"]->Fill(v.z, yresol);
  mes_[label][prefix + "_ResolZ_vs_Z"]->Fill(v.z, zresol);
  mes_[label][prefix + "_ResolPt2_vs_Z"]->Fill(v.z, pt2resol);
  mes_[label][prefix + "_PullX_vs_Z"]->Fill(v.z, xpull);
  mes_[label][prefix + "_PullY_vs_Z"]->Fill(v.z, ypull);
  mes_[label][prefix + "_PullZ_vs_Z"]->Fill(v.z, zpull);

  mes_[label][prefix + "_ResolX_vs_Pt"]->Fill(v.pt, xresol);
  mes_[label][prefix + "_ResolY_vs_Pt"]->Fill(v.pt, yresol);
  mes_[label][prefix + "_ResolZ_vs_Pt"]->Fill(v.pt, zresol);
  mes_[label][prefix + "_ResolPt2_vs_Pt"]->Fill(v.pt, pt2resol);
  mes_[label][prefix + "_PullX_vs_Pt"]->Fill(v.pt, xpull);
  mes_[label][prefix + "_PullY_vs_Pt"]->Fill(v.pt, ypull);
  mes_[label][prefix + "_PullZ_vs_Pt"]->Fill(v.pt, zpull);
}

bool PrimaryVertexAnalyzer4PUSlimmed::matchRecoTrack2SimSignal(const reco::TrackBaseRef& recoTrack) {
  auto found = r2s_->find(recoTrack);

  // reco track not matched to any TP
  if (found == r2s_->end())
    return false;

  // reco track matched to some TP from signal vertex
  for (const auto& tp : found->val) {
    if (tp.first->eventId().bunchCrossing() == 0 && tp.first->eventId().event() == 0)
      return true;
  }

  // reco track not matched to any TP from signal vertex
  return false;
}

void PrimaryVertexAnalyzer4PUSlimmed::calculatePurityAndFillHistograms(const std::string& label,
                                                                       std::vector<recoPrimaryVertex>& recopvs,
                                                                       int genpv_position_in_reco_collection,
                                                                       bool signal_is_highest_pt) {
  if (recopvs.empty())
    return;

  std::vector<double> vtx_sumpt_sigmatched;
  std::vector<double> vtx_sumpt2_sigmatched;

  vtx_sumpt_sigmatched.reserve(recopvs.size());
  vtx_sumpt2_sigmatched.reserve(recopvs.size());

  // Calculate purity
  for (auto& v : recopvs) {
    double sumpt_all = 0;
    double sumpt_sigmatched = 0;
    double sumpt2_sigmatched = 0;
    const reco::Vertex* vertex = v.recVtx;
    for (auto iTrack = vertex->tracks_begin(); iTrack != vertex->tracks_end(); ++iTrack) {
      double pt = (*iTrack)->pt();
      sumpt_all += pt;
      if (matchRecoTrack2SimSignal(*iTrack)) {
        sumpt_sigmatched += pt;
        sumpt2_sigmatched += pt * pt;
      }
    }
    v.purity = sumpt_sigmatched / sumpt_all;

    vtx_sumpt_sigmatched.push_back(sumpt_sigmatched);
    vtx_sumpt2_sigmatched.push_back(sumpt2_sigmatched);
  }

  const double vtxAll_sumpt_sigmatched = std::accumulate(vtx_sumpt_sigmatched.begin(), vtx_sumpt_sigmatched.end(), 0.0);
  const double vtxNot0_sumpt_sigmatched = vtxAll_sumpt_sigmatched - vtx_sumpt_sigmatched[0];
  const double missing = vtxNot0_sumpt_sigmatched / vtxAll_sumpt_sigmatched;

  // Fill purity
  std::string prefix = "RecoPVAssoc2GenPVNotMatched_";
  if (genpv_position_in_reco_collection == 0)
    prefix = "RecoPVAssoc2GenPVMatched_";

  mes_[label][prefix + "Purity"]->Fill(recopvs[0].purity);
  mes_[label][prefix + "Missing"]->Fill(missing);
  auto hpurity = mes_[label][prefix + "Purity_vs_Index"];
  for (size_t i = 0; i < recopvs.size(); ++i) {
    hpurity->Fill(i, recopvs[i].purity);
  }

  // Fill sumpt2
  for (size_t i = 0; i < recopvs.size(); ++i) {
    if (static_cast<int>(i) == genpv_position_in_reco_collection) {
      mes_[label]["RecoAssoc2GenPVMatched_Pt2"]->Fill(recopvs[i].ptsq);
    } else {
      double pt2 = recopvs[i].ptsq;
      mes_[label]["RecoAssoc2GenPVNotMatched_Pt2"]->Fill(pt2);
      // Subtract hard-scatter track pt2 from the pileup pt2
      double pt2_pu = pt2 - vtx_sumpt2_sigmatched[i];
      mes_[label]["RecoAssoc2GenPVNotMatched_GenPVTracksRemoved_Pt2"]->Fill(pt2_pu);
    }
  }
  if (!signal_is_highest_pt && genpv_position_in_reco_collection >= 0)
    mes_[label]["RecoAssoc2GenPVMatchedNotHighest_Pt2"]->Fill(recopvs[genpv_position_in_reco_collection].ptsq);
}

/* Extract information form TrackingParticles/TrackingVertex and fill
 * the helper class simPrimaryVertex with proper generation-level
 * information */
std::vector<PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex> PrimaryVertexAnalyzer4PUSlimmed::getSimPVs(
    const edm::Handle<TrackingVertexCollection>& tVC) {
  std::vector<PrimaryVertexAnalyzer4PUSlimmed::simPrimaryVertex> simpv;
  int current_event = -1;

  if (verbose_) {
    std::cout << "getSimPVs TrackingVertexCollection " << std::endl;
  }

  for (TrackingVertexCollection::const_iterator v = tVC->begin(); v != tVC->end(); ++v) {
    if (verbose_) {
      std::cout << "BunchX.EventId: " << v->eventId().bunchCrossing() << "." << (v->eventId()).event()
                << " Position: " << v->position() << " G4/HepMC Vertices: " << v->g4Vertices().size() << "/"
                << v->genVertices().size() << "   t = " << v->position().t() * 1.e12
                << "    == 0:" << (v->position().t() > 0) << std::endl;
      for (TrackingVertex::g4v_iterator gv = v->g4Vertices_begin(); gv != v->g4Vertices_end(); gv++) {
        std::cout << *gv << std::endl;
      }
      std::cout << "----------" << std::endl;

    }  // end of verbose_ session

    // I'd rather change this and select only vertices that come from
    // BX=0.  We should keep only the first vertex from all the events
    // at BX=0.
    if (v->eventId().bunchCrossing() != 0)
      continue;
    if (v->eventId().event() != current_event) {
      current_event = v->eventId().event();
    } else {
      continue;
    }
    // TODO(rovere) is this really necessary?
    if (fabs(v->position().z()) > 1000)
      continue;  // skip funny junk vertices

    // could be a new vertex, check  all primaries found so far to avoid
    // multiple entries
    simPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z());
    sv.eventId = v->eventId();
    sv.sim_vertex = TrackingVertexRef(tVC, std::distance(tVC->begin(), v));

    for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin(); iTrack != v->daughterTracks_end();
         ++iTrack) {
      // TODO(rovere) isn't it always the case? Is it really worth
      // checking this out?
      // sv.eventId = (**iTrack).eventId();
      assert((**iTrack).eventId().bunchCrossing() == 0);
    }
    // TODO(rovere) maybe get rid of this old logic completely ... ?
    simPrimaryVertex* vp = nullptr;  // will become non-NULL if a vertex
                                     // is found and then point to it
    for (std::vector<simPrimaryVertex>::iterator v0 = simpv.begin(); v0 != simpv.end(); v0++) {
      if ((sv.eventId == v0->eventId) && (fabs(sv.x - v0->x) < 1e-5) && (fabs(sv.y - v0->y) < 1e-5) &&
          (fabs(sv.z - v0->z) < 1e-5)) {
        vp = &(*v0);
        break;
      }
    }
    if (!vp) {
      // this is a new vertex, add it to the list of sim-vertices
      simpv.push_back(sv);
      vp = &simpv.back();
      if (verbose_) {
        std::cout << "this is a new vertex " << sv.eventId.event() << "   " << sv.x << " " << sv.y << " " << sv.z
                  << std::endl;
      }
    } else {
      if (verbose_) {
        std::cout << "this is not a new vertex" << sv.x << " " << sv.y << " " << sv.z << std::endl;
      }
    }

    // Loop over daughter track(s) as Tracking Particles
    for (TrackingVertex::tp_iterator iTP = v->daughterTracks_begin(); iTP != v->daughterTracks_end(); ++iTP) {
      auto momentum = (*(*iTP)).momentum();
      const reco::Track* matched_best_reco_track = nullptr;
      double match_quality = -1;
      if (use_only_charged_tracks_ && (**iTP).charge() == 0)
        continue;
      if (s2r_->find(*iTP) != s2r_->end()) {
        matched_best_reco_track = (*s2r_)[*iTP][0].first.get();
        match_quality = (*s2r_)[*iTP][0].second;
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
      if (((**iTP).pt() > 0.2) && (fabs((**iTP).eta()) < 2.5) && (**iTP).charge() != 0) {
        vp->nGenTrk++;
      }
    }  // End of for loop on daughters sim-particles
    if (vp->num_matched_reco_tracks)
      vp->average_match_quality /= static_cast<float>(vp->num_matched_reco_tracks);
    if (verbose_) {
      std::cout << "average number of associated tracks: "
                << vp->num_matched_reco_tracks / static_cast<float>(vp->nGenTrk)
                << " with average quality: " << vp->average_match_quality << std::endl;
    }
  }  // End of for loop on tracking vertices

  if (verbose_) {
    std::cout << "------- PrimaryVertexAnalyzer4PUSlimmed simPVs from "
                 "TrackingVertices "
                 "-------"
              << std::endl;
    for (std::vector<simPrimaryVertex>::iterator v0 = simpv.begin(); v0 != simpv.end(); v0++) {
      std::cout << "z=" << v0->z << "  event=" << v0->eventId.event() << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;
  }  // End of for summary on discovered simulated primary vertices.

  // In case of no simulated vertices, break here
  if (simpv.empty())
    return simpv;

  // Now compute the closest distance in z between all simulated vertex
  // first initialize
  auto prev_z = simpv.back().z;
  for (simPrimaryVertex& vsim : simpv) {
    vsim.closest_vertex_distance_z = std::abs(vsim.z - prev_z);
    prev_z = vsim.z;
  }
  // then calculate
  for (std::vector<simPrimaryVertex>::iterator vsim = simpv.begin(); vsim != simpv.end(); vsim++) {
    std::vector<simPrimaryVertex>::iterator vsim2 = vsim;
    vsim2++;
    for (; vsim2 != simpv.end(); vsim2++) {
      double distance = std::abs(vsim->z - vsim2->z);
      // need both to be complete
      vsim->closest_vertex_distance_z = std::min(vsim->closest_vertex_distance_z, distance);
      vsim2->closest_vertex_distance_z = std::min(vsim2->closest_vertex_distance_z, distance);
    }
  }
  return simpv;
}

/* Extract information form recoVertex and fill the helper class
 * recoPrimaryVertex with proper reco-level information */
std::vector<PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex> PrimaryVertexAnalyzer4PUSlimmed::getRecoPVs(
    const edm::Handle<edm::View<reco::Vertex>>& tVC) {
  std::vector<PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex> recopv;

  if (verbose_) {
    std::cout << "getRecoPVs TrackingVertexCollection " << std::endl;
  }

  for (auto v = tVC->begin(); v != tVC->end(); ++v) {
    if (verbose_) {
      std::cout << " Position: " << v->position() << std::endl;
    }

    // Skip junk vertices
    if (fabs(v->z()) > 1000)
      continue;
    if (v->isFake() || !v->isValid())
      continue;

    recoPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z());
    sv.recVtx = &(*v);
    sv.recVtxRef = reco::VertexBaseRef(tVC, std::distance(tVC->begin(), v));
    // this is a new vertex, add it to the list of reco-vertices
    recopv.push_back(sv);
    PrimaryVertexAnalyzer4PUSlimmed::recoPrimaryVertex* vp = &recopv.back();

    // Loop over daughter track(s)
    for (auto iTrack = v->tracks_begin(); iTrack != v->tracks_end(); ++iTrack) {
      auto momentum = (*(*iTrack)).innerMomentum();
      // TODO(rovere) better handle the pixelVertices, whose tracks
      // do not have the innerMomentum defined. This is a temporary
      // hack to overcome this problem.
      if (momentum.mag2() == 0)
        momentum = (*(*iTrack)).momentum();
      if (verbose_) {
        std::cout << "  Daughter momentum:      " << momentum;
        std::cout << std::endl;
      }
      vp->pt += std::sqrt(momentum.perp2());
      vp->ptsq += (momentum.perp2());
      vp->nRecoTrk++;

      auto matched = r2s_->find(*iTrack);
      if (matched != r2s_->end()) {
        vp->num_matched_sim_tracks++;
      }

    }  // End of for loop on daughters reconstructed tracks
  }    // End of for loop on tracking vertices

  if (verbose_) {
    std::cout << "------- PrimaryVertexAnalyzer4PUSlimmed recoPVs from "
                 "VertexCollection "
                 "-------"
              << std::endl;
    for (std::vector<recoPrimaryVertex>::iterator v0 = recopv.begin(); v0 != recopv.end(); v0++) {
      std::cout << "z=" << v0->z << std::endl;
    }
    std::cout << "-----------------------------------------------" << std::endl;
  }  // End of for summary on reconstructed primary vertices.

  // In case of no reco vertices, break here
  if (recopv.empty())
    return recopv;

  // Now compute the closest distance in z between all reconstructed vertex
  // first initialize
  auto prev_z = recopv.back().z;
  for (recoPrimaryVertex& vreco : recopv) {
    vreco.closest_vertex_distance_z = std::abs(vreco.z - prev_z);
    prev_z = vreco.z;
  }
  for (std::vector<recoPrimaryVertex>::iterator vreco = recopv.begin(); vreco != recopv.end(); vreco++) {
    std::vector<recoPrimaryVertex>::iterator vreco2 = vreco;
    vreco2++;
    for (; vreco2 != recopv.end(); vreco2++) {
      double distance = std::abs(vreco->z - vreco2->z);
      // need both to be complete
      vreco->closest_vertex_distance_z = std::min(vreco->closest_vertex_distance_z, distance);
      vreco2->closest_vertex_distance_z = std::min(vreco2->closest_vertex_distance_z, distance);
    }
  }
  return recopv;
}

void PrimaryVertexAnalyzer4PUSlimmed::resetSimPVAssociation(std::vector<simPrimaryVertex>& simpv) {
  for (auto& v : simpv) {
    v.rec_vertices.clear();
  }
}

// ------------ method called to produce the data  ------------
void PrimaryVertexAnalyzer4PUSlimmed::matchSim2RecoVertices(std::vector<simPrimaryVertex>& simpv,
                                                            const reco::VertexSimToRecoCollection& vertex_s2r) {
  if (verbose_) {
    std::cout << "PrimaryVertexAnalyzer4PUSlimmed::matchSim2RecoVertices " << std::endl;
  }
  for (std::vector<simPrimaryVertex>::iterator vsim = simpv.begin(); vsim != simpv.end(); vsim++) {
    auto matched = vertex_s2r.find(vsim->sim_vertex);
    if (matched != vertex_s2r.end()) {
      for (const auto& vertexRefQuality : matched->val) {
        vsim->rec_vertices.push_back(&(*(vertexRefQuality.first)));
      }
    }

    if (verbose_) {
      if (!vsim->rec_vertices.empty()) {
        for (auto const& v : vsim->rec_vertices) {
          std::cout << "Found a matching vertex for genVtx " << vsim->z << " at " << v->z()
                    << " with sign: " << fabs(v->z() - vsim->z) / v->zError() << std::endl;
        }
      } else {
        std::cout << "No matching vertex for " << vsim->z << std::endl;
      }
    }
  }  // end for loop on simulated vertices
  if (verbose_) {
    std::cout << "Done with matching sim vertices" << std::endl;
  }
}

void PrimaryVertexAnalyzer4PUSlimmed::matchReco2SimVertices(std::vector<recoPrimaryVertex>& recopv,
                                                            const reco::VertexRecoToSimCollection& vertex_r2s,
                                                            const std::vector<simPrimaryVertex>& simpv) {
  for (std::vector<recoPrimaryVertex>::iterator vrec = recopv.begin(); vrec != recopv.end(); vrec++) {
    auto matched = vertex_r2s.find(vrec->recVtxRef);
    if (matched != vertex_r2s.end()) {
      for (const auto& vertexRefQuality : matched->val) {
        const auto tvPtr = &(*(vertexRefQuality.first));
        vrec->sim_vertices.push_back(tvPtr);
      }

      for (const TrackingVertex* tv : vrec->sim_vertices) {
        // Set pointers to internal simVertex objects
        for (const auto& vv : simpv) {
          if (&(*(vv.sim_vertex)) == tv) {
            vrec->sim_vertices_internal.push_back(&vv);
            continue;
          }
        }

        // Calculate number of shared tracks
        vrec->sim_vertices_num_shared_tracks.push_back(calculateVertexSharedTracks(*(vrec->recVtx), *tv, *r2s_));
      }
    }

    if (verbose_) {
      for (auto v : vrec->sim_vertices) {
        std::cout << "Found a matching vertex for reco: " << vrec->z << " at gen:" << v->position().z()
                  << " with sign: " << fabs(vrec->z - v->position().z()) / vrec->recVtx->zError() << std::endl;
      }
    }
  }  // end for loop on reconstructed vertices
}

void PrimaryVertexAnalyzer4PUSlimmed::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using edm::Handle;
  using edm::View;
  using std::cout;
  using std::endl;
  using std::vector;
  using namespace reco;

  std::vector<float> pileUpInfo_z;

  // get the pileup information
  edm::Handle<std::vector<PileupSummaryInfo>> puinfoH;
  if (iEvent.getByToken(vecPileupSummaryInfoToken_, puinfoH)) {
    for (auto const& pu_info : *puinfoH.product()) {
      if (do_generic_sim_plots_) {
        mes_["root_folder"]["GenVtx_vs_BX"]->Fill(pu_info.getBunchCrossing(), pu_info.getPU_NumInteractions());
      }
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

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleCollectionToken_, TPCollectionH);
  if (!TPCollectionH.isValid())
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "TPCollectionH is not valid";

  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(trackingVertexCollectionToken_, TVCollectionH);
  if (!TVCollectionH.isValid())
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "TVCollectionH is not valid";

  // TODO(rovere) the idea is to put in case a track-selector in front
  // of this module and then use its label to get the selected tracks
  // out of the event instead of making an hard-coded selection in the
  // code.

  edm::Handle<reco::SimToRecoCollection> simToRecoH;
  iEvent.getByToken(simToRecoAssociationToken_, simToRecoH);
  if (simToRecoH.isValid())
    s2r_ = simToRecoH.product();
  else
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "simToRecoH is not valid";

  edm::Handle<reco::RecoToSimCollection> recoToSimH;
  iEvent.getByToken(recoToSimAssociationToken_, recoToSimH);
  if (recoToSimH.isValid())
    r2s_ = recoToSimH.product();
  else
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "recoToSimH is not valid";

  // Vertex associator
  edm::Handle<reco::VertexToTrackingVertexAssociator> vertexAssociatorH;
  iEvent.getByToken(vertexAssociatorToken_, vertexAssociatorH);
  if (!vertexAssociatorH.isValid()) {
    edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed") << "vertexAssociatorH is not valid";
    return;
  }
  const reco::VertexToTrackingVertexAssociator& vertexAssociator = *(vertexAssociatorH.product());

  std::vector<simPrimaryVertex> simpv;  // a list of simulated primary
                                        // MC vertices
  // TODO(rovere) use move semantic?
  simpv = getSimPVs(TVCollectionH);
  // TODO(rovere) 1 vertex is not, by definition, pileup, and should
  // probably be subtracted?
  int kind_of_signal_vertex = 0;
  int num_pileup_vertices = simpv.size();
  if (do_generic_sim_plots_)
    mes_["root_folder"]["GenAllV_NumVertices"]->Fill(simpv.size());
  bool signal_is_highest_pt =
      std::max_element(simpv.begin(), simpv.end(), [](const simPrimaryVertex& lhs, const simPrimaryVertex& rhs) {
        return lhs.ptsq < rhs.ptsq;
      }) == simpv.begin();
  kind_of_signal_vertex |= (signal_is_highest_pt << HIGHEST_PT);
  if (do_generic_sim_plots_) {
    mes_["root_folder"]["SignalIsHighestPt2"]->Fill(signal_is_highest_pt ? 1. : 0.);
    computePairDistance(simpv, mes_["root_folder"]["GenAllV_PairDistanceZ"]);
  }

  int label_index = -1;
  for (size_t iToken = 0, endToken = reco_vertex_collection_tokens_.size(); iToken < endToken; ++iToken) {
    auto const& vertex_token = reco_vertex_collection_tokens_[iToken];
    std::vector<recoPrimaryVertex> recopv;  // a list of reconstructed
                                            // primary MC vertices
    std::string label = reco_vertex_collections_[++label_index].label();
    edm::Handle<edm::View<reco::Vertex>> recVtxs;
    if (!iEvent.getByToken(vertex_token, recVtxs)) {
      if (!errorPrintedForColl_[iToken]) {
        edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed")
            << "Skipping vertex collection: " << label << " since it is missing.";
        errorPrintedForColl_[iToken] = true;
      }
      continue;
    }

    {
      // check upfront that refs to track are (likely) to be valid
      bool ok = true;
      for (const auto& v : *recVtxs) {
        if (v.tracksSize() > 0) {
          const auto& ref = v.trackRefAt(0);
          if (ref.isNull() || !ref.isAvailable()) {
            if (!errorPrintedForColl_[iToken]) {
              edm::LogWarning("PrimaryVertexAnalyzer4PUSlimmed")
                  << "Skipping vertex collection: " << label
                  << " since likely the track collection the vertex has refs pointing to is missing (at least the "
                     "first TrackBaseRef is null or not available)";
              errorPrintedForColl_[iToken] = true;
            }
            ok = false;
          }
        }
      }
      if (!ok)
        continue;
    }

    reco::VertexRecoToSimCollection vertex_r2s = vertexAssociator.associateRecoToSim(recVtxs, TVCollectionH);
    reco::VertexSimToRecoCollection vertex_s2r = vertexAssociator.associateSimToReco(recVtxs, TVCollectionH);

    resetSimPVAssociation(simpv);
    matchSim2RecoVertices(simpv, vertex_s2r);
    recopv = getRecoPVs(recVtxs);
    computePairDistance(recopv, mes_[label]["RecoAllAssoc2Gen_PairDistanceZ"]);
    matchReco2SimVertices(recopv, vertex_r2s, simpv);

    int num_total_gen_vertices_assoc2reco = 0;
    int num_total_reco_vertices_assoc2gen = 0;
    int num_total_gen_vertices_multiassoc2reco = 0;
    int num_total_reco_vertices_multiassoc2gen = 0;
    int num_total_reco_vertices_duplicate = 0;
    int genpv_position_in_reco_collection = -1;
    for (auto const& v : simpv) {
      float mistag = 1.;
      // TODO(rovere) put selectors here in front of fill* methods.
      if (v.eventId.event() == 0) {
        if (!recVtxs->empty() && std::find(v.rec_vertices.begin(), v.rec_vertices.end(), &((*recVtxs.product())[0])) !=
                                     v.rec_vertices.end()) {
          mistag = 0.;
          kind_of_signal_vertex |= (1 << IS_ASSOC2FIRST_RECO);
        } else {
          if (!v.rec_vertices.empty()) {
            kind_of_signal_vertex |= (1 << IS_ASSOC2ANY_RECO);
          }
        }
        mes_[label]["KindOfSignalPV"]->Fill(kind_of_signal_vertex);
        mes_[label]["MisTagRate"]->Fill(mistag);
        mes_[label]["MisTagRate_vs_PU"]->Fill(simpv.size(), mistag);
        mes_[label]["MisTagRate_vs_sum-pt2"]->Fill(v.ptsq, mistag);
        mes_[label]["MisTagRate_vs_Z"]->Fill(v.z, mistag);
        mes_[label]["MisTagRate_vs_R"]->Fill(v.r, mistag);
        mes_[label]["MisTagRate_vs_NumTracks"]->Fill(v.nGenTrk, mistag);
        if (signal_is_highest_pt) {
          mes_[label]["MisTagRateSignalIsHighest"]->Fill(mistag);
          mes_[label]["MisTagRateSignalIsHighest_vs_PU"]->Fill(simpv.size(), mistag);
          mes_[label]["MisTagRateSignalIsHighest_vs_sum-pt2"]->Fill(v.ptsq, mistag);
          mes_[label]["MisTagRateSignalIsHighest_vs_Z"]->Fill(v.z, mistag);
          mes_[label]["MisTagRateSignalIsHighest_vs_R"]->Fill(v.r, mistag);
          mes_[label]["MisTagRateSignalIsHighest_vs_NumTracks"]->Fill(v.nGenTrk, mistag);
        } else {
          mes_[label]["MisTagRateSignalIsNotHighest"]->Fill(mistag);
          mes_[label]["MisTagRateSignalIsNotHighest_vs_PU"]->Fill(simpv.size(), mistag);
          mes_[label]["MisTagRateSignalIsNotHighest_vs_sum-pt2"]->Fill(v.ptsq, mistag);
          mes_[label]["MisTagRateSignalIsNotHighest_vs_Z"]->Fill(v.z, mistag);
          mes_[label]["MisTagRateSignalIsNotHighest_vs_R"]->Fill(v.r, mistag);
          mes_[label]["MisTagRateSignalIsNotHighest_vs_NumTracks"]->Fill(v.nGenTrk, mistag);
        }
        // Now check at which location the Simulated PV has been
        // reconstructed in the primary vertex collection
        // at-hand. Mark it with fake index -1 if it was not
        // reconstructed at all.

        auto iv = (*recVtxs.product()).begin();
        for (int pv_position_in_reco_collection = 0; iv != (*recVtxs.product()).end();
             ++pv_position_in_reco_collection, ++iv) {
          if (std::find(v.rec_vertices.begin(), v.rec_vertices.end(), &(*iv)) != v.rec_vertices.end()) {
            mes_[label]["TruePVLocationIndex"]->Fill(pv_position_in_reco_collection);
            const bool genPVMatchedToRecoPV = (pv_position_in_reco_collection == 0);
            mes_[label]["TruePVLocationIndexCumulative"]->Fill(genPVMatchedToRecoPV ? 0 : 1);

            if (signal_is_highest_pt) {
              mes_[label]["TruePVLocationIndexSignalIsHighest"]->Fill(pv_position_in_reco_collection);
            } else {
              mes_[label]["TruePVLocationIndexSignalIsNotHighest"]->Fill(pv_position_in_reco_collection);
            }

            fillRecoAssociatedGenPVHistograms(label, v, genPVMatchedToRecoPV);
            if (genPVMatchedToRecoPV) {
              auto pv = recopv[0];
              assert(pv.recVtx == &(*iv));
              fillResolutionAndPullHistograms(label, num_pileup_vertices, pv, true);
            }
            genpv_position_in_reco_collection = pv_position_in_reco_collection;
            break;
          }
        }

        // If we reached the end, it means that the Simulated PV has not
        // been associated to any reconstructed vertex: mark it as
        // missing in the reconstructed vertex collection using the fake
        // index -1.
        if (iv == (*recVtxs.product()).end()) {
          mes_[label]["TruePVLocationIndex"]->Fill(-1.);
          mes_[label]["TruePVLocationIndexCumulative"]->Fill(-1.);
          if (signal_is_highest_pt)
            mes_[label]["TruePVLocationIndexSignalIsHighest"]->Fill(-1.);
          else
            mes_[label]["TruePVLocationIndexSignalIsNotHighest"]->Fill(-1.);
        }
      }

      if (!v.rec_vertices.empty())
        num_total_gen_vertices_assoc2reco++;
      if (v.rec_vertices.size() > 1)
        num_total_gen_vertices_multiassoc2reco++;
      // No need to N-tplicate the Gen-related cumulative histograms:
      // fill them only at the first iteration
      if (do_generic_sim_plots_ && label_index == 0)
        fillGenericGenVertexHistograms(v);
      fillRecoAssociatedGenVertexHistograms(label, v);
    }
    calculatePurityAndFillHistograms(label, recopv, genpv_position_in_reco_collection, signal_is_highest_pt);

    mes_[label]["GenAllAssoc2Reco_NumVertices"]->Fill(simpv.size(), simpv.size());
    mes_[label]["GenAllAssoc2RecoMatched_NumVertices"]->Fill(simpv.size(), num_total_gen_vertices_assoc2reco);
    mes_[label]["GenAllAssoc2RecoMultiMatched_NumVertices"]->Fill(simpv.size(), num_total_gen_vertices_multiassoc2reco);
    for (auto& v : recopv) {
      fillGenAssociatedRecoVertexHistograms(label, num_pileup_vertices, v);
      if (!v.sim_vertices.empty()) {
        num_total_reco_vertices_assoc2gen++;
        if (v.sim_vertices_internal[0]->rec_vertices.size() > 1) {
          num_total_reco_vertices_duplicate++;
        }
      }
      if (v.sim_vertices.size() > 1)
        num_total_reco_vertices_multiassoc2gen++;
    }
    mes_[label]["RecoAllAssoc2Gen_NumVertices"]->Fill(recopv.size(), recopv.size());
    mes_[label]["RecoAllAssoc2GenMatched_NumVertices"]->Fill(recopv.size(), num_total_reco_vertices_assoc2gen);
    mes_[label]["RecoAllAssoc2GenMultiMatched_NumVertices"]->Fill(recopv.size(),
                                                                  num_total_reco_vertices_multiassoc2gen);
    mes_[label]["RecoAllAssoc2MultiMatchedGen_NumVertices"]->Fill(recopv.size(), num_total_reco_vertices_duplicate);
    mes_[label]["RecoVtx_vs_GenVtx"]->Fill(simpv.size(), recopv.size());
    mes_[label]["MatchedRecoVtx_vs_GenVtx"]->Fill(simpv.size(), num_total_reco_vertices_assoc2gen);
  }
}  // end of analyze

template <class T>
void PrimaryVertexAnalyzer4PUSlimmed::computePairDistance(const T& collection, MonitorElement* me) {
  for (unsigned int i = 0; i < collection.size(); ++i) {
    for (unsigned int j = i + 1; j < collection.size(); ++j) {
      me->Fill(std::abs(collection[i].z - collection[j].z));
    }
  }
}
