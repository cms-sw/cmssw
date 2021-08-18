#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Validation/MuonCSCDigis/interface/CSCStubEfficiencyValidation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"

CSCStubEfficiencyValidation::CSCStubEfficiencyValidation(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
    : CSCBaseValidation(pset) {
  const auto& simVertex = pset.getParameter<edm::ParameterSet>("simVertex");
  simVertexInput_ = iC.consumes<edm::SimVertexContainer>(simVertex.getParameter<edm::InputTag>("inputTag"));
  const auto& simTrack = pset.getParameter<edm::ParameterSet>("simTrack");
  simTrackInput_ = iC.consumes<edm::SimTrackContainer>(simTrack.getParameter<edm::InputTag>("inputTag"));
  simTrackMinPt_ = simTrack.getParameter<double>("minPt");
  simTrackMinEta_ = simTrack.getParameter<double>("minEta");
  simTrackMaxEta_ = simTrack.getParameter<double>("maxEta");

  // all CSC TPs have the same label
  const auto& stubConfig = pset.getParameterSet("cscALCT");
  inputTag_ = stubConfig.getParameter<edm::InputTag>("inputTag");
  alcts_Token_ = iC.consumes<CSCALCTDigiCollection>(inputTag_);
  clcts_Token_ = iC.consumes<CSCCLCTDigiCollection>(inputTag_);
  lcts_Token_ = iC.consumes<CSCCorrelatedLCTDigiCollection>(inputTag_);

  // Initialize stub matcher
  cscStubMatcher_.reset(new CSCStubMatcher(pset, std::move(iC)));

  // get the eta ranges
  etaMins_ = pset.getParameter<std::vector<double>>("etaMins");
  etaMaxs_ = pset.getParameter<std::vector<double>>("etaMaxs");
}

CSCStubEfficiencyValidation::~CSCStubEfficiencyValidation() {}

void CSCStubEfficiencyValidation::bookHistograms(DQMStore::IBooker& iBooker) {
  iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/Stub/Occupancy/");

  for (int i = 1; i <= 10; ++i) {
    int j = i - 1;
    const std::string cn(CSCDetId::chamberName(i));

    std::string t1 = "ALCTEtaDenom_" + cn;
    std::string t2 = "CLCTEtaDenom_" + cn;
    std::string t3 = "LCTEtaDenom_" + cn;

    etaALCTDenom[j] = iBooker.book1D(t1, t1 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaCLCTDenom[j] = iBooker.book1D(t2, t2 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaLCTDenom[j] = iBooker.book1D(t3, t3 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);

    t1 = "ALCTEtaNum_" + cn;
    t2 = "CLCTEtaNum_" + cn;
    t3 = "LCTEtaNum_" + cn;

    etaALCTNum[j] = iBooker.book1D(t1, t1 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaCLCTNum[j] = iBooker.book1D(t2, t2 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaLCTNum[j] = iBooker.book1D(t3, t3 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
  }
}

void CSCStubEfficiencyValidation::analyze(const edm::Event& e, const edm::EventSetup& eventSetup) {
  // Define handles
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  edm::Handle<CSCCorrelatedLCTDigiCollection> lcts;

  // Use token to retreive event information
  e.getByToken(simTrackInput_, sim_tracks);
  e.getByToken(simVertexInput_, sim_vertices);
  e.getByToken(alcts_Token_, alcts);
  e.getByToken(clcts_Token_, clcts);
  e.getByToken(lcts_Token_, lcts);

  // Initialize StubMatcher
  cscStubMatcher_->init(e, eventSetup);

  const edm::SimTrackContainer& sim_track = *sim_tracks.product();
  const edm::SimVertexContainer& sim_vert = *sim_vertices.product();

  if (!alcts.isValid()) {
    edm::LogError("CSCStubEfficiencyValidation") << "Cannot get ALCTs by label " << inputTag_.encode();
  }
  if (!clcts.isValid()) {
    edm::LogError("CSCStubEfficiencyValidation") << "Cannot get CLCTs by label " << inputTag_.encode();
  }
  if (!lcts.isValid()) {
    edm::LogError("CSCStubEfficiencyValidation") << "Cannot get LCTs by label " << inputTag_.encode();
  }

  // select simtracks for true muons
  edm::SimTrackContainer sim_track_selected;
  for (const auto& t : sim_track) {
    if (!isSimTrackGood(t))
      continue;
    sim_track_selected.push_back(t);
  }

  // Skip events with no selected simtracks
  if (sim_track_selected.empty())
    return;

  // Loop through good tracks, use corresponding vetrex to match stubs, then fill hists of chambers where the stub appears.
  for (const auto& t : sim_track_selected) {
    std::vector<bool> hitALCT(10);
    std::vector<bool> hitCLCT(10);
    std::vector<bool> hitLCT(10);

    // Match track to stubs with appropriate vertex
    cscStubMatcher_->match(t, sim_vert[t.vertIndex()]);

    // Store matched stubs.
    // Key: ChamberID, Value : CSCStubDigiContainer
    const auto& alcts = cscStubMatcher_->alcts();
    const auto& clcts = cscStubMatcher_->clcts();
    const auto& lcts = cscStubMatcher_->lcts();

    // denominator histograms
    for (int i = 0; i < 10; ++i) {
      etaALCTDenom[i]->Fill(t.momentum().eta());
      etaCLCTDenom[i]->Fill(t.momentum().eta());
      etaLCTDenom[i]->Fill(t.momentum().eta());
    }

    for (auto& [id, container] : alcts) {
      const CSCDetId cscId(id);
      const unsigned chamberType(cscId.iChamberType());
      hitALCT[chamberType - 1] = true;
    }

    for (auto& [id, container] : clcts) {
      const CSCDetId cscId(id);
      const unsigned chamberType(cscId.iChamberType());
      hitCLCT[chamberType - 1] = true;
    }

    for (auto& [id, container] : lcts) {
      const CSCDetId cscId(id);
      const unsigned chamberType(cscId.iChamberType());
      hitLCT[chamberType - 1] = true;
    }

    // numerator histograms
    for (int i = 0; i < 10; ++i) {
      if (hitALCT[i])
        etaALCTNum[i]->Fill(t.momentum().eta());
      if (hitCLCT[i])
        etaCLCTNum[i]->Fill(t.momentum().eta());
      if (hitLCT[i])
        etaLCTNum[i]->Fill(t.momentum().eta());
    }
  }
}

bool CSCStubEfficiencyValidation::isSimTrackGood(const SimTrack& t) {
  // SimTrack selection
  if (t.noVertex())
    return false;
  if (t.noGenpart())
    return false;
  // only muons
  if (std::abs(t.type()) != 13)
    return false;
  // pt selection
  if (t.momentum().pt() < simTrackMinPt_)
    return false;
  // eta selection
  const float eta(std::abs(t.momentum().eta()));
  if (eta > simTrackMaxEta_ || eta < simTrackMinEta_)
    return false;
  return true;
}
