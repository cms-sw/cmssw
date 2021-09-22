#include <memory>

#include "Validation/MuonCSCDigis/interface/CSCStubEfficiencyValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStubMatcher.h"

CSCStubEfficiencyValidation::CSCStubEfficiencyValidation(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
    : CSCBaseValidation(pset) {
  const auto& simVertex = pset.getParameter<edm::ParameterSet>("simVertex");
  simVertexInput_ = iC.consumes<edm::SimVertexContainer>(simVertex.getParameter<edm::InputTag>("inputTag"));
  const auto& simTrack = pset.getParameter<edm::ParameterSet>("simTrack");
  simTrackInput_ = iC.consumes<edm::SimTrackContainer>(simTrack.getParameter<edm::InputTag>("inputTag"));

  // Initialize stub matcher
  cscStubMatcher_ = std::make_unique<CSCStubMatcher>(pset, std::move(iC));

  // get the eta ranges
  etaMins_ = pset.getParameter<std::vector<double>>("etaMins");
  etaMaxs_ = pset.getParameter<std::vector<double>>("etaMaxs");
}

CSCStubEfficiencyValidation::~CSCStubEfficiencyValidation() {}

void CSCStubEfficiencyValidation::bookHistograms(DQMStore::IBooker& iBooker) {
  for (int i = 1; i <= 10; ++i) {
    int j = i - 1;
    const std::string cn(CSCDetId::chamberName(i));

    std::string t1 = "ALCTEtaDenom_" + cn;
    std::string t2 = "CLCTEtaDenom_" + cn;
    std::string t3 = "LCTEtaDenom_" + cn;

    iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/ALCT/Occupancy/");
    etaALCTDenom[j] = iBooker.book1D(t1, t1 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaALCTDenom[j]->getTH1()->SetMinimum(0);
    t1 = "ALCTEtaNum_" + cn;
    etaALCTNum[j] = iBooker.book1D(t1, t1 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaALCTNum[j]->getTH1()->SetMinimum(0);

    iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/CLCT/Occupancy/");
    etaCLCTDenom[j] = iBooker.book1D(t2, t2 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaCLCTDenom[j]->getTH1()->SetMinimum(0);
    t2 = "CLCTEtaNum_" + cn;
    etaCLCTNum[j] = iBooker.book1D(t2, t2 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaCLCTNum[j]->getTH1()->SetMinimum(0);

    iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/LCT/Occupancy/");
    etaLCTDenom[j] = iBooker.book1D(t3, t3 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaLCTDenom[j]->getTH1()->SetMinimum(0);
    t3 = "LCTEtaNum_" + cn;
    etaLCTNum[j] = iBooker.book1D(t3, t3 + ";True Muon |#eta|; Entries", 50, etaMins_[j], etaMaxs_[j]);
    etaLCTNum[j]->getTH1()->SetMinimum(0);
  }
}

void CSCStubEfficiencyValidation::analyze(const edm::Event& e, const edm::EventSetup& eventSetup) {
  // Define handles
  edm::Handle<edm::SimTrackContainer> sim_tracks;
  edm::Handle<edm::SimVertexContainer> sim_vertices;

  // Use token to retreive event information
  e.getByToken(simTrackInput_, sim_tracks);
  e.getByToken(simVertexInput_, sim_vertices);

  // Initialize StubMatcher
  cscStubMatcher_->init(e, eventSetup);

  const edm::SimTrackContainer& sim_track = *sim_tracks.product();
  const edm::SimVertexContainer& sim_vert = *sim_vertices.product();

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
