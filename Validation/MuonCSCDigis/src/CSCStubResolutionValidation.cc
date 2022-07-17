#include <memory>

#include "Validation/MuonCSCDigis/interface/CSCStubResolutionValidation.h"
#include "Validation/MuonCSCDigis/interface/CSCStubMatcher.h"

CSCStubResolutionValidation::CSCStubResolutionValidation(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
    : CSCBaseValidation(pset) {
  const auto& simVertex = pset.getParameter<edm::ParameterSet>("simVertex");
  simVertexInput_ = iC.consumes<edm::SimVertexContainer>(simVertex.getParameter<edm::InputTag>("inputTag"));
  const auto& simTrack = pset.getParameter<edm::ParameterSet>("simTrack");
  simTrackInput_ = iC.consumes<edm::SimTrackContainer>(simTrack.getParameter<edm::InputTag>("inputTag"));

  // Initialize stub matcher
  cscStubMatcher_ = std::make_unique<CSCStubMatcher>(pset, std::move(iC));
}

CSCStubResolutionValidation::~CSCStubResolutionValidation() {}

//create folder for resolution histograms and book them
void CSCStubResolutionValidation::bookHistograms(DQMStore::IBooker& iBooker) {
  for (int i = 1; i <= 10; ++i) {
    int j = i - 1;
    const std::string cn(CSCDetId::chamberName(i));

    //Position resolution; CLCT
    std::string t1 = "CLCTPosRes_hs_" + cn;
    std::string t2 = "CLCTPosRes_qs_" + cn;
    std::string t3 = "CLCTPosRes_es_" + cn;

    iBooker.setCurrentFolder("MuonCSCDigisV/CSCDigiTask/CLCT/Resolution/");
    posresCLCT_hs[j] = iBooker.book1D(
        t1, cn + " CLCT Position Resolution (1/2-strip prec.); Strip_{L1T} - Strip_{SIM}; Entries", 50, -1, 1);
    posresCLCT_qs[j] = iBooker.book1D(
        t2, cn + " CLCT Position Resolution (1/4-strip prec.); Strip_{L1T} - Strip_{SIM}; Entries", 50, -1, 1);
    posresCLCT_es[j] = iBooker.book1D(
        t3, cn + " CLCT Position Resolution (1/8-strip prec.); Strip_{L1T} - Strip_{SIM}; Entries", 50, -1, 1);

    //Slope resolution; CLCT
    std::string t4 = "CLCTBendRes_" + cn;

    bendresCLCT[j] =
        iBooker.book1D(t4, cn + " CLCT Bend Resolution; Slope_{L1T} - Slope_{SIM}; Entries", 50, -0.5, 0.5);
  }
}

void CSCStubResolutionValidation::analyze(const edm::Event& e, const edm::EventSetup& eventSetup) {
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

  // Loop through good tracks, use corresponding vertex to match stubs, then fill hists of chambers where the stub appears.
  for (const auto& t : sim_track_selected) {
    std::vector<bool> hitCLCT(10);

    std::vector<float> delta_fhs_clct(10);
    std::vector<float> delta_fqs_clct(10);
    std::vector<float> delta_fes_clct(10);

    std::vector<float> dslope_clct(10);

    // Match track to stubs with appropriate vertex
    cscStubMatcher_->match(t, sim_vert[t.vertIndex()]);

    // Store matched stubs.
    // Key: ChamberID, Value : CSCStubDigiContainer
    const auto& clcts = cscStubMatcher_->clcts();

    // CLCTs
    for (auto& [id, container] : clcts) {
      const CSCDetId cscId(id);

      // get the best clct in chamber
      const auto& clct = cscStubMatcher_->bestClctInChamber(id);
      if (!clct.isValid())
        continue;

      // ME1a CLCTs are saved in ME1b container. So the DetId need to be specified
      const bool isME11(cscId.station() == 1 and (cscId.ring() == 4 or cscId.ring() == 1));
      const bool isME1a(isME11 and clct.getKeyStrip() > CSCConstants::MAX_HALF_STRIP_ME1B);
      int ring = cscId.ring();
      if (isME1a)
        ring = 4;
      else if (isME11)
        ring = 1;
      CSCDetId cscId2(cscId.endcap(), cscId.station(), ring, cscId.chamber(), 0);
      auto id2 = cscId2.rawId();

      // calculate deltastrip for ME1/a. Basically, we need to subtract 64 from the CLCT key strip to
      // compare with key strip as obtained through the fit to simhits positions.
      int deltaStrip = 0;
      if (isME1a)
        deltaStrip = CSCConstants::NUM_STRIPS_ME1B;

      // fractional strip
      const float fhs_clct = clct.getFractionalStrip(2);
      const float fqs_clct = clct.getFractionalStrip(4);
      const float fes_clct = clct.getFractionalStrip(8);

      // in half-strips per layer
      const float slopeHalfStrip(clct.getFractionalSlope());
      const float slopeStrip(slopeHalfStrip / 2.);

      // get the fit hits in chamber for true value
      float stripIntercept, stripSlope;
      cscStubMatcher_->cscDigiMatcher()->muonSimHitMatcher()->fitHitsInChamber(id2, stripIntercept, stripSlope);

      // add offset of +0.25 strips for non-ME1/1 chambers
      if (!isME11) {
        stripIntercept -= 0.25;
      }

      const float strip_csc_sh = stripIntercept;
      const float bend_csc_sh = stripSlope;

      const unsigned chamberType(cscId2.iChamberType());
      hitCLCT[chamberType - 1] = true;

      delta_fhs_clct[chamberType - 1] = fhs_clct - deltaStrip - strip_csc_sh;
      delta_fqs_clct[chamberType - 1] = fqs_clct - deltaStrip - strip_csc_sh;
      delta_fes_clct[chamberType - 1] = fes_clct - deltaStrip - strip_csc_sh;

      dslope_clct[chamberType - 1] = slopeStrip - bend_csc_sh;
    }

    for (int i = 0; i < 10; i++) {
      if (hitCLCT[i]) {
        //fill histograms
        posresCLCT_hs[i]->Fill(delta_fhs_clct[i]);
        posresCLCT_qs[i]->Fill(delta_fqs_clct[i]);
        posresCLCT_es[i]->Fill(delta_fes_clct[i]);

        bendresCLCT[i]->Fill(dslope_clct[i]);
      }
    }
  }
}
