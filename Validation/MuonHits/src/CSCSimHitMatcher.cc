#include "Validation/MuonHits/interface/CSCSimHitMatcher.h"
#include "TGraphErrors.h"
#include "TF1.h"

using namespace std;

CSCSimHitMatcher::CSCSimHitMatcher(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC)
    : MuonSimHitMatcher(ps, std::move(iC)) {
  simHitPSet_ = ps.getParameterSet("cscSimHit");
  verbose_ = simHitPSet_.getParameter<int>("verbose");
  simMuOnly_ = simHitPSet_.getParameter<bool>("simMuOnly");
  discardEleHits_ = simHitPSet_.getParameter<bool>("discardEleHits");

  simHitInput_ = iC.consumes<edm::PSimHitContainer>(simHitPSet_.getParameter<edm::InputTag>("inputTag"));
  geomToken_ = iC.esConsumes<CSCGeometry, MuonGeometryRecord>();
}

/// initialize the event
void CSCSimHitMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  geometry_ = &iSetup.getData(geomToken_);
  MuonSimHitMatcher::init(iEvent, iSetup);
}

/// do the matching
void CSCSimHitMatcher::match(const SimTrack& track, const SimVertex& vertex) {
  clear();

  // instantiates the track ids and simhits
  MuonSimHitMatcher::match(track, vertex);

  // hard cut on non-CSC muons
  if (std::abs(track.momentum().eta()) < 0.9)
    return;
  if (std::abs(track.momentum().eta()) > 2.45)
    return;

  matchSimHitsToSimTrack();

  if (verbose_) {
    edm::LogInfo("CSCSimHitMatcher") << "nTrackIds " << track_ids_.size() << " nSelectedCSCSimHits " << hits_.size();
    edm::LogInfo("CSCSimHitMatcher") << "detids CSC " << detIds(0).size();

    for (const auto& id : detIds(0)) {
      const auto& simhits = hitsInDetId(id);
      const auto& simhits_gp = simHitsMeanPosition(simhits);
      const auto& strips = hitStripsInDetId(id);
      CSCDetId cscid(id);
      if (cscid.station() == 1 and (cscid.ring() == 1 or cscid.ring() == 4)) {
        edm::LogInfo("CSCSimHitMatcher") << "cscdetid " << CSCDetId(id) << ": " << simhits.size() << " "
                                         << simhits_gp.phi() << " " << detid_to_hits_[id].size();
        edm::LogInfo("CSCSimHitMatcher") << "nStrip " << strips.size();
        edm::LogInfo("CSCSimHitMatcher") << "strips : ";
        for (const auto& p : strips) {
          edm::LogInfo("CSCSimHitMatcher") << p;
        }
      }
    }
  }
}

void CSCSimHitMatcher::matchSimHitsToSimTrack() {
  for (const auto& track_id : track_ids_) {
    for (const auto& h : simHits_) {
      if (h.trackId() != track_id)
        continue;
      int pdgid = h.particleType();
      if (simMuOnly_ && std::abs(pdgid) != 13)
        continue;
      // discard electron hits in the CSC chambers
      if (discardEleHits_ && std::abs(pdgid) == 11)
        continue;

      const CSCDetId& layer_id(h.detUnitId());
      hits_.push_back(h);
      detid_to_hits_[h.detUnitId()].push_back(h);
      chamber_to_hits_[layer_id.chamberId().rawId()].push_back(h);
    }
  }
}

std::set<unsigned int> CSCSimHitMatcher::detIds(int csc_type) const {
  std::set<unsigned int> result;
  for (const auto& p : detid_to_hits_) {
    const auto& id = p.first;
    if (csc_type > 0) {
      CSCDetId detId(id);
      if (MuonHitHelper::toCSCType(detId.station(), detId.ring()) != csc_type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

std::set<unsigned int> CSCSimHitMatcher::chamberIds(int csc_type) const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_hits_) {
    const auto& id = p.first;
    if (csc_type > 0) {
      CSCDetId detId(id);
      if (MuonHitHelper::toCSCType(detId.station(), detId.ring()) != csc_type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

int CSCSimHitMatcher::nLayersWithHitsInChamber(unsigned int detid) const {
  set<int> layers_with_hits;
  for (const auto& h : MuonSimHitMatcher::hitsInChamber(detid)) {
    const CSCDetId& idd(h.detUnitId());
    layers_with_hits.insert(idd.layer());
  }
  return layers_with_hits.size();
}

bool CSCSimHitMatcher::hitStation(int st, int nlayers) const {
  int nst = 0;
  for (const auto& ddt : chamberIds()) {
    const CSCDetId id(ddt);
    int ri(id.ring());
    if (id.station() != st)
      continue;

    // ME1/a case - check the ME1/b chamber
    if (st == 1 and ri == 4) {
      CSCDetId idME1b(id.endcap(), id.station(), 1, id.chamber(), 0);
      const int nl1a(nLayersWithHitsInChamber(id.rawId()));
      const int nl1b(nLayersWithHitsInChamber(idME1b.rawId()));
      if (nl1a + nl1b < nlayers)
        continue;
      ++nst;
    }
    // ME1/b case - check the ME1/a chamber
    else if (st == 1 and ri == 1) {
      CSCDetId idME1a(id.endcap(), id.station(), 4, id.chamber(), 0);
      const int nl1a(nLayersWithHitsInChamber(idME1a.rawId()));
      const int nl1b(nLayersWithHitsInChamber(id.rawId()));
      if (nl1a + nl1b < nlayers)
        continue;
      ++nst;
    }
    // default case
    else {
      const int nl(nLayersWithHitsInChamber(id.rawId()));
      if (nl < nlayers)
        continue;
      ++nst;
    }
  }
  return nst;
}

int CSCSimHitMatcher::nStations(int nlayers) const {
  return (hitStation(1, nlayers) + hitStation(2, nlayers) + hitStation(3, nlayers) + hitStation(4, nlayers));
}

float CSCSimHitMatcher::LocalBendingInChamber(unsigned int detid) const {
  const CSCDetId cscid(detid);
  if (nLayersWithHitsInChamber(detid) < 6)
    return -100;
  float phi_layer1 = -10;
  float phi_layer6 = 10;

  if (cscid.station() == 1 and (cscid.ring() == 1 or cscid.ring() == 4)) {
    // phi in layer 1
    const CSCDetId cscid1a(cscid.endcap(), cscid.station(), 4, cscid.chamber(), 1);
    const CSCDetId cscid1b(cscid.endcap(), cscid.station(), 1, cscid.chamber(), 1);
    const edm::PSimHitContainer& hits1a = MuonSimHitMatcher::hitsInDetId(cscid1a.rawId());
    const edm::PSimHitContainer& hits1b = MuonSimHitMatcher::hitsInDetId(cscid1b.rawId());
    const GlobalPoint& gp1a = simHitsMeanPosition(MuonSimHitMatcher::hitsInDetId(cscid1a.rawId()));
    const GlobalPoint& gp1b = simHitsMeanPosition(MuonSimHitMatcher::hitsInDetId(cscid1b.rawId()));
    if (!hits1a.empty() and !hits1b.empty())
      phi_layer1 = (gp1a.phi() + gp1b.phi()) / 2.0;
    else if (!hits1a.empty())
      phi_layer1 = gp1a.phi();
    else if (!hits1b.empty())
      phi_layer1 = gp1b.phi();

    // phi in layer 6
    const CSCDetId cscid6a(cscid.endcap(), cscid.station(), 4, cscid.chamber(), 6);
    const CSCDetId cscid6b(cscid.endcap(), cscid.station(), 1, cscid.chamber(), 6);
    const edm::PSimHitContainer& hits6a = MuonSimHitMatcher::hitsInDetId(cscid6a.rawId());
    const edm::PSimHitContainer& hits6b = MuonSimHitMatcher::hitsInDetId(cscid6b.rawId());
    const GlobalPoint& gp6a = simHitsMeanPosition(MuonSimHitMatcher::hitsInDetId(cscid6a.rawId()));
    const GlobalPoint& gp6b = simHitsMeanPosition(MuonSimHitMatcher::hitsInDetId(cscid6b.rawId()));
    if (!hits6a.empty() and !hits6b.empty())
      phi_layer6 = (gp6a.phi() + gp6b.phi()) / 2.0;
    else if (!hits6a.empty())
      phi_layer6 = gp6a.phi();
    else if (!hits6b.empty())
      phi_layer6 = gp6b.phi();

  } else {
    // phi in layer 1
    const CSCDetId cscid1(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 1);
    const GlobalPoint& gp1 = simHitsMeanPosition(MuonSimHitMatcher::hitsInDetId(cscid1.rawId()));
    phi_layer1 = gp1.phi();

    // phi in layer 6
    const CSCDetId cscid6(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 6);
    const GlobalPoint& gp6 = simHitsMeanPosition(MuonSimHitMatcher::hitsInDetId(cscid6.rawId()));
    phi_layer6 = gp6.phi();
  }
  return deltaPhi(phi_layer6, phi_layer1);
}

// difference in strip per layer
void CSCSimHitMatcher::fitHitsInChamber(unsigned int detid, float& intercept, float& slope) const {
  const CSCDetId cscid(detid);

  const auto& sim_hits = hitsInChamber(detid);

  if (sim_hits.empty())
    return;

  vector<float> x;
  vector<float> y;
  vector<float> xe;
  vector<float> ye;

  const float HALF_STRIP_ERROR = 0.288675;

  for (const auto& h : sim_hits) {
    const LocalPoint& lp = h.entryPoint();
    const auto& d = h.detUnitId();
    float s = dynamic_cast<const CSCGeometry*>(geometry_)->layer(d)->geometry()->strip(lp);
    // shift to key half strip layer (layer 3)
    x.push_back(CSCDetId(d).layer() - 3);
    y.push_back(s);
    xe.push_back(float(0));
    ye.push_back(2 * HALF_STRIP_ERROR);
  }
  if (x.size() < 2)
    return;

  std::unique_ptr<TGraphErrors> gr(new TGraphErrors(x.size(), &x[0], &y[0], &xe[0], &ye[0]));
  std::unique_ptr<TF1> fit(new TF1("fit", "pol1", -3, 4));
  gr->Fit("fit", "EMQ");

  intercept = fit->GetParameter(0);
  slope = fit->GetParameter(1);
}

float CSCSimHitMatcher::simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const {
  if (sim_hits.empty())
    return -1.f;

  float sums = 0.f;
  size_t n = 0;
  for (const auto& h : sim_hits) {
    const LocalPoint& lp = h.entryPoint();
    float s;
    const auto& d = h.detUnitId();
    s = dynamic_cast<const CSCGeometry*>(geometry_)->layer(d)->geometry()->strip(lp);
    // convert to half-strip:
    s *= 2.;
    sums += s;
    ++n;
  }
  if (n == 0)
    return -1.f;
  return sums / n;
}

float CSCSimHitMatcher::simHitsMeanWG(const edm::PSimHitContainer& sim_hits) const {
  if (sim_hits.empty())
    return -1.f;

  float sums = 0.f;
  size_t n = 0;
  for (const auto& h : sim_hits) {
    const LocalPoint& lp = h.entryPoint();
    float s;
    const auto& d = h.detUnitId();
    // find nearest wire
    const auto& layerG(dynamic_cast<const CSCGeometry*>(geometry_)->layer(d)->geometry());
    int nearestWire(layerG->nearestWire(lp));
    // then find the corresponding wire group
    s = layerG->wireGroup(nearestWire);
    sums += s;
    ++n;
  }
  if (n == 0)
    return -1.f;
  return sums / n;
}

std::set<int> CSCSimHitMatcher::hitStripsInDetId(unsigned int detid, int margin_n_strips) const {
  set<int> result;
  const auto& simhits = MuonSimHitMatcher::hitsInDetId(detid);
  CSCDetId id(detid);
  int max_nstrips = dynamic_cast<const CSCGeometry*>(geometry_)->layer(id)->geometry()->numberOfStrips();
  for (const auto& h : simhits) {
    const LocalPoint& lp = h.entryPoint();
    int central_strip = dynamic_cast<const CSCGeometry*>(geometry_)->layer(id)->geometry()->nearestStrip(lp);
    int smin = central_strip - margin_n_strips;
    smin = (smin > 0) ? smin : 1;
    int smax = central_strip + margin_n_strips;
    smax = (smax <= max_nstrips) ? smax : max_nstrips;
    for (int ss = smin; ss <= smax; ++ss)
      result.insert(ss);
  }
  return result;
}

std::set<int> CSCSimHitMatcher::hitWiregroupsInDetId(unsigned int detid, int margin_n_wg) const {
  set<int> result;
  const auto& simhits = MuonSimHitMatcher::hitsInDetId(detid);
  CSCDetId id(detid);
  int max_n_wg = dynamic_cast<const CSCGeometry*>(geometry_)->layer(id)->geometry()->numberOfWireGroups();
  for (const auto& h : simhits) {
    const LocalPoint& lp = h.entryPoint();
    const auto& layer_geo = dynamic_cast<const CSCGeometry*>(geometry_)->layer(id)->geometry();
    int central_wg = layer_geo->wireGroup(layer_geo->nearestWire(lp));
    int wg_min = central_wg - margin_n_wg;
    wg_min = (wg_min > 0) ? wg_min : 1;
    int wg_max = central_wg + margin_n_wg;
    wg_max = (wg_max <= max_n_wg) ? wg_max : max_n_wg;
    for (int wg = wg_min; wg <= wg_max; ++wg)
      result.insert(wg);
  }
  return result;
}

int CSCSimHitMatcher::nCoincidenceChambers(int min_n_layers) const {
  int result = 0;
  const auto& chamber_ids = chamberIds(0);
  for (const auto& id : chamber_ids) {
    if (nLayersWithHitsInChamber(id) >= min_n_layers)
      result += 1;
  }
  return result;
}

void CSCSimHitMatcher::chamberIdsToString(const std::set<unsigned int>& set) const {
  for (const auto& p : set) {
    CSCDetId detId(p);
    edm::LogInfo("CSCSimHitMatcher") << " " << detId << "\n";
  }
}

std::set<unsigned int> CSCSimHitMatcher::chamberIdsStation(int station) const {
  set<unsigned int> result;
  switch (station) {
    case 1: {
      const auto& p1(chamberIds(MuonHitHelper::CSC_ME1a));
      const auto& p2(chamberIds(MuonHitHelper::CSC_ME1b));
      const auto& p3(chamberIds(MuonHitHelper::CSC_ME12));
      const auto& p4(chamberIds(MuonHitHelper::CSC_ME13));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      result.insert(p3.begin(), p3.end());
      result.insert(p4.begin(), p4.end());
      break;
    }
    case 2: {
      const auto& p1(chamberIds(MuonHitHelper::CSC_ME21));
      const auto& p2(chamberIds(MuonHitHelper::CSC_ME22));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      break;
    }
    case 3: {
      const auto& p1(chamberIds(MuonHitHelper::CSC_ME31));
      const auto& p2(chamberIds(MuonHitHelper::CSC_ME32));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      break;
    }
    case 4: {
      const auto& p1(chamberIds(MuonHitHelper::CSC_ME41));
      const auto& p2(chamberIds(MuonHitHelper::CSC_ME42));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      break;
    }
  };
  return result;
}

void CSCSimHitMatcher::clear() { MuonSimHitMatcher::clear(); }
