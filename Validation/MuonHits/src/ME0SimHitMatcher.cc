#include "Validation/MuonHits/interface/ME0SimHitMatcher.h"

using namespace std;

ME0SimHitMatcher::ME0SimHitMatcher(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC)
    : MuonSimHitMatcher(ps, std::move(iC)) {
  simHitPSet_ = ps.getParameterSet("me0SimHit");
  verbose_ = simHitPSet_.getParameter<int>("verbose");
  simMuOnly_ = simHitPSet_.getParameter<bool>("simMuOnly");
  discardEleHits_ = simHitPSet_.getParameter<bool>("discardEleHits");

  simHitInput_ = iC.consumes<edm::PSimHitContainer>(simHitPSet_.getParameter<edm::InputTag>("inputTag"));
}

/// initialize the event
void ME0SimHitMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iSetup.get<MuonGeometryRecord>().get(me0_geom_);
  if (me0_geom_.isValid()) {
    geometry_ = &*me0_geom_;
  } else {
    hasGeometry_ = false;
    edm::LogWarning("ME0SimHitMatcher") << "+++ Info: ME0 geometry is unavailable. +++\n";
  }
  MuonSimHitMatcher::init(iEvent, iSetup);
}

/// do the matching
void ME0SimHitMatcher::match(const SimTrack& track, const SimVertex& vertex) {
  // instantiates the track ids and simhits
  MuonSimHitMatcher::match(track, vertex);

  if (hasGeometry_) {
    matchSimHitsToSimTrack();

    if (verbose_) {
      edm::LogInfo("ME0SimHitMatcher") << "nTrackIds " << track_ids_.size() << " nSelectedME0SimHits " << hits_.size()
                                       << endl;
      edm::LogInfo("ME0SimHitMatcher") << "detids ME0 " << detIds().size() << endl;

      const auto& me0_ch_ids = detIds();
      for (const auto& id : me0_ch_ids) {
        const auto& me0_simhits = MuonSimHitMatcher::hitsInChamber(id);
        const auto& me0_simhits_gp = simHitsMeanPosition(me0_simhits);
        edm::LogInfo("ME0SimHitMatcher") << "me0chid " << ME0DetId(id) << ": nHits " << me0_simhits.size() << " phi "
                                         << me0_simhits_gp.phi() << " nCh " << chamber_to_hits_[id].size() << endl;
        const auto& strips = hitStripsInDetId(id);
        edm::LogInfo("ME0SimHitMatcher") << "nStrip " << strips.size() << endl;
        edm::LogInfo("ME0SimHitMatcher") << "strips : ";
        for (const auto& p : strips) {
          edm::LogInfo("ME0SimHitMatcher") << p;
        }
      }
    }
  }
}

void ME0SimHitMatcher::matchSimHitsToSimTrack() {
  for (const auto& track_id : track_ids_) {
    for (const auto& h : simHits_) {
      if (h.trackId() != track_id)
        continue;
      int pdgid = h.particleType();
      if (simMuOnly_ && std::abs(pdgid) != 13)
        continue;
      // discard electron hits in the ME0 chambers
      if (discardEleHits_ && std::abs(pdgid) == 11)
        continue;

      const ME0DetId& layer_id(h.detUnitId());
      detid_to_hits_[h.detUnitId()].push_back(h);
      hits_.push_back(h);
      chamber_to_hits_[layer_id.layerId().rawId()].push_back(h);
      superChamber_to_hits_[layer_id.chamberId().rawId()].push_back(h);
    }
  }

  // find pads with hits
  const auto& detids = detIds();
  for (const auto& d : detids) {
    ME0DetId id(d);
    const auto& hits = hitsInDetId(d);
    const auto& roll = dynamic_cast<const ME0Geometry*>(geometry_)->etaPartition(id);
    // int max_npads = roll->npads();
    set<int> pads;
    for (const auto& h : hits) {
      const LocalPoint& lp = h.entryPoint();
      pads.insert(1 + static_cast<int>(roll->padTopology().channel(lp)));
    }
    detids_to_pads_[d] = pads;
  }
}

std::set<unsigned int> ME0SimHitMatcher::detIds() const {
  std::set<unsigned int> result;
  for (const auto& p : detid_to_hits_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> ME0SimHitMatcher::chamberIds() const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_hits_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> ME0SimHitMatcher::superChamberIds() const {
  std::set<unsigned int> result;
  for (const auto& p : superChamber_to_hits_)
    result.insert(p.first);
  return result;
}

const edm::PSimHitContainer& ME0SimHitMatcher::hitsInSuperChamber(unsigned int detid) const {
  if (superChamber_to_hits_.find(detid) == superChamber_to_hits_.end())
    return no_hits_;
  return superChamber_to_hits_.at(detid);
}

int ME0SimHitMatcher::nLayersWithHitsInSuperChamber(unsigned int detid) const {
  set<int> layers_with_hits;
  const auto& hits = hitsInSuperChamber(detid);
  for (const auto& h : hits) {
    const ME0DetId& idd(h.detUnitId());
    layers_with_hits.insert(idd.layer());
  }
  return layers_with_hits.size();
}

std::set<unsigned int> ME0SimHitMatcher::superChamberIdsCoincidences(int min_n_layers) const {
  set<unsigned int> result;
  // loop over the super chamber Ids
  for (const auto& p : superChamberIds()) {
    if (nLayersWithHitsInSuperChamber(p) >= min_n_layers) {
      result.insert(p);
    }
  }
  return result;
}

int ME0SimHitMatcher::nCoincidenceChambers(int min_n_layers) const {
  return superChamberIdsCoincidences(min_n_layers).size();
}

float ME0SimHitMatcher::simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const {
  if (sim_hits.empty())
    return -1.f;

  float sums = 0.f;
  size_t n = 0;
  for (const auto& h : sim_hits) {
    const LocalPoint& lp = h.entryPoint();
    float s;
    const auto& d = h.detUnitId();
    s = dynamic_cast<const ME0Geometry*>(geometry_)->etaPartition(d)->strip(lp);
    sums += s;
    ++n;
  }
  if (n == 0)
    return -1.f;
  return sums / n;
}

std::set<int> ME0SimHitMatcher::hitStripsInDetId(unsigned int detid, int margin_n_strips) const {
  set<int> result;
  const auto& simhits = hitsInDetId(detid);
  ME0DetId id(detid);
  int max_nstrips = dynamic_cast<const ME0Geometry*>(geometry_)->etaPartition(id)->nstrips();
  for (const auto& h : simhits) {
    const LocalPoint& lp = h.entryPoint();
    int central_strip =
        1 + static_cast<int>(dynamic_cast<const ME0Geometry*>(geometry_)->etaPartition(id)->topology().channel(lp));
    int smin = central_strip - margin_n_strips;
    smin = (smin > 0) ? smin : 1;
    int smax = central_strip + margin_n_strips;
    smax = (smax <= max_nstrips) ? smax : max_nstrips;
    for (int ss = smin; ss <= smax; ++ss)
      result.insert(ss);
  }
  return result;
}

std::set<int> ME0SimHitMatcher::hitPadsInDetId(unsigned int detid) const {
  set<int> none;
  if (detids_to_pads_.find(detid) == detids_to_pads_.end())
    return none;
  return detids_to_pads_.at(detid);
}

std::set<int> ME0SimHitMatcher::hitPartitions() const {
  std::set<int> result;

  const auto& detids = detIds();
  for (const auto& id : detids) {
    ME0DetId idd(id);
    result.insert(idd.roll());
  }
  return result;
}

int ME0SimHitMatcher::nPadsWithHits() const {
  int result = 0;
  const auto& pad_ids = detIds();
  for (const auto& id : pad_ids) {
    result += hitPadsInDetId(id).size();
  }
  return result;
}
