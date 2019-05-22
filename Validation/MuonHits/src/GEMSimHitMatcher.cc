#include "Validation/MuonHits/interface/GEMSimHitMatcher.h"

using namespace std;

GEMSimHitMatcher::GEMSimHitMatcher(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC)
    : MuonSimHitMatcher(ps, std::move(iC)) {
  simHitPSet_ = ps.getParameterSet("gemSimHit");
  verbose_ = simHitPSet_.getParameter<int>("verbose");
  simMuOnly_ = simHitPSet_.getParameter<bool>("simMuOnly");
  discardEleHits_ = simHitPSet_.getParameter<bool>("discardEleHits");

  simHitInput_ = iC.consumes<edm::PSimHitContainer>(simHitPSet_.getParameter<edm::InputTag>("inputTag"));
}

/// initialize the event
void GEMSimHitMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iSetup.get<MuonGeometryRecord>().get(gem_geom_);
  if (gem_geom_.isValid()) {
    geometry_ = dynamic_cast<const GEMGeometry*>(&*gem_geom_);
  } else {
    hasGeometry_ = false;
    edm::LogWarning("GEMSimHitMatcher") << "+++ Info: GEM geometry is unavailable. +++\n";
  }
  MuonSimHitMatcher::init(iEvent, iSetup);
}

/// do the matching
void GEMSimHitMatcher::match(const SimTrack& track, const SimVertex& vertex) {
  // instantiates the track ids and simhits
  MuonSimHitMatcher::match(track, vertex);

  if (hasGeometry_) {
    matchSimHitsToSimTrack();

    if (verbose_) {
      edm::LogInfo("GEMSimHitMatcher") << "nTrackIds " << track_ids_.size() << " nSelectedGEMSimHits " << hits_.size()
                                       << endl;
      edm::LogInfo("GEMSimHitMatcher") << "detids GEM " << detIds(0).size() << endl;

      const auto& gem_ch_ids = detIds();
      for (const auto& id : gem_ch_ids) {
        const auto& gem_simhits = MuonSimHitMatcher::hitsInDetId(id);
        const auto& gem_simhits_gp = simHitsMeanPosition(gem_simhits);
        edm::LogInfo("GEMSimHitMatcher") << "gemchid " << GEMDetId(id) << ": nHits " << gem_simhits.size() << " phi "
                                         << gem_simhits_gp.phi() << " nCh " << chamber_to_hits_[id].size() << endl;
        const auto& strips = hitStripsInDetId(id);
        edm::LogInfo("GEMSimHitMatcher") << "nStrip " << strips.size() << endl;
        edm::LogInfo("GEMSimHitMatcher") << "strips : ";
        for (const auto& p : strips) {
          edm::LogInfo("GEMSimHitMatcher") << p;
        }
      }
      const auto& gem_sch_ids = superChamberIds();
      for (const auto& id : gem_sch_ids) {
        const auto& gem_simhits = hitsInSuperChamber(id);
        const auto& gem_simhits_gp = simHitsMeanPosition(gem_simhits);
        edm::LogInfo("GEMSimHitMatcher") << "gemschid " << GEMDetId(id) << ": " << nCoincidencePadsWithHits() << " | "
                                         << gem_simhits.size() << " " << gem_simhits_gp.phi() << " "
                                         << superchamber_to_hits_[id].size() << endl;
      }
    }
  }
}

void GEMSimHitMatcher::matchSimHitsToSimTrack() {
  for (const auto& track_id : track_ids_) {
    for (const auto& h : simHits_) {
      if (h.trackId() != track_id)
        continue;
      int pdgid = h.particleType();
      if (simMuOnly_ && std::abs(pdgid) != 13)
        continue;
      // discard electron hits in the GEM chambers
      if (discardEleHits_ && pdgid == 11)
        continue;

      const GEMDetId& p_id(h.detUnitId());
      detid_to_hits_[h.detUnitId()].push_back(h);
      hits_.push_back(h);
      chamber_to_hits_[p_id.chamberId().rawId()].push_back(h);
      superchamber_to_hits_[p_id.superChamberId().rawId()].push_back(h);
    }
  }

  // find pads with hits
  const auto& detids = detIds();
  // find 2-layer coincidence pads with hits
  for (const auto& d : detids) {
    GEMDetId id(d);
    const auto& hits = hitsInDetId(d);
    const auto& roll = dynamic_cast<const GEMGeometry*>(geometry_)->etaPartition(id);
    // int max_npads = roll->npads();
    set<int> pads;
    for (const auto& h : hits) {
      const LocalPoint& lp = h.entryPoint();
      pads.insert(1 + static_cast<int>(roll->padTopology().channel(lp)));
    }
    detids_to_pads_[d] = pads;
  }

  // find 2-layer coincidence pads with hits
  for (const auto& d : detids) {
    GEMDetId id1(d);
    if (id1.layer() != 1)
      continue;

    // find pads with hits in layer1
    const auto& hits1 = hitsInDetId(d);
    const auto& roll1 = dynamic_cast<const GEMGeometry*>(geometry_)->etaPartition(id1);
    set<int> pads1;
    set<int> pads2;
    set<int> copads;

    for (const auto& h : hits1) {
      const LocalPoint& lp = h.entryPoint();
      pads1.insert(1 + static_cast<int>(roll1->padTopology().channel(lp)));
      if (verbose_)
        edm::LogInfo("GEMSimHitMatcher") << "GEMHits detid1 " << id1 << " pad1 "
                                         << 1 + static_cast<int>(roll1->padTopology().channel(lp)) << std::endl;
    }

    // find pads with hits in layer2
    for (const auto& d2 : detids) {
      // staggered geometry???? improve here !!
      GEMDetId id2(d2);
      // does layer 2 has simhits?
      if (id2.layer() != 2 or id2.region() != id1.region() or id2.ring() != id1.ring() or
          id2.station() != id1.station() or abs(id2.roll() - id1.roll()) > 1)
        continue;
      const auto& hits2 = hitsInDetId(id2());
      const auto& roll2 = dynamic_cast<const GEMGeometry*>(geometry_)->etaPartition(id2);
      for (const auto& h : hits2) {
        const LocalPoint& lp = h.entryPoint();
        pads2.insert(1 + static_cast<int>(roll2->padTopology().channel(lp)));
        if (verbose_)
          edm::LogInfo("GEMSimHitMatcher") << "GEMHits detid2 " << id2 << " pad2 "
                                           << 1 + static_cast<int>(roll2->padTopology().channel(lp)) << std::endl;
      }
    }

    for (const auto& pad1 : pads1) {
      for (const auto& pad2 : pads2) {
        if (abs(pad1 - pad2) <= 2) {
          if (copads.find(pad1) == copads.end())
            copads.insert(pad1);
          if (copads.find(pad2) == copads.end())
            copads.insert(pad2);
        }
      }
    }

    if (copads.empty())
      continue;

    // detids here is layer1 id
    detids_to_copads_[d] = copads;
  }
}

std::set<unsigned int> GEMSimHitMatcher::detIds(int gem_type) const {
  std::set<unsigned int> result;
  for (const auto& p : detid_to_hits_) {
    const auto& id = p.first;
    if (gem_type > 0) {
      GEMDetId detId(id);
      if (MuonHitHelper::toGEMType(detId.station(), detId.ring()) != gem_type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

std::set<unsigned int> GEMSimHitMatcher::detIdsCoincidences() const {
  std::set<unsigned int> result;
  for (const auto& p : detids_to_copads_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> GEMSimHitMatcher::chamberIds(int gem_type) const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_hits_) {
    const auto& id = p.first;
    if (gem_type > 0) {
      GEMDetId detId(id);
      if (MuonHitHelper::toGEMType(detId.station(), detId.ring()) != gem_type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

std::set<unsigned int> GEMSimHitMatcher::superChamberIds() const {
  std::set<unsigned int> result;
  for (const auto& p : superchamber_to_hits_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> GEMSimHitMatcher::superChamberIdsCoincidences() const {
  std::set<unsigned int> result;
  for (const auto& p : detids_to_copads_) {
    const GEMDetId& p_id(p.first);
    result.insert(p_id.superChamberId().rawId());
  }
  return result;
}

const edm::PSimHitContainer& GEMSimHitMatcher::hitsInSuperChamber(unsigned int detid) const {
  if (MuonHitHelper::isGEM(detid)) {
    const GEMDetId id(detid);
    if (superchamber_to_hits_.find(id.chamberId().rawId()) == superchamber_to_hits_.end())
      return no_hits_;
    return superchamber_to_hits_.at(id.chamberId().rawId());
  }

  return no_hits_;
}

int GEMSimHitMatcher::nLayersWithHitsInSuperChamber(unsigned int detid) const {
  set<int> layers_with_hits;
  const auto& hits = hitsInSuperChamber(detid);
  for (const auto& h : hits) {
    const GEMDetId& idd(h.detUnitId());
    layers_with_hits.insert(idd.layer());
  }
  return layers_with_hits.size();
}

bool GEMSimHitMatcher::hitStation(int st, int nlayers) const {
  int nst = 0;
  for (const auto& ddt : chamberIds()) {
    const GEMDetId id(ddt);
    if (id.station() != st)
      continue;

    const int nl(nLayersWithHitsInSuperChamber(id.rawId()));
    if (nl < nlayers)
      continue;
    ++nst;
  }
  return nst;
}

int GEMSimHitMatcher::nStations(int nlayers) const { return (hitStation(1, nlayers) + hitStation(2, nlayers)); }

float GEMSimHitMatcher::simHitsGEMCentralPosition(const edm::PSimHitContainer& sim_hits) const {
  if (sim_hits.empty())
    return -0.0;  // point "zero"

  float central = -0.0;
  size_t n = 0;
  for (const auto& h : sim_hits) {
    LocalPoint lp(0., 0., 0.);  // local central
    GlobalPoint gp;
    if (MuonHitHelper::isGEM(h.detUnitId())) {
      gp = geometry_->idToDet(h.detUnitId())->surface().toGlobal(lp);
    }
    central = gp.perp();
    if (n >= 1)
      edm::LogWarning("GEMSimHitMatcher") << "warning! find more than one simhits in GEM chamber " << std::endl;
    ++n;
  }

  return central;
}

float GEMSimHitMatcher::simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const {
  if (sim_hits.empty())
    return -1.f;

  float sums = 0.f;
  size_t n = 0;
  for (const auto& h : sim_hits) {
    const LocalPoint& lp = h.entryPoint();
    const auto& d = h.detUnitId();
    sums += dynamic_cast<const GEMGeometry*>(geometry_)->etaPartition(d)->strip(lp);
    ++n;
  }
  if (n == 0)
    return -1.f;
  return sums / n;
}

std::set<int> GEMSimHitMatcher::hitStripsInDetId(unsigned int detid, int margin_n_strips) const {
  set<int> result;
  const auto& simhits = MuonSimHitMatcher::hitsInDetId(detid);
  GEMDetId id(detid);
  int max_nstrips = dynamic_cast<const GEMGeometry*>(geometry_)->etaPartition(id)->nstrips();
  for (const auto& h : simhits) {
    const LocalPoint& lp = h.entryPoint();
    int central_strip =
        static_cast<int>(dynamic_cast<const GEMGeometry*>(geometry_)->etaPartition(id)->topology().channel(lp));
    int smin = central_strip - margin_n_strips;
    smin = (smin > 0) ? smin : 1;
    int smax = central_strip + margin_n_strips;
    smax = (smax <= max_nstrips) ? smax : max_nstrips;
    for (int ss = smin; ss <= smax; ++ss)
      result.insert(ss);
  }
  return result;
}

std::set<int> GEMSimHitMatcher::hitPadsInDetId(unsigned int detid) const {
  set<int> none;
  if (detids_to_pads_.find(detid) == detids_to_pads_.end())
    return none;
  return detids_to_pads_.at(detid);
}

std::set<int> GEMSimHitMatcher::hitCoPadsInDetId(unsigned int detid) const {
  set<int> none;
  if (detids_to_copads_.find(detid) == detids_to_copads_.end())
    return none;
  return detids_to_copads_.at(detid);
}

std::set<int> GEMSimHitMatcher::hitPartitions() const {
  std::set<int> result;

  const auto& detids = detIds();
  for (const auto& id : detids) {
    GEMDetId idd(id);
    result.insert(idd.roll());
  }
  return result;
}

int GEMSimHitMatcher::nPadsWithHits() const {
  int result = 0;
  const auto& pad_ids = detIds();
  for (const auto& id : pad_ids) {
    result += hitPadsInDetId(id).size();
  }
  return result;
}

int GEMSimHitMatcher::nCoincidencePadsWithHits() const {
  int result = 0;
  const auto& copad_ids = detIdsCoincidences();
  for (const auto& id : copad_ids) {
    result += hitCoPadsInDetId(id).size();
  }
  return result;
}
