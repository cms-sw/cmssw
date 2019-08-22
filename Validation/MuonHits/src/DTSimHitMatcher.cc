#include "Validation/MuonHits/interface/DTSimHitMatcher.h"

using namespace std;

DTSimHitMatcher::DTSimHitMatcher(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC)
    : MuonSimHitMatcher(ps, std::move(iC)) {
  simHitPSet_ = ps.getParameterSet("dtSimHit");
  verbose_ = simHitPSet_.getParameter<int>("verbose");
  simMuOnly_ = simHitPSet_.getParameter<bool>("simMuOnly");
  discardEleHits_ = simHitPSet_.getParameter<bool>("discardEleHits");

  simHitInput_ = iC.consumes<edm::PSimHitContainer>(simHitPSet_.getParameter<edm::InputTag>("inputTag"));
}

/// initialize the event
void DTSimHitMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iSetup.get<MuonGeometryRecord>().get(dt_geom_);
  if (dt_geom_.isValid()) {
    geometry_ = &*dt_geom_;
  } else {
    hasGeometry_ = false;
    edm::LogWarning("DTSimHitMatcher") << "+++ Info: DT geometry is unavailable. +++\n";
  }
  MuonSimHitMatcher::init(iEvent, iSetup);
}

/// do the matching
void DTSimHitMatcher::match(const SimTrack& track, const SimVertex& vertex) {
  // instantiates the track ids and simhits
  MuonSimHitMatcher::match(track, vertex);

  if (hasGeometry_) {
    matchSimHitsToSimTrack();

    if (verbose_) {
      edm::LogInfo("DTSimHitMatcher") << "nTrackIds " << track_ids_.size() << " nSelectedDTSimHits " << hits_.size()
                                      << endl;
      edm::LogInfo("DTSimHitMatcher") << "detids DT " << detIds(0).size() << endl;

      const auto& dt_det_ids = detIds(0);
      for (const auto& id : dt_det_ids) {
        const auto& dt_simhits = MuonSimHitMatcher::hitsInDetId(id);
        const auto& dt_simhits_gp = simHitsMeanPosition(dt_simhits);
        edm::LogInfo("DTSimHitMatcher") << "DTWireId " << DTWireId(id) << ": nHits " << dt_simhits.size() << " eta "
                                        << dt_simhits_gp.eta() << " phi " << dt_simhits_gp.phi() << " nCh "
                                        << chamber_to_hits_[id].size() << endl;
      }
    }
  }
}

void DTSimHitMatcher::matchSimHitsToSimTrack() {
  for (const auto& track_id : track_ids_) {
    for (const auto& h : simHits_) {
      if (h.trackId() != track_id)
        continue;
      int pdgid = h.particleType();
      if (simMuOnly_ && std::abs(pdgid) != 13)
        continue;
      // discard electron hits in the DT chambers
      if (discardEleHits_ && pdgid == 11)
        continue;

      const DTWireId layer_id(h.detUnitId());
      detid_to_hits_[h.detUnitId()].push_back(h);
      hits_.push_back(h);
      layer_to_hits_[layer_id.layerId().rawId()].push_back(h);
      superlayer_to_hits_[layer_id.superlayerId().rawId()].push_back(h);
      chamber_to_hits_[layer_id.chamberId().rawId()].push_back(h);
    }
  }
}

std::set<unsigned int> DTSimHitMatcher::detIds(int dt_type) const {
  std::set<unsigned int> result;
  for (const auto& p : detid_to_hits_) {
    const auto& id = p.first;
    if (dt_type > 0) {
      DTWireId detId(id);
      if (MuonHitHelper::toDTType(detId.wheel(), detId.station()) != dt_type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

std::set<unsigned int> DTSimHitMatcher::chamberIds(int dt_type) const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_hits_) {
    const auto& id = p.first;
    if (dt_type > 0) {
      DTChamberId detId(id);
      if (MuonHitHelper::toDTType(detId.wheel(), detId.station()) != dt_type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

std::set<unsigned int> DTSimHitMatcher::layerIds() const {
  std::set<unsigned int> result;
  for (const auto& p : layer_to_hits_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> DTSimHitMatcher::superlayerIds() const {
  std::set<unsigned int> result;
  for (const auto& p : superlayer_to_hits_)
    result.insert(p.first);
  return result;
}

const edm::PSimHitContainer& DTSimHitMatcher::hitsInLayer(unsigned int detid) const {
  if (!MuonHitHelper::isDT(detid))
    return no_hits_;

  const DTWireId id(detid);
  if (layer_to_hits_.find(id.layerId().rawId()) == layer_to_hits_.end())
    return no_hits_;
  return layer_to_hits_.at(id.layerId().rawId());
}

const edm::PSimHitContainer& DTSimHitMatcher::hitsInSuperLayer(unsigned int detid) const {
  if (!MuonHitHelper::isDT(detid))
    return no_hits_;

  const DTWireId id(detid);
  if (superlayer_to_hits_.find(id.superlayerId().rawId()) == superlayer_to_hits_.end())
    return no_hits_;
  return superlayer_to_hits_.at(id.superlayerId().rawId());
}

const edm::PSimHitContainer& DTSimHitMatcher::hitsInChamber(unsigned int detid) const {
  if (!MuonHitHelper::isDT(detid))
    return no_hits_;

  const DTWireId id(detid);
  if (chamber_to_hits_.find(id.chamberId().rawId()) == chamber_to_hits_.end())
    return no_hits_;
  return chamber_to_hits_.at(id.chamberId().rawId());
}

bool DTSimHitMatcher::hitStation(int st, int nsuperlayers, int nlayers) const {
  int nst = 0;
  for (const auto& ddt : chamberIds()) {
    const DTChamberId id(ddt);
    if (id.station() != st)
      continue;

    // require at least 1 superlayer
    const int nsl(nSuperLayersWithHitsInChamber(id.rawId()));
    if (nsl < nsuperlayers)
      continue;

    // require at least 3 layers hit per chamber
    const int nl(nLayersWithHitsInChamber(id.rawId()));
    if (nl < nlayers)
      continue;
    ++nst;
  }
  return nst;
}

int DTSimHitMatcher::nStations(int nsuperlayers, int nlayers) const {
  return (hitStation(1, nsuperlayers, nlayers) + hitStation(2, nsuperlayers, nlayers) +
          hitStation(3, nsuperlayers, nlayers) + hitStation(4, nsuperlayers, nlayers));
}

int DTSimHitMatcher::nCellsWithHitsInLayer(unsigned int detid) const {
  set<int> layers_with_hits;
  const auto& hits = hitsInLayer(detid);
  for (const auto& h : hits) {
    if (MuonHitHelper::isDT(detid)) {
      const DTWireId idd(h.detUnitId());
      layers_with_hits.insert(idd.wire());
    }
  }
  return layers_with_hits.size();
}

int DTSimHitMatcher::nLayersWithHitsInSuperLayer(unsigned int detid) const {
  set<int> layers_with_hits;
  const auto& hits = hitsInSuperLayer(detid);
  for (const auto& h : hits) {
    if (MuonHitHelper::isDT(detid)) {
      const DTLayerId idd(h.detUnitId());
      layers_with_hits.insert(idd.layer());
    }
  }
  return layers_with_hits.size();
}

int DTSimHitMatcher::nSuperLayersWithHitsInChamber(unsigned int detid) const {
  set<int> sl_with_hits;
  const auto& hits = MuonSimHitMatcher::hitsInChamber(detid);
  for (const auto& h : hits) {
    if (MuonHitHelper::isDT(detid)) {
      const DTSuperLayerId idd(h.detUnitId());
      sl_with_hits.insert(idd.superLayer());
    }
  }
  return sl_with_hits.size();
}

int DTSimHitMatcher::nLayersWithHitsInChamber(unsigned int detid) const {
  int nLayers = 0;
  const auto& superLayers(dynamic_cast<const DTGeometry*>(geometry_)->chamber(DTChamberId(detid))->superLayers());
  for (const auto& sl : superLayers) {
    nLayers += nLayersWithHitsInSuperLayer(sl->id().rawId());
  }
  return nLayers;
}
float DTSimHitMatcher::simHitsMeanWire(const edm::PSimHitContainer& sim_hits) const {
  if (sim_hits.empty())
    return -1.f;

  float sums = 0.f;
  size_t n = 0;
  for (const auto& h : sim_hits) {
    const LocalPoint& lp = h.entryPoint();
    float s;
    const auto& d = h.detUnitId();
    if (MuonHitHelper::isDT(d)) {
      // find nearest wire
      s = dynamic_cast<const DTGeometry*>(geometry_)->layer(DTLayerId(d))->specificTopology().channel(lp);
    } else
      continue;
    sums += s;
    ++n;
  }
  if (n == 0)
    return -1.f;
  return sums / n;
}

std::set<unsigned int> DTSimHitMatcher::hitWiresInDTLayerId(unsigned int detid, int margin_n_wires) const {
  set<unsigned int> result;
  if (MuonHitHelper::isDT(detid)) {
    DTLayerId id(detid);
    int max_nwires = dynamic_cast<const DTGeometry*>(geometry_)->layer(id)->specificTopology().channels();
    for (int wn = 0; wn <= max_nwires; ++wn) {
      DTWireId wid(id, wn);
      for (const auto& h : MuonSimHitMatcher::hitsInDetId(wid.rawId())) {
        if (verbose_)
          edm::LogInfo("DTSimHitMatcher") << "central DTWireId " << wid << " simhit " << h << endl;
        int smin = wn - margin_n_wires;
        smin = (smin > 0) ? smin : 1;
        int smax = wn + margin_n_wires;
        smax = (smax <= max_nwires) ? smax : max_nwires;
        for (int ss = smin; ss <= smax; ++ss) {
          DTWireId widd(id, ss);
          if (verbose_)
            edm::LogInfo("DTSimHitMatcher") << "\tadding DTWireId to collection " << widd << endl;
          result.insert(widd.rawId());
        }
      }
    }
  }
  return result;
}

std::set<unsigned int> DTSimHitMatcher::hitWiresInDTSuperLayerId(unsigned int detid, int margin_n_wires) const {
  set<unsigned int> result;
  const auto& layers(dynamic_cast<const DTGeometry*>(geometry_)->superLayer(DTSuperLayerId(detid))->layers());
  for (const auto& l : layers) {
    if (verbose_)
      edm::LogInfo("DTSimHitMatcher") << "hitWiresInDTSuperLayerId::l id " << l->id() << endl;
    const auto& p(hitWiresInDTLayerId(l->id().rawId(), margin_n_wires));
    result.insert(p.begin(), p.end());
  }
  return result;
}

std::set<unsigned int> DTSimHitMatcher::hitWiresInDTChamberId(unsigned int detid, int margin_n_wires) const {
  set<unsigned int> result;
  const auto& superLayers(dynamic_cast<const DTGeometry*>(geometry_)->chamber(DTChamberId(detid))->superLayers());
  for (const auto& sl : superLayers) {
    if (verbose_)
      edm::LogInfo("DTSimHitMatcher") << "hitWiresInDTChamberId::sl id " << sl->id() << endl;
    const auto& p(hitWiresInDTSuperLayerId(sl->id().rawId(), margin_n_wires));
    result.insert(p.begin(), p.end());
  }
  return result;
}

void DTSimHitMatcher::dtChamberIdsToString(const std::set<unsigned int>& set) const {
  for (const auto& p : set) {
    DTChamberId detId(p);
    edm::LogInfo("DTSimHitMatcher") << " " << detId << "\n";
  }
}

std::set<unsigned int> DTSimHitMatcher::chamberIdsStation(int station) const {
  set<unsigned int> result;
  switch (station) {
    case 1: {
      const auto& p1(chamberIds(MuonHitHelper::DT_MB21p));
      const auto& p2(chamberIds(MuonHitHelper::DT_MB11p));
      const auto& p3(chamberIds(MuonHitHelper::DT_MB01));
      const auto& p4(chamberIds(MuonHitHelper::DT_MB11n));
      const auto& p5(chamberIds(MuonHitHelper::DT_MB21n));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      result.insert(p3.begin(), p3.end());
      result.insert(p4.begin(), p4.end());
      result.insert(p5.begin(), p5.end());
      break;
    }
    case 2: {
      const auto& p1(chamberIds(MuonHitHelper::DT_MB22p));
      const auto& p2(chamberIds(MuonHitHelper::DT_MB12p));
      const auto& p3(chamberIds(MuonHitHelper::DT_MB02));
      const auto& p4(chamberIds(MuonHitHelper::DT_MB12n));
      const auto& p5(chamberIds(MuonHitHelper::DT_MB22n));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      result.insert(p3.begin(), p3.end());
      result.insert(p4.begin(), p4.end());
      result.insert(p5.begin(), p5.end());
      break;
    }
    case 3: {
      const auto& p1(chamberIds(MuonHitHelper::DT_MB23p));
      const auto& p2(chamberIds(MuonHitHelper::DT_MB13p));
      const auto& p3(chamberIds(MuonHitHelper::DT_MB03));
      const auto& p4(chamberIds(MuonHitHelper::DT_MB13n));
      const auto& p5(chamberIds(MuonHitHelper::DT_MB23n));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      result.insert(p3.begin(), p3.end());
      result.insert(p4.begin(), p4.end());
      result.insert(p5.begin(), p5.end());
      break;
    }
    case 4: {
      const auto& p1(chamberIds(MuonHitHelper::DT_MB24p));
      const auto& p2(chamberIds(MuonHitHelper::DT_MB14p));
      const auto& p3(chamberIds(MuonHitHelper::DT_MB04));
      const auto& p4(chamberIds(MuonHitHelper::DT_MB14n));
      const auto& p5(chamberIds(MuonHitHelper::DT_MB24n));
      result.insert(p1.begin(), p1.end());
      result.insert(p2.begin(), p2.end());
      result.insert(p3.begin(), p3.end());
      result.insert(p4.begin(), p4.end());
      result.insert(p5.begin(), p5.end());
      break;
    }
  };
  return result;
}
