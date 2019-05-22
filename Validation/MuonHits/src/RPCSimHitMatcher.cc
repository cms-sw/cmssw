#include "Validation/MuonHits/interface/RPCSimHitMatcher.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

using namespace std;

RPCSimHitMatcher::RPCSimHitMatcher(const edm::ParameterSet& ps, edm::ConsumesCollector&& iC)
    : MuonSimHitMatcher(ps, std::move(iC)) {
  simHitPSet_ = ps.getParameterSet("rpcSimHit");
  verbose_ = simHitPSet_.getParameter<int>("verbose");
  simMuOnly_ = simHitPSet_.getParameter<bool>("simMuOnly");
  discardEleHits_ = simHitPSet_.getParameter<bool>("discardEleHits");

  simHitInput_ = iC.consumes<edm::PSimHitContainer>(simHitPSet_.getParameter<edm::InputTag>("inputTag"));
}

/// initialize the event
void RPCSimHitMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iSetup.get<MuonGeometryRecord>().get(rpc_geom_);
  if (rpc_geom_.isValid()) {
    geometry_ = &*rpc_geom_;
  } else {
    hasGeometry_ = false;
    edm::LogWarning("RPCSimHitMatcher") << "+++ Info: RPC geometry is unavailable. +++\n";
  }
  MuonSimHitMatcher::init(iEvent, iSetup);
}

/// do the matching
void RPCSimHitMatcher::match(const SimTrack& track, const SimVertex& vertex) {
  // instantiates the track ids and simhits
  MuonSimHitMatcher::match(track, vertex);

  if (hasGeometry_) {
    matchSimHitsToSimTrack();

    if (verbose_) {
      edm::LogInfo("RPCSimHitMatcher") << "nSimHits " << simHits_.size() << " nTrackIds " << track_ids_.size() << endl;
      edm::LogInfo("RPCSimHitMatcher") << "detids RPC " << detIds().size() << endl;

      const auto& ch_ids = chamberIds();
      for (const auto& id : ch_ids) {
        const auto& simhits = MuonSimHitMatcher::hitsInChamber(id);
        const auto& simhits_gp = simHitsMeanPosition(simhits);
        edm::LogInfo("RPCSimHitMatcher") << "RPCDetId " << RPCDetId(id) << ": nHits " << simhits.size() << " eta "
                                         << simhits_gp.eta() << " phi " << simhits_gp.phi() << " nCh "
                                         << chamber_to_hits_[id].size() << endl;
        const auto& strips = hitStripsInDetId(id);
        edm::LogInfo("RPCSimHitMatcher") << "nStrips " << strips.size() << endl;
        edm::LogInfo("RPCSimHitMatcher") << "strips : ";
        for (const auto& p : strips) {
          edm::LogInfo("RPCSimHitMatcher") << p;
        }
      }
    }
  }
}

void RPCSimHitMatcher::matchSimHitsToSimTrack() {
  for (const auto& track_id : track_ids_) {
    for (const auto& h : simHits_) {
      if (h.trackId() != track_id)
        continue;
      int pdgid = h.particleType();
      if (simMuOnly_ && std::abs(pdgid) != 13)
        continue;
      // discard electron hits in the RPC chambers
      if (discardEleHits_ && pdgid == 11)
        continue;

      const RPCDetId& layer_id(h.detUnitId());
      detid_to_hits_[h.detUnitId()].push_back(h);
      hits_.push_back(h);
      chamber_to_hits_[layer_id.chamberId().rawId()].push_back(h);
    }
  }
}

std::set<unsigned int> RPCSimHitMatcher::detIds(int type) const {
  std::set<unsigned int> result;
  for (const auto& p : detid_to_hits_) {
    const auto& id = p.first;
    if (type > 0) {
      RPCDetId detId(id);
      if (MuonHitHelper::toRPCType(detId.region(), detId.station(), detId.ring()) != type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

std::set<unsigned int> RPCSimHitMatcher::chamberIds(int type) const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_hits_) {
    const auto& id = p.first;
    if (type > 0) {
      RPCDetId detId(id);
      if (MuonHitHelper::toRPCType(detId.region(), detId.station(), detId.ring()) != type)
        continue;
    }
    result.insert(id);
  }
  return result;
}

bool RPCSimHitMatcher::hitStation(int st) const {
  int nst = 0;
  for (const auto& ddt : chamberIds(0)) {
    const RPCDetId id(ddt);
    if (id.station() != st)
      continue;
    ++nst;
  }
  return nst;
}

int RPCSimHitMatcher::nStations() const { return (hitStation(1) + hitStation(2) + hitStation(3) + hitStation(4)); }

float RPCSimHitMatcher::simHitsMeanStrip(const edm::PSimHitContainer& sim_hits) const {
  if (sim_hits.empty())
    return -1.f;

  float sums = 0.f;
  size_t n = 0;
  for (const auto& h : sim_hits) {
    const LocalPoint& lp = h.entryPoint();
    const auto& d = h.detUnitId();
    sums += dynamic_cast<const RPCGeometry*>(geometry_)->roll(d)->strip(lp);
    ++n;
  }
  if (n == 0)
    return -1.f;
  return sums / n;
}

std::set<int> RPCSimHitMatcher::hitStripsInDetId(unsigned int detid, int margin_n_strips) const {
  set<int> result;
  RPCDetId id(detid);
  for (const auto& roll : dynamic_cast<const RPCGeometry*>(geometry_)->chamber(id)->rolls()) {
    int max_nstrips = roll->nstrips();
    for (const auto& h : MuonSimHitMatcher::hitsInDetId(roll->id().rawId())) {
      const LocalPoint& lp = h.entryPoint();
      int central_strip = static_cast<int>(roll->topology().channel(lp));
      int smin = central_strip - margin_n_strips;
      smin = (smin > 0) ? smin : 1;
      int smax = central_strip + margin_n_strips;
      smax = (smax <= max_nstrips) ? smax : max_nstrips;
      for (int ss = smin; ss <= smax; ++ss)
        result.insert(ss);
    }
  }
  return result;
}
