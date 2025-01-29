#include <memory>

#include "Validation/MuonGEMDigis/interface/GEMDigiMatcher.h"

using namespace std;

GEMDigiMatcher::GEMDigiMatcher(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
  const auto& gemSimLink = pset.getParameterSet("gemSimLink");
  simMuOnly_ = gemSimLink.getParameter<bool>("simMuOnly");
  discardEleHits_ = gemSimLink.getParameter<bool>("discardEleHits");
  verboseSimLink_ = gemSimLink.getParameter<int>("verbose");

  const auto& gemDigi = pset.getParameterSet("gemStripDigi");
  minBXDigi_ = gemDigi.getParameter<int>("minBX");
  maxBXDigi_ = gemDigi.getParameter<int>("maxBX");
  matchDeltaStrip_ = gemDigi.getParameter<int>("matchDeltaStrip");
  verboseDigi_ = gemDigi.getParameter<int>("verbose");
  matchToSimLink_ = gemDigi.getParameter<bool>("matchToSimLink");

  const auto& gemPad = pset.getParameterSet("gemPadDigi");
  minBXPad_ = gemPad.getParameter<int>("minBX");
  maxBXPad_ = gemPad.getParameter<int>("maxBX");
  verbosePad_ = gemPad.getParameter<int>("verbose");

  const auto& gemCluster = pset.getParameterSet("gemPadCluster");
  minBXCluster_ = gemCluster.getParameter<int>("minBX");
  maxBXCluster_ = gemCluster.getParameter<int>("maxBX");
  verboseCluster_ = gemCluster.getParameter<int>("verbose");

  const auto& gemCoPad = pset.getParameterSet("gemCoPadDigi");
  minBXCoPad_ = gemCoPad.getParameter<int>("minBX");
  maxBXCoPad_ = gemCoPad.getParameter<int>("maxBX");
  verboseCoPad_ = gemCoPad.getParameter<int>("verbose");

  // make a new simhits matcher
  muonSimHitMatcher_ = std::make_shared<GEMSimHitMatcher>(pset, std::move(iC));

  if (matchToSimLink_)
    gemSimLinkToken_ =
        iC.consumes<edm::DetSetVector<GEMDigiSimLink>>(gemSimLink.getParameter<edm::InputTag>("inputTag"));
  gemDigiToken_ = iC.consumes<GEMDigiCollection>(gemDigi.getParameter<edm::InputTag>("inputTag"));
  gemPadToken_ = iC.consumes<GEMPadDigiCollection>(gemPad.getParameter<edm::InputTag>("inputTag"));
  gemClusterToken_ = iC.consumes<GEMPadDigiClusterCollection>(gemCluster.getParameter<edm::InputTag>("inputTag"));
  gemCoPadToken_ = iC.consumes<GEMCoPadDigiCollection>(gemCoPad.getParameter<edm::InputTag>("inputTag"));

  geomToken_ = iC.esConsumes<GEMGeometry, MuonGeometryRecord>();
}

void GEMDigiMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  muonSimHitMatcher_->init(iEvent, iSetup);

  if (matchToSimLink_)
    iEvent.getByToken(gemSimLinkToken_, gemDigisSLH_);
  iEvent.getByToken(gemDigiToken_, gemDigisH_);
  iEvent.getByToken(gemPadToken_, gemPadsH_);
  iEvent.getByToken(gemClusterToken_, gemClustersH_);
  iEvent.getByToken(gemCoPadToken_, gemCoPadsH_);

  const auto gemH = iSetup.getHandle(geomToken_);
  if (!gemH.isValid()) {
    gemGeometry_ = nullptr;
    edm::LogError("GEMDigiMatcher") << "Failed to initialize GEM geometry.";
  }
  gemGeometry_ = gemH.product();
}

/// do the matching
void GEMDigiMatcher::match(const SimTrack& t, const SimVertex& v) {
  // match simhits first
  muonSimHitMatcher_->match(t, v);

  // get the digi collections
  const GEMDigiCollection& gemDigis = *gemDigisH_.product();
  const GEMPadDigiCollection& gemPads = *gemPadsH_.product();
  const GEMPadDigiClusterCollection& gemClusters = *gemClustersH_.product();
  const GEMCoPadDigiCollection& gemCoPads = *gemCoPadsH_.product();

  clear();

  // hard cut on non-GEM muons
  if (std::abs(t.momentum().eta()) < 1.55)
    return;

  // now match the digis
  if (matchToSimLink_) {
    const edm::DetSetVector<GEMDigiSimLink>& gemDigisSL = *gemDigisSLH_.product();
    matchDigisSLToSimTrack(gemDigisSL);
  }
  matchDigisToSimTrack(gemDigis);
  matchPadsToSimTrack(gemPads);
  matchClustersToSimTrack(gemClusters);
  matchCoPadsToSimTrack(gemCoPads);
}

void GEMDigiMatcher::matchDigisSLToSimTrack(const edm::DetSetVector<GEMDigiSimLink>& digisSL) {
  if (verboseSimLink_)
    edm::LogInfo("GEMDigiMatcher") << "Matching simtrack to GEM simlinks" << endl;

  // loop on the simlinks
  for (auto itsimlink = digisSL.begin(); itsimlink != digisSL.end(); itsimlink++) {
    GEMDetId p_id(itsimlink->id);
    for (auto sl = itsimlink->data.begin(); sl != itsimlink->data.end(); ++sl) {
      // ignore simlinks in non-matched chambers
      const auto& detids(muonSimHitMatcher_->detIds());
      if (detids.find(p_id.rawId()) == detids.end())
        continue;

      // no simhits in this chamber!
      if (muonSimHitMatcher_->hitsInDetId(p_id.rawId()).empty())
        continue;

      if (verboseSimLink_)
        edm::LogInfo("GEMDigiMatcher") << "GEMDigiSimLink " << p_id << " " << sl->getStrip() << " " << sl->getBx()
                                       << " " << sl->getTrackId() << std::endl;

      // consider only the muon hits
      if (simMuOnly_ && std::abs(sl->getParticleType()) != 13)
        continue;

      // discard electron hits in the GEM chambers
      if (discardEleHits_ && std::abs(sl->getParticleType()) == 11)
        continue;

      // loop on the matched simhits
      for (const auto& simhit : muonSimHitMatcher_->hitsInDetId(p_id.rawId())) {
        // check if the simhit properties agree
        if (simhit.trackId() == sl->getTrackId() and simhit.particleType() == sl->getParticleType()) {
          detid_to_simLinks_[p_id.rawId()].push_back(*sl);
          if (verboseSimLink_)
            edm::LogInfo("GEMDigiMatcher") << "...was matched!" << endl;
          break;
        }
      }
    }
  }
}

void GEMDigiMatcher::matchDigisToSimTrack(const GEMDigiCollection& digis) {
  if (verboseDigi_)
    edm::LogInfo("GEMDigiMatcher") << "Matching simtrack to GEM digis" << endl;
  for (auto id : muonSimHitMatcher_->detIds()) {
    GEMDetId p_id(id);
    const auto& hit_strips = muonSimHitMatcher_->hitStripsInDetId(id, matchDeltaStrip_);
    const auto& digis_in_det = digis.get(p_id);

    for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
      bool isMatched = false;

      // check that the digi is within BX range
      if (d->bx() < minBXDigi_ || d->bx() > maxBXDigi_)
        continue;

      if (verboseDigi_)
        edm::LogInfo("GEMDigiMatcher") << "GEMDigi " << p_id << " " << *d << endl;

      // GEN-SIM-DIGI-RAW monte carlo
      if (matchToSimLink_) {
        // check that the digi matches to at least one GEMDigiSimLink
        for (const auto& sl : detid_to_simLinks_[p_id.rawId()]) {
          if (sl.getStrip() == d->strip() and sl.getBx() == d->bx()) {
            isMatched = true;
            break;
          }
        }
      }
      // GEN-SIM-RAW monte carlo
      else {
        // check that it matches a strip that was hit by SimHits from our track
        if (hit_strips.find(d->strip()) != hit_strips.end()) {
          isMatched = true;
        }
      }
      if (isMatched) {
        detid_to_digis_[p_id.rawId()].push_back(*d);
        chamber_to_digis_[p_id.chamberId().rawId()].push_back(*d);
        superchamber_to_digis_[p_id.superChamberId().rawId()].push_back(*d);
        if (verboseDigi_)
          edm::LogInfo("GEMDigiMatcher") << "...was matched!" << endl;
      }
    }
  }
}
void GEMDigiMatcher::matchPadsToSimTrack(const GEMPadDigiCollection& pads) {
  for (auto it = pads.begin(); it != pads.end(); ++it) {
    const GEMDetId& p_id = (*it).first;
    const auto& padsvec = (*it).second;

    for (auto pad = padsvec.first; pad != padsvec.second; ++pad) {
      // check that the pad BX is within the range
      if (pad->bx() < minBXPad_ || pad->bx() > maxBXPad_)
        continue;

      if (verbosePad_)
        edm::LogInfo("GEMDigiMatcher") << "GEMPad " << p_id << " " << *pad << endl;

      auto digivec = detid_to_digis_[p_id.rawId()];
      /*
        For GE1/1 and ME0 (and obsolete 8-partition GE2/1) geometries, the matching
        is pretty simple. Each simhit is converted to a digi. Two digis in neighboring
        strips are ed together into a pad. So for these geometries you just need
        to match pads to simtracks that are in the same eta partition (detid) as the
        simhit is in.  By convention, all pads in the 16-partition GE2/1 geometry are
        assigned to odd partition numbers. For the 16-partition GE2/1 geometry, you may
        have a track with simhits in eta partition 1 and 2 on the same strip and only
        produce 1 pad associated to eta partition 1. Therefore, for pads with odd
        partition number N, you need to consider the digis in the even partition number
        N+1.
      */
      if (p_id.roll() % 2 == 1 && p_id.isGE21() && pad->nPartitions() == GEMPadDigi::GE21SplitStrip) {
        // make the GEMDetId for the neighboring partition
        GEMDetId p_id2(p_id.region(), p_id.ring(), p_id.station(), p_id.layer(), p_id.chamber(), p_id.roll() + 1);

        // make a temporary container for its digis
        auto digivec2 = detid_to_digis_[p_id2.rawId()];

        // now add it to the container for the digis in the odd partition number
        digivec.insert(digivec.end(), digivec2.begin(), digivec2.end());
      }
      for (const auto& digi : digivec) {
        // for 8-partition geometries, the pad number equals the strip number divided by two
        const bool match8Partition(digi.strip() / 2 == pad->pad());

        // for 16-partition geometries, the pad number is the strip number itself
        const bool match16Partition(digi.strip() == pad->pad());

        // now consider the different cases separately
        const bool matchGE0(p_id.isME0() and match8Partition);
        const bool matchGE11(p_id.isGE11() and match8Partition);
        const bool matchGE21_8(p_id.isGE21() and pad->nPartitions() == GEMPadDigi::GE21 and match8Partition);
        const bool matchGE21_16(p_id.isGE21() and pad->nPartitions() == GEMPadDigi::GE21SplitStrip and
                                match16Partition);

        // OR them together
        if (matchGE0 or matchGE11 or matchGE21_8 or matchGE21_16) {
          detid_to_pads_[p_id.rawId()].push_back(*pad);
          chamber_to_pads_[p_id.chamberId().rawId()].push_back(*pad);
          superchamber_to_pads_[p_id.superChamberId().rawId()].push_back(*pad);
          if (verbosePad_)
            edm::LogInfo("GEMDigiMatcher") << "...was matched!" << endl;
          break;
        }
      }
    }
  }
}

void GEMDigiMatcher::matchClustersToSimTrack(const GEMPadDigiClusterCollection& clusters) {
  for (auto it = clusters.begin(); it != clusters.end(); ++it) {
    const GEMDetId p_id = (*it).first;
    const auto clvec = (*it).second;
    for (auto cluster = clvec.first; cluster != clvec.second; ++cluster) {
      bool isMatched = false;

      // check that the cluster BX is within the range
      if (cluster->bx() < minBXCluster_ || cluster->bx() > maxBXCluster_)
        continue;

      if (verboseCluster_)
        edm::LogInfo("GEMDigiMatcher") << "GEMCluster " << p_id << " " << *cluster << endl;

      // check that at least one pad was hit by the track
      for (const auto& p : cluster->pads()) {
        for (const auto& pad : detid_to_pads_[p_id.rawId()]) {
          if (pad.pad() == p) {
            isMatched = true;
          }
        }
      }
      if (isMatched) {
        detid_to_clusters_[p_id.rawId()].push_back(*cluster);
        chamber_to_clusters_[p_id.chamberId().rawId()].push_back(*cluster);
        superchamber_to_clusters_[p_id.superChamberId().rawId()].push_back(*cluster);
        if (verboseCluster_)
          edm::LogInfo("GEMDigiMatcher") << "...was matched!" << endl;
      }
    }
  }
}

void GEMDigiMatcher::matchCoPadsToSimTrack(const GEMCoPadDigiCollection& co_pads) {
  // loop on the GEM detids
  for (auto d : superChamberIdsPad()) {
    GEMDetId id(d);

    const auto& co_pads_in_det = co_pads.get(id);
    for (auto copad = co_pads_in_det.first; copad != co_pads_in_det.second; ++copad) {
      // check that the cluster BX is within the range
      if (copad->bx(1) < minBXCoPad_ || copad->bx(1) > maxBXCoPad_)
        continue;

      if (verboseCoPad_)
        edm::LogInfo("GEMDigiMatcher") << "GEMCoPadDigi: " << id << " " << *copad << endl;

      bool isMatchedL1 = false;
      bool isMatchedL2 = false;
      GEMDetId gemL1_id(id.region(), 1, id.station(), 1, id.chamber(), copad->roll());
      GEMDetId gemL2_id(id.region(), 1, id.station(), 2, id.chamber(), 0);

      // first pad is tightly matched
      for (const auto& p : padsInDetId(gemL1_id.rawId())) {
        if (p == copad->first()) {
          isMatchedL1 = true;
        }
      }

      // second pad can only be loosely matched
      for (const auto& p : padsInChamber(gemL2_id.rawId())) {
        if (p == copad->second()) {
          isMatchedL2 = true;
        }
      }
      if (isMatchedL1 and isMatchedL2) {
        superchamber_to_copads_[id.rawId()].push_back(*copad);
        if (verboseCoPad_)
          edm::LogInfo("GEMDigiMatcher") << "...was matched! " << endl;
      }
    }
  }
}

std::set<unsigned int> GEMDigiMatcher::detIdsSimLink(int gem_type) const {
  return selectDetIds(detid_to_simLinks_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::detIdsDigi(int gem_type) const {
  return selectDetIds(detid_to_digis_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::detIdsPad(int gem_type) const { return selectDetIds(detid_to_pads_, gem_type); }

std::set<unsigned int> GEMDigiMatcher::detIdsCluster(int gem_type) const {
  return selectDetIds(detid_to_clusters_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::chamberIdsDigi(int gem_type) const {
  return selectDetIds(chamber_to_digis_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::chamberIdsPad(int gem_type) const {
  return selectDetIds(chamber_to_pads_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::chamberIdsCluster(int gem_type) const {
  return selectDetIds(chamber_to_clusters_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::superChamberIdsDigi(int gem_type) const {
  return selectDetIds(superchamber_to_digis_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::superChamberIdsPad(int gem_type) const {
  return selectDetIds(superchamber_to_pads_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::superChamberIdsCluster(int gem_type) const {
  return selectDetIds(superchamber_to_clusters_, gem_type);
}

std::set<unsigned int> GEMDigiMatcher::superChamberIdsCoPad(int gem_type) const {
  return selectDetIds(superchamber_to_copads_, gem_type);
}

const GEMDigiContainer& GEMDigiMatcher::digisInDetId(unsigned int detid) const {
  if (detid_to_digis_.find(detid) == detid_to_digis_.end())
    return no_gem_digis_;
  return detid_to_digis_.at(detid);
}

const GEMDigiContainer& GEMDigiMatcher::digisInChamber(unsigned int detid) const {
  if (chamber_to_digis_.find(detid) == chamber_to_digis_.end())
    return no_gem_digis_;
  return chamber_to_digis_.at(detid);
}

const GEMDigiContainer& GEMDigiMatcher::digisInSuperChamber(unsigned int detid) const {
  if (superchamber_to_digis_.find(detid) == superchamber_to_digis_.end())
    return no_gem_digis_;
  return superchamber_to_digis_.at(detid);
}

const GEMPadDigiContainer& GEMDigiMatcher::padsInDetId(unsigned int detid) const {
  if (detid_to_pads_.find(detid) == detid_to_pads_.end())
    return no_gem_pads_;
  return detid_to_pads_.at(detid);
}

const GEMPadDigiContainer& GEMDigiMatcher::padsInChamber(unsigned int detid) const {
  if (chamber_to_pads_.find(detid) == chamber_to_pads_.end())
    return no_gem_pads_;
  return chamber_to_pads_.at(detid);
}

const GEMPadDigiContainer& GEMDigiMatcher::padsInSuperChamber(unsigned int detid) const {
  if (superchamber_to_pads_.find(detid) == superchamber_to_pads_.end())
    return no_gem_pads_;
  return superchamber_to_pads_.at(detid);
}

const GEMPadDigiClusterContainer& GEMDigiMatcher::clustersInDetId(unsigned int detid) const {
  if (detid_to_clusters_.find(detid) == detid_to_clusters_.end())
    return no_gem_clusters_;
  return detid_to_clusters_.at(detid);
}

const GEMPadDigiClusterContainer& GEMDigiMatcher::clustersInChamber(unsigned int detid) const {
  if (chamber_to_clusters_.find(detid) == chamber_to_clusters_.end())
    return no_gem_clusters_;
  return chamber_to_clusters_.at(detid);
}

const GEMPadDigiClusterContainer& GEMDigiMatcher::clustersInSuperChamber(unsigned int detid) const {
  if (superchamber_to_clusters_.find(detid) == superchamber_to_clusters_.end())
    return no_gem_clusters_;
  return superchamber_to_clusters_.at(detid);
}

const GEMCoPadDigiContainer& GEMDigiMatcher::coPadsInSuperChamber(unsigned int detid) const {
  if (superchamber_to_copads_.find(detid) == superchamber_to_copads_.end())
    return no_gem_copads_;
  return superchamber_to_copads_.at(detid);
}

int GEMDigiMatcher::nLayersWithDigisInSuperChamber(unsigned int detid) const {
  set<int> layers;
  GEMDetId sch_id(detid);
  for (int iLayer = 1; iLayer <= 2; iLayer++) {
    GEMDetId ch_id(sch_id.region(), sch_id.ring(), sch_id.station(), iLayer, sch_id.chamber(), 0);
    // get the digis in this chamber
    const auto& digis = digisInChamber(ch_id.rawId());
    // at least one digi in this layer!
    if (!digis.empty()) {
      layers.insert(iLayer);
    }
  }
  return layers.size();
}

int GEMDigiMatcher::nLayersWithPadsInSuperChamber(unsigned int detid) const {
  set<int> layers;
  GEMDetId sch_id(detid);
  for (int iLayer = 1; iLayer <= 2; iLayer++) {
    GEMDetId ch_id(sch_id.region(), sch_id.ring(), sch_id.station(), iLayer, sch_id.chamber(), 0);
    // get the pads in this chamber
    const auto& pads = padsInChamber(ch_id.rawId());
    // at least one digi in this layer!
    if (!pads.empty()) {
      layers.insert(iLayer);
    }
  }
  return layers.size();
}

int GEMDigiMatcher::nLayersWithClustersInSuperChamber(unsigned int detid) const {
  set<int> layers;
  GEMDetId sch_id(detid);
  for (int iLayer = 1; iLayer <= 2; iLayer++) {
    GEMDetId ch_id(sch_id.region(), sch_id.ring(), sch_id.station(), iLayer, sch_id.chamber(), 0);
    // get the pads in this chamber
    const auto& clusters = clustersInChamber(ch_id.rawId());
    // at least one digi in this layer!
    if (!clusters.empty()) {
      layers.insert(iLayer);
    }
  }
  return layers.size();
}

int GEMDigiMatcher::nPads() const {
  int n = 0;
  const auto& ids = superChamberIdsPad();
  for (const auto& id : ids) {
    n += padsInSuperChamber(id).size();
  }
  return n;
}

int GEMDigiMatcher::nCoPads() const {
  int n = 0;
  const auto& ids = superChamberIdsCoPad();
  for (const auto& id : ids) {
    n += coPadsInSuperChamber(id).size();
  }
  return n;
}

std::set<int> GEMDigiMatcher::stripNumbersInDetId(unsigned int detid) const {
  set<int> result;
  const auto& digis = digisInDetId(detid);
  for (const auto& d : digis) {
    result.insert(d.strip());
  }
  return result;
}

std::set<int> GEMDigiMatcher::padNumbersInDetId(unsigned int detid) const {
  set<int> result;
  const auto& digis = padsInDetId(detid);
  for (const auto& d : digis) {
    result.insert(d.pad());
  }
  return result;
}

std::set<int> GEMDigiMatcher::partitionNumbers() const {
  std::set<int> result;

  const auto& detids = detIdsDigi();
  for (const auto& id : detids) {
    const GEMDetId& idd(id);
    result.insert(idd.roll());
  }
  return result;
}

std::set<int> GEMDigiMatcher::partitionNumbersWithCoPads() const {
  std::set<int> result;

  const auto& detids = superChamberIdsCoPad();
  for (const auto& id : detids) {
    const GEMDetId& idd(id);
    result.insert(idd.roll());
  }
  return result;
}

GlobalPoint GEMDigiMatcher::getGlobalPointDigi(unsigned int rawId, const GEMDigi& d) const {
  GEMDetId gem_id(rawId);
  const LocalPoint& gem_lp = gemGeometry_->etaPartition(gem_id)->centreOfStrip(d.strip());
  const GlobalPoint& gem_gp = gemGeometry_->idToDet(gem_id)->surface().toGlobal(gem_lp);
  return gem_gp;
}

GlobalPoint GEMDigiMatcher::getGlobalPointPad(unsigned int rawId, const GEMPadDigi& tp) const {
  GEMDetId gem_id(rawId);
  const LocalPoint& gem_lp = gemGeometry_->etaPartition(gem_id)->centreOfPad(tp.pad());
  const GlobalPoint& gem_gp = gemGeometry_->idToDet(gem_id)->surface().toGlobal(gem_lp);
  return gem_gp;
}

void GEMDigiMatcher::clear() {
  detid_to_simLinks_.clear();

  detid_to_digis_.clear();
  chamber_to_digis_.clear();
  superchamber_to_digis_.clear();

  detid_to_pads_.clear();
  chamber_to_pads_.clear();
  superchamber_to_pads_.clear();

  detid_to_clusters_.clear();
  chamber_to_clusters_.clear();
  superchamber_to_clusters_.clear();

  superchamber_to_copads_.clear();
}
