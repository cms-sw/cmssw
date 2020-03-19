#include "Validation/MuonGEMRecHits/interface/GEMRecHitMatcher.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

using namespace std;

GEMRecHitMatcher::GEMRecHitMatcher(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
  const auto& gemRecHit = pset.getParameterSet("gemRecHit");
  minBX_ = gemRecHit.getParameter<int>("minBX");
  maxBX_ = gemRecHit.getParameter<int>("maxBX");
  verbose_ = gemRecHit.getParameter<int>("verbose");

  // make a new digi matcher
  gemDigiMatcher_.reset(new GEMDigiMatcher(pset, std::move(iC)));

  gemRecHitToken_ = iC.consumes<GEMRecHitCollection>(gemRecHit.getParameter<edm::InputTag>("inputTag"));
}

void GEMRecHitMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  gemDigiMatcher_->init(iEvent, iSetup);

  iEvent.getByToken(gemRecHitToken_, gemRecHitH_);

  iSetup.get<MuonGeometryRecord>().get(gem_geom_);
  if (gem_geom_.isValid()) {
    gemGeometry_ = &*gem_geom_;
  } else {
    std::cout << "+++ Info: GEM geometry is unavailable. +++\n";
  }
}

/// do the matching
void GEMRecHitMatcher::match(const SimTrack& t, const SimVertex& v) {
  // match digis first
  gemDigiMatcher_->match(t, v);

  // get the rechit collection
  const GEMRecHitCollection& gemRecHits = *gemRecHitH_.product();

  // now match the rechits
  matchRecHitsToSimTrack(gemRecHits);
}

void GEMRecHitMatcher::matchRecHitsToSimTrack(const GEMRecHitCollection& rechits) {
  // get the matched ids with digis
  const auto& det_ids = gemDigiMatcher_->detIdsDigi();

  // loop on those ids
  for (auto id : det_ids) {
    // now check the digis in this detid
    const auto& hit_digis = gemDigiMatcher_->stripNumbersInDetId(id);
    if (verbose()) {
      cout << "hit_digis_fat ";
      copy(hit_digis.begin(), hit_digis.end(), ostream_iterator<int>(cout, " "));
      cout << endl;
    }

    GEMDetId p_id(id);
    const auto& rechits_in_det = rechits.get(p_id);

    for (auto d = rechits_in_det.first; d != rechits_in_det.second; ++d) {
      if (verbose())
        cout << "recHit " << p_id << " " << *d << endl;

      // check that the rechit is within BX range
      if (d->BunchX() < minBX_ || d->BunchX() > maxBX_)
        continue;

      int firstStrip = d->firstClusterStrip();
      int cls = d->clusterSize();
      bool stripFound = false;

      // check that it matches a strip that was hit by digis from our track
      for (int i = firstStrip; i < (firstStrip + cls); i++) {
        if (hit_digis.find(i) != hit_digis.end())
          stripFound = true;
      }

      // this rechit did not correspond with any previously matched digi
      if (!stripFound)
        continue;
      if (verbose())
        cout << "oki" << endl;

      recHits_.push_back(*d);
      detid_to_recHits_[id].push_back(*d);
      chamber_to_recHits_[p_id.chamberId().rawId()].push_back(*d);
      superchamber_to_recHits_[p_id.superChamberId().rawId()].push_back(*d);
    }
  }
}

std::set<unsigned int> GEMRecHitMatcher::detIds() const {
  std::set<unsigned int> result;
  for (const auto& p : detid_to_recHits_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> GEMRecHitMatcher::chamberIds() const {
  std::set<unsigned int> result;
  for (const auto& p : chamber_to_recHits_)
    result.insert(p.first);
  return result;
}

std::set<unsigned int> GEMRecHitMatcher::superChamberIds() const {
  std::set<unsigned int> result;
  for (const auto& p : superchamber_to_recHits_)
    result.insert(p.first);
  return result;
}

const GEMRecHitContainer& GEMRecHitMatcher::recHitsInDetId(unsigned int detid) const {
  if (detid_to_recHits_.find(detid) == detid_to_recHits_.end())
    return no_recHits_;
  return detid_to_recHits_.at(detid);
}

const GEMRecHitContainer& GEMRecHitMatcher::recHitsInChamber(unsigned int detid) const {
  if (chamber_to_recHits_.find(detid) == chamber_to_recHits_.end())
    return no_recHits_;
  return chamber_to_recHits_.at(detid);
}

const GEMRecHitContainer& GEMRecHitMatcher::recHitsInSuperChamber(unsigned int detid) const {
  if (superchamber_to_recHits_.find(detid) == superchamber_to_recHits_.end())
    return no_recHits_;
  return superchamber_to_recHits_.at(detid);
}

int GEMRecHitMatcher::nLayersWithRecHitsInSuperChamber(unsigned int detid) const {
  set<int> layers;
  for (const auto& d : recHitsInSuperChamber(detid)) {
    layers.insert(d.gemId().layer());
  }
  return layers.size();
}

std::set<int> GEMRecHitMatcher::stripNumbersInDetId(unsigned int detid) const {
  set<int> result;
  for (const auto& d : recHitsInDetId(detid)) {
    // loop on all strips hit in this rechit
    for (int iStrip = d.firstClusterStrip(); iStrip < d.firstClusterStrip() + d.clusterSize(); ++iStrip) {
      result.insert(iStrip);
    }
  }
  return result;
}

std::set<int> GEMRecHitMatcher::partitionNumbers() const {
  std::set<int> result;

  for (auto id : detIds()) {
    GEMDetId idd(id);
    result.insert(idd.roll());
  }
  return result;
}

GlobalPoint GEMRecHitMatcher::recHitPosition(const GEMRecHit& rechit) const {
  const GEMDetId& idd = rechit.gemId();
  const LocalPoint& lp = rechit.localPosition();
  return gemGeometry_->idToDet(idd)->surface().toGlobal(lp);
}

GlobalPoint GEMRecHitMatcher::recHitMeanPosition(const GEMRecHitContainer& rechit) const {
  GlobalPoint point_zero;
  if (rechit.empty())
    return point_zero;  // point "zero"

  float sumx, sumy, sumz;
  sumx = sumy = sumz = 0.f;
  size_t n = 0;
  for (const auto& d : rechit) {
    GlobalPoint gp = recHitPosition(d);
    if (gp == point_zero)
      continue;

    sumx += gp.x();
    sumy += gp.y();
    sumz += gp.z();
    ++n;
  }
  if (n == 0)
    return GlobalPoint();
  return GlobalPoint(sumx / n, sumy / n, sumz / n);
}

bool GEMRecHitMatcher::recHitInContainer(const GEMRecHit& rh, const GEMRecHitContainer& c) const {
  bool isSame = false;
  for (const auto& thisRH : c)
    if (areGEMRecHitSame(thisRH, rh))
      isSame = true;
  return isSame;
}

bool GEMRecHitMatcher::isGEMRecHitMatched(const GEMRecHit& thisRh) const {
  return recHitInContainer(thisRh, recHits());
}

bool GEMRecHitMatcher::areGEMRecHitSame(const GEMRecHit& l, const GEMRecHit& r) const {
  return l.localPosition() == r.localPosition() and l.BunchX() == r.BunchX();
}
