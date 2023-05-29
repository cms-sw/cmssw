#include "Validation/MuonCSCDigis/interface/CSCDigiMatcher.h"

using namespace std;

CSCDigiMatcher::CSCDigiMatcher(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
  const auto& wireDigi = pset.getParameterSet("cscWireDigi");
  verboseWG_ = wireDigi.getParameter<int>("verbose");
  minBXWire_ = wireDigi.getParameter<int>("minBX");
  maxBXWire_ = wireDigi.getParameter<int>("maxBX");
  matchDeltaWG_ = wireDigi.getParameter<int>("matchDeltaWG");

  const auto& comparatorDigi = pset.getParameterSet("cscComparatorDigi");
  verboseComparator_ = comparatorDigi.getParameter<int>("verbose");
  minBXComparator_ = comparatorDigi.getParameter<int>("minBX");
  maxBXComparator_ = comparatorDigi.getParameter<int>("maxBX");
  matchDeltaComparator_ = comparatorDigi.getParameter<int>("matchDeltaStrip");

  const auto& stripDigi = pset.getParameterSet("cscStripDigi");
  verboseStrip_ = stripDigi.getParameter<int>("verbose");
  minBXStrip_ = stripDigi.getParameter<int>("minBX");
  maxBXStrip_ = stripDigi.getParameter<int>("maxBX");
  matchDeltaStrip_ = stripDigi.getParameter<int>("matchDeltaStrip");

  // make a new simhits matcher
  muonSimHitMatcher_.reset(new CSCSimHitMatcher(pset, std::move(iC)));

  comparatorDigiInput_ =
      iC.consumes<CSCComparatorDigiCollection>(comparatorDigi.getParameter<edm::InputTag>("inputTag"));
  stripDigiInput_ = iC.consumes<CSCStripDigiCollection>(stripDigi.getParameter<edm::InputTag>("inputTag"));
  wireDigiInput_ = iC.consumes<CSCWireDigiCollection>(wireDigi.getParameter<edm::InputTag>("inputTag"));
}

void CSCDigiMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  muonSimHitMatcher_->init(iEvent, iSetup);

  iEvent.getByToken(comparatorDigiInput_, comparatorDigisH_);
  iEvent.getByToken(stripDigiInput_, stripDigisH_);
  iEvent.getByToken(wireDigiInput_, wireDigisH_);
}

/// do the matching
void CSCDigiMatcher::match(const SimTrack& t, const SimVertex& v) {
  // match simhits first
  muonSimHitMatcher_->match(t, v);

  // get the digi collections
  const CSCComparatorDigiCollection& comparators = *comparatorDigisH_.product();
  const CSCStripDigiCollection& strips = *stripDigisH_.product();
  const CSCWireDigiCollection& wires = *wireDigisH_.product();

  clear();

  // now match the digis
  matchComparatorsToSimTrack(comparators);
  matchStripsToSimTrack(strips);
  matchWiresToSimTrack(wires);
}

void CSCDigiMatcher::matchComparatorsToSimTrack(const CSCComparatorDigiCollection& comparators) {
  const auto& det_ids = muonSimHitMatcher_->detIds(0);
  for (const auto& id : det_ids) {
    CSCDetId layer_id(id);

    const auto& hit_comparators = muonSimHitMatcher_->hitStripsInDetId(id, matchDeltaStrip_);
    if (verboseComparator_) {
      cout << "hit_comparators_fat, CSCid " << layer_id << " ";
      copy(hit_comparators.begin(), hit_comparators.end(), ostream_iterator<int>(cout, " "));
      cout << endl;
    }

    int ndigis = 0;
    const auto& comp_digis_in_det = comparators.get(layer_id);
    for (auto c = comp_digis_in_det.first; c != comp_digis_in_det.second; ++c) {
      if (verboseComparator_)
        edm::LogInfo("CSCDigiMatcher") << "sdigi " << layer_id << " (comparator, comparator, Tbin ) " << *c;

      // check that the first BX for this digi wasn't too early or too late
      if (c->getTimeBin() < minBXComparator_ || c->getTimeBin() > maxBXComparator_)
        continue;

      ndigis++;

      int comparator = c->getStrip();  // comparators are counted from 1
      // check that it matches a comparator that was hit by SimHits from our track
      if (hit_comparators.find(comparator) == hit_comparators.end())
        continue;

      if (verboseComparator_)
        edm::LogInfo("CSCDigiMatcher") << "Matched comparator " << *c;
      detid_to_comparators_[id].push_back(*c);
      chamber_to_comparators_[layer_id.chamberId().rawId()].push_back(*c);
    }
    detid_to_totalcomparators_[id] = ndigis;  //id to totalcomparators
  }
}

void CSCDigiMatcher::matchStripsToSimTrack(const CSCStripDigiCollection& strips) {
  for (auto detUnitIt = strips.begin(); detUnitIt != strips.end(); ++detUnitIt) {
    const CSCDetId& id = (*detUnitIt).first;
    const auto& range = (*detUnitIt).second;
    for (auto digiIt = range.first; digiIt != range.second; ++digiIt) {
      if (id.station() == 1 and (id.ring() == 1 or id.ring() == 4))
        if (verboseStrip_)
          edm::LogInfo("CSCDigiMatcher") << "CSCid " << id << " Strip digi (strip, strip, Tbin ) " << (*digiIt);
    }
  }

  const auto& det_ids = muonSimHitMatcher_->detIds(0);
  for (const auto& id : det_ids) {
    CSCDetId layer_id(id);

    const auto& hit_strips = muonSimHitMatcher_->hitStripsInDetId(id, matchDeltaStrip_);
    if (verboseStrip_) {
      cout << "hit_strips_fat, CSCid " << layer_id << " ";
      copy(hit_strips.begin(), hit_strips.end(), ostream_iterator<int>(cout, " "));
      cout << endl;
    }

    int ndigis = 0;

    const auto& strip_digis_in_det = strips.get(layer_id);
    for (auto c = strip_digis_in_det.first; c != strip_digis_in_det.second; ++c) {
      //next is to remove the strips with pulse at noise level, with ACD info
      //the detail may be from CSC rechits algorithm
      if (verboseStrip_)
        edm::LogInfo("CSCDigiMatcher") << "sdigi " << layer_id << " (strip, ADC ) " << *c;

      ndigis++;

      int strip = c->getStrip();  // strips are counted from 1
      // check that it matches a strip that was hit by SimHits from our track
      if (hit_strips.find(strip) == hit_strips.end())
        continue;

      if (verboseStrip_)
        edm::LogInfo("CSCDigiMatcher") << "Matched strip " << *c;
      detid_to_strips_[id].push_back(*c);
      chamber_to_strips_[layer_id.chamberId().rawId()].push_back(*c);
    }
    detid_to_totalstrips_[id] = ndigis;
  }
}

void CSCDigiMatcher::matchWiresToSimTrack(const CSCWireDigiCollection& wires) {
  const auto& det_ids = muonSimHitMatcher_->detIds(0);
  for (const auto& id : det_ids) {
    CSCDetId layer_id(id);

    const auto& hit_wires = muonSimHitMatcher_->hitWiregroupsInDetId(id, matchDeltaWG_);
    if (verboseWG_) {
      cout << "hit_wires ";
      copy(hit_wires.begin(), hit_wires.end(), ostream_iterator<int>(cout, " "));
      cout << endl;
    }

    int ndigis = 0;

    const auto& wire_digis_in_det = wires.get(layer_id);
    for (auto w = wire_digis_in_det.first; w != wire_digis_in_det.second; ++w) {
      if (verboseStrip_)
        edm::LogInfo("CSCDigiMatcher") << "wdigi " << layer_id << " (wire, Tbin ) " << *w;

      // check that the first BX for this digi wasn't too early or too late
      if (w->getTimeBin() < minBXWire_ || w->getTimeBin() > maxBXWire_)
        continue;

      ndigis++;

      int wg = w->getWireGroup();  // wiregroups are counted from 1
      // check that it matches a strip that was hit by SimHits from our track
      if (hit_wires.find(wg) == hit_wires.end())
        continue;

      if (verboseStrip_)
        edm::LogInfo("CSCDigiMatcher") << "Matched wire digi " << *w << endl;
      detid_to_wires_[id].push_back(*w);
      chamber_to_wires_[layer_id.chamberId().rawId()].push_back(*w);
    }
    detid_to_totalwires_[id] = ndigis;
  }
}

std::set<unsigned int> CSCDigiMatcher::detIdsComparator(int csc_type) const {
  return selectDetIds(detid_to_comparators_, csc_type);
}

std::set<unsigned int> CSCDigiMatcher::detIdsStrip(int csc_type) const {
  return selectDetIds(detid_to_strips_, csc_type);
}

std::set<unsigned int> CSCDigiMatcher::detIdsWire(int csc_type) const {
  return selectDetIds(detid_to_wires_, csc_type);
}

std::set<unsigned int> CSCDigiMatcher::chamberIdsComparator(int csc_type) const {
  return selectDetIds(chamber_to_comparators_, csc_type);
}

std::set<unsigned int> CSCDigiMatcher::chamberIdsStrip(int csc_type) const {
  return selectDetIds(chamber_to_strips_, csc_type);
}

std::set<unsigned int> CSCDigiMatcher::chamberIdsWire(int csc_type) const {
  return selectDetIds(chamber_to_wires_, csc_type);
}

const CSCComparatorDigiContainer& CSCDigiMatcher::comparatorDigisInDetId(unsigned int detid) const {
  if (detid_to_comparators_.find(detid) == detid_to_comparators_.end())
    return no_comparators_;
  return detid_to_comparators_.at(detid);
}

const CSCComparatorDigiContainer& CSCDigiMatcher::comparatorDigisInChamber(unsigned int detid) const {
  if (chamber_to_comparators_.find(detid) == chamber_to_comparators_.end())
    return no_comparators_;
  return chamber_to_comparators_.at(detid);
}

const CSCStripDigiContainer& CSCDigiMatcher::stripDigisInDetId(unsigned int detid) const {
  if (detid_to_strips_.find(detid) == detid_to_strips_.end())
    return no_strips_;
  return detid_to_strips_.at(detid);
}

const CSCStripDigiContainer& CSCDigiMatcher::stripDigisInChamber(unsigned int detid) const {
  if (chamber_to_strips_.find(detid) == chamber_to_strips_.end())
    return no_strips_;
  return chamber_to_strips_.at(detid);
}

const CSCWireDigiContainer& CSCDigiMatcher::wireDigisInDetId(unsigned int detid) const {
  if (detid_to_wires_.find(detid) == detid_to_wires_.end())
    return no_wires_;
  return detid_to_wires_.at(detid);
}

const CSCWireDigiContainer& CSCDigiMatcher::wireDigisInChamber(unsigned int detid) const {
  if (chamber_to_wires_.find(detid) == chamber_to_wires_.end())
    return no_wires_;
  return chamber_to_wires_.at(detid);
}

int CSCDigiMatcher::nLayersWithComparatorInChamber(unsigned int detid) const {
  int nLayers = 0;
  CSCDetId chamberId(detid);
  for (int i = 1; i <= 6; ++i) {
    CSCDetId layerId(chamberId.endcap(), chamberId.station(), chamberId.ring(), chamberId.chamber(), i);
    if (!comparatorDigisInDetId(layerId.rawId()).empty()) {
      nLayers++;
    }
  }
  return nLayers;
}

int CSCDigiMatcher::nLayersWithStripInChamber(unsigned int detid) const {
  int nLayers = 0;
  CSCDetId chamberId(detid);
  for (int i = 1; i <= 6; ++i) {
    CSCDetId layerId(chamberId.endcap(), chamberId.station(), chamberId.ring(), chamberId.chamber(), i);
    if (!stripDigisInDetId(layerId.rawId()).empty()) {
      nLayers++;
    }
  }
  return nLayers;
}

int CSCDigiMatcher::nLayersWithWireInChamber(unsigned int detid) const {
  int nLayers = 0;
  CSCDetId chamberId(detid);
  for (int i = 1; i <= 6; ++i) {
    CSCDetId layerId(chamberId.endcap(), chamberId.station(), chamberId.ring(), chamberId.chamber(), i);
    if (!wireDigisInDetId(layerId.rawId()).empty()) {
      nLayers++;
    }
  }
  return nLayers;
}

int CSCDigiMatcher::nCoincidenceComparatorChambers(int min_n_layers) const {
  int result = 0;
  const auto& chamber_ids = chamberIdsComparator();
  for (const auto& id : chamber_ids) {
    if (nLayersWithComparatorInChamber(id) >= min_n_layers)
      result += 1;
  }
  return result;
}

int CSCDigiMatcher::nCoincidenceStripChambers(int min_n_layers) const {
  int result = 0;
  const auto& chamber_ids = chamberIdsStrip();
  for (const auto& id : chamber_ids) {
    if (nLayersWithStripInChamber(id) >= min_n_layers)
      result += 1;
  }
  return result;
}

int CSCDigiMatcher::nCoincidenceWireChambers(int min_n_layers) const {
  int result = 0;
  const auto& chamber_ids = chamberIdsWire();
  for (const auto& id : chamber_ids) {
    if (nLayersWithWireInChamber(id) >= min_n_layers)
      result += 1;
  }
  return result;
}

std::set<int> CSCDigiMatcher::comparatorsInDetId(unsigned int detid) const {
  set<int> result;
  const auto& digis = comparatorDigisInDetId(detid);
  for (const auto& d : digis) {
    result.insert(d.getHalfStrip());
  }
  return result;
}

std::set<int> CSCDigiMatcher::stripsInDetId(unsigned int detid) const {
  set<int> result;
  const auto& digis = stripDigisInDetId(detid);
  for (const auto& d : digis) {
    result.insert(d.getStrip());
  }
  return result;
}

std::set<int> CSCDigiMatcher::wiregroupsInDetId(unsigned int detid) const {
  set<int> result;
  const auto& digis = wireDigisInDetId(detid);
  for (const auto& d : digis) {
    result.insert(d.getWireGroup());
  }
  return result;
}

std::set<int> CSCDigiMatcher::comparatorsInChamber(unsigned int detid, int max_gap_to_fill) const {
  set<int> result;
  const auto& digis = comparatorDigisInChamber(detid);
  for (const auto& d : digis) {
    result.insert(d.getStrip());
  }
  if (max_gap_to_fill > 0) {
    int prev = -111;
    for (const auto& s : result) {
      if (s - prev > 1 && s - prev - 1 <= max_gap_to_fill) {
        for (int fill_s = prev + 1; fill_s < s; ++fill_s)
          result.insert(fill_s);
      }
      prev = s;
    }
  }

  return result;
}

std::set<int> CSCDigiMatcher::stripsInChamber(unsigned int detid, int max_gap_to_fill) const {
  set<int> result;
  const auto& digis = stripDigisInChamber(detid);
  for (const auto& d : digis) {
    result.insert(d.getStrip());
  }
  if (max_gap_to_fill > 0) {
    int prev = -111;
    for (const auto& s : result) {
      if (s - prev > 1 && s - prev - 1 <= max_gap_to_fill) {
        for (int fill_s = prev + 1; fill_s < s; ++fill_s)
          result.insert(fill_s);
      }
      prev = s;
    }
  }

  return result;
}

std::set<int> CSCDigiMatcher::wiregroupsInChamber(unsigned int detid, int max_gap_to_fill) const {
  set<int> result;
  const auto& digis = wireDigisInChamber(detid);
  for (const auto& d : digis) {
    result.insert(d.getWireGroup());
  }
  if (max_gap_to_fill > 0) {
    int prev = -111;
    for (const auto& w : result) {
      if (w - prev > 1 && w - prev - 1 <= max_gap_to_fill) {
        for (int fill_w = prev + 1; fill_w < w; ++fill_w)
          result.insert(fill_w);
      }
      prev = w;
    }
  }
  return result;
}

int CSCDigiMatcher::totalComparators(unsigned int detid) const {
  if (detid_to_totalcomparators_.find(detid) == detid_to_totalcomparators_.end())
    return 0;
  return detid_to_totalcomparators_.at(detid);
}

int CSCDigiMatcher::totalStrips(unsigned int detid) const {
  if (detid_to_totalstrips_.find(detid) == detid_to_totalstrips_.end())
    return 0;
  return detid_to_totalstrips_.at(detid);
}

int CSCDigiMatcher::totalWires(unsigned int detid) const {
  if (detid_to_totalwires_.find(detid) == detid_to_totalwires_.end())
    return 0;
  return detid_to_totalwires_.at(detid);
}

void CSCDigiMatcher::clear() {
  detid_to_comparators_.clear();
  chamber_to_comparators_.clear();

  detid_to_strips_.clear();
  chamber_to_strips_.clear();

  detid_to_wires_.clear();
  chamber_to_wires_.clear();
}
