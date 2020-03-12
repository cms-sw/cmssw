#include "Validation/MuonCSCDigis/interface/CSCStubMatcher.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <algorithm>

using namespace std;

CSCStubMatcher::CSCStubMatcher(const edm::ParameterSet& pSet, edm::ConsumesCollector&& iC) {
  const auto& cscCLCT = pSet.getParameter<edm::ParameterSet>("cscCLCT");
  minBXCLCT_ = cscCLCT.getParameter<int>("minBX");
  maxBXCLCT_ = cscCLCT.getParameter<int>("maxBX");
  verboseCLCT_ = cscCLCT.getParameter<int>("verbose");
  minNHitsChamberCLCT_ = cscCLCT.getParameter<int>("minNHitsChamber");

  const auto& cscALCT = pSet.getParameter<edm::ParameterSet>("cscALCT");
  minBXALCT_ = cscALCT.getParameter<int>("minBX");
  maxBXALCT_ = cscALCT.getParameter<int>("maxBX");
  verboseALCT_ = cscALCT.getParameter<int>("verbose");
  minNHitsChamberALCT_ = cscALCT.getParameter<int>("minNHitsChamber");

  const auto& cscLCT = pSet.getParameter<edm::ParameterSet>("cscLCT");
  minBXLCT_ = cscLCT.getParameter<int>("minBX");
  maxBXLCT_ = cscLCT.getParameter<int>("maxBX");
  verboseLCT_ = cscLCT.getParameter<int>("verbose");
  minNHitsChamberLCT_ = cscLCT.getParameter<int>("minNHitsChamber");
  hsFromSimHitMean_ = cscLCT.getParameter<bool>("hsFromSimHitMean");

  const auto& cscMPLCT = pSet.getParameter<edm::ParameterSet>("cscMPLCT");
  minBXMPLCT_ = cscMPLCT.getParameter<int>("minBX");
  maxBXMPLCT_ = cscMPLCT.getParameter<int>("maxBX");
  verboseMPLCT_ = cscMPLCT.getParameter<int>("verbose");
  minNHitsChamberMPLCT_ = cscMPLCT.getParameter<int>("minNHitsChamber");

  gemDigiMatcher_.reset(new GEMDigiMatcher(pSet, std::move(iC)));
  cscDigiMatcher_.reset(new CSCDigiMatcher(pSet, std::move(iC)));

  clctToken_ = iC.consumes<CSCCLCTDigiCollection>(cscCLCT.getParameter<edm::InputTag>("inputTag"));
  alctToken_ = iC.consumes<CSCALCTDigiCollection>(cscALCT.getParameter<edm::InputTag>("inputTag"));
  lctToken_ = iC.consumes<CSCCorrelatedLCTDigiCollection>(cscLCT.getParameter<edm::InputTag>("inputTag"));
  mplctToken_ = iC.consumes<CSCCorrelatedLCTDigiCollection>(cscMPLCT.getParameter<edm::InputTag>("inputTag"));
}

void CSCStubMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  gemDigiMatcher_->init(iEvent, iSetup);
  cscDigiMatcher_->init(iEvent, iSetup);

  iEvent.getByToken(clctToken_, clctsH_);
  iEvent.getByToken(alctToken_, alctsH_);
  iEvent.getByToken(lctToken_, lctsH_);
  iEvent.getByToken(mplctToken_, mplctsH_);

  iSetup.get<MuonGeometryRecord>().get(csc_geom_);
  if (csc_geom_.isValid()) {
    cscGeometry_ = &*csc_geom_;
  } else {
    std::cout << "+++ Info: CSC geometry is unavailable. +++\n";
  }
}

// do the matching
void CSCStubMatcher::match(const SimTrack& t, const SimVertex& v) {
  // match simhits first
  gemDigiMatcher_->match(t, v);
  cscDigiMatcher_->match(t, v);

  const CSCCLCTDigiCollection& clcts = *clctsH_.product();
  const CSCALCTDigiCollection& alcts = *alctsH_.product();
  const CSCCorrelatedLCTDigiCollection& lcts = *lctsH_.product();
  const CSCCorrelatedLCTDigiCollection& mplcts = *mplctsH_.product();

  matchCLCTsToSimTrack(clcts);
  matchALCTsToSimTrack(alcts);
  matchLCTsToSimTrack(lcts);
  matchMPLCTsToSimTrack(mplcts);
}

void CSCStubMatcher::matchCLCTsToSimTrack(const CSCCLCTDigiCollection& clcts) {
  const auto& cathode_ids = cscDigiMatcher_->chamberIdsStrip(0);
  int n_minLayers = 0;
  for (const auto& id : cathode_ids) {
    CSCDetId ch_id(id);
    if (verboseCLCT_) {
      cout << "To check CSC chamber " << ch_id << endl;
    }
    if (cscDigiMatcher_->nLayersWithStripInChamber(id) >= minNHitsChamberCLCT_)
      ++n_minLayers;

    // fill 1 half-strip wide gaps
    const auto& digi_strips = cscDigiMatcher_->stripsInChamber(id, 1);
    if (verboseCLCT_) {
      cout << "clct: digi_strips " << ch_id << " Nlayers " << cscDigiMatcher_->nLayersWithStripInChamber(id) << " ";
      copy(digi_strips.begin(), digi_strips.end(), ostream_iterator<int>(cout, " "));
      cout << endl;
    }

    int ring = ch_id.ring();
    if (ring == 4)
      ring = 1;  //use ME1b id to get CLCTs
    CSCDetId ch_id2(ch_id.endcap(), ch_id.station(), ring, ch_id.chamber(), 0);

    const auto& clcts_in_det = clcts.get(ch_id2);

    for (auto c = clcts_in_det.first; c != clcts_in_det.second; ++c) {
      if (verboseCLCT_)
        cout << "clct " << ch_id << " " << *c << endl;

      if (!c->isValid())
        continue;

      // check that the BX for this stub wasn't too early or too late
      if (c->getBX() < minBXCLCT_ || c->getBX() > maxBXCLCT_)
        continue;

      int half_strip = c->getKeyStrip() + 1;  // CLCT halfstrip numbers start from 0
      if (ch_id.ring() == 4 and ch_id.station() == 1 and half_strip > 128)
        half_strip = half_strip - 128;

      // store all CLCTs in this chamber
      chamber_to_clcts_all_[id].push_back(*c);

      // match by half-strip with the digis
      if (digi_strips.find(half_strip) == digi_strips.end()) {
        if (verboseCLCT_)
          cout << "clctBAD, half_strip " << half_strip << endl;
        continue;
      }
      if (verboseCLCT_)
        cout << "clctGOOD" << endl;

      // store matching CLCTs in this chamber
      chamber_to_clcts_[id].push_back(*c);
    }
    if (chamber_to_clcts_[id].size() > 2) {
      cout << "WARNING!!! too many CLCTs " << chamber_to_clcts_[id].size() << " in " << ch_id << endl;
      for (auto& c : chamber_to_clcts_[id])
        cout << "  " << c << endl;
    }
  }
}

void CSCStubMatcher::matchALCTsToSimTrack(const CSCALCTDigiCollection& alcts) {
  const auto& anode_ids = cscDigiMatcher_->chamberIdsWire(0);
  int n_minLayers = 0;
  for (const auto& id : anode_ids) {
    if (cscDigiMatcher_->nLayersWithWireInChamber(id) >= minNHitsChamberALCT_)
      ++n_minLayers;
    CSCDetId ch_id(id);

    // fill 1 WG wide gaps
    const auto& digi_wgs = cscDigiMatcher_->wiregroupsInChamber(id, 1);
    if (verboseALCT_) {
      cout << "alct: digi_wgs " << ch_id << " ";
      copy(digi_wgs.begin(), digi_wgs.end(), ostream_iterator<int>(cout, " "));
      cout << endl;
    }

    int ring = ch_id.ring();
    if (ring == 4)
      ring = 1;  //use ME1b id to get CLCTs
    CSCDetId ch_id2(ch_id.endcap(), ch_id.station(), ring, ch_id.chamber(), 0);

    const auto& alcts_in_det = alcts.get(ch_id2);
    for (auto a = alcts_in_det.first; a != alcts_in_det.second; ++a) {
      if (!a->isValid())
        continue;

      if (verboseALCT_)
        cout << "alct " << ch_id << " " << *a << endl;

      // check that the BX for stub wasn't too early or too late
      if (a->getBX() < minBXALCT_ || a->getBX() > maxBXALCT_)
        continue;

      int wg = a->getKeyWG() + 1;  // as ALCT wiregroups numbers start from 0

      // store all ALCTs in this chamber
      chamber_to_alcts_all_[id].push_back(*a);

      // match by wiregroup with the digis
      if (digi_wgs.find(wg) == digi_wgs.end()) {
        if (verboseALCT_)
          cout << "alctBAD" << endl;
        continue;
      }
      if (verboseALCT_)
        cout << "alctGOOD" << endl;

      // store matching ALCTs in this chamber
      chamber_to_alcts_[id].push_back(*a);
    }
    if (chamber_to_alcts_[id].size() > 2) {
      cout << "WARNING!!! too many ALCTs " << chamber_to_alcts_[id].size() << " in " << ch_id << endl;
      for (auto& a : chamber_to_alcts_[id])
        cout << "  " << a << endl;
    }
  }
}

void CSCStubMatcher::matchLCTsToSimTrack(const CSCCorrelatedLCTDigiCollection& lcts) {
  // only look for stubs in chambers that already have CLCT and ALCT
  const auto& cathode_ids = chamberIdsAllCLCT(0);
  const auto& anode_ids = chamberIdsAllALCT(0);

  std::set<int> cathode_and_anode_ids;
  std::set_union(cathode_ids.begin(),
                 cathode_ids.end(),
                 anode_ids.begin(),
                 anode_ids.end(),
                 std::inserter(cathode_and_anode_ids, cathode_and_anode_ids.end()));

  for (const auto& id : cathode_and_anode_ids) {
    int iLct = -1;

    CSCDetId ch_id(id);

    //use ME1b id to get LCTs
    int ring = ch_id.ring();
    if (ring == 4)
      ring = 1;
    CSCDetId ch_id2(ch_id.endcap(), ch_id.station(), ring, ch_id.chamber(), 0);

    const auto& lcts_in_det = lcts.get(ch_id2);

    // collect all LCTs in a handy container
    CSCCorrelatedLCTDigiContainer lcts_tmp;
    for (auto lct = lcts_in_det.first; lct != lcts_in_det.second; ++lct) {
      lcts_tmp.push_back(*lct);
    }

    for (const auto& lct : lcts_tmp) {
      iLct++;

      bool lct_matched(false);
      bool lct_clct_match(false);
      bool lct_alct_match(false);
      bool lct_gem1_match(false);
      bool lct_gem2_match(false);

      if (verboseLCT_)
        cout << "in LCT, getCLCT " << lct.getCLCT() << " getALCT " << lct.getALCT() << endl;

      // Check if matched to an CLCT
      for (const auto& p : clctsInChamber(id)) {
        if (p == lct.getCLCT()) {
          lct_clct_match = true;
          break;
        }
      }

      // Check if matched to an ALCT
      for (const auto& p : alctsInChamber(id)) {
        if (p == lct.getALCT()) {
          lct_alct_match = true;
          break;
        }
      }

      // fixME here: double check the timing of GEMPad
      if (ch_id.ring() == 1 and (ch_id.station() == 1 or ch_id.station() == 2)) {
        // Check if matched to an GEM pad L1
        const GEMDetId gemDetIdL1(ch_id.zendcap(), 1, ch_id.station(), 1, ch_id.chamber(), 0);
        for (const auto& p : gemDigiMatcher_->padsInChamber(gemDetIdL1.rawId())) {
          if (p == lct.getGEM1()) {
            lct_gem1_match = true;
            break;
          }
        }

        // Check if matched to an GEM pad L2
        const GEMDetId gemDetIdL2(ch_id.zendcap(), 1, ch_id.station(), 2, ch_id.chamber(), 0);
        for (const auto& p : gemDigiMatcher_->padsInChamber(gemDetIdL2.rawId())) {
          if (p == lct.getGEM2()) {
            lct_gem2_match = true;
            break;
          }
        }
      }

      lct_matched = ((lct_clct_match and lct_alct_match) or (lct_alct_match and lct_gem1_match and lct_gem2_match) or
                     (lct_clct_match and lct_gem1_match and lct_gem2_match));

      if (lct_matched) {
        if (verboseLCT_)
          cout << "this LCT matched to simtrack in chamber " << ch_id << endl;
        chamber_to_lcts_[id].emplace_back(lct);
      }
    }  // lct loop over
  }
}

void CSCStubMatcher::matchMPLCTsToSimTrack(const CSCCorrelatedLCTDigiCollection& mplcts) {
  // match simtrack to MPC LCT by looking only in chambers
  // that already have LCTs matched to this simtrack
  const auto& lcts_ids = chamberIdsLCT(0);

  // loop on the detids
  for (const auto& id : lcts_ids) {
    const auto& mplcts_in_det = mplcts.get(id);

    // loop on the MPC LCTs in this detid
    for (auto lct = mplcts_in_det.first; lct != mplcts_in_det.second; ++lct) {
      if (!lct->isValid())
        continue;

      // std::cout << "MPC Stub ALL " << *lct << std::endl;
      chamber_to_mplcts_all_[id].emplace_back(*lct);

      // check if this stub corresponds with a previously matched stub
      for (const auto& sim_stub : lctsInChamber(id)) {
        if (sim_stub == *lct) {
          chamber_to_mplcts_[id].emplace_back(*lct);
        }
      }
    }
  }
}

std::set<unsigned int> CSCStubMatcher::chamberIdsAllCLCT(int csc_type) const {
  return selectDetIds(chamber_to_clcts_all_, csc_type);
}

std::set<unsigned int> CSCStubMatcher::chamberIdsAllALCT(int csc_type) const {
  return selectDetIds(chamber_to_alcts_all_, csc_type);
}

std::set<unsigned int> CSCStubMatcher::chamberIdsAllLCT(int csc_type) const {
  return selectDetIds(chamber_to_lcts_all_, csc_type);
}

std::set<unsigned int> CSCStubMatcher::chamberIdsAllMPLCT(int csc_type) const {
  return selectDetIds(chamber_to_mplcts_all_, csc_type);
}

std::set<unsigned int> CSCStubMatcher::chamberIdsCLCT(int csc_type) const {
  return selectDetIds(chamber_to_clcts_, csc_type);
}

std::set<unsigned int> CSCStubMatcher::chamberIdsALCT(int csc_type) const {
  return selectDetIds(chamber_to_alcts_, csc_type);
}

std::set<unsigned int> CSCStubMatcher::chamberIdsLCT(int csc_type) const {
  return selectDetIds(chamber_to_lcts_, csc_type);
}

std::set<unsigned int> CSCStubMatcher::chamberIdsMPLCT(int csc_type) const {
  return selectDetIds(chamber_to_mplcts_, csc_type);
}

const CSCCLCTDigiContainer& CSCStubMatcher::allCLCTsInChamber(unsigned int detid) const {
  if (chamber_to_clcts_all_.find(detid) == chamber_to_clcts_all_.end())
    return no_clcts_;
  return chamber_to_clcts_all_.at(detid);
}

const CSCALCTDigiContainer& CSCStubMatcher::allALCTsInChamber(unsigned int detid) const {
  if (chamber_to_alcts_all_.find(detid) == chamber_to_alcts_all_.end())
    return no_alcts_;
  return chamber_to_alcts_all_.at(detid);
}

const CSCCorrelatedLCTDigiContainer& CSCStubMatcher::allLCTsInChamber(unsigned int detid) const {
  if (chamber_to_lcts_all_.find(detid) == chamber_to_lcts_all_.end())
    return no_lcts_;
  return chamber_to_lcts_all_.at(detid);
}

const CSCCorrelatedLCTDigiContainer& CSCStubMatcher::allMPLCTsInChamber(unsigned int detid) const {
  if (chamber_to_mplcts_all_.find(detid) == chamber_to_mplcts_all_.end())
    return no_mplcts_;
  return chamber_to_mplcts_all_.at(detid);
}

const CSCCLCTDigiContainer& CSCStubMatcher::clctsInChamber(unsigned int detid) const {
  if (chamber_to_clcts_.find(detid) == chamber_to_clcts_.end())
    return no_clcts_;
  return chamber_to_clcts_.at(detid);
}

const CSCALCTDigiContainer& CSCStubMatcher::alctsInChamber(unsigned int detid) const {
  if (chamber_to_alcts_.find(detid) == chamber_to_alcts_.end())
    return no_alcts_;
  return chamber_to_alcts_.at(detid);
}

const CSCCorrelatedLCTDigiContainer& CSCStubMatcher::lctsInChamber(unsigned int detid) const {
  if (chamber_to_lcts_.find(detid) == chamber_to_lcts_.end())
    return no_lcts_;
  return chamber_to_lcts_.at(detid);
}

const CSCCorrelatedLCTDigiContainer& CSCStubMatcher::mplctsInChamber(unsigned int detid) const {
  if (chamber_to_mplcts_.find(detid) == chamber_to_mplcts_.end())
    return no_mplcts_;
  return chamber_to_mplcts_.at(detid);
}

CSCCLCTDigi CSCStubMatcher::bestClctInChamber(unsigned int detid) const {
  //sort stubs based on quality
  const auto& input(clctsInChamber(detid));
  int bestQ = 0;
  int index = -1;
  for (unsigned int i = 0; i < input.size(); ++i) {
    int quality = input[i].getQuality();
    if (quality > bestQ) {
      bestQ = quality;
      index = i;
    }
  }
  if (index != -1)
    return input[index];
  return CSCCLCTDigi();
}

CSCALCTDigi CSCStubMatcher::bestAlctInChamber(unsigned int detid) const {
  //sort stubs based on quality
  const auto& input(alctsInChamber(detid));
  int bestQ = 0;
  int index = -1;
  for (unsigned int i = 0; i < input.size(); ++i) {
    int quality = input[i].getQuality();
    if (quality > bestQ) {
      bestQ = quality;
      index = i;
    }
  }
  if (index != -1)
    return input[index];
  return CSCALCTDigi();
}

CSCCorrelatedLCTDigi CSCStubMatcher::bestLctInChamber(unsigned int detid) const {
  //sort stubs based on quality
  const auto& input(lctsInChamber(detid));
  int bestQ = 0;
  int index = -1;
  for (unsigned int i = 0; i < input.size(); ++i) {
    int quality = input[i].getQuality();
    if (quality > bestQ) {
      bestQ = quality;
      index = i;
    }
  }
  if (index != -1)
    return input[index];
  return CSCCorrelatedLCTDigi();
}

float CSCStubMatcher::zpositionOfLayer(unsigned int detid, int layer) const {
  const auto& id = CSCDetId(detid);
  const auto& chamber(cscGeometry_->chamber(id));
  return fabs(chamber->layer(layer)->centerOfStrip(20).z());
}

int CSCStubMatcher::nChambersWithCLCT(int min_quality) const {
  int result = 0;
  const auto& chamber_ids = chamberIdsCLCT();
  for (const auto& id : chamber_ids) {
    int nStubChamber = 0;
    const auto& clcts = clctsInChamber(id);
    for (const auto& clct : clcts) {
      if (!clct.isValid())
        continue;
      if (clct.getQuality() >= min_quality) {
        nStubChamber++;
      }
    }
    if (nStubChamber > 0) {
      ++result;
    }
  }
  return result;
}

int CSCStubMatcher::nChambersWithALCT(int min_quality) const {
  int result = 0;
  const auto& chamber_ids = chamberIdsALCT();
  for (const auto& id : chamber_ids) {
    int nStubChamber = 0;
    const auto& alcts = alctsInChamber(id);
    for (const auto& alct : alcts) {
      if (!alct.isValid())
        continue;
      if (alct.getQuality() >= min_quality) {
        nStubChamber++;
      }
    }
    if (nStubChamber > 0) {
      ++result;
    }
  }
  return result;
}

int CSCStubMatcher::nChambersWithLCT(int min_quality) const {
  int result = 0;
  const auto& chamber_ids = chamberIdsLCT();
  for (const auto& id : chamber_ids) {
    int nStubChamber = 0;
    const auto& lcts = lctsInChamber(id);
    for (const auto& lct : lcts) {
      if (!lct.isValid())
        continue;
      if (lct.getQuality() >= min_quality) {
        nStubChamber++;
      }
    }
    if (nStubChamber > 0) {
      ++result;
    }
  }
  return result;
}

int CSCStubMatcher::nChambersWithMPLCT(int min_quality) const {
  int result = 0;
  const auto& chamber_ids = chamberIdsMPLCT();
  for (const auto& id : chamber_ids) {
    int nStubChamber = 0;
    const auto& mplcts = mplctsInChamber(id);
    for (const auto& mplct : mplcts) {
      if (!mplct.isValid())
        continue;
      if (mplct.getQuality() >= min_quality) {
        nStubChamber++;
      }
    }
    if (nStubChamber > 0) {
      ++result;
    }
  }
  return result;
}

bool CSCStubMatcher::lctInChamber(const CSCDetId& id, const CSCCorrelatedLCTDigi& lct) const {
  for (const auto& stub : lctsInChamber(id.rawId())) {
    if (stub == lct)
      return true;
  }
  return false;
}

GlobalPoint CSCStubMatcher::getGlobalPosition(unsigned int rawId, const CSCCorrelatedLCTDigi& lct) const {
  CSCDetId cscId(rawId);
  CSCDetId key_id(cscId.endcap(), cscId.station(), cscId.ring(), cscId.chamber(), CSCConstants::KEY_CLCT_LAYER);
  const auto& chamber = cscGeometry_->chamber(cscId);
  float fractional_strip = lct.getFractionalStrip();
  const auto& layer_geo = chamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
  // LCT::getKeyWG() also starts from 0
  float wire = layer_geo->middleWireOfGroup(lct.getKeyWG() + 1);
  const LocalPoint& csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
  const GlobalPoint& csc_gp = cscGeometry_->idToDet(key_id)->surface().toGlobal(csc_intersect);
  return csc_gp;
}
