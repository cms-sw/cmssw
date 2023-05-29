#include "Validation/MuonCSCDigis/interface/CSCStubMatcher.h"
#include "DataFormats/CSCDigi/interface/CSCConstants.h"
#include <algorithm>

using namespace std;

CSCStubMatcher::CSCStubMatcher(const edm::ParameterSet& pSet, edm::ConsumesCollector&& iC) {
  useGEMs_ = pSet.getParameter<bool>("useGEMs");

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
  matchTypeTightLCT_ = cscLCT.getParameter<bool>("matchTypeTight");
  verboseLCT_ = cscLCT.getParameter<int>("verbose");
  minNHitsChamberLCT_ = cscLCT.getParameter<int>("minNHitsChamber");
  addGhostLCTs_ = cscLCT.getParameter<bool>("addGhosts");

  const auto& cscMPLCT = pSet.getParameter<edm::ParameterSet>("cscMPLCT");
  minBXMPLCT_ = cscMPLCT.getParameter<int>("minBX");
  maxBXMPLCT_ = cscMPLCT.getParameter<int>("maxBX");
  verboseMPLCT_ = cscMPLCT.getParameter<int>("verbose");
  minNHitsChamberMPLCT_ = cscMPLCT.getParameter<int>("minNHitsChamber");

  if (useGEMs_)
    gemDigiMatcher_.reset(new GEMDigiMatcher(pSet, std::move(iC)));
  cscDigiMatcher_.reset(new CSCDigiMatcher(pSet, std::move(iC)));

  clctInputTag_ = cscCLCT.getParameter<edm::InputTag>("inputTag");
  alctInputTag_ = cscALCT.getParameter<edm::InputTag>("inputTag");
  lctInputTag_ = cscLCT.getParameter<edm::InputTag>("inputTag");
  mplctInputTag_ = cscMPLCT.getParameter<edm::InputTag>("inputTag");

  clctToken_ = iC.consumes<CSCCLCTDigiCollection>(clctInputTag_);
  alctToken_ = iC.consumes<CSCALCTDigiCollection>(alctInputTag_);
  lctToken_ = iC.consumes<CSCCorrelatedLCTDigiCollection>(lctInputTag_);
  mplctToken_ = iC.consumes<CSCCorrelatedLCTDigiCollection>(mplctInputTag_);

  geomToken_ = iC.esConsumes<CSCGeometry, MuonGeometryRecord>();
}

void CSCStubMatcher::init(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (useGEMs_)
    gemDigiMatcher_->init(iEvent, iSetup);
  cscDigiMatcher_->init(iEvent, iSetup);

  iEvent.getByToken(clctToken_, clctsH_);
  iEvent.getByToken(alctToken_, alctsH_);
  iEvent.getByToken(lctToken_, lctsH_);
  iEvent.getByToken(mplctToken_, mplctsH_);

  cscGeometry_ = &iSetup.getData(geomToken_);
}

// do the matching
void CSCStubMatcher::match(const SimTrack& t, const SimVertex& v) {
  // match simhits first
  if (useGEMs_)
    gemDigiMatcher_->match(t, v);
  cscDigiMatcher_->match(t, v);

  const CSCCLCTDigiCollection& clcts = *clctsH_.product();
  const CSCALCTDigiCollection& alcts = *alctsH_.product();
  const CSCCorrelatedLCTDigiCollection& lcts = *lctsH_.product();
  const CSCCorrelatedLCTDigiCollection& mplcts = *mplctsH_.product();

  // clear collections
  clear();

  if (!alctsH_.isValid()) {
    edm::LogError("CSCStubMatcher") << "Cannot get ALCTs with label " << alctInputTag_.encode();
  } else {
    matchALCTsToSimTrack(alcts);
  }

  if (!clctsH_.isValid()) {
    edm::LogError("CSCStubMatcher") << "Cannot get CLCTs with label " << clctInputTag_.encode();
  } else {
    matchCLCTsToSimTrack(clcts);
  }

  if (!lctsH_.isValid()) {
    edm::LogError("CSCStubMatcher") << "Cannot get LCTs with label " << lctInputTag_.encode();
  } else {
    matchLCTsToSimTrack(lcts);
  }

  if (!mplctsH_.isValid()) {
    edm::LogError("CSCStubMatcher") << "Cannot get MPLCTs with label " << mplctInputTag_.encode();
  } else {
    matchMPLCTsToSimTrack(mplcts);
  }
}

void CSCStubMatcher::matchCLCTsToSimTrack(const CSCCLCTDigiCollection& clcts) {
  const auto& cathode_ids = cscDigiMatcher_->chamberIdsStrip(0);

  for (const auto& id : cathode_ids) {
    CSCDetId ch_id(id);
    if (verboseCLCT_) {
      edm::LogInfo("CSCStubMatcher") << "To check CSC chamber " << ch_id;
    }

    int ring = ch_id.ring();

    // do not consider CSCs with too few hits
    if (cscDigiMatcher_->nLayersWithStripInChamber(ch_id) < minNHitsChamberCLCT_)
      continue;

    // get the comparator digis in this chamber
    std::vector<CSCComparatorDigiContainer> comps;
    for (int ilayer = CSCDetId::minLayerId(); ilayer <= CSCDetId::maxLayerId(); ilayer++) {
      CSCDetId layerid(ch_id.endcap(), ch_id.station(), ring, ch_id.chamber(), ilayer);
      comps.push_back(cscDigiMatcher_->comparatorDigisInDetId(layerid));
    }

    // print out the digis
    if (verboseCLCT_) {
      edm::LogInfo("CSCStubMatcher") << "clct: comparators " << ch_id;
      int layer = 0;
      for (const auto& p : comps) {
        layer++;
        for (const auto& q : p) {
          edm::LogInfo("CSCStubMatcher") << "L" << layer << " " << q << " " << q.getHalfStrip() << " ";
        }
      }
    }

    //use ME1b id to get CLCTs
    const bool isME1a(ch_id.station() == 1 and ch_id.ring() == 4);
    if (isME1a)
      ring = 1;
    CSCDetId ch_id2(ch_id.endcap(), ch_id.station(), ring, ch_id.chamber(), 0);
    auto id2 = ch_id2.rawId();  // CLCTs should be sorted into the det of the CLCTs.

    const auto& clcts_in_det = clcts.get(ch_id2);

    for (auto c = clcts_in_det.first; c != clcts_in_det.second; ++c) {
      if (verboseCLCT_)
        edm::LogInfo("CSCStubMatcher") << "clct " << ch_id2 << " " << *c;

      if (!c->isValid())
        continue;

      // check that the BX for this stub wasn't too early or too late
      if (c->getBX() < minBXCLCT_ || c->getBX() > maxBXCLCT_)
        continue;

      // store all CLCTs in this chamber
      chamber_to_clcts_all_[id2].push_back(*c);

      // check that at least 3 comparator digis were matched!
      int nMatches = 0;
      int layer = 0;
      for (const auto& p : comps) {
        layer++;
        for (const auto& q : p) {
          if (verboseCLCT_)
            edm::LogInfo("CSCStubMatcher") << "L" << layer << " " << q << " " << q.getHalfStrip() << " " << std::endl;
          for (const auto& clctComp : (*c).getHits()[layer - 1]) {
            if (clctComp == 65535)
              continue;
            if (verboseCLCT_) {
              edm::LogInfo("CSCStubMatcher") << "\t" << clctComp << " ";
            }
            if (q.getHalfStrip() == clctComp or (isME1a and q.getHalfStrip() + 128 == clctComp)) {
              nMatches++;
              if (verboseCLCT_) {
                edm::LogInfo("CSCStubMatcher") << "\t\tnMatches " << nMatches << std::endl;
              }
            }
          }
        }
      }

      // require at least 3 good matches
      if (nMatches < 3)
        continue;

      if (verboseCLCT_)
        edm::LogInfo("CSCStubMatcher") << "clctGOOD";

      // store matching CLCTs in this chamber
      if (std::find(chamber_to_clcts_[id2].begin(), chamber_to_clcts_[id2].end(), *c) == chamber_to_clcts_[id2].end()) {
        chamber_to_clcts_[id2].push_back(*c);
      }
    }
    if (chamber_to_clcts_[id2].size() > 2) {
      edm::LogInfo("CSCStubMatcher") << "WARNING!!! too many CLCTs " << chamber_to_clcts_[id2].size() << " in "
                                     << ch_id2;
      for (auto& c : chamber_to_clcts_[id2])
        edm::LogInfo("CSCStubMatcher") << "  " << c;
    }
  }
}

void CSCStubMatcher::matchALCTsToSimTrack(const CSCALCTDigiCollection& alcts) {
  const auto& anode_ids = cscDigiMatcher_->chamberIdsWire(0);
  for (const auto& id : anode_ids) {
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
      ring = 1;  //use ME1b id to get ALCTs
    CSCDetId ch_id2(ch_id.endcap(), ch_id.station(), ring, ch_id.chamber(), 0);
    auto id2 = ch_id2.rawId();  // ALCTs should be sorted into the det of the ALCTs.

    const auto& alcts_in_det = alcts.get(ch_id2);
    for (auto a = alcts_in_det.first; a != alcts_in_det.second; ++a) {
      if (!a->isValid())
        continue;

      if (verboseALCT_)
        edm::LogInfo("CSCStubMatcher") << "alct " << ch_id << " " << *a;

      // check that the BX for stub wasn't too early or too late
      if (a->getBX() < minBXALCT_ || a->getBX() > maxBXALCT_)
        continue;

      int wg = a->getKeyWG() + 1;  // as ALCT wiregroups numbers start from 0

      // store all ALCTs in this chamber
      chamber_to_alcts_all_[id2].push_back(*a);

      // match by wiregroup with the digis
      if (digi_wgs.find(wg) == digi_wgs.end()) {
        continue;
      }
      if (verboseALCT_)
        edm::LogInfo("CSCStubMatcher") << "alctGOOD";

      // store matching ALCTs in this chamber
      if (std::find(chamber_to_alcts_[id2].begin(), chamber_to_alcts_[id2].end(), *a) == chamber_to_alcts_[id2].end()) {
        chamber_to_alcts_[id2].push_back(*a);
      }
    }
    if (chamber_to_alcts_[id2].size() > 2) {
      edm::LogInfo("CSCStubMatcher") << "WARNING!!! too many ALCTs " << chamber_to_alcts_[id2].size() << " in "
                                     << ch_id;
      for (auto& a : chamber_to_alcts_[id2])
        edm::LogInfo("CSCStubMatcher") << "  " << a;
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
    CSCDetId ch_id(id);

    //use ME1b id to get LCTs
    int ring = ch_id.ring();
    if (ring == 4)
      ring = 1;
    CSCDetId ch_id2(ch_id.endcap(), ch_id.station(), ring, ch_id.chamber(), 0);
    auto id2 = ch_id2.rawId();  // LCTs should be sorted into the det of the LCTs.

    const auto& lcts_in_det = lcts.get(ch_id2);

    std::map<int, CSCCorrelatedLCTDigiContainer> bx_to_lcts;

    // collect all valid LCTs in a handy container
    CSCCorrelatedLCTDigiContainer lcts_tmp;
    for (auto lct = lcts_in_det.first; lct != lcts_in_det.second; ++lct) {
      if (!lct->isValid())
        continue;
      lcts_tmp.push_back(*lct);
      int bx = lct->getBX();
      bx_to_lcts[bx].push_back(*lct);

      // Add ghost LCTs when there are two in bx
      // and the two don't share half-strip or wiregroup
      if (bx_to_lcts[bx].size() == 2 and addGhostLCTs_) {
        auto lct11 = bx_to_lcts[bx][0];
        auto lct22 = bx_to_lcts[bx][1];
        addGhostLCTs(lct11, lct22, lcts_tmp);
      }
    }

    for (const auto& lct : lcts_tmp) {
      bool lct_clct_match(false);
      bool lct_alct_match(false);
      bool lct_gem1_match(false);
      bool lct_gem2_match(false);

      if (verboseLCT_) {
        edm::LogInfo("CSCStubMatcher") << ch_id << " " << ch_id2;
        edm::LogInfo("CSCStubMatcher") << lct;
        edm::LogInfo("CSCStubMatcher") << "getCLCT " << lct.getCLCT() << "\ngetALCT " << lct.getALCT() << "\ngetGEM1 "
                                       << lct.getGEM1() << "\ngetGEM2 " << lct.getGEM2();
      }
      // Check if matched to an CLCT
      for (const auto& p : clctsInChamber(id)) {
        if (p == lct.getCLCT()) {
          lct_clct_match = true;
          if (verboseLCT_)
            edm::LogInfo("CSCStubMatcher") << "\t...lct_clct_match";
          break;
        }
      }

      // Check if matched to an ALCT
      for (const auto& p : alctsInChamber(id)) {
        if (p == lct.getALCT()) {
          lct_alct_match = true;
          if (verboseLCT_)
            edm::LogInfo("CSCStubMatcher") << "\t...lct_alct_match";
          break;
        }
      }

      if (useGEMs_) {
        // fixME here: double check the timing of GEMPad
        if (ch_id.ring() == 1 and (ch_id.station() == 1 or ch_id.station() == 2)) {
          // Check if matched to an GEM pad L1
          const GEMDetId gemDetIdL1(ch_id.zendcap(), 1, ch_id.station(), 1, ch_id.chamber(), 0);
          for (const auto& p : gemDigiMatcher_->padsInChamber(gemDetIdL1.rawId())) {
            if (p == lct.getGEM1()) {
              lct_gem1_match = true;
              if (verboseLCT_)
                edm::LogInfo("CSCStubMatcher") << "\t...lct_gem1_match";
              break;
            }
          }

          // Check if matched to an GEM pad L2
          const GEMDetId gemDetIdL2(ch_id.zendcap(), 1, ch_id.station(), 2, ch_id.chamber(), 0);
          for (const auto& p : gemDigiMatcher_->padsInChamber(gemDetIdL2.rawId())) {
            if (p == lct.getGEM2()) {
              lct_gem2_match = true;
              if (verboseLCT_)
                edm::LogInfo("CSCStubMatcher") << "\t...lct_gem2_match";
              break;
            }
          }
        }
      }

      const bool alct_clct = lct_clct_match and lct_alct_match;
      const bool alct_gem = lct_alct_match and lct_gem1_match and lct_gem2_match;
      const bool clct_gem = lct_clct_match and lct_gem1_match and lct_gem2_match;

      bool lct_tight_matched = alct_clct or alct_gem or clct_gem;
      bool lct_loose_matched = lct_clct_match or lct_alct_match;
      bool lct_matched = matchTypeTightLCT_ ? lct_tight_matched : lct_loose_matched;

      if (lct_matched) {
        if (verboseLCT_)
          edm::LogInfo("CSCStubMatcher") << "...was matched";
        if (std::find(chamber_to_lcts_[id2].begin(), chamber_to_lcts_[id2].end(), lct) == chamber_to_lcts_[id2].end()) {
          chamber_to_lcts_[id2].emplace_back(lct);
        }
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

      chamber_to_mplcts_all_[id].emplace_back(*lct);

      // check if this stub corresponds with a previously matched stub
      for (const auto& sim_stub : lctsInChamber(id)) {
        if (sim_stub == *lct) {
          if (std::find(chamber_to_mplcts_[id].begin(), chamber_to_mplcts_[id].end(), *lct) ==
              chamber_to_mplcts_[id].end()) {
            chamber_to_mplcts_[id].emplace_back(*lct);
          }
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
  CSCDetId keyId(cscId.endcap(), cscId.station(), cscId.ring(), cscId.chamber(), CSCConstants::KEY_CLCT_LAYER);
  float fractional_strip = lct.getFractionalStrip();
  // case ME1/1
  if (cscId.station() == 1 and (cscId.ring() == 4 || cscId.ring() == 1)) {
    int ring = 1;  // Default to ME1/b
    if (lct.getStrip() > CSCConstants::MAX_HALF_STRIP_ME1B) {
      ring = 4;  // Change to ME1/a if the HalfStrip Number exceeds the range of ME1/b
      fractional_strip -= CSCConstants::NUM_STRIPS_ME1B;
    }
    CSCDetId cscId_(cscId.endcap(), cscId.station(), ring, cscId.chamber(), cscId.layer());
    cscId = cscId_;
  }
  // regular cases
  const auto& chamber = cscGeometry_->chamber(cscId);
  const auto& layer_geo = chamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
  // LCT::getKeyWG() also starts from 0
  float wire = layer_geo->middleWireOfGroup(lct.getKeyWG() + 1);
  const LocalPoint& csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
  const GlobalPoint& csc_gp = cscGeometry_->idToDet(keyId)->surface().toGlobal(csc_intersect);
  return csc_gp;
}

void CSCStubMatcher::clear() {
  chamber_to_clcts_all_.clear();
  chamber_to_alcts_all_.clear();
  chamber_to_lcts_all_.clear();
  chamber_to_mplcts_all_.clear();

  chamber_to_clcts_.clear();
  chamber_to_alcts_.clear();
  chamber_to_lcts_.clear();
  chamber_to_mplcts_.clear();
}

void CSCStubMatcher::addGhostLCTs(const CSCCorrelatedLCTDigi& lct11,
                                  const CSCCorrelatedLCTDigi& lct22,
                                  CSCCorrelatedLCTDigiContainer& lcts_tmp) const {
  int wg1 = lct11.getKeyWG();
  int wg2 = lct22.getKeyWG();
  int hs1 = lct11.getStrip();
  int hs2 = lct22.getStrip();

  if (!(wg1 == wg2 || hs1 == hs2)) {
    // flip the ALCTs
    CSCCorrelatedLCTDigi lct12 = lct11;
    lct12.setWireGroup(wg2);
    lct12.setALCT(lct22.getALCT());
    lct12.setCLCT(lct11.getCLCT());
    lcts_tmp.push_back(lct12);

    CSCCorrelatedLCTDigi lct21 = lct22;
    lct21.setWireGroup(wg1);
    lct21.setALCT(lct11.getALCT());
    lct21.setCLCT(lct22.getCLCT());
    lcts_tmp.push_back(lct21);
  }
}
