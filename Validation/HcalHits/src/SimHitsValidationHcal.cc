#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/HcalCommonData/interface/HcalHitRelabeller.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Validation/HcalHits/interface/SimHitsValidationHcal.h"

//#define DebugLog

SimHitsValidationHcal::SimHitsValidationHcal(const edm::ParameterSet &ps) {
  g4Label_ = ps.getParameter<std::string>("ModuleLabel");
  hcalHits_ = ps.getParameter<std::string>("HitCollection");
  verbose_ = ps.getParameter<bool>("Verbose");
  testNumber_ = ps.getParameter<bool>("TestNumber");

  tok_hits_ = consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, hcalHits_));

  edm::LogVerbatim("HitsValidationHcal") << "Module Label: " << g4Label_ << "   Hits: " << hcalHits_
                                         << " TestNumbering " << testNumber_;
}

SimHitsValidationHcal::~SimHitsValidationHcal() {}

void SimHitsValidationHcal::bookHistograms(DQMStore::IBooker &ib, edm::Run const &run, edm::EventSetup const &es) {
  edm::ESHandle<HcalDDDRecConstants> pHRNDC;
  es.get<HcalRecNumberingRecord>().get(pHRNDC);
  hcons = &(*pHRNDC);
  maxDepthHB_ = hcons->getMaxDepth(0);
  maxDepthHE_ = hcons->getMaxDepth(1);
  maxDepthHF_ = hcons->getMaxDepth(2);
  maxDepthHO_ = hcons->getMaxDepth(3);

  // Get Phi segmentation from geometry, use the max phi number so that all iphi
  // values are included.

  int NphiMax = hcons->getNPhi(0);

  NphiMax = (hcons->getNPhi(1) > NphiMax ? hcons->getNPhi(1) : NphiMax);
  NphiMax = (hcons->getNPhi(2) > NphiMax ? hcons->getNPhi(2) : NphiMax);
  NphiMax = (hcons->getNPhi(3) > NphiMax ? hcons->getNPhi(3) : NphiMax);

  // Center the iphi bins on the integers
  float iphi_min = 0.5;
  float iphi_max = NphiMax + 0.5;
  int iphi_bins = (int)(iphi_max - iphi_min);

  int iEtaHBMax = hcons->getEtaRange(0).second;
  int iEtaHEMax = std::max(hcons->getEtaRange(1).second, 1);
  int iEtaHFMax = hcons->getEtaRange(2).second;
  int iEtaHOMax = hcons->getEtaRange(3).second;

  // Retain classic behavior, all plots have same ieta range.
  // Comment out	code to	allow each subdetector to have its on range

  int iEtaMax = (iEtaHBMax > iEtaHEMax ? iEtaHBMax : iEtaHEMax);
  iEtaMax = (iEtaMax > iEtaHFMax ? iEtaMax : iEtaHFMax);
  iEtaMax = (iEtaMax > iEtaHOMax ? iEtaMax : iEtaHOMax);

  iEtaHBMax = iEtaMax;
  iEtaHEMax = iEtaMax;
  iEtaHFMax = iEtaMax;
  iEtaHOMax = iEtaMax;

  // Give an empty bin around the subdet ieta range to make it clear that all
  // ieta rings have been included float ieta_min_HB = -iEtaHBMax - 1.5; float
  // ieta_max_HB = iEtaHBMax + 1.5; int ieta_bins_HB = (int) (ieta_max_HB -
  // ieta_min_HB);

  // float ieta_min_HE = -iEtaHEMax - 1.5;
  // float ieta_max_HE = iEtaHEMax + 1.5;
  // int ieta_bins_HE = (int) (ieta_max_HE - ieta_min_HE);

  // float ieta_min_HF = -iEtaHFMax - 1.5;
  // float ieta_max_HF = iEtaHFMax + 1.5;
  // int ieta_bins_HF = (int) (ieta_max_HF - ieta_min_HF);

  // float ieta_min_HO = -iEtaHOMax - 1.5;
  // float ieta_max_HO = iEtaHOMax + 1.5;
  // int ieta_bins_HO = (int) (ieta_max_HO - ieta_min_HO);

#ifdef DebugLog
  edm::LogVerbatim("HitsValidationHcal") << " Maximum Depths HB:" << maxDepthHB_ << " HE:" << maxDepthHE_
                                         << " HO:" << maxDepthHO_ << " HF:" << maxDepthHF_;
#endif
  std::vector<std::pair<std::string, std::string>> divisions = getHistogramTypes();

  edm::LogVerbatim("HitsValidationHcal") << "Booking the Histograms";
  ib.setCurrentFolder("HcalHitsV/SimHitsValidationHcal");

  // Histograms for Hits

  std::string name, title;
  for (unsigned int i = 0; i < types.size(); ++i) {
    etaRange limit = getLimits(types[i]);
    name = "HcalHitEta" + divisions[i].first;
    title = "Hit energy as a function of eta tower index in " + divisions[i].second;
    meHcalHitEta_.push_back(ib.book1D(name, title, limit.bins, limit.low, limit.high));

    name = "HcalHitTimeAEta" + divisions[i].first;
    title = "Hit time as a function of eta tower index in" + divisions[i].second;
    meHcalHitTimeEta_.push_back(ib.book1D(name, title, limit.bins, limit.low, limit.high));

    name = "HcalHitE25" + divisions[i].first;
    title = "Energy in time window 0 to 25 for a tower in " + divisions[i].second;
    meHcalEnergyl25_.push_back(
        ib.book2D(name, title, limit.bins, limit.low, limit.high, iphi_bins, iphi_min, iphi_max));

    name = "HcalHitE50" + divisions[i].first;
    title = "Energy in time window 0 to 50 for a tower in " + divisions[i].second;
    meHcalEnergyl50_.push_back(
        ib.book2D(name, title, limit.bins, limit.low, limit.high, iphi_bins, iphi_min, iphi_max));

    name = "HcalHitE100" + divisions[i].first;
    title = "Energy in time window 0 to 100 for a tower in " + divisions[i].second;
    meHcalEnergyl100_.push_back(
        ib.book2D(name, title, limit.bins, limit.low, limit.high, iphi_bins, iphi_min, iphi_max));

    name = "HcalHitE250" + divisions[i].first;
    title = "Energy in time window 0 to 250 for a tower in " + divisions[i].second;
    meHcalEnergyl250_.push_back(
        ib.book2D(name, title, limit.bins, limit.low, limit.high, iphi_bins, iphi_min, iphi_max));
  }

  name = "Energy_HB";
  meEnergy_HB = ib.book1D(name, name, 100, 0, 1);
  name = "Energy_HE";
  meEnergy_HE = ib.book1D(name, name, 100, 0, 1);
  name = "Energy_HO";
  meEnergy_HO = ib.book1D(name, name, 100, 0, 1);
  name = "Energy_HF";
  meEnergy_HF = ib.book1D(name, name, 100, 0, 50);

  name = "Time_HB";
  metime_HB = ib.book1D(name, name, 300, -150, 150);
  name = "Time_HE";
  metime_HE = ib.book1D(name, name, 300, -150, 150);
  name = "Time_HO";
  metime_HO = ib.book1D(name, name, 300, -150, 150);
  name = "Time_HF";
  metime_HF = ib.book1D(name, name, 300, -150, 150);

  name = "Time_Enweighted_HB";
  metime_enweighted_HB = ib.book1D(name, name, 300, -150, 150);
  name = "Time_Enweighted_HE";
  metime_enweighted_HE = ib.book1D(name, name, 300, -150, 150);
  name = "Time_Enweighted_HO";
  metime_enweighted_HO = ib.book1D(name, name, 300, -150, 150);
  name = "Time_Enweighted_HF";
  metime_enweighted_HF = ib.book1D(name, name, 300, -150, 150);
}

void SimHitsValidationHcal::analyze(const edm::Event &e, const edm::EventSetup &) {
#ifdef DebugLog
  edm::LogVerbatim("HitsValidationHcal") << "Run = " << e.id().run() << " Event = " << e.id().event();
#endif
  std::vector<PCaloHit> caloHits;
  edm::Handle<edm::PCaloHitContainer> hitsHcal;

  bool getHits = false;
  e.getByToken(tok_hits_, hitsHcal);
  if (hitsHcal.isValid())
    getHits = true;
#ifdef DebugLog
  edm::LogVerbatim("HitsValidationHcal") << "HitsValidationHcal.: Input flags Hits " << getHits;
#endif
  if (getHits) {
    caloHits.insert(caloHits.end(), hitsHcal->begin(), hitsHcal->end());
#ifdef DebugLog
    edm::LogVerbatim("HitsValidationHcal") << "testNumber_:" << testNumber_;
#endif
    if (testNumber_) {
      for (unsigned int i = 0; i < caloHits.size(); ++i) {
        unsigned int id_ = caloHits[i].id();
        HcalDetId hid = HcalHitRelabeller::relabel(id_, hcons);
        caloHits[i].setID(hid.rawId());
#ifdef DebugLog
        edm::LogVerbatim("HitsValidationHcal") << "Hit[" << i << "] " << hid;
#endif
      }
    }
#ifdef DebugLog
    edm::LogVerbatim("HitsValidationHcal") << "HitsValidationHcal: Hit buffer " << caloHits.size();
#endif
    analyzeHits(caloHits);
  }
}

void SimHitsValidationHcal::analyzeHits(std::vector<PCaloHit> &hits) {
  int nHit = hits.size();
  double entotHB = 0, entotHE = 0, entotHF = 0, entotHO = 0;
  double timetotHB = 0, timetotHE = 0, timetotHF = 0, timetotHO = 0;
  int nHB = 0, nHE = 0, nHO = 0, nHF = 0;

  std::map<std::pair<HcalDetId, unsigned int>, energysum> map_try;
  map_try.clear();
  std::map<std::pair<HcalDetId, unsigned int>, energysum>::iterator itr;

  for (int i = 0; i < nHit; i++) {
    double energy = hits[i].energy();
    double time = hits[i].time();
    HcalDetId id = HcalDetId(hits[i].id());
    int itime = (int)(time);
    int subdet = id.subdet();
    int depth = id.depth();
    int eta = id.ieta();
    unsigned int dep = hits[i].depth();

    std::pair<int, int> types = histId(subdet, eta, depth, dep);
    if (subdet == static_cast<int>(HcalBarrel)) {
      entotHB += energy;
      timetotHB += time;
      nHB++;
    } else if (subdet == static_cast<int>(HcalEndcap)) {
      entotHE += energy;
      timetotHE += time;
      nHE++;
    } else if (subdet == static_cast<int>(HcalOuter)) {
      entotHO += energy;
      timetotHO += time;
      nHO++;
    } else if (subdet == static_cast<int>(HcalForward)) {
      entotHF += energy;
      timetotHF += time;
      nHF++;
    }

    std::pair<HcalDetId, unsigned int> id0(id, dep);
    energysum ensum;
    if (map_try.count(id0) != 0)
      ensum = map_try[id0];
    if (itime < 250) {
      ensum.e250 += energy;
      if (itime < 100) {
        ensum.e100 += energy;
        if (itime < 50) {
          ensum.e50 += energy;
          if (itime < 25)
            ensum.e25 += energy;
        }
      }
    }
    map_try[id0] = ensum;

#ifdef DebugLog
    edm::LogVerbatim("HitsValidationHcal")
        << "Hit[" << i << "] ID " << std::dec << " " << id << std::dec << " Det " << id.det() << " Sub " << subdet
        << " depth " << depth << " depthX " << dep << " Eta " << eta << " Phi " << id.iphi() << " E " << energy
        << " time " << time << " type " << types.first << " " << types.second;
#endif

    double etax = eta - 0.5;
    if (eta < 0)
      etax += 1;
    if (types.first >= 0) {
      meHcalHitEta_[types.first]->Fill(etax, energy);
      meHcalHitTimeEta_[types.first]->Fill(etax, time);
    }
    if (types.second >= 0) {
      meHcalHitEta_[types.second]->Fill(etax, energy);
      meHcalHitTimeEta_[types.second]->Fill(etax, time);
    }
  }

  meEnergy_HB->Fill(entotHB);
  meEnergy_HE->Fill(entotHE);
  meEnergy_HF->Fill(entotHF);
  meEnergy_HO->Fill(entotHO);

  metime_HB->Fill(timetotHB);
  metime_HE->Fill(timetotHE);
  metime_HF->Fill(timetotHF);
  metime_HO->Fill(timetotHO);

  metime_enweighted_HB->Fill(timetotHB, entotHB);
  metime_enweighted_HE->Fill(timetotHE, entotHE);
  metime_enweighted_HF->Fill(timetotHF, entotHF);
  metime_enweighted_HO->Fill(timetotHO, entotHO);

  for (itr = map_try.begin(); itr != map_try.end(); ++itr) {
    HcalDetId id = (*itr).first.first;
    energysum ensum = (*itr).second;
    std::pair<int, int> types = histId((int)(id.subdet()), id.ieta(), id.depth(), (*itr).first.second);
    int eta = id.ieta();
    int phi = id.iphi();
    double etax = eta - 0.5;
    double phix = phi - 0.5;
    if (types.first >= 0) {
      meHcalEnergyl25_[types.first]->Fill(etax, phix, ensum.e25);
      meHcalEnergyl50_[types.first]->Fill(etax, phix, ensum.e50);
      meHcalEnergyl100_[types.first]->Fill(etax, phix, ensum.e100);
      meHcalEnergyl250_[types.first]->Fill(etax, phix, ensum.e250);
    }
    if (types.second >= 0) {
      meHcalEnergyl25_[types.second]->Fill(etax, phix, ensum.e25);
      meHcalEnergyl50_[types.second]->Fill(etax, phix, ensum.e50);
      meHcalEnergyl100_[types.second]->Fill(etax, phix, ensum.e100);
      meHcalEnergyl250_[types.second]->Fill(etax, phix, ensum.e250);
    }

#ifdef DebugLog
    edm::LogVerbatim("HitsValidationHcal")
        << " energy of tower =" << (*itr).first.first << " in time 25ns is == " << (*itr).second.e25
        << " in time 25-50ns == " << (*itr).second.e50 << " in time 50-100ns == " << (*itr).second.e100
        << " in time 100-250 ns == " << (*itr).second.e250;
#endif
  }
}

SimHitsValidationHcal::etaRange SimHitsValidationHcal::getLimits(idType type) {
  int bins;
  std::pair<int, int> range;
  double low, high;

  if (type.subdet == HcalBarrel) {
    range = hcons->getEtaRange(0);
    low = -range.second;
    high = range.second;
    bins = (high - low);
  } else if (type.subdet == HcalEndcap) {
    range = hcons->getEtaRange(1);
    bins = range.second - range.first;
    if (type.z == 1) {
      low = range.first;
      high = range.second;
    } else {
      low = -range.second;
      high = -range.first;
    }
  } else if (type.subdet == HcalOuter) {
    range = hcons->getEtaRange(3);
    low = -range.second;
    high = range.second;
    bins = high - low;
  } else if (type.subdet == HcalForward) {
    range = hcons->getEtaRange(2);
    bins = range.second - range.first;
    if (type.z == 1) {
      low = range.first;
      high = range.second;
    } else {
      low = -range.second;
      high = -range.first;
    }
  } else {
    bins = 82;
    low = -41;
    high = 41;
  }
#ifdef DebugLog
  edm::LogVerbatim("HitsValidationHcal") << "Subdetector:" << type.subdet << " z:" << type.z
                                         << " range.first:" << range.first << " and second:" << range.second;
  edm::LogVerbatim("HitsValidationHcal") << "bins: " << bins << " low:" << low << " high:" << high;
#endif
  return SimHitsValidationHcal::etaRange(bins, low, high);
}

std::pair<int, int> SimHitsValidationHcal::histId(int subdet, int eta, int depth, unsigned int dep) {
  int id1(-1), id2(-1);
  for (unsigned int k = 0; k < types.size(); ++k) {
    if (subdet == HcalForward) {
      if (subdet == (int)(types[k].subdet) && depth == types[k].depth1 && eta * types[k].z > 0 &&
          dep == (unsigned int)(types[k].depth2)) {
        id1 = k;
        break;
      }
    } else if (subdet == HcalEndcap) {
      if (subdet == (int)(types[k].subdet) && depth == types[k].depth1 && eta * types[k].z > 0) {
        id1 = k;
        break;
      }
    } else {
      if (subdet == (int)(types[k].subdet) && depth == types[k].depth1) {
        id1 = k;
        break;
      }
    }
  }
  if (subdet == HcalForward)
    depth += 2 * dep;
  for (unsigned int k = 0; k < types.size(); ++k) {
    if (types[k].subdet == HcalEmpty && types[k].depth1 == depth) {
      id2 = k;
      break;
    }
  }
  return std::pair<int, int>(id1, id2);
}

std::vector<std::pair<std::string, std::string>> SimHitsValidationHcal::getHistogramTypes() {
  int maxDepth = std::max(maxDepthHB_, maxDepthHE_);
  maxDepth = std::max(maxDepth, maxDepthHF_);
  maxDepth = std::max(maxDepth, maxDepthHO_);

  std::vector<std::pair<std::string, std::string>> divisions;
  // divisions and types need to be in sync
  types.clear();
  std::pair<std::string, std::string> names;
  char name1[40], name2[40];
  SimHitsValidationHcal::idType type;
  // first overall Hcal
  for (int depth = 0; depth < maxDepth; ++depth) {
    snprintf(name1, 40, "HC%d", depth);
    snprintf(name2, 40, "HCAL depth%d", depth + 1);
    names = std::pair<std::string, std::string>(std::string(name1), std::string(name2));
    type = SimHitsValidationHcal::idType(HcalEmpty, 0, depth + 1, depth + 1);
    divisions.push_back(names);
    types.push_back(type);
  }
  // HB
  for (int depth = 0; depth < maxDepthHB_; ++depth) {
    snprintf(name1, 40, "HB%d", depth);
    snprintf(name2, 40, "HB depth%d", depth + 1);
    names = std::pair<std::string, std::string>(std::string(name1), std::string(name2));
    type = SimHitsValidationHcal::idType(HcalBarrel, 0, depth + 1, depth + 1);
    divisions.push_back(names);
    types.push_back(type);
  }
  // HE
  for (int depth = 0; depth < maxDepthHE_; ++depth) {
    snprintf(name1, 40, "HE%d+z", depth);
    snprintf(name2, 40, "HE +z depth%d", depth + 1);
    names = std::pair<std::string, std::string>(std::string(name1), std::string(name2));
    type = SimHitsValidationHcal::idType(HcalEndcap, 1, depth + 1, depth + 1);
    divisions.push_back(names);
    types.push_back(type);
    snprintf(name1, 40, "HE%d-z", depth);
    snprintf(name2, 40, "HE -z depth%d", depth + 1);
    names = std::pair<std::string, std::string>(std::string(name1), std::string(name2));
    type = SimHitsValidationHcal::idType(HcalEndcap, -1, depth + 1, depth + 1);
    divisions.push_back(names);
    types.push_back(type);
  }
  // HO
  {
    int depth = maxDepthHO_;
    snprintf(name1, 40, "HO%d", depth);
    snprintf(name2, 40, "HO depth%d", depth);
    names = std::pair<std::string, std::string>(std::string(name1), std::string(name2));
    type = SimHitsValidationHcal::idType(HcalOuter, 0, depth, depth);
    divisions.push_back(names);
    types.push_back(type);
  }
  // HF (first absorber, then different types of abnormal hits)
  std::string hfty1[4] = {"A", "W", "B", "J"};
  std::string hfty2[4] = {"Absorber", "Window", "Bundle", "Jungle"};
  int dept0[4] = {0, 1, 2, 3};
  for (int k = 0; k < 4; ++k) {
    for (int depth = 0; depth < maxDepthHF_; ++depth) {
      snprintf(name1, 40, "HF%s%d+z", hfty1[k].c_str(), depth);
      snprintf(name2, 40, "HF (%s) +z depth%d", hfty2[k].c_str(), depth + 1);
      names = std::pair<std::string, std::string>(std::string(name1), std::string(name2));
      type = SimHitsValidationHcal::idType(HcalForward, 1, depth + 1, dept0[k]);
      divisions.push_back(names);
      types.push_back(type);
      snprintf(name1, 40, "HF%s%d-z", hfty1[k].c_str(), depth);
      snprintf(name2, 40, "HF (%s) -z depth%d", hfty2[k].c_str(), depth + 1);
      names = std::pair<std::string, std::string>(std::string(name1), std::string(name2));
      type = SimHitsValidationHcal::idType(HcalForward, -1, depth + 1, dept0[k]);
      divisions.push_back(names);
      types.push_back(type);
    }
  }

  return divisions;
}

void SimHitsValidationHcal::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ModuleLabel", "g4SimHits");
  desc.add<std::string>("HitCollection", "HcalHits");
  desc.add<bool>("Verbose", false);
  desc.add<bool>("TestNumber", false);

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(SimHitsValidationHcal);
