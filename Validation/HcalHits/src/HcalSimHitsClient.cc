#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/HcalHits/interface/HcalSimHitsClient.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

HcalSimHitsClient::HcalSimHitsClient(const edm::ParameterSet &iConfig) {
  dirName_ = iConfig.getParameter<std::string>("DQMDirName");
  verbose_ = iConfig.getUntrackedParameter<bool>("Verbosity", false);
  tok_HRNDC_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord, edm::Transition::BeginRun>();
}

HcalSimHitsClient::~HcalSimHitsClient() {}

void HcalSimHitsClient::beginRun(edm::Run const &run, edm::EventSetup const &c) {
  const auto &pHRNDC = c.getData(tok_HRNDC_);
  hcons = &pHRNDC;
  maxDepthHB_ = hcons->getMaxDepth(0);
  maxDepthHE_ = hcons->getMaxDepth(1);
  maxDepthHF_ = hcons->getMaxDepth(2);
  maxDepthHO_ = hcons->getMaxDepth(3);

  edm::LogInfo("HitsValidationHcal") << " Maximum Depths HB:" << maxDepthHB_ << " HE:" << maxDepthHE_
                                     << " HO:" << maxDepthHO_ << " HF:" << maxDepthHF_;
}

void HcalSimHitsClient::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) { runClient_(ib, ig); }

void HcalSimHitsClient::runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig) {
  ig.setCurrentFolder(dirName_);

  if (verbose_)
    edm::LogInfo("HitsValidationHcal") << "runClient";

  std::vector<MonitorElement *> hcalMEs;

  std::vector<std::string> fullPathHLTFolders = ig.getSubdirs();
  for (unsigned int i = 0; i < fullPathHLTFolders.size(); i++) {
    if (verbose_)
      edm::LogInfo("HitsValidationHcal") << "fullPath: " << fullPathHLTFolders[i];
    ig.setCurrentFolder(fullPathHLTFolders[i]);

    std::vector<std::string> fullSubPathHLTFolders = ig.getSubdirs();
    for (unsigned int j = 0; j < fullSubPathHLTFolders.size(); j++) {
      if (verbose_)
        edm::LogInfo("HitsValidationHcal") << "fullSub: " << fullSubPathHLTFolders[j];

      if (strcmp(fullSubPathHLTFolders[j].c_str(), "HcalHitsV/SimHitsValidationHcal") == 0) {
        hcalMEs = ig.getContents(fullSubPathHLTFolders[j]);
        if (verbose_)
          edm::LogInfo("HitsValidationHcal") << "hltMES size : " << hcalMEs.size();
        if (!SimHitsEndjob(hcalMEs))
          edm::LogWarning("HitsValidationHcal") << "Error in SimhitEndjob!";
      }
    }
  }
}

// called after entering the  directory
// hcalMEs are within that directory
int HcalSimHitsClient::SimHitsEndjob(const std::vector<MonitorElement *> &hcalMEs) {
  std::vector<std::string> divisions = getHistogramTypes();
  MonitorElement *Occupancy_map[nTime][divisions.size()];
  MonitorElement *Energy[nType1], *Time_weighteden[nType1];
  MonitorElement *HitEnergyvsieta[divisions.size()], *HitTimevsieta[divisions.size()];

  std::string time[nTime] = {"25", "50", "100", "250"};
  std::string detdivision[nType1] = {"HB", "HE", "HF", "HO"};
  char name[40], name1[40], name2[40], name3[40], name4[40];

  for (int k = 0; k < nType1; k++) {
    Energy[k] = nullptr;
    Time_weighteden[k] = nullptr;
    for (unsigned int ih = 0; ih < hcalMEs.size(); ih++) {
      sprintf(name1, "Energy_%s", detdivision[k].c_str());
      sprintf(name2, "Time_Enweighted_%s", detdivision[k].c_str());
      if (strcmp(hcalMEs[ih]->getName().c_str(), name1) == 0) {
        Energy[k] = hcalMEs[ih];
      }
      if (strcmp(hcalMEs[ih]->getName().c_str(), name2) == 0) {
        Time_weighteden[k] = hcalMEs[ih];
      }
    }
  }

  for (int i = 0; i < nTime; i++) {
    for (unsigned int j = 0; j < divisions.size(); j++) {
      for (unsigned int ih = 0; ih < hcalMEs.size(); ih++) {
        sprintf(name, "HcalHitE%s%s", time[i].c_str(), divisions[j].c_str());
        if (strcmp(hcalMEs[ih]->getName().c_str(), name) == 0) {
          Occupancy_map[i][j] = hcalMEs[ih];
        }
      }
    }
  }

  for (unsigned int k = 0; k < divisions.size(); k++) {
    HitEnergyvsieta[k] = nullptr;
    HitTimevsieta[k] = nullptr;
    for (unsigned int ih = 0; ih < hcalMEs.size(); ih++) {
      sprintf(name3, "HcalHitEta%s", divisions[k].c_str());
      sprintf(name4, "HcalHitTimeAEta%s", divisions[k].c_str());
      if (strcmp(hcalMEs[ih]->getName().c_str(), name3) == 0) {
        HitEnergyvsieta[k] = hcalMEs[ih];
      }
      if (strcmp(hcalMEs[ih]->getName().c_str(), name4) == 0) {
        HitTimevsieta[k] = hcalMEs[ih];
      }
    }
  }

  // mean energy

  double nevent = Energy[0]->getEntries();
  if (verbose_)
    edm::LogInfo("HitsValidationHcal") << "nevent : " << nevent;

  float cont[nTime][divisions.size()];
  float en[nType1], tme[nType1];
  float hitenergy[divisions.size()], hittime[divisions.size()];
  float fev = float(nevent);

  for (int dettype = 0; dettype < nType1; dettype++) {
    int nx1 = Energy[dettype]->getNbinsX();
    for (int i = 0; i <= nx1; i++) {
      en[dettype] = Energy[dettype]->getBinContent(i) / fev;
      Energy[dettype]->setBinContent(i, en[dettype]);
    }
    int nx2 = Time_weighteden[dettype]->getNbinsX();
    for (int i = 0; i <= nx2; i++) {
      tme[dettype] = Time_weighteden[dettype]->getBinContent(i) / fev;
      Time_weighteden[dettype]->setBinContent(i, tme[dettype]);
    }
  }

  for (unsigned int dettype = 0; dettype < divisions.size(); dettype++) {
    int nx1 = HitEnergyvsieta[dettype]->getNbinsX();
    for (int i = 0; i <= nx1; i++) {
      hitenergy[dettype] = HitEnergyvsieta[dettype]->getBinContent(i) / fev;
      HitEnergyvsieta[dettype]->setBinContent(i, hitenergy[dettype]);
    }
    int nx2 = HitTimevsieta[dettype]->getNbinsX();
    for (int i = 0; i <= nx2; i++) {
      hittime[dettype] = HitTimevsieta[dettype]->getBinContent(i) / fev;
      HitTimevsieta[dettype]->setBinContent(i, hittime[dettype]);
    }
  }

  for (int itime = 0; itime < nTime; itime++) {
    for (unsigned int det = 0; det < divisions.size(); det++) {
      int ny = Occupancy_map[itime][det]->getNbinsY();
      int nx = Occupancy_map[itime][det]->getNbinsX();
      for (int i = 1; i < nx + 1; i++) {
        for (int j = 1; j < ny + 1; j++) {
          cont[itime][det] = Occupancy_map[itime][det]->getBinContent(i, j) / fev;
          Occupancy_map[itime][det]->setBinContent(i, j, cont[itime][det]);
        }
      }
    }
  }

  return 1;
}

std::vector<std::string> HcalSimHitsClient::getHistogramTypes() {
  int maxDepth = std::max(maxDepthHB_, maxDepthHE_);
  if (verbose_)
    edm::LogInfo("HitsValidationHcal") << "Max depth 1st step:: " << maxDepth;
  maxDepth = std::max(maxDepth, maxDepthHF_);
  if (verbose_)
    edm::LogInfo("HitsValidationHcal") << "Max depth 2nd step:: " << maxDepth;
  maxDepth = std::max(maxDepth, maxDepthHO_);
  if (verbose_)
    edm::LogInfo("HitsValidationHcal") << "Max depth 3rd step:: " << maxDepth;
  std::vector<std::string> divisions;
  char name1[20];

  // first overall Hcal
  for (int depth = 0; depth < maxDepth; ++depth) {
    sprintf(name1, "HC%d", depth);
    divisions.push_back(std::string(name1));
  }
  // HB
  for (int depth = 0; depth < maxDepthHB_; ++depth) {
    sprintf(name1, "HB%d", depth);
    divisions.push_back(std::string(name1));
  }
  // HE
  for (int depth = 0; depth < maxDepthHE_; ++depth) {
    sprintf(name1, "HE%d+z", depth);
    divisions.push_back(std::string(name1));
    sprintf(name1, "HE%d-z", depth);
    divisions.push_back(std::string(name1));
  }
  // HO
  {
    int depth = maxDepthHO_;
    sprintf(name1, "HO%d", depth);
    divisions.push_back(std::string(name1));
  }
  // HF (first absorber, then different types of abnormal hits)
  std::string hfty1[4] = {"A", "W", "B", "J"};
  for (int k = 0; k < 4; ++k) {
    for (int depth = 0; depth < maxDepthHF_; ++depth) {
      sprintf(name1, "HF%s%d+z", hfty1[k].c_str(), depth);
      divisions.push_back(std::string(name1));
      sprintf(name1, "HF%s%d-z", hfty1[k].c_str(), depth);
      divisions.push_back(std::string(name1));
    }
  }
  return divisions;
}

DEFINE_FWK_MODULE(HcalSimHitsClient);
