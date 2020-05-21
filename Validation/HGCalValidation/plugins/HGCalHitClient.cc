#include <memory>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"

//#define EDM_ML_DEBUG

class HGCalHitClient : public DQMEDHarvester {
public:
  explicit HGCalHitClient(const edm::ParameterSet&);
  ~HGCalHitClient() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& c) override {}
  void dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) override;

private:
  std::string subDirectory_;

  int geometryEndjob(const std::vector<MonitorElement*>& hcalMEs);
};

HGCalHitClient::HGCalHitClient(const edm::ParameterSet& iConfig) {
  subDirectory_ = iConfig.getParameter<std::string>("DirectoryName");
}

HGCalHitClient::~HGCalHitClient() {}

void HGCalHitClient::dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
  ig.setCurrentFolder("/");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalValid") << "HGCalHitValidation :: runClient";
#endif
  std::vector<MonitorElement*> hgcalMEs;
  std::vector<std::string> fullDirPath = ig.getSubdirs();

  for (const auto& i : fullDirPath) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalValid") << "HGCalHitValidation::fullPath: " << fullDirPath.at(i);
#endif
    ig.setCurrentFolder(i);
    std::vector<std::string> fullSubDirPath = ig.getSubdirs();

    for (auto& j : fullSubDirPath) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalValid") << "HGCalHitValidation:: fullSubPath: " << fullSubDirPath.at(j);
#endif
      if (strcmp(j.c_str(), subDirectory_.c_str()) == 0) {
        hgcalMEs = ig.getContents(j);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalValid") << "HGCalHitValidation:: hgcalMES size : " << hgcalMEs.size();
#endif
        if (!geometryEndjob(hgcalMEs))
          edm::LogWarning("HGCalValid") << "\nError in HGcalHitEndjob!";
      }
    }
  }
}

int HGCalHitClient::geometryEndjob(const std::vector<MonitorElement*>& hgcalMEs) {
  std::string dets[3] = {"hee", "hef", "heb"};
  std::string hist1[2] = {"EnRec", "EnSim"};
  std::string hist2[7] = {"dzVsZ", "dyVsY", "dxVsX", "RecVsSimZ", "RecVsSimY", "RecVsSimX", "EnSimRec"};

  std::vector<MonitorElement*> hist1_;
  std::vector<MonitorElement*> hist2_;

  //Normalize the histograms
  for (auto& det : dets) {
    char name[100];
    for (auto& kh : hist1) {
      sprintf(name, "%s%s", det.c_str(), kh.c_str());
      for (auto hgcalME : hgcalMEs) {
        if (strcmp(hgcalME->getName().c_str(), name) == 0) {
          hist1_.push_back(hgcalME);
          double nevent = hist1_.back()->getEntries();
          int nbinsx = hist1_.back()->getNbinsX();
          for (int i = 1; i <= nbinsx; ++i) {
            double binValue = hist1_.back()->getBinContent(i) / nevent;
            hist1_.back()->setBinContent(i, binValue);
          }
        }
      }
    }
    for (auto& kh : hist2) {
      sprintf(name, "%s%s", det.c_str(), kh.c_str());
      for (auto hgcalME : hgcalMEs) {
        if (strcmp(hgcalME->getName().c_str(), name) == 0) {
          hist2_.push_back(hgcalME);
          double nevent = hist2_.back()->getEntries();
          int nbinsx = hist2_.back()->getNbinsX();
          int nbinsy = hist2_.back()->getNbinsY();
          for (int i = 1; i <= nbinsx; ++i) {
            for (int j = 1; j <= nbinsy; ++j) {
              double binValue = hist2_.back()->getBinContent(i, j) / nevent;
              hist2_.back()->setBinContent(i, j, binValue);
            }
          }
        }
      }
    }
  }

  return 1;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalHitClient);
