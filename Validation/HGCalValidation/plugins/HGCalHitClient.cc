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
  ~HGCalHitClient() override = default;

  void beginRun(const edm::Run& run, const edm::EventSetup& c) override {}
  void dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) override;

private:
  const std::string subDirectory_;

  int geometryEndjob(const std::vector<MonitorElement*>& hcalMEs);
};

HGCalHitClient::HGCalHitClient(const edm::ParameterSet& iConfig)
    : subDirectory_(iConfig.getParameter<std::string>("DirectoryName")) {}

void HGCalHitClient::dqmEndJob(DQMStore::IBooker& ib, DQMStore::IGetter& ig) {
  ig.setCurrentFolder("/");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalValid") << "HGCalHitValidation :: runClient";
#endif
  std::vector<MonitorElement*> hgcalMEs;
  std::vector<std::string> fullDirPath = ig.getSubdirs();

  for (unsigned int i = 0; i < fullDirPath.size(); i++) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalValid") << "HGCalHitValidation::fullPath: " << fullDirPath.at(i);
#endif
    ig.setCurrentFolder(fullDirPath.at(i));
    std::vector<std::string> fullSubDirPath = ig.getSubdirs();

    for (unsigned int j = 0; j < fullSubDirPath.size(); j++) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalValid") << "HGCalHitValidation:: fullSubPath: " << fullSubDirPath.at(j);
#endif
      if (strcmp(fullSubDirPath.at(j).c_str(), subDirectory_.c_str()) == 0) {
        hgcalMEs = ig.getContents(fullSubDirPath.at(j));
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
  for (unsigned int idet = 0; idet < 3; ++idet) {
    char name[100];
    for (unsigned int kh = 0; kh < 2; ++kh) {
      sprintf(name, "%s%s", dets[idet].c_str(), hist1[kh].c_str());
      for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
        if (strcmp(hgcalMEs[ih]->getName().c_str(), name) == 0) {
          hist1_.push_back(hgcalMEs[ih]);
          double nevent = hist1_.back()->getEntries();
          int nbinsx = hist1_.back()->getNbinsX();
          for (int i = 1; i <= nbinsx; ++i) {
            double binValue = hist1_.back()->getBinContent(i) / nevent;
            hist1_.back()->setBinContent(i, binValue);
          }
        }
      }
    }
    for (unsigned int kh = 0; kh < 7; ++kh) {
      sprintf(name, "%s%s", dets[idet].c_str(), hist2[kh].c_str());
      for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
        if (strcmp(hgcalMEs[ih]->getName().c_str(), name) == 0) {
          hist2_.push_back(hgcalMEs[ih]);
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
