#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalRecHitsClient : public DQMEDHarvester {
private:
  //member data
  const std::string nameDetector_;
  const int verbosity_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> ddc_token_;
  unsigned int layers_;

public:
  explicit HGCalRecHitsClient(const edm::ParameterSet &);
  ~HGCalRecHitsClient() override = default;

  void beginRun(const edm::Run &run, const edm::EventSetup &c) override;
  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) override;
  virtual void runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig);

  int recHitsEndjob(const std::vector<MonitorElement *> &hgcalMEs);
};

HGCalRecHitsClient::HGCalRecHitsClient(const edm::ParameterSet &iConfig)
    : nameDetector_(iConfig.getParameter<std::string>("DetectorName")),
      verbosity_(iConfig.getUntrackedParameter<int>("Verbosity", 0)),
      ddc_token_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})) {}

void HGCalRecHitsClient::beginRun(const edm::Run &run, const edm::EventSetup &iSetup) {
  const HGCalDDDConstants &hgcons_ = iSetup.getData(ddc_token_);
  layers_ = hgcons_.layers(true);
}

void HGCalRecHitsClient::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) { runClient_(ib, ig); }

void HGCalRecHitsClient::runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig) {
  ig.setCurrentFolder("/");
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << "\nrunClient";
  std::vector<MonitorElement *> hgcalMEs;
  std::vector<std::string> fullDirPath = ig.getSubdirs();

  for (unsigned int i = 0; i < fullDirPath.size(); i++) {
    if (verbosity_ > 0)
      edm::LogVerbatim("HGCalValidation") << "\nfullPath: " << fullDirPath.at(i);
    ig.setCurrentFolder(fullDirPath.at(i));
    std::vector<std::string> fullSubDirPath = ig.getSubdirs();

    for (unsigned int j = 0; j < fullSubDirPath.size(); j++) {
      if (verbosity_ > 1)
        edm::LogVerbatim("HGCalValidation") << "fullSubPath: " << fullSubDirPath.at(j);
      std::string nameDirectory = "HGCAL/HGCalRecHitsV/" + nameDetector_;
      if (strcmp(fullSubDirPath.at(j).c_str(), nameDirectory.c_str()) == 0) {
        hgcalMEs = ig.getContents(fullSubDirPath.at(j));
        if (verbosity_ > 1)
          edm::LogVerbatim("HGCalValidation") << "hgcalMES size : " << hgcalMEs.size();
        if (!recHitsEndjob(hgcalMEs))
          edm::LogWarning("HGCalValidation") << "\nError in RecHitsEndjob!";
      }
    }
  }
}

int HGCalRecHitsClient::recHitsEndjob(const std::vector<MonitorElement *> &hgcalMEs) {
  std::vector<MonitorElement *> energy_;
  std::vector<MonitorElement *> EtaPhi_Plus_;
  std::vector<MonitorElement *> EtaPhi_Minus_;
  std::vector<MonitorElement *> HitOccupancy_Plus_;
  std::vector<MonitorElement *> HitOccupancy_Minus_;
  std::vector<MonitorElement *> MeanHitOccupancy_Plus_;
  std::vector<MonitorElement *> MeanHitOccupancy_Minus_;

  std::ostringstream name;
  double nevent;
  int nbinsx, nbinsy;

  for (unsigned int ilayer = 0; ilayer < layers_; ilayer++) {
    name.str("");
    name << "energy_layer_" << ilayer;
    for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
        energy_.push_back(hgcalMEs[ih]);
      }
    }

    //normalization
    nevent = energy_.at(ilayer)->getEntries();
    nbinsx = energy_.at(ilayer)->getNbinsX();
    for (int i = 1; i <= nbinsx; i++) {
      double binValue = energy_.at(ilayer)->getBinContent(i) / nevent;
      energy_.at(ilayer)->setBinContent(i, binValue);
    }

    //EtaPhi 2d plots
    name.str("");
    name << "EtaPhi_Plus_"
         << "layer_" << ilayer;
    for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
        EtaPhi_Plus_.push_back(hgcalMEs[ih]);
      }
    }

    name.str("");
    name << "EtaPhi_Minus_"
         << "layer_" << ilayer;
    for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
        EtaPhi_Minus_.push_back(hgcalMEs[ih]);
      }
    }

    //normalization EtaPhi
    nevent = EtaPhi_Plus_.at(ilayer)->getEntries();
    nbinsx = EtaPhi_Plus_.at(ilayer)->getNbinsX();
    nbinsy = EtaPhi_Plus_.at(ilayer)->getNbinsY();
    for (int i = 1; i <= nbinsx; ++i) {
      for (int j = 1; j <= nbinsy; ++j) {
        double binValue = EtaPhi_Plus_.at(ilayer)->getBinContent(i, j) / nevent;
        EtaPhi_Plus_.at(ilayer)->setBinContent(i, j, binValue);
      }
    }

    nevent = EtaPhi_Minus_.at(ilayer)->getEntries();
    nbinsx = EtaPhi_Minus_.at(ilayer)->getNbinsX();
    nbinsy = EtaPhi_Plus_.at(ilayer)->getNbinsY();
    for (int i = 1; i <= nbinsx; ++i) {
      for (int j = 1; j <= nbinsy; ++j) {
        double binValue = EtaPhi_Minus_.at(ilayer)->getBinContent(i, j) / nevent;
        EtaPhi_Minus_.at(ilayer)->setBinContent(i, j, binValue);
      }
    }

    //HitOccupancy
    name.str("");
    name << "HitOccupancy_Plus_layer_" << ilayer;
    for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
        HitOccupancy_Plus_.push_back(hgcalMEs[ih]);
      }
    }

    name.str("");
    name << "HitOccupancy_Minus_layer_" << ilayer;
    for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
      if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
        HitOccupancy_Minus_.push_back(hgcalMEs[ih]);
      }
    }

    //normalization of hit occupancy histos
    nevent = HitOccupancy_Plus_.at(ilayer)->getEntries();
    nbinsx = HitOccupancy_Plus_.at(ilayer)->getNbinsX();
    for (int i = 1; i <= nbinsx; ++i) {
      double binValue = HitOccupancy_Plus_.at(ilayer)->getBinContent(i) / nevent;
      HitOccupancy_Plus_.at(ilayer)->setBinContent(i, binValue);
    }

    nevent = HitOccupancy_Minus_.at(ilayer)->getEntries();
    nbinsx = HitOccupancy_Minus_.at(ilayer)->getNbinsX();
    for (int i = 1; i <= nbinsx; ++i) {
      double binValue = HitOccupancy_Minus_.at(ilayer)->getBinContent(i) / nevent;
      HitOccupancy_Minus_.at(ilayer)->setBinContent(i, binValue);
    }

  }  //loop over layers

  name.str("");
  name << "SUMOfRecHitOccupancy_Plus";
  for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
      MeanHitOccupancy_Plus_.push_back(hgcalMEs[ih]);
      unsigned int indx = MeanHitOccupancy_Plus_.size() - 1;
      for (int ilayer = 0; ilayer < static_cast<int>(layers_); ++ilayer) {
        double meanVal = HitOccupancy_Plus_.at(ilayer)->getMean();
        MeanHitOccupancy_Plus_[indx]->setBinContent(ilayer + 1, meanVal);
      }
      break;
    }
  }

  name.str("");
  name << "SUMOfRecHitOccupancy_Plus";
  for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
      MeanHitOccupancy_Minus_.push_back(hgcalMEs[ih]);
      unsigned indx = MeanHitOccupancy_Minus_.size() - 1;
      for (int ilayer = 0; ilayer < static_cast<int>(layers_); ++ilayer) {
        double meanVal = HitOccupancy_Minus_.at(ilayer)->getMean();
        MeanHitOccupancy_Minus_[indx]->setBinContent(ilayer + 1, meanVal);
      }
      break;
    }
  }

  return 1;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalRecHitsClient);
