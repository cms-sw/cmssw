#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalSimHitsClient : public DQMEDHarvester {
private:
  //member data
  const std::string nameDetector_;
  const int nTimes_, verbosity_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> tok_hgcal_;
  unsigned int layers_;

public:
  explicit HGCalSimHitsClient(const edm::ParameterSet &);
  ~HGCalSimHitsClient() override = default;

  void beginRun(const edm::Run &run, const edm::EventSetup &c) override;
  void dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) override;
  virtual void runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig);
  int simHitsEndjob(const std::vector<MonitorElement *> &hgcalMEs);
};

HGCalSimHitsClient::HGCalSimHitsClient(const edm::ParameterSet &iConfig)
    : nameDetector_(iConfig.getParameter<std::string>("DetectorName")),
      nTimes_(iConfig.getParameter<int>("TimeSlices")),
      verbosity_(iConfig.getUntrackedParameter<int>("Verbosity", 0)),
      tok_hgcal_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})) {}

void HGCalSimHitsClient::beginRun(const edm::Run &run, const edm::EventSetup &iSetup) {
  const HGCalDDDConstants *hgcons = &iSetup.getData(tok_hgcal_);
  layers_ = hgcons->layers(true);
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << "Initialize HGCalSimHitsClient for " << nameDetector_ << " : " << layers_;
}

void HGCalSimHitsClient::dqmEndJob(DQMStore::IBooker &ib, DQMStore::IGetter &ig) { runClient_(ib, ig); }

void HGCalSimHitsClient::runClient_(DQMStore::IBooker &ib, DQMStore::IGetter &ig) {
  ig.setCurrentFolder("/");
  if (verbosity_ > 0)
    edm::LogVerbatim("HGCalValidation") << " runClient";
  std::vector<MonitorElement *> hgcalMEs;
  std::vector<std::string> fullDirPath = ig.getSubdirs();

  for (unsigned int i = 0; i < fullDirPath.size(); i++) {
    if (verbosity_ > 0)
      edm::LogVerbatim("HGCalValidation") << "fullPath: " << fullDirPath.at(i);
    ig.setCurrentFolder(fullDirPath.at(i));
    std::vector<std::string> fullSubDirPath = ig.getSubdirs();

    for (unsigned int j = 0; j < fullSubDirPath.size(); j++) {
      if (verbosity_ > 0)
        edm::LogVerbatim("HGCalValidation") << "fullSubPath: " << fullSubDirPath.at(j);
      std::string nameDirectory = "HGCAL/HGCalSimHitsV/" + nameDetector_;

      if (strcmp(fullSubDirPath.at(j).c_str(), nameDirectory.c_str()) == 0) {
        hgcalMEs = ig.getContents(fullSubDirPath.at(j));
        if (verbosity_ > 0)
          edm::LogVerbatim("HGCalValidation") << "hgcalMES size : " << hgcalMEs.size();
        if (!simHitsEndjob(hgcalMEs))
          edm::LogWarning("HGCalValidation") << "\nError in SimhitsEndjob!";
      }
    }
  }
}

int HGCalSimHitsClient::simHitsEndjob(const std::vector<MonitorElement *> &hgcalMEs) {
  std::vector<MonitorElement *> energy_[6];
  std::vector<MonitorElement *> EtaPhi_Plus_, EtaPhi_Minus_;
  std::vector<MonitorElement *> HitOccupancy_Plus_, HitOccupancy_Minus_;
  MonitorElement *MeanHitOccupancy_Plus_, *MeanHitOccupancy_Minus_;

  std::ostringstream name;
  double nevent;
  int nbinsx, nbinsy;
  for (unsigned int ilayer = 0; ilayer < layers_; ilayer++) {
    for (int itimeslice = 0; itimeslice < nTimes_; itimeslice++) {
      //Energy
      name.str("");
      name << "energy_time_" << itimeslice << "_layer_" << ilayer;
      for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
        if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
          energy_[itimeslice].push_back(hgcalMEs[ih]);
        }
      }
      //normalization
      nevent = energy_[itimeslice].at(ilayer)->getEntries();
      nbinsx = energy_[itimeslice].at(ilayer)->getNbinsX();
      for (int i = 1; i <= nbinsx; i++) {
        double binValue = energy_[itimeslice].at(ilayer)->getBinContent(i) / nevent;
        energy_[itimeslice].at(ilayer)->setBinContent(i, binValue);
      }
    }  ///loop over timeslice

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
  name << "MeanHitOccupancy_Plus";
  for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
      MeanHitOccupancy_Plus_ = hgcalMEs[ih];
      for (int ilayer = 0; ilayer < static_cast<int>(layers_); ++ilayer) {
        double meanVal = HitOccupancy_Plus_.at(ilayer)->getMean();
        MeanHitOccupancy_Plus_->setBinContent(ilayer + 1, meanVal);
      }
      break;
    }
  }

  name.str("");
  name << "MeanHitOccupancy_Minus";
  for (unsigned int ih = 0; ih < hgcalMEs.size(); ih++) {
    if (strcmp(hgcalMEs[ih]->getName().c_str(), name.str().c_str()) == 0) {
      MeanHitOccupancy_Minus_ = hgcalMEs[ih];
      for (int ilayer = 0; ilayer < static_cast<int>(layers_); ++ilayer) {
        double meanVal = HitOccupancy_Minus_.at(ilayer)->getMean();
        MeanHitOccupancy_Minus_->setBinContent(ilayer + 1, meanVal);
      }
      break;
    }
  }

  return 1;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HGCalSimHitsClient);
