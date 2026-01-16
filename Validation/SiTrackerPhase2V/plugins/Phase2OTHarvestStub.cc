
// Package:    Validation/SiTrackerPhase2V
// Class:      Phase2OTHarvestStub

/**
 * This class is part of the Phase 2 Tracker validation framework and performs
 * the harvesting step for stub validation. It processes histograms
 * created during the earlier validation steps to calculate efficiencies and
 * resolutions for stub reconstruction.
 *
 * Usage:
 * To generate histograms from this code, run the test configuration files
 * provided in the DQM/SiTrackerPhase2/test directory. The generated histograms
 * can then be analyzed or visualized.
 */

// Created by: Brandi Skipworth, 2025

#include <algorithm>

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Validation/SiTrackerPhase2V/interface/TrackerPhase2HistUtil.h"

class Phase2OTHarvestStub : public DQMEDHarvester {
public:
  explicit Phase2OTHarvestStub(const edm::ParameterSet &);
  ~Phase2OTHarvestStub() override;
  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  // ----------member data ---------------------------
  std::string topFolderName_;
};

Phase2OTHarvestStub::Phase2OTHarvestStub(const edm::ParameterSet &iConfig)
    : topFolderName_(iConfig.getParameter<std::string>("TopFolderName")) {}

Phase2OTHarvestStub::~Phase2OTHarvestStub() {}

// ------------ method called once each job just after ending the event loop
// ------------
void Phase2OTHarvestStub::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  using namespace edm;

  // Find all monitor elements for histograms
  MonitorElement *meN_clus_barrel = igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_if_stub_barrel");
  MonitorElement *meD_clus_barrel = igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_barrel");
  MonitorElement *meN_clus_zoom_barrel =
      igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_if_stub_zoom_barrel");
  MonitorElement *meD_clus_zoom_barrel =
      igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_zoom_barrel");
  MonitorElement *meN_clus_endcaps =
      igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_if_stub_endcaps");
  MonitorElement *meD_clus_endcaps = igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_endcaps");
  MonitorElement *meN_clus_zoom_endcaps =
      igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_if_stub_zoom_endcaps");
  MonitorElement *meD_clus_zoom_endcaps =
      igetter.get(topFolderName_ + "/EfficiencyIngredients/gen_clusters_zoom_endcaps");

  if (meN_clus_barrel && meD_clus_barrel) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_clus_barrel->getTH1F();
    TH1F *denominator = meD_clus_barrel->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_clus_barrel = ibooker.book1D("StubEfficiencyBarrel",
                                                          "Stub Efficiency Barrel",
                                                          numerator->GetNbinsX(),
                                                          numerator->GetXaxis()->GetXmin(),
                                                          numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_clus_barrel, "tracking particle pT [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for stub efficiency barrel cannot be found!\n";
  }

  if (meN_clus_zoom_barrel && meD_clus_zoom_barrel) {
    // Get the numerator and denominator histograms
    TH1F *numerator_zoom = meN_clus_zoom_barrel->getTH1F();
    TH1F *denominator_zoom = meD_clus_zoom_barrel->getTH1F();
    numerator_zoom->Sumw2();
    denominator_zoom->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_clus_zoom_barrel = ibooker.book1D("StubEfficiencyZoomBarrel",
                                                               "Stub Efficiency Zoom Barrel",
                                                               numerator_zoom->GetNbinsX(),
                                                               numerator_zoom->GetXaxis()->GetXmin(),
                                                               numerator_zoom->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(
        numerator_zoom, denominator_zoom, me_effic_clus_zoom_barrel, "tracking particle pT [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for stub zoom barrel "
                                       "efficiency cannot be found!\n";
  }

  if (meN_clus_endcaps && meD_clus_endcaps) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_clus_endcaps->getTH1F();
    TH1F *denominator = meD_clus_endcaps->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_clus_endcaps = ibooker.book1D("StubEfficiencyEndcaps",
                                                           "Stub Efficiency Endcaps",
                                                           numerator->GetNbinsX(),
                                                           numerator->GetXaxis()->GetXmin(),
                                                           numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_clus_endcaps, "tracking particle pT [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for stub efficiency endcaps cannot be found!\n";
  }

  if (meN_clus_zoom_endcaps && meD_clus_zoom_endcaps) {
    // Get the numerator and denominator histograms
    TH1F *numerator_zoom = meN_clus_zoom_endcaps->getTH1F();
    TH1F *denominator_zoom = meD_clus_zoom_endcaps->getTH1F();
    numerator_zoom->Sumw2();
    denominator_zoom->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_clus_zoom_endcaps = ibooker.book1D("StubEfficiencyZoomEndcaps",
                                                                "Stub Efficiency Zoom Endcaps",
                                                                numerator_zoom->GetNbinsX(),
                                                                numerator_zoom->GetXaxis()->GetXmin(),
                                                                numerator_zoom->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(
        numerator_zoom, denominator_zoom, me_effic_clus_zoom_endcaps, "tracking particle pT [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for stub zoom endcaps "
                                       "efficiency cannot be found!\n";
  }
}  // end dqmEndJob

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTHarvestStub::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTStubV");
  descriptions.add("Phase2OTHarvestStub", desc);
}
DEFINE_FWK_MODULE(Phase2OTHarvestStub);
