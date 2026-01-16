

// Package:    Validation/SiTrackerPhase2V
// Class:      Phase2OTHarvestTracks

/**
 * This class is part of the Phase 2 Tracker validation framework and performs
 * the harvesting step for L1 tracks validation. It processes histograms
 * created during the earlier validation steps to calculate efficiencies and
 * resolutions for tracking performance.
 *
 * Usage:
 * To generate histograms from this code, run the test configuration files
 * provided in the DQM/SiTrackerPhase2/test directory. The generated histograms
 * can then be analyzed or visualized.
 */

// Updated by: Brandi Skipworth, 2025

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

class Phase2OTHarvestTracks : public DQMEDHarvester {
public:
  explicit Phase2OTHarvestTracks(const edm::ParameterSet &);
  ~Phase2OTHarvestTracks() override;
  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  // ----------member data ---------------------------
  std::string topFolderName_;
};

Phase2OTHarvestTracks::Phase2OTHarvestTracks(const edm::ParameterSet &iConfig)
    : topFolderName_(iConfig.getParameter<std::string>("TopFolderName")) {}

Phase2OTHarvestTracks::~Phase2OTHarvestTracks() {}

// ------------ method called once each job just after ending the event loop
// ------------
void Phase2OTHarvestTracks::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  using namespace edm;

  float eta_bins[] = {0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4};
  int eta_binnum = 6;

  // Find all monitor elements for histograms
  MonitorElement *meN_eta = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/match_tp_eta");
  MonitorElement *meD_eta = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/tp_eta");
  MonitorElement *meN_pt = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/match_tp_pt");
  MonitorElement *meD_pt = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/tp_pt");
  MonitorElement *meN_pt_zoom = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/match_tp_pt_zoom");
  MonitorElement *meD_pt_zoom = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/tp_pt_zoom");
  MonitorElement *meN_d0 = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/match_tp_d0");
  MonitorElement *meD_d0 = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/tp_d0");
  MonitorElement *meN_Lxy = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/match_tp_Lxy");
  MonitorElement *meD_Lxy = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/tp_Lxy");
  MonitorElement *meN_z0 = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/match_tp_z0");
  MonitorElement *meD_z0 = igetter.get(topFolderName_ + "/Nominal_L1TF/EfficiencyIngredients/tp_z0");

  // Extended collection
  MonitorElement *meN_prompt_eta =
      igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients/match_prompt_tp_eta");
  MonitorElement *meN_prompt_pt =
      igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients/match_prompt_tp_pt");
  MonitorElement *meN_prompt_pt_zoom =
      igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients/match_prompt_tp_pt_zoom");
  MonitorElement *meN_prompt_d0 =
      igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients/match_prompt_tp_d0");
  MonitorElement *meN_prompt_Lxy =
      igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients/match_prompt_tp_Lxy");
  MonitorElement *meN_prompt_z0 =
      igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/EfficiencyIngredients/match_prompt_tp_z0");

  MonitorElement *meN_displaced_eta =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/match_displaced_tp_eta");
  MonitorElement *meD_displaced_eta =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/tp_eta_for_dis");
  MonitorElement *meN_displaced_pt =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/match_displaced_tp_pt");
  MonitorElement *meD_displaced_pt =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/tp_pt_for_dis");
  MonitorElement *meN_displaced_pt_zoom = igetter.get(topFolderName_ +
                                                      "/Extended_L1TF/Displaced/EfficiencyIngredients/"
                                                      "match_displaced_tp_pt_zoom");
  MonitorElement *meD_displaced_pt_zoom =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/tp_pt_zoom_for_dis");
  MonitorElement *meN_displaced_d0 =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/match_displaced_tp_d0");
  MonitorElement *meD_displaced_d0 =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/tp_d0_for_dis");
  MonitorElement *meN_displaced_Lxy =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/match_displaced_tp_Lxy");
  MonitorElement *meN_displaced_z0 =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/match_displaced_tp_z0");
  MonitorElement *meD_displaced_z0 =
      igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/EfficiencyIngredients/tp_z0_for_dis");

  std::string eta_ranges[6] = {"eta0to0p7", "eta0p7to1", "eta1to1p2", "eta1p2to1p6", "eta1p6to2", "eta2to2p4"};
  // nominal collection
  std::vector<MonitorElement *> respt_pt2to3 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> respt_pt3to8 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> respt_pt8toInf = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> mereseta_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> meresphi_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> meresz0_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> meresd0_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  // extended collection
  std::vector<MonitorElement *> prompt_respt_pt2to3 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> prompt_respt_pt3to8 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> prompt_respt_pt8toInf = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> prompt_mereseta_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> prompt_meresphi_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> prompt_meresz0_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> prompt_meresd0_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  std::vector<MonitorElement *> displaced_respt_pt2to3 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> displaced_respt_pt3to8 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> displaced_respt_pt8toInf = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> displaced_mereseta_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> displaced_meresphi_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> displaced_meresz0_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  std::vector<MonitorElement *> displaced_meresd0_vect = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  for (int i = 0; i < 6; i++) {
    // nominal collection
    respt_pt2to3[i] =
        igetter.get(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients/respt_" + eta_ranges[i] + "_pt2to3");
    respt_pt3to8[i] =
        igetter.get(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients/respt_" + eta_ranges[i] + "_pt3to8");
    respt_pt8toInf[i] =
        igetter.get(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients/respt_" + eta_ranges[i] + "_pt8toInf");
    mereseta_vect[i] = igetter.get(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients/reseta_" + eta_ranges[i]);
    meresphi_vect[i] = igetter.get(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients/resphi_" + eta_ranges[i]);
    meresz0_vect[i] = igetter.get(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients/resz0_" + eta_ranges[i]);
    meresd0_vect[i] = igetter.get(topFolderName_ + "/Nominal_L1TF/ResolutionIngredients/resd0_" + eta_ranges[i]);

    // extended collection
    prompt_respt_pt2to3[i] = igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients/respt_prompt_" +
                                         eta_ranges[i] + "_pt2to3");
    prompt_respt_pt3to8[i] = igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients/respt_prompt_" +
                                         eta_ranges[i] + "_pt3to8");
    prompt_respt_pt8toInf[i] = igetter.get(
        topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients/respt_prompt_" + eta_ranges[i] + "_pt8toInf");
    prompt_mereseta_vect[i] =
        igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients/reseta_prompt_" + eta_ranges[i]);
    prompt_meresphi_vect[i] =
        igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients/resphi_prompt_" + eta_ranges[i]);
    prompt_meresz0_vect[i] =
        igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients/resz0_prompt_" + eta_ranges[i]);
    prompt_meresd0_vect[i] =
        igetter.get(topFolderName_ + "/Extended_L1TF/Prompt/ResolutionIngredients/resd0_prompt_" + eta_ranges[i]);

    displaced_respt_pt2to3[i] = igetter.get(
        topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients/respt_displaced_" + eta_ranges[i] + "_pt2to3");
    displaced_respt_pt3to8[i] = igetter.get(
        topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients/respt_displaced_" + eta_ranges[i] + "_pt3to8");
    displaced_respt_pt8toInf[i] =
        igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients/respt_displaced_" + eta_ranges[i] +
                    "_pt8toInf");
    displaced_mereseta_vect[i] = igetter.get(
        topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients/reseta_displaced_" + eta_ranges[i]);
    displaced_meresphi_vect[i] = igetter.get(
        topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients/resphi_displaced_" + eta_ranges[i]);
    displaced_meresz0_vect[i] =
        igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients/resz0_displaced_" + eta_ranges[i]);
    displaced_meresd0_vect[i] =
        igetter.get(topFolderName_ + "/Extended_L1TF/Displaced/ResolutionIngredients/resd0_displaced_" + eta_ranges[i]);
  }

  // nominal collection
  if (meN_eta && meD_eta) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_eta->getTH1F();
    TH1F *denominator = meD_eta->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_eta = ibooker.book1D("EtaEfficiency",
                                                  "#eta efficiency",
                                                  numerator->GetNbinsX(),
                                                  numerator->GetXaxis()->GetXmin(),
                                                  numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_eta, "tracking particle #eta");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for eta efficiency cannot be found!\n";
  }

  // extended collection
  if (meN_prompt_eta && meD_eta) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_prompt_eta->getTH1F();
    TH1F *denominator = meD_eta->getTH1F();
    numerator->Sumw2();
    if (!denominator->GetSumw2N()) {
      denominator->Sumw2();
    }

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_eta = ibooker.book1D("EtaEfficiency",
                                                  "Prompt #eta efficiency",
                                                  numerator->GetNbinsX(),
                                                  numerator->GetXaxis()->GetXmin(),
                                                  numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_eta, "tracking particle #eta");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended prompt "
                                       "eta efficiency cannot be found!\n";
  }

  if (meN_displaced_eta && meD_displaced_eta) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_displaced_eta->getTH1F();
    TH1F *denominator = meD_displaced_eta->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_eta = ibooker.book1D("EtaEfficiency",
                                                  "Displaced #eta efficiency",
                                                  numerator->GetNbinsX(),
                                                  numerator->GetXaxis()->GetXmin(),
                                                  numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_eta, "tracking particle #eta");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended displaced eta efficiency cannot be "
                                       "found!\n";
  }

  // nominal collection
  if (meN_pt && meD_pt) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_pt->getTH1F();
    TH1F *denominator = meD_pt->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_pt = ibooker.book1D("PtEfficiency",
                                                 "p_{T} efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_pt, "Tracking particle p_{T} [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT efficiency cannot be found!\n";
  }

  // extended collection
  if (meN_prompt_pt && meD_pt) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_prompt_pt->getTH1F();
    TH1F *denominator = meD_pt->getTH1F();
    numerator->Sumw2();
    if (!denominator->GetSumw2N()) {
      denominator->Sumw2();
    }

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_pt = ibooker.book1D("PtEfficiency",
                                                 "Prompt p_{T} efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_pt, "Tracking particle p_{T} [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended prompt "
                                       "pT efficiency cannot be found!\n";
  }

  if (meN_displaced_pt && meD_displaced_pt) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_displaced_pt->getTH1F();
    TH1F *denominator = meD_displaced_pt->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_pt = ibooker.book1D("PtEfficiency",
                                                 "Displaced p_{T} efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_pt, "Tracking particle p_{T} [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended displaced pT efficiency cannot be "
                                       "found!\n";
  }

  // nominal collection
  if (meN_pt_zoom && meD_pt_zoom) {
    // Get the numerator and denominator histograms
    TH1F *numerator_zoom = meN_pt_zoom->getTH1F();
    TH1F *denominator_zoom = meD_pt_zoom->getTH1F();
    numerator_zoom->Sumw2();
    denominator_zoom->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_pt_zoom = ibooker.book1D("PtEfficiency_zoom",
                                                      "p_{T} efficiency",
                                                      numerator_zoom->GetNbinsX(),
                                                      numerator_zoom->GetXaxis()->GetXmin(),
                                                      numerator_zoom->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator_zoom, denominator_zoom, me_effic_pt_zoom, "Tracking particle p_{T} [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for zoom pT efficiency cannot be found!\n";
  }

  // extended collection
  if (meN_prompt_pt_zoom && meD_pt_zoom) {
    // Get the numerator and denominator histograms
    TH1F *numerator_zoom = meN_prompt_pt_zoom->getTH1F();
    TH1F *denominator_zoom = meD_pt_zoom->getTH1F();
    numerator_zoom->Sumw2();
    if (!denominator_zoom->GetSumw2N()) {
      denominator_zoom->Sumw2();
    }

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_pt_zoom = ibooker.book1D("PtEfficiency_zoom",
                                                      "Prompt p_{T} efficiency",
                                                      numerator_zoom->GetNbinsX(),
                                                      numerator_zoom->GetXaxis()->GetXmin(),
                                                      numerator_zoom->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator_zoom, denominator_zoom, me_effic_pt_zoom, "Tracking particle p_{T} [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended prompt "
                                       "zoom pT efficiency cannot be found!\n";
  }

  if (meN_displaced_pt_zoom && meD_displaced_pt_zoom) {
    // Get the numerator and denominator histograms
    TH1F *numerator_zoom = meN_displaced_pt_zoom->getTH1F();
    TH1F *denominator_zoom = meD_displaced_pt_zoom->getTH1F();
    numerator_zoom->Sumw2();
    denominator_zoom->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_pt_zoom = ibooker.book1D("PtEfficiency_zoom",
                                                      "Displaced p_{T} efficiency",
                                                      numerator_zoom->GetNbinsX(),
                                                      numerator_zoom->GetXaxis()->GetXmin(),
                                                      numerator_zoom->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator_zoom, denominator_zoom, me_effic_pt_zoom, "Tracking particle p_{T} [GeV]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended displaced zoom pT efficiency cannot "
                                       "be found!\n";
  }

  // nominal collection
  if (meN_d0 && meD_d0) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_d0->getTH1F();
    TH1F *denominator = meD_d0->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_d0 = ibooker.book1D("d0Efficiency",
                                                 "d_{0} efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_d0, "Tracking particle d_{0} [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for d0 efficiency cannot be found!\n";
  }

  // extended collection
  if (meN_prompt_d0 && meD_d0) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_prompt_d0->getTH1F();
    TH1F *denominator = meD_d0->getTH1F();
    numerator->Sumw2();
    if (!denominator->GetSumw2N()) {
      denominator->Sumw2();
    }

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_d0 = ibooker.book1D("d0Efficiency",
                                                 "Prompt d_{0} efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_d0, "Tracking particle d_{0} [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended prompt "
                                       "d0 efficiency cannot be found!\n";
  }

  if (meN_displaced_d0 && meD_displaced_d0) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_displaced_d0->getTH1F();
    TH1F *denominator = meD_displaced_d0->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_d0 = ibooker.book1D("d0Efficiency",
                                                 "Displaced d_{0} efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_d0, "Tracking particle d_{0} [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended displaced d0 efficiency cannot be "
                                       "found!\n";
  }

  // nominal collection
  if (meN_Lxy && meD_Lxy) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_Lxy->getTH1F();
    TH1F *denominator = meD_Lxy->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_Lxy = ibooker.book1D("LxyEfficiency",
                                                  "Lxy efficiency",
                                                  numerator->GetNbinsX(),
                                                  numerator->GetXaxis()->GetXmin(),
                                                  numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_Lxy, "Tracking particle Lxy [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for Lxy efficiency cannot be found!\n";
  }

  // extended collecion
  if (meN_prompt_Lxy && meD_Lxy) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_prompt_Lxy->getTH1F();
    TH1F *denominator = meD_Lxy->getTH1F();
    numerator->Sumw2();
    if (!denominator->GetSumw2N()) {
      denominator->Sumw2();
    }

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_Lxy = ibooker.book1D("LxyEfficiency",
                                                  "Prompt Lxy efficiency",
                                                  numerator->GetNbinsX(),
                                                  numerator->GetXaxis()->GetXmin(),
                                                  numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_Lxy, "Tracking particle Lxy [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended prompt "
                                       "Lxy efficiency cannot be found!\n";
  }

  if (meN_displaced_Lxy && meD_Lxy) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_displaced_Lxy->getTH1F();
    TH1F *denominator = meD_Lxy->getTH1F();
    numerator->Sumw2();
    if (!denominator->GetSumw2N()) {
      denominator->Sumw2();
    }

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_Lxy = ibooker.book1D("LxyEfficiency",
                                                  "Displaced Lxy efficiency",
                                                  numerator->GetNbinsX(),
                                                  numerator->GetXaxis()->GetXmin(),
                                                  numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_Lxy, "Tracking particle Lxy [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended displaced Lxy efficiency cannot be "
                                       "found!\n";
  }

  // nominal collection
  if (meN_z0 && meD_z0) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_z0->getTH1F();
    TH1F *denominator = meD_z0->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_z0 = ibooker.book1D("z0Efficiency",
                                                 "z0 efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_z0, "Tracking particle z0 [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for z0 efficiency cannot be found!\n";
  }

  // extended collection
  if (meN_prompt_z0 && meD_z0) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_prompt_z0->getTH1F();
    TH1F *denominator = meD_z0->getTH1F();
    numerator->Sumw2();
    if (!denominator->GetSumw2N()) {
      denominator->Sumw2();
    }

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_z0 = ibooker.book1D("z0Efficiency",
                                                 "Prompt z0 efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_z0, "Tracking particle z0 [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended prompt "
                                       "d0 efficiency cannot be found!\n";
  }

  if (meN_displaced_z0 && meD_displaced_z0) {
    // Get the numerator and denominator histograms
    TH1F *numerator = meN_displaced_z0->getTH1F();
    TH1F *denominator = meD_displaced_z0->getTH1F();
    numerator->Sumw2();
    denominator->Sumw2();

    // Set the current directory
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalEfficiency");

    // Book the new histogram to contain the results
    MonitorElement *me_effic_z0 = ibooker.book1D("z0Efficiency",
                                                 "Displaced z0 efficiency",
                                                 numerator->GetNbinsX(),
                                                 numerator->GetXaxis()->GetXmin(),
                                                 numerator->GetXaxis()->GetXmax());

    // Calculate the efficiency
    phase2tkutil::makeEfficiencyME(numerator, denominator, me_effic_z0, "Tracking particle z0 [cm]");
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended displaced z0 efficiency cannot be "
                                       "found!\n";
  }

  // nominal collection
  if (std::find(respt_pt2to3.begin(), respt_pt2to3.end(), nullptr) == respt_pt2to3.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt1 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt1[i] = respt_pt2to3[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt1 =
        ibooker.book1D("pTResVsEta_2-3", "p_{T} resolution vs |#eta|, for p_{T}: 2-3 GeV", eta_binnum, eta_bins);
    TH1F *resPt1 = me_res_pt1->getTH1F();
    resPt1->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt1->GetYaxis()->SetTitle("#sigma(#Delta p_{T}/p_{T})");
    resPt1->SetMinimum(0.0);
    resPt1->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt1->SetBinContent(i + 1, vResPt1[i]->GetStdDev());
      resPt1->SetBinError(i + 1, vResPt1[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution (2-3) cannot be found!\n";
  }

  // extended collection
  if (std::find(prompt_respt_pt2to3.begin(), prompt_respt_pt2to3.end(), nullptr) == prompt_respt_pt2to3.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt1 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt1[i] = prompt_respt_pt2to3[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt1 =
        ibooker.book1D("pTResVsEta_2-3_prompt", "p_{T} resolution vs |#eta|, for p_{T}: 2-3 GeV", eta_binnum, eta_bins);
    TH1F *resPt1 = me_res_pt1->getTH1F();
    resPt1->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt1->GetYaxis()->SetTitle("#sigma(#Delta p_{T}/p_{T})");
    resPt1->SetMinimum(0.0);
    resPt1->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt1->SetBinContent(i + 1, vResPt1[i]->GetStdDev());
      resPt1->SetBinError(i + 1, vResPt1[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended prompt "
                                       "pT resolution (2-3) cannot be found!\n";
  }

  if (std::find(displaced_respt_pt2to3.begin(), displaced_respt_pt2to3.end(), nullptr) ==
      displaced_respt_pt2to3.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt1 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt1[i] = displaced_respt_pt2to3[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt1 = ibooker.book1D(
        "pTResVsEta_2-3_displaced", "p_{T} resolution vs |#eta|, for p_{T}: 2-3 GeV", eta_binnum, eta_bins);
    TH1F *resPt1 = me_res_pt1->getTH1F();
    resPt1->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt1->GetYaxis()->SetTitle("#sigma(#Delta p_{T}/p_{T})");
    resPt1->SetMinimum(0.0);
    resPt1->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt1->SetBinContent(i + 1, vResPt1[i]->GetStdDev());
      resPt1->SetBinError(i + 1, vResPt1[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for extended displaced pT resolution (2-3) cannot "
                                       "be found!\n";
  }

  // nominal collection
  if (std::find(respt_pt3to8.begin(), respt_pt3to8.end(), nullptr) == respt_pt3to8.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt2 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt2[i] = respt_pt3to8[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt2 =
        ibooker.book1D("pTResVsEta_3-8", "p_{T} resolution vs |#eta|, for p_{T}: 3-8 GeV", eta_binnum, eta_bins);
    TH1F *resPt2 = me_res_pt2->getTH1F();
    resPt2->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt2->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
    resPt2->SetMinimum(0.0);
    resPt2->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt2->SetBinContent(i + 1, vResPt2[i]->GetStdDev());
      resPt2->SetBinError(i + 1, vResPt2[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution (3-8) cannot be found!\n";
  }

  // extended collection
  // prompt
  if (std::find(prompt_respt_pt3to8.begin(), prompt_respt_pt3to8.end(), nullptr) == prompt_respt_pt3to8.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt2 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt2[i] = prompt_respt_pt3to8[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt2 =
        ibooker.book1D("pTResVsEta_3-8_prompt", "p_{T} resolution vs |#eta|, for p_{T}: 3-8 GeV", eta_binnum, eta_bins);
    TH1F *resPt2 = me_res_pt2->getTH1F();
    resPt2->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt2->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
    resPt2->SetMinimum(0.0);
    resPt2->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt2->SetBinContent(i + 1, vResPt2[i]->GetStdDev());
      resPt2->SetBinError(i + 1, vResPt2[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution "
                                       "(3-8) (prompt) cannot be found!\n";
  }

  // displaced
  if (std::find(displaced_respt_pt3to8.begin(), displaced_respt_pt3to8.end(), nullptr) ==
      displaced_respt_pt3to8.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt2 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt2[i] = displaced_respt_pt3to8[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt2 = ibooker.book1D(
        "pTResVsEta_3-8_displaced", "p_{T} resolution vs |#eta|, for p_{T}: 3-8 GeV", eta_binnum, eta_bins);
    TH1F *resPt2 = me_res_pt2->getTH1F();
    resPt2->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt2->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
    resPt2->SetMinimum(0.0);
    resPt2->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt2->SetBinContent(i + 1, vResPt2[i]->GetStdDev());
      resPt2->SetBinError(i + 1, vResPt2[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution "
                                       "(3-8) (displaced) cannot be found!\n";
  }

  // nominal collection
  if (std::find(respt_pt8toInf.begin(), respt_pt8toInf.end(), nullptr) == respt_pt8toInf.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt3 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt3[i] = respt_pt8toInf[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt3 =
        ibooker.book1D("pTResVsEta_8-inf", "p_{T} resolution vs |#eta|, for p_{T}: >8 GeV", eta_binnum, eta_bins);
    TH1F *resPt3 = me_res_pt3->getTH1F();
    resPt3->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt3->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
    resPt3->SetMinimum(0.0);
    resPt3->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt3->SetBinContent(i + 1, vResPt3[i]->GetStdDev());
      resPt3->SetBinError(i + 1, vResPt3[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution (8-inf) cannot be found!\n";
  }

  // extended collection
  // prompt
  if (std::find(prompt_respt_pt8toInf.begin(), prompt_respt_pt8toInf.end(), nullptr) == prompt_respt_pt8toInf.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt3 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt3[i] = prompt_respt_pt8toInf[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt3 = ibooker.book1D(
        "pTResVsEta_8-inf_prompt", "p_{T} resolution vs |#eta|, for p_{T}: >8 GeV", eta_binnum, eta_bins);
    TH1F *resPt3 = me_res_pt3->getTH1F();
    resPt3->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt3->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
    resPt3->SetMinimum(0.0);
    resPt3->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt3->SetBinContent(i + 1, vResPt3[i]->GetStdDev());
      resPt3->SetBinError(i + 1, vResPt3[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution "
                                       "(8-inf) (prompt) cannot be found!\n";
  }

  // displaced
  if (std::find(displaced_respt_pt8toInf.begin(), displaced_respt_pt8toInf.end(), nullptr) ==
      displaced_respt_pt8toInf.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPt3 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPt3[i] = displaced_respt_pt8toInf[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_pt3 = ibooker.book1D(
        "pTResVsEta_8-inf_displaced", "p_{T} resolution vs |#eta|, for p_{T}: >8 GeV", eta_binnum, eta_bins);
    TH1F *resPt3 = me_res_pt3->getTH1F();
    resPt3->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPt3->GetYaxis()->SetTitle("#sigma(#Deltap_{T}/p_{T})");
    resPt3->SetMinimum(0.0);
    resPt3->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPt3->SetBinContent(i + 1, vResPt3[i]->GetStdDev());
      resPt3->SetBinError(i + 1, vResPt3[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for pT resolution "
                                       "(8-inf) (displaced) cannot be found!\n";
  }

  // nominal collection eta resolution
  if (std::find(mereseta_vect.begin(), mereseta_vect.end(), nullptr) == mereseta_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResEta = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResEta[i] = mereseta_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_eta = ibooker.book1D("EtaResolution", "#eta resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resEta = me_res_eta->getTH1F();
    resEta->GetXaxis()->SetTitle("tracking particle |#eta|");
    resEta->GetYaxis()->SetTitle("#sigma(#Delta#eta)");
    resEta->SetMinimum(0.0);
    resEta->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resEta->SetBinContent(i + 1, vResEta[i]->GetStdDev());
      resEta->SetBinError(i + 1, vResEta[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for eta resolution cannot be found!\n";
  }

  // extended collection eta resolution
  // prompt
  if (std::find(prompt_mereseta_vect.begin(), prompt_mereseta_vect.end(), nullptr) == prompt_mereseta_vect.end()) {
    // Set the current director
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalResolution");
    std::vector<TH1F *> vResEta(6, nullptr);
    for (int i = 0; i < 6; i++) {
      vResEta[i] = prompt_mereseta_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_eta =
        ibooker.book1D("EtaResolution_prompt", "#eta resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resEta = me_res_eta->getTH1F();
    resEta->GetXaxis()->SetTitle("tracking particle |#eta|");
    resEta->GetYaxis()->SetTitle("#sigma(#Delta#eta)");
    resEta->SetMinimum(0.0);
    resEta->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resEta->SetBinContent(i + 1, vResEta[i]->GetStdDev());
      resEta->SetBinError(i + 1, vResEta[i]->GetStdDevError());
    }
  } else {
    edm::LogWarning("DataNotFound") << "Monitor elements for eta resolution (prompt) cannot be found!";
  }

  // displaced
  if (std::find(displaced_mereseta_vect.begin(), displaced_mereseta_vect.end(), nullptr) ==
      displaced_mereseta_vect.end()) {
    // Set the current director
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalResolution");
    std::vector<TH1F *> vResEta(6, nullptr);
    for (int i = 0; i < 6; i++) {
      vResEta[i] = displaced_mereseta_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_eta =
        ibooker.book1D("EtaResolution_displaced", "#eta resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resEta = me_res_eta->getTH1F();
    resEta->GetXaxis()->SetTitle("tracking particle |#eta|");
    resEta->GetYaxis()->SetTitle("#sigma(#Delta#eta)");
    resEta->SetMinimum(0.0);
    resEta->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resEta->SetBinContent(i + 1, vResEta[i]->GetStdDev());
      resEta->SetBinError(i + 1, vResEta[i]->GetStdDevError());
    }
  } else {
    edm::LogWarning("DataNotFound") << "Monitor elements for eta resolution (displaced) cannot be found!";
  }

  // nominal collection phi resolution
  if (std::find(meresphi_vect.begin(), meresphi_vect.end(), nullptr) == meresphi_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPhi = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPhi[i] = meresphi_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_phi = ibooker.book1D("PhiResolution", "#phi resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resPhi = me_res_phi->getTH1F();
    resPhi->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPhi->GetYaxis()->SetTitle("#sigma(#Delta#phi)");
    resPhi->SetMinimum(0.0);
    resPhi->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPhi->SetBinContent(i + 1, vResPhi[i]->GetStdDev());
      resPhi->SetBinError(i + 1, vResPhi[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for phi resolution cannot be found!\n";
  }

  // extended collection phi resolution
  // prompt
  if (std::find(prompt_meresphi_vect.begin(), prompt_meresphi_vect.end(), nullptr) == prompt_meresphi_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPhi = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPhi[i] = prompt_meresphi_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_phi =
        ibooker.book1D("PhiResolution_prompt", "#phi resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resPhi = me_res_phi->getTH1F();
    resPhi->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPhi->GetYaxis()->SetTitle("#sigma(#Delta#phi)");
    resPhi->SetMinimum(0.0);
    resPhi->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPhi->SetBinContent(i + 1, vResPhi[i]->GetStdDev());
      resPhi->SetBinError(i + 1, vResPhi[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for phi resolution (prompt) cannot be found!\n";
  }

  // displaced
  if (std::find(displaced_meresphi_vect.begin(), displaced_meresphi_vect.end(), nullptr) ==
      displaced_meresphi_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResPhi = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResPhi[i] = displaced_meresphi_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_phi =
        ibooker.book1D("PhiResolution_displaced", "#phi resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resPhi = me_res_phi->getTH1F();
    resPhi->GetXaxis()->SetTitle("tracking particle |#eta|");
    resPhi->GetYaxis()->SetTitle("#sigma(#Delta#phi)");
    resPhi->SetMinimum(0.0);
    resPhi->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resPhi->SetBinContent(i + 1, vResPhi[i]->GetStdDev());
      resPhi->SetBinError(i + 1, vResPhi[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for phi resolution (displaced) cannot be found!\n";
  }

  // nominal collection z0 resolution
  if (std::find(meresz0_vect.begin(), meresz0_vect.end(), nullptr) == meresz0_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResz0 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResz0[i] = meresz0_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_z0 = ibooker.book1D("z0Resolution", "z0 resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resz0 = me_res_z0->getTH1F();
    resz0->GetXaxis()->SetTitle("tracking particle |#eta|");
    resz0->GetYaxis()->SetTitle("#sigma(#Deltaz0) [cm]");
    resz0->SetMinimum(0.0);
    resz0->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resz0->SetBinContent(i + 1, vResz0[i]->GetStdDev());
      resz0->SetBinError(i + 1, vResz0[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for z0 resolution cannot be found!\n";
  }

  // extended collection z0 resolution
  // prompt
  if (std::find(prompt_meresz0_vect.begin(), prompt_meresz0_vect.end(), nullptr) == prompt_meresz0_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResz0 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResz0[i] = prompt_meresz0_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_z0 = ibooker.book1D("z0Resolution_prompt", "z0 resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resz0 = me_res_z0->getTH1F();
    resz0->GetXaxis()->SetTitle("tracking particle |#eta|");
    resz0->GetYaxis()->SetTitle("#sigma(#Deltaz0) [cm]");
    resz0->SetMinimum(0.0);
    resz0->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resz0->SetBinContent(i + 1, vResz0[i]->GetStdDev());
      resz0->SetBinError(i + 1, vResz0[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for z0 resolution (prompt) cannot be found!\n";
  }

  // displaced
  if (std::find(displaced_meresz0_vect.begin(), displaced_meresz0_vect.end(), nullptr) ==
      displaced_meresz0_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResz0 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResz0[i] = displaced_meresz0_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_z0 =
        ibooker.book1D("z0Resolution_displaced", "z0 resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resz0 = me_res_z0->getTH1F();
    resz0->GetXaxis()->SetTitle("tracking particle |#eta|");
    resz0->GetYaxis()->SetTitle("#sigma(#Deltaz0) [cm]");
    resz0->SetMinimum(0.0);
    resz0->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resz0->SetBinContent(i + 1, vResz0[i]->GetStdDev());
      resz0->SetBinError(i + 1, vResz0[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for z0 resolution (displaced) cannot be found!\n";
  }

  // nominal collection d0 resolution
  if (std::find(meresd0_vect.begin(), meresd0_vect.end(), nullptr) == meresd0_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Nominal_L1TF/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResD0 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResD0[i] = meresd0_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_d0 = ibooker.book1D("d0Resolution", "d_{0} resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resd0 = me_res_d0->getTH1F();
    resd0->GetXaxis()->SetTitle("tracking particle |#eta|");
    resd0->GetYaxis()->SetTitle("#sigma(#Deltad_{0}) [cm]");
    resd0->SetMinimum(0.0);
    resd0->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resd0->SetBinContent(i + 1, vResD0[i]->GetStdDev());
      resd0->SetBinError(i + 1, vResD0[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for d0 resolution cannot be found!\n";
  }

  // extended collection d0 resolution
  // prompt
  if (std::find(prompt_meresd0_vect.begin(), prompt_meresd0_vect.end(), nullptr) == prompt_meresd0_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Prompt/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResD0 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResD0[i] = prompt_meresd0_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_d0 =
        ibooker.book1D("d0Resolution_prompt", "d_{0} resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resd0 = me_res_d0->getTH1F();
    resd0->GetXaxis()->SetTitle("tracking particle |#eta|");
    resd0->GetYaxis()->SetTitle("#sigma(#Deltad_{0}) [cm]");
    resd0->SetMinimum(0.0);
    resd0->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resd0->SetBinContent(i + 1, vResD0[i]->GetStdDev());
      resd0->SetBinError(i + 1, vResD0[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for d0 resolution (prompt) cannot be found!\n";
  }

  // displaced
  if (std::find(displaced_meresd0_vect.begin(), displaced_meresd0_vect.end(), nullptr) ==
      displaced_meresd0_vect.end()) {
    // Set the current directoy
    igetter.setCurrentFolder(topFolderName_ + "/Extended_L1TF/Displaced/FinalResolution");

    // Grab the histograms
    std::vector<TH1F *> vResD0 = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int i = 0; i < 6; i++) {
      vResD0[i] = displaced_meresd0_vect[i]->getTH1F();
    }

    // Book the new histogram to contain the results
    MonitorElement *me_res_d0 =
        ibooker.book1D("d0Resolution_displaced", "d_{0} resolution vs |#eta|", eta_binnum, eta_bins);
    TH1F *resd0 = me_res_d0->getTH1F();
    resd0->GetXaxis()->SetTitle("tracking particle |#eta|");
    resd0->GetYaxis()->SetTitle("#sigma(#Deltad_{0}) [cm]");
    resd0->SetMinimum(0.0);
    resd0->SetStats(false);

    for (int i = 0; i < 6; i++) {
      resd0->SetBinContent(i + 1, vResD0[i]->GetStdDev());
      resd0->SetBinError(i + 1, vResD0[i]->GetStdDevError());
    }
  }  // if ME found
  else {
    edm::LogWarning("DataNotFound") << "Monitor elements for d0 resolution (displaced) cannot be found!\n";
  }
}  // end dqmEndJob

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
void Phase2OTHarvestTracks::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("TopFolderName", "TrackerPhase2OTL1TrackV");
  descriptions.add("Phase2OTHarvestTracks", desc);
}
DEFINE_FWK_MODULE(Phase2OTHarvestTracks);
