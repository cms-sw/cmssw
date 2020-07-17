///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoManager.cc
// Description: Histogram managing class in HcalTestAnalysis (HcalTest)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HcalTestHistoManager.h"

#include "FWCore/PluginManager/interface/PluginManager.h"

#include <cmath>
#include <iostream>
#include <memory>


//#define EDM_ML_DEBUG

HcalTestHistoManager::HcalTestHistoManager(const std::string& file) : tree_(nullptr), kount_(0) {
  if (fs_.isAvailable()) {
    h_ = std::make_unique<HcalTestHistoClass>();

    tree_ = fs_->make<TTree>("HcalTest", "HcalTest");
    tree_->SetAutoSave(10000);
    tree_->Branch("HcalTestHisto", "HcalTestHistoClass", &h_);
    edm::LogVerbatim("HcalSim") << "HcalTestHistoManager:===>>>  Book the Tree";
  } else {
    edm::LogVerbatim("HcalSim") << "HcalTestHistoManager:===>>> No file provided";
  }
}

HcalTestHistoManager::~HcalTestHistoManager() {
  edm::LogVerbatim("HcalSim") << "================================================================="
                              << "====================\n=== HcalTestHistoManager: Start writing user "
                              << "histograms after " << kount_ << " events ";
}

void HcalTestHistoManager::fillTree(HcalTestHistoClass* histos) {
  ++kount_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalSim") << "HcalTestHistoManager: tree pointer for " << kount_ << " = " << histos;
#endif
  if (tree_) {
    h_.reset(histos);
    tree_->Fill();
  }
}
