///////////////////////////////////////////////////////////////////////////////
// File: HcalTestHistoManager.cc
// Description: Histogram managing class in HcalTestAnalysis (HcalTest)
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HcalTestHistoManager.h"

#include "FWCore/PluginManager/interface/PluginManager.h"

#include <iostream>
#include <cmath>

HcalTestHistoManager::HcalTestHistoManager(const std::string & file) :
  tree(0), h(0), kount(0) {
  if (fs.isAvailable()) {
    h    = new HcalTestHistoClass();

    tree = fs->make<TTree>("HcalTest", "HcalTest");
    tree->SetAutoSave(10000);
    tree->Branch("HcalTestHisto", "HcalTestHistoClass", &h); 
    edm::LogInfo("HcalSim") << "HcalTestHistoManager:===>>>  Book the Tree";
  } else {
    edm::LogInfo("HcalSim") << "HcalTestHistoManager:===>>> No file provided";
  }
}

HcalTestHistoManager::~HcalTestHistoManager() {

  edm::LogInfo("HcalSim") << "============================================="
			  << "========================================\n"
			  << "=== HcalTestHistoManager: Start writing user "
			  << "histograms after " << kount << " events ";
  if (h) delete h;
}

void HcalTestHistoManager::fillTree(HcalTestHistoClass *  histos) {

  kount++;
  LogDebug("HcalSim") << "HcalTestHistoManager: tree pointer for " << kount 
		      << " = " << histos;
  if (tree) {
    h = histos;
    tree->Fill();
  }
}
