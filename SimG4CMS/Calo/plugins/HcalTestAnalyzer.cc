///////////////////////////////////////////////////////////////////////////////
// File: HcalTestAnalyzer.h
// Histogram managing class for analysis in HcalTest
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/CaloTest/interface/HcalTestHistoClass.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>

class HcalTestAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  HcalTestAnalyzer(const edm::ParameterSet&);
  ~HcalTestAnalyzer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  TTree* tree_;
  HcalTestHistoClass h_;
  const edm::EDGetTokenT<HcalTestHistoClass> tokHist_;
  int kount_;
};

HcalTestAnalyzer::HcalTestAnalyzer(const edm::ParameterSet&)
    : tree_(nullptr), tokHist_(consumes<HcalTestHistoClass>(edm::InputTag("g4SimHits"))), kount_(0) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  if (fs.isAvailable()) {
    tree_ = fs->make<TTree>("HcalTest", "HcalTest");
    tree_->SetAutoSave(10000);
    tree_->Branch("HcalTestHistoClass", &h_);
    edm::LogVerbatim("HcalSim") << "HcalTestAnalyzer:===>>>  Book the Tree";
  } else {
    edm::LogVerbatim("HcalSim") << "HcalTestAnalyzer:===>>> No file provided";
  }
}

HcalTestAnalyzer::~HcalTestAnalyzer() {
  edm::LogVerbatim("HcalSim") << "================================================================="
                              << "====================\n=== HcalTestAnalyzer: Start writing user "
                              << "histograms after " << kount_ << " events ";
}

void HcalTestAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("hcalTestAnalyzer", desc);
}

void HcalTestAnalyzer::analyze(const edm::Event& e, const edm::EventSetup&) {
  ++kount_;
  const auto& histos = e.getHandle(tokHist_);
  edm::LogVerbatim("HcalSim") << "HcalTestAnalyzer: [" << kount_ << "] event " << e.id().event() << " with "
                              << histos.isValid();

  if ((tree_) && histos.isValid()) {
    auto histo = histos.product();
    edm::LogVerbatim("HcalSim") << "HcalTestAnalyzer: tree pointer = " << histo;
    h_ = *histo;
    tree_->Fill();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalTestAnalyzer);
