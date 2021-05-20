#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CaloHit/interface/MaterialInformation.h"

#include "TProfile.h"
#include "TProfile2D.h"

#include <iostream>
#include <string>
#include <vector>

//#define EDM_ML_DEBUG

using namespace geant_units::operators;

class MaterialBudgetVolumeAnalysis : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit MaterialBudgetVolumeAnalysis(edm::ParameterSet const&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void bookHisto();

  const std::vector<std::string> names_;
  const edm::InputTag tag_;
  const int binEta_, binPhi_;
  const double etaLow_, etaHigh_, phiLow_, phiHigh_;
  edm::EDGetTokenT<edm::MaterialInformationContainer> tok_info_;
  std::vector<TProfile*> meStepEta_, meStepPhi_;
  std::vector<TProfile*> meRadLEta_, meRadLPhi_;
  std::vector<TProfile*> meIntLEta_, meIntLPhi_;
  std::vector<TProfile2D*> meStepEtaPhi_, meRadLEtaPhi_, meIntLEtaPhi_;
};

MaterialBudgetVolumeAnalysis::MaterialBudgetVolumeAnalysis(const edm::ParameterSet& p)
    : names_(p.getParameter<std::vector<std::string> >("names")),
      tag_(p.getParameter<edm::InputTag>("inputTag")),
      binEta_(p.getParameter<int>("nBinEta")),
      binPhi_(p.getParameter<int>("nBinPhi")),
      etaLow_(p.getParameter<double>("etaLow")),
      etaHigh_(p.getParameter<double>("etaHigh")),
      phiLow_(-1._pi),
      phiHigh_(1._pi) {
  usesResource(TFileService::kSharedResource);
  tok_info_ = consumes<edm::MaterialInformationContainer>(tag_);

  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolumeAnalysis: Eta plot: NX " << binEta_ << " Range "
                                     << -etaLow_ << ":" << etaHigh_ << " Phi plot: NX " << binPhi_ << " Range "
                                     << -1._pi << ":" << 1._pi << " for " << names_.size() << " detectors from "
                                     << tag_;
  std::ostringstream st1;
  for (unsigned int k = 0; k < names_.size(); ++k)
    st1 << " [" << k << "] " << names_[k];
  edm::LogVerbatim("MaterialBudget") << "MaterialBudgetVolume: " << st1.str();
  bookHisto();
}

void MaterialBudgetVolumeAnalysis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names = {
      "BEAM", "BEAM1", "BEAM2", "BEAM3", "BEAM4", "Tracker", "ECAL", "HCal", "MUON", "VCAL", "MGNT", "OQUA", "CALOEC"};
  desc.add<std::vector<std::string> >("names", names);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("g4SimHits", "MaterialInformation"));
  desc.add<int>("nBinEta", 300);
  desc.add<int>("nBinPhi", 180);
  desc.add<double>("etaLow", -6.0);
  desc.add<double>("etaHigh", 6.0);
  descriptions.add("materialBudgetVolumeAnalysis", desc);
}

void MaterialBudgetVolumeAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<edm::MaterialInformationContainer> materialInformationContainer;
  iEvent.getByToken(tok_info_, materialInformationContainer);
#ifdef EDM_ML_DEBUG
  unsigned int nsize(0), ntot(0), nused(0);
#endif
  if (materialInformationContainer.isValid()) {
#ifdef EDM_ML_DEBUG
    nsize = materialInformationContainer->size();
#endif
    for (const auto& it : *(materialInformationContainer.product())) {
#ifdef EDM_ML_DEBUG
      ntot++;
#endif
      if (std::find(names_.begin(), names_.end(), it.vname()) != names_.end()) {
#ifdef EDM_ML_DEBUG
        nused++;
#endif
        unsigned int k =
            static_cast<unsigned int>(std::find(names_.begin(), names_.end(), it.vname()) - names_.begin());
        meStepEta_[k]->Fill(it.trackEta(), it.stepLength());
        meRadLEta_[k]->Fill(it.trackEta(), it.radiationLength());
        meIntLEta_[k]->Fill(it.trackEta(), it.interactionLength());
        meStepPhi_[k]->Fill(it.trackPhi(), it.stepLength());
        meRadLPhi_[k]->Fill(it.trackPhi(), it.radiationLength());
        meIntLPhi_[k]->Fill(it.trackPhi(), it.interactionLength());
        meStepEtaPhi_[k]->Fill(it.trackEta(), it.trackPhi(), it.stepLength());
        meRadLEtaPhi_[k]->Fill(it.trackEta(), it.trackPhi(), it.radiationLength());
        meIntLEtaPhi_[k]->Fill(it.trackEta(), it.trackPhi(), it.interactionLength());
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MaterialBudget") << "MaterialInformation with " << nsize << ":" << ntot << " elements of which "
                                     << nused << " are used";
#endif
}

void MaterialBudgetVolumeAnalysis::bookHisto() {
  edm::Service<TFileService> fs;
  char name[40], title[100];
  for (unsigned int k = 0; k < names_.size(); ++k) {
    sprintf(name, "stepEta%s", names_[k].c_str());
    sprintf(title, "MB(Step) vs #eta for %s", names_[k].c_str());
    meStepEta_.emplace_back(fs->make<TProfile>(name, title, binEta_, etaLow_, etaHigh_));
    sprintf(name, "radlEta%s", names_[k].c_str());
    sprintf(title, "MB(X0) vs #eta for %s", names_[k].c_str());
    meRadLEta_.emplace_back(fs->make<TProfile>(name, title, binEta_, etaLow_, etaHigh_));
    sprintf(name, "intlEta%s", names_[k].c_str());
    sprintf(title, "MB(L0) vs #eta for %s", names_[k].c_str());
    meIntLEta_.emplace_back(fs->make<TProfile>(name, title, binEta_, etaLow_, etaHigh_));
    sprintf(name, "stepPhi%s", names_[k].c_str());
    sprintf(title, "MB(Step) vs #phi for %s", names_[k].c_str());
    meStepPhi_.emplace_back(fs->make<TProfile>(name, title, binPhi_, phiLow_, phiHigh_));
    sprintf(name, "radlPhi%s", names_[k].c_str());
    sprintf(title, "MB(X0) vs #phi for %s", names_[k].c_str());
    meRadLPhi_.emplace_back(fs->make<TProfile>(name, title, binPhi_, phiLow_, phiHigh_));
    sprintf(name, "intlPhi%s", names_[k].c_str());
    sprintf(title, "MB(L0) vs #phi for %s", names_[k].c_str());
    meIntLPhi_.emplace_back(fs->make<TProfile>(name, title, binPhi_, phiLow_, phiHigh_));
    sprintf(name, "stepEtaPhi%s", names_[k].c_str());
    sprintf(title, "MB(Step) vs #eta and #phi for %s", names_[k].c_str());
    meStepEtaPhi_.emplace_back(
        fs->make<TProfile2D>(name, title, binEta_ / 2, etaLow_, etaHigh_, binPhi_ / 2, phiLow_, phiHigh_));
    sprintf(name, "radlEtaPhi%s", names_[k].c_str());
    sprintf(title, "MB(X0) vs #eta and #phi for %s", names_[k].c_str());
    meRadLEtaPhi_.emplace_back(
        fs->make<TProfile2D>(name, title, binEta_ / 2, etaLow_, etaHigh_, binPhi_ / 2, phiLow_, phiHigh_));
    sprintf(name, "intlEtaPhi%s", names_[k].c_str());
    sprintf(title, "MB(L0) vs #eta and #phi for %s", names_[k].c_str());
    meIntLEtaPhi_.emplace_back(
        fs->make<TProfile2D>(name, title, binEta_ / 2, etaLow_, etaHigh_, binPhi_ / 2, phiLow_, phiHigh_));
  }
}

// define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MaterialBudgetVolumeAnalysis);
