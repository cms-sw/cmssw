// -*- C++ -*-
//
//
// Original Author:  Andreas Gruber
//         Created:  Mon, 16 Oct 2023 14:24:35 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "TH1.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/SimTauCPLink.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class SimTauAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SimTauAnalyzer(const edm::ParameterSet&);
  ~SimTauAnalyzer() override = default;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<std::vector<SimTauCPLink>> simTau_token_;
  TH1D* DM_histo;
};

SimTauAnalyzer::SimTauAnalyzer(const edm::ParameterSet& iConfig)
    : simTau_token_(consumes<std::vector<SimTauCPLink>>(iConfig.getParameter<edm::InputTag>("simTau"))) {
  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  DM_histo = fs->make<TH1D>("DM_histo", "DM_histo", 20, -1, 19);
}

// ------------ method called for each event  ------------
void SimTauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<std::vector<SimTauCPLink>> simTau_h;
  iEvent.getByToken(simTau_token_, simTau_h);

  const auto& simTaus = *simTau_h;

  for (auto const& simTau : simTaus) {
#ifdef EDM_ML_DEBUG
    simTau.dumpFullDecay();
    simTau.dump();
#endif
    DM_histo->Fill(simTau.decayMode);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimTauAnalyzer);
