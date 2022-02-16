#ifndef GENWEIGHTVALIDATION_H
#define GENWEIGHTVALIDATION_H

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "Validation/EventGenerator/interface/WeightManager.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"

class GenWeightValidation : public DQMEDAnalyzer {
public:
  explicit GenWeightValidation(const edm::ParameterSet &);
  ~GenWeightValidation() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &r, const edm::EventSetup &c) override;

private:
  void bookTemplates(std::vector<MonitorElement*>& tmps, std::string name, std::string title, int nbin, float low, float high, std::string xtitle, std::string ytitle);
  void fillTemplates(std::vector<MonitorElement*>& tmps, float obs);
  WeightManager wmanager_;
  DQMHelper* dqm;

  double weight;
  std::vector<std::vector<double>> weights;

  MonitorElement* nEvt;
  MonitorElement* nlogWgt;
  MonitorElement* wgtVal;
  std::vector<MonitorElement*> leadLepPtTemp;
  std::vector<MonitorElement*> leadLepEtaTemp;
  std::vector<MonitorElement*> jetMultTemp;
  std::vector<MonitorElement*> leadJetPtTemp;
  std::vector<MonitorElement*> leadJetEtaTemp;

  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken;
  edm::EDGetTokenT<reco::GenJetCollection> genJetToken;

  int idxGenEvtInfo, idxFSRup, idxFSRdown, idxISRup, idxISRdown, leadLepPtNbin, rapidityNbin;
  int idxMax;
  double leadLepPtRange, leadLepPtCut, lepEtaCut, rapidityRange;
  int nJetsNbin, jetPtNbin;
  double jetPtCut, jetEtaCut, jetPtRange;
};

#endif
