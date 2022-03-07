#ifndef GENWEIGHTVALIDATION_H
#define GENWEIGHTVALIDATION_H

// framework & common header files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/Handle.h"
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
  ~GenWeightValidation() override = default;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;

private:
  void bookTemplates(DQMHelper &aDqmHelper,
                     std::vector<MonitorElement *> &tmps,
                     const std::string &name,
                     const std::string &title,
                     int nbin,
                     float low,
                     float high,
                     const std::string &xtitle,
                     const std::string &ytitle);
  void fillTemplates(std::vector<MonitorElement *> &tmps, float obs);
  WeightManager wmanager_;

  double weight_;
  std::vector<std::vector<double>> weights_;

  MonitorElement *nEvt_;
  MonitorElement *nlogWgt_;
  MonitorElement *wgtVal_;
  std::vector<MonitorElement *> leadLepPtTemp_;
  std::vector<MonitorElement *> leadLepEtaTemp_;
  std::vector<MonitorElement *> jetMultTemp_;
  std::vector<MonitorElement *> leadJetPtTemp_;
  std::vector<MonitorElement *> leadJetEtaTemp_;

  const edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
  const edm::EDGetTokenT<reco::GenJetCollection> genJetToken_;

  const int idxGenEvtInfo_, idxFSRup_, idxFSRdown_, idxISRup_, idxISRdown_, leadLepPtNbin_, rapidityNbin_;
  const double leadLepPtRange_, leadLepPtCut_, lepEtaCut_, rapidityRange_;
  const int nJetsNbin_, jetPtNbin_;
  const double jetPtCut_, jetEtaCut_, jetPtRange_;
  int idxMax_;
};

#endif
