#ifndef LHEWEIGHTVALIDATION_H
#define LHEWEIGHTVALIDATION_H

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
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "Validation/EventGenerator/interface/DQMHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

class TH1F;  // forward declaration for ROOT

class LheWeightValidation : public DQMOneEDAnalyzer<> {
public:
  explicit LheWeightValidation(const edm::ParameterSet&);
  ~LheWeightValidation() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void dqmEndRun(const edm::Run&, const edm::EventSetup&) override;

private:
  void bookTemplates(std::vector<std::unique_ptr<TH1F>>& scaleVar,
                     std::vector<std::unique_ptr<TH1F>>& pdfVar,
                     std::vector<MonitorElement*>& tmps,
                     std::string name,
                     std::string title,
                     int nbin,
                     float low,
                     float high,
                     std::string xtitle,
                     std::string ytitle);

  void fillTemplates(std::vector<std::unique_ptr<TH1F>>& scaleVar,
                     std::vector<std::unique_ptr<TH1F>>& pdfVar,
                     std::vector<MonitorElement*>& tmps,
                     float obs);

  void envelop(const std::vector<std::unique_ptr<TH1F>>& var, std::vector<MonitorElement*>& tmps);
  void pdfRMS(const std::vector<std::unique_ptr<TH1F>>& var, std::vector<MonitorElement*>& tmps);
  DQMHelper* dqm_;

  double weight_, orgWgt_;
  std::vector<LHEEventProduct::WGT> weights_;

  MonitorElement* nEvt_;
  MonitorElement* nlogWgt_;
  MonitorElement* wgtVal_;
  std::vector<MonitorElement*> leadLepPtTemp_;
  std::vector<MonitorElement*> leadLepEtaTemp_;
  std::vector<MonitorElement*> jetMultTemp_;
  std::vector<MonitorElement*> leadJetPtTemp_;
  std::vector<MonitorElement*> leadJetEtaTemp_;

  std::vector<std::unique_ptr<TH1F>> leadLepPtScaleVar_;
  std::vector<std::unique_ptr<TH1F>> leadLepPtPdfVar_;
  std::vector<std::unique_ptr<TH1F>> leadLepEtaScaleVar_;
  std::vector<std::unique_ptr<TH1F>> leadLepEtaPdfVar_;
  std::vector<std::unique_ptr<TH1F>> jetMultScaleVar_;
  std::vector<std::unique_ptr<TH1F>> jetMultPdfVar_;
  std::vector<std::unique_ptr<TH1F>> leadJetPtScaleVar_;
  std::vector<std::unique_ptr<TH1F>> leadJetPtPdfVar_;
  std::vector<std::unique_ptr<TH1F>> leadJetEtaScaleVar_;
  std::vector<std::unique_ptr<TH1F>> leadJetEtaPdfVar_;

  edm::InputTag lheLabel_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
  edm::EDGetTokenT<LHEEventProduct> lheEvtToken_;
  edm::EDGetTokenT<LHERunInfoProduct> lheRunToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genJetToken_;

  bool dumpLHEheader_;
  int leadLepPtNbin_, rapidityNbin_;
  double leadLepPtRange_, leadLepPtCut_, lepEtaCut_, rapidityRange_;
  int nJetsNbin_, jetPtNbin_;
  double jetPtCut_, jetEtaCut_, jetPtRange_;

  int nScaleVar_;  // including Nominal
  int idxPdfStart_, idxPdfEnd_, nPdfVar_;
};

#endif
