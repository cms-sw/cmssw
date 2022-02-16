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

class TH1F; // forward declaration for ROOT

class LheWeightValidation : public DQMOneEDAnalyzer<> {
public:
  explicit LheWeightValidation(const edm::ParameterSet &);
  ~LheWeightValidation() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &r, const edm::EventSetup &c) override;
  void dqmEndRun(const edm::Run& r, const edm::EventSetup& c) override;

private:
  void bookTemplates(std::vector<std::unique_ptr<TH1F>>& scaleVar, std::vector<std::unique_ptr<TH1F>>& pdfVar, std::vector<MonitorElement*>& tmps,
    std::string name, std::string title, int nbin, float low, float high, std::string xtitle, std::string ytitle);

  void fillTemplates(std::vector<std::unique_ptr<TH1F>>& scaleVar, std::vector<std::unique_ptr<TH1F>>& pdfVar, std::vector<MonitorElement*>& tmps, float obs);

  void envelop(const std::vector<std::unique_ptr<TH1F>>& var, std::vector<MonitorElement*>& tmps);
  void pdfRMS(const std::vector<std::unique_ptr<TH1F>>& var, std::vector<MonitorElement*>& tmps);
  DQMHelper* dqm;

  double weight, orgWgt;
  std::vector<LHEEventProduct::WGT> weights;

  MonitorElement* nEvt;
  MonitorElement* nlogWgt;
  MonitorElement* wgtVal;
  std::vector<MonitorElement*> leadLepPtTemp;
  std::vector<MonitorElement*> leadLepEtaTemp;
  std::vector<MonitorElement*> jetMultTemp;
  std::vector<MonitorElement*> leadJetPtTemp;
  std::vector<MonitorElement*> leadJetEtaTemp;

  std::vector<std::unique_ptr<TH1F>> leadLepPtScaleVar;
  std::vector<std::unique_ptr<TH1F>> leadLepPtPdfVar;
  std::vector<std::unique_ptr<TH1F>> leadLepEtaScaleVar;
  std::vector<std::unique_ptr<TH1F>> leadLepEtaPdfVar;
  std::vector<std::unique_ptr<TH1F>> jetMultScaleVar;
  std::vector<std::unique_ptr<TH1F>> jetMultPdfVar;
  std::vector<std::unique_ptr<TH1F>> leadJetPtScaleVar;
  std::vector<std::unique_ptr<TH1F>> leadJetPtPdfVar;
  std::vector<std::unique_ptr<TH1F>> leadJetEtaScaleVar;
  std::vector<std::unique_ptr<TH1F>> leadJetEtaPdfVar;

  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken;
  edm::EDGetTokenT<LHEEventProduct> lheEvtToken;
  edm::EDGetTokenT<LHERunInfoProduct> lheRunToken;
  edm::EDGetTokenT<reco::GenJetCollection> genJetToken;

  bool dumpLHEheader;
  int leadLepPtNbin, rapidityNbin;
  double leadLepPtRange, leadLepPtCut, lepEtaCut, rapidityRange;
  int nJetsNbin, jetPtNbin;
  double jetPtCut, jetEtaCut, jetPtRange;

  int nScaleVar; // including Nominal
  int idxPdfStart, idxPdfEnd, nPdfVar;
};

#endif
