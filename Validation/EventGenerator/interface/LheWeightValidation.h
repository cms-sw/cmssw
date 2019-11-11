#ifndef LHEWEIGHTVALIDATION_H
#define LHEWEIGHTVALIDATION_H

// framework & common header files
#include <algorithm>
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

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "Validation/EventGenerator/interface/DQMHelper.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

class TH1F; // forward declaration for ROOT

class LheWeightValidation : public DQMEDAnalyzer {
public:
  explicit LheWeightValidation(const edm::ParameterSet &);
  ~LheWeightValidation() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run &r, const edm::EventSetup &c) override;
  void dqmEndRun(const edm::Run& r, const edm::EventSetup& c) override;

private:
  void bookTemplates(std::vector<TH1F*>& scaleVar, std::vector<TH1F*>& pdfVar, std::vector<MonitorElement*>& tmps,
    std::string name, std::string title, int nbin, float low, float high, std::string xtitle, std::string ytitle);

  void fillTemplates(std::vector<TH1F*>& scaleVar, std::vector<TH1F*>& pdfVar, std::vector<MonitorElement*>& tmps, float obs);

  void envelop(const std::vector<TH1F*>& var, std::vector<MonitorElement*>& tmps);
  void pdfRMS(const std::vector<TH1F*>& var, std::vector<MonitorElement*>& tmps);
  DQMHelper* dqm;

  /// PDT table
  edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable;

  double weight, orgWgt;
  std::vector<LHEEventProduct::WGT> weights;

  MonitorElement* nEvt;
  MonitorElement* nlogWgt;
  MonitorElement* wgtVal;
  std::vector<MonitorElement*> leadLepPtTemp;
  std::vector<MonitorElement*> leadLepEtaTemp;
  std::vector<MonitorElement*> ZptTemp;
  std::vector<MonitorElement*> ZmassTemp;
  std::vector<MonitorElement*> ZrapidityTemp;
  std::vector<MonitorElement*> jetMultTemp;
  std::vector<MonitorElement*> leadJetPtTemp;
  std::vector<MonitorElement*> leadJetEtaTemp;

  std::vector<TH1F*> leadLepPtScaleVar;
  std::vector<TH1F*> leadLepPtPdfVar;
  std::vector<TH1F*> leadLepEtaScaleVar;
  std::vector<TH1F*> leadLepEtaPdfVar;
  std::vector<TH1F*> ZptScaleVar;
  std::vector<TH1F*> ZptPdfVar;
  std::vector<TH1F*> ZmassScaleVar;
  std::vector<TH1F*> ZmassPdfVar;
  std::vector<TH1F*> ZrapidityScaleVar;
  std::vector<TH1F*> ZrapidityPdfVar;
  std::vector<TH1F*> jetMultScaleVar;
  std::vector<TH1F*> jetMultPdfVar;
  std::vector<TH1F*> leadJetPtScaleVar;
  std::vector<TH1F*> leadJetPtPdfVar;
  std::vector<TH1F*> leadJetEtaScaleVar;
  std::vector<TH1F*> leadJetEtaPdfVar;

  edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken;
  edm::EDGetTokenT<LHEEventProduct> lheEvtToken;
  edm::EDGetTokenT<LHERunInfoProduct> lheRunToken;
  edm::EDGetTokenT<reco::GenJetCollection> genJetToken;

  bool dumpLHEheader;
  int leadLepPtNbin, ZptNbin, ZmassNbin, rapidityNbin;
  double leadLepPtRange, leadLepPtCut, subLeadLepPtCut, lepEtaCut, FSRdRCut;
  double ZptRange, ZmassLow, ZmassHigh, rapidityRange;
  int nJetsNbin, jetPtNbin;
  double jetPtCut, jetEtaCut, jetPtRange;

  int nScaleVar; // including Nominal
  int idxPdfStart, idxPdfEnd, nPdfVar;
};

#endif
