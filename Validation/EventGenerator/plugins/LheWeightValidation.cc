#include "Validation/EventGenerator/interface/LheWeightValidation.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"

using namespace edm;

LheWeightValidation::LheWeightValidation(const edm::ParameterSet& iPSet)
    : lheLabel_(iPSet.getParameter<edm::InputTag>("lheProduct")),
      genParticleToken_(consumes<reco::GenParticleCollection>(iPSet.getParameter<edm::InputTag>("genParticles"))),
      lheEvtToken_(consumes<LHEEventProduct>(lheLabel_)),
      lheRunToken_(consumes<LHERunInfoProduct, edm::InRun>(lheLabel_)),
      genJetToken_(consumes<reco::GenJetCollection>(iPSet.getParameter<edm::InputTag>("genJets"))),
      dumpLHEheader_(iPSet.getParameter<bool>("dumpLHEheader")),
      leadLepPtNbin_(iPSet.getParameter<int>("leadLepPtNbin")),
      rapidityNbin_(iPSet.getParameter<int>("rapidityNbin")),
      leadLepPtRange_(iPSet.getParameter<double>("leadLepPtRange")),
      leadLepPtCut_(iPSet.getParameter<double>("leadLepPtCut")),
      lepEtaCut_(iPSet.getParameter<double>("lepEtaCut")),
      rapidityRange_(iPSet.getParameter<double>("rapidityRange")),
      nJetsNbin_(iPSet.getParameter<int>("nJetsNbin")),
      jetPtNbin_(iPSet.getParameter<int>("jetPtNbin")),
      jetPtCut_(iPSet.getParameter<double>("jetPtCut")),
      jetEtaCut_(iPSet.getParameter<double>("jetEtaCut")),
      jetPtRange_(iPSet.getParameter<double>("jetPtRange")),
      nScaleVar_(iPSet.getParameter<int>("nScaleVar")),
      idxPdfStart_(iPSet.getParameter<int>("idxPdfStart")),
      idxPdfEnd_(iPSet.getParameter<int>("idxPdfEnd")),
      nPdfVar_(idxPdfEnd_ - idxPdfStart_ + 1) {}

void LheWeightValidation::bookHistograms(DQMStore::IBooker& iBook, edm::Run const& iRun, edm::EventSetup const&) {
  // check LHE product exists
  edm::Handle<LHERunInfoProduct> lheInfo;
  // getByToken throws an exception unless we're in the endRun (see https://github.com/cms-sw/cmssw/pull/18499)
  iRun.getByLabel(lheLabel_, lheInfo);

  if (!lheInfo.isValid())
    return;

  ///Setting the DQM top directories
  std::string folderName = "Generator/LHEWeight";
  DQMHelper aDqmHelper(&iBook);
  iBook.setCurrentFolder(folderName);

  // Number of analyzed events
  nEvt_ = aDqmHelper.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1., "bin", "Number of Events");
  nlogWgt_ = aDqmHelper.book1dHisto("nlogWgt", "Log10(n weights)", 100, 0., 5., "log_{10}(nWgts)", "Number of Events");
  wgtVal_ = aDqmHelper.book1dHisto("wgtVal", "weights", 100, -1.5, 3., "weight", "Number of Weights");

  bookTemplates(aDqmHelper,
                leadLepPtScaleVar_,
                leadLepPtPdfVar_,
                leadLepPtTemp_,
                "leadLepPt",
                "leading lepton Pt",
                leadLepPtNbin_,
                0.,
                leadLepPtRange_,
                "Pt_{l} (GeV)",
                "Number of Events");
  bookTemplates(aDqmHelper,
                leadLepEtaScaleVar_,
                leadLepEtaPdfVar_,
                leadLepEtaTemp_,
                "leadLepEta",
                "leading lepton #eta",
                rapidityNbin_,
                -rapidityRange_,
                rapidityRange_,
                "#eta_{l}",
                "Number of Events");
  bookTemplates(aDqmHelper,
                jetMultScaleVar_,
                jetMultPdfVar_,
                jetMultTemp_,
                "JetMultiplicity",
                "Gen jet multiplicity",
                nJetsNbin_,
                0,
                nJetsNbin_,
                "n",
                "Number of Events");
  bookTemplates(aDqmHelper,
                leadJetPtScaleVar_,
                leadJetPtPdfVar_,
                leadJetPtTemp_,
                "leadJetPt",
                "leading Gen jet Pt",
                jetPtNbin_,
                0.,
                jetPtRange_,
                "Pt_{j} (GeV)",
                "Number of Events");
  bookTemplates(aDqmHelper,
                leadJetEtaScaleVar_,
                leadJetEtaPdfVar_,
                leadJetEtaTemp_,
                "leadJetEta",
                "leading Gen jet #eta",
                rapidityNbin_,
                -rapidityRange_,
                rapidityRange_,
                "#eta_{j}",
                "Number of Events");

  return;
}

void LheWeightValidation::bookTemplates(DQMHelper& aDqmHelper,
                                        std::vector<std::unique_ptr<TH1F>>& scaleVar,
                                        std::vector<std::unique_ptr<TH1F>>& pdfVar,
                                        std::vector<MonitorElement*>& tmps,
                                        const std::string& name,
                                        const std::string& title,
                                        int nbin,
                                        float low,
                                        float high,
                                        const std::string& xtitle,
                                        const std::string& ytitle) {
  tmps.push_back(aDqmHelper.book1dHisto(name, title, nbin, low, high, xtitle, ytitle));
  tmps.push_back(aDqmHelper.book1dHisto(name + "ScaleUp", title + " scale up", nbin, low, high, xtitle, ytitle));
  tmps.at(1)->getTH1()->Sumw2(false);
  tmps.push_back(aDqmHelper.book1dHisto(name + "ScaleDn", title + " scale down", nbin, low, high, xtitle, ytitle));
  tmps.at(2)->getTH1()->Sumw2(false);
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "ScaleUp_ratio", "Ratio of " + title + " scale upper envelop / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(3)->setEfficiencyFlag();
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "ScaleDn_ratio", "Ratio of " + title + " scale lower envelop / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(4)->setEfficiencyFlag();
  tmps.push_back(aDqmHelper.book1dHisto(name + "PdfUp", title + " PDF upper RMS", nbin, low, high, xtitle, ytitle));
  tmps.at(5)->getTH1()->Sumw2(false);
  tmps.push_back(aDqmHelper.book1dHisto(name + "PdfDn", title + " PDF lower RMS", nbin, low, high, xtitle, ytitle));
  tmps.at(6)->getTH1()->Sumw2(false);
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "PdfUp_ratio", "Ratio of " + title + " PDF upper RMS / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(7)->setEfficiencyFlag();
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "PdfDn_ratio", "Ratio of " + title + " PDF lower RMS / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(8)->setEfficiencyFlag();

  for (int idx = 0; idx < nScaleVar_; idx++) {
    scaleVar.push_back(
        std::make_unique<TH1F>(std::string(name + "Scale" + std::to_string(idx)).c_str(),
                               std::string(";" + std::string(xtitle) + ";" + std::string(ytitle)).c_str(),
                               nbin,
                               low,
                               high));
    scaleVar.at(idx)->Sumw2();
  }

  for (int idx = 0; idx < nPdfVar_; idx++) {
    pdfVar.push_back(std::make_unique<TH1F>(std::string(name + "Pdf" + std::to_string(idx)).c_str(),
                                            std::string(";" + std::string(xtitle) + ";" + std::string(ytitle)).c_str(),
                                            nbin,
                                            low,
                                            high));
    pdfVar.at(idx)->Sumw2();
  }
}  // to get ratio plots correctly - need to modify PostProcessor_cff.py as well!

void LheWeightValidation::dqmBeginRun(const edm::Run&, const edm::EventSetup&) {}

void LheWeightValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<LHEEventProduct> lheEvt;

  if (!lheEvtToken_.isUninitialized())
    iEvent.getByToken(lheEvtToken_, lheEvt);

  if (!lheEvt.isValid())
    return;  // do nothing if there is no LHE product

  orgWgt_ = lheEvt->originalXWGTUP();
  weight_ = orgWgt_ / std::abs(orgWgt_);
  weights_ = lheEvt->weights();

  nEvt_->Fill(0.5, weight_);
  nlogWgt_->Fill(std::log10(lheEvt->weights().size()), weight_);

  for (unsigned idx = 0; idx < lheEvt->weights().size(); idx++)
    wgtVal_->Fill(weights_[idx].wgt / orgWgt_);

  edm::Handle<reco::GenParticleCollection> ptcls;
  iEvent.getByToken(genParticleToken_, ptcls);
  edm::Handle<reco::GenJetCollection> genjets;
  iEvent.getByToken(genJetToken_, genjets);

  std::vector<reco::GenParticleRef> leptons;

  for (unsigned iptc = 0; iptc < ptcls->size(); iptc++) {
    reco::GenParticleRef ptc(ptcls, iptc);
    if (ptc->status() == 1 && (std::abs(ptc->pdgId()) == 11 || std::abs(ptc->pdgId()) == 13)) {
      if (ptc->pt() > leadLepPtCut_ && std::abs(ptc->eta()) < lepEtaCut_)
        leptons.push_back(ptc);
    }
  }

  std::sort(leptons.begin(), leptons.end(), HepMCValidationHelper::sortByPtRef<reco::GenParticleRef>);

  if (!leptons.empty()) {
    reco::GenParticleRef leadLep = leptons.at(0);
    fillTemplates(leadLepPtScaleVar_, leadLepPtPdfVar_, leadLepPtTemp_, leadLep->pt());
    fillTemplates(leadLepEtaScaleVar_, leadLepEtaPdfVar_, leadLepEtaTemp_, leadLep->eta());
  }

  std::vector<reco::GenJetRef> genjetVec;

  for (unsigned igj = 0; igj < genjets->size(); igj++) {
    reco::GenJetRef genjet(genjets, igj);

    if (genjet->pt() > jetPtCut_ && std::abs(genjet->eta()) < jetEtaCut_)
      genjetVec.push_back(genjet);
  }

  fillTemplates(jetMultScaleVar_, jetMultPdfVar_, jetMultTemp_, (float)genjetVec.size());

  if (!genjetVec.empty()) {
    std::sort(genjetVec.begin(), genjetVec.end(), HepMCValidationHelper::sortByPtRef<reco::GenJetRef>);

    auto leadJet = genjetVec.at(0);
    fillTemplates(leadJetPtScaleVar_, leadJetPtPdfVar_, leadJetPtTemp_, leadJet->pt());
    fillTemplates(leadJetEtaScaleVar_, leadJetEtaPdfVar_, leadJetEtaTemp_, leadJet->eta());
  }
}  //analyze

void LheWeightValidation::fillTemplates(std::vector<std::unique_ptr<TH1F>>& scaleVar,
                                        std::vector<std::unique_ptr<TH1F>>& pdfVar,
                                        std::vector<MonitorElement*>& tmps,
                                        float obs) {
  tmps.at(0)->Fill(obs, weight_);

  if (static_cast<int>(weights_.size()) >= nScaleVar_) {
    for (int iWgt = 0; iWgt < nScaleVar_; iWgt++)
      scaleVar.at(iWgt)->Fill(obs, weights_[iWgt].wgt / orgWgt_);
  }

  if (static_cast<int>(weights_.size()) >= idxPdfEnd_) {
    for (int iWgt = 0; iWgt < nPdfVar_; iWgt++)
      pdfVar.at(iWgt)->Fill(obs, weights_[idxPdfStart_ + iWgt].wgt / orgWgt_);
  }
}

void LheWeightValidation::dqmEndRun(const edm::Run& iRun, const edm::EventSetup&) {
  if (lheRunToken_.isUninitialized())
    return;

  edm::Handle<LHERunInfoProduct> lheInfo;
  iRun.getByToken(lheRunToken_, lheInfo);

  if (!lheInfo.isValid())
    return;

  envelop(leadLepPtScaleVar_, leadLepPtTemp_);
  pdfRMS(leadLepPtPdfVar_, leadLepPtTemp_);
  envelop(leadLepEtaScaleVar_, leadLepEtaTemp_);
  pdfRMS(leadLepEtaPdfVar_, leadLepEtaTemp_);
  envelop(jetMultScaleVar_, jetMultTemp_);
  pdfRMS(jetMultPdfVar_, jetMultTemp_);
  envelop(leadJetPtScaleVar_, leadJetPtTemp_);
  pdfRMS(leadJetPtPdfVar_, leadJetPtTemp_);
  envelop(leadJetEtaScaleVar_, leadJetEtaTemp_);
  pdfRMS(leadJetEtaPdfVar_, leadJetEtaTemp_);

  if (dumpLHEheader_) {
    for (auto it = lheInfo->headers_begin(); it != lheInfo->headers_end(); it++) {
      std::cout << "Header start" << std::endl;
      std::cout << "Tag: " << it->tag() << std::endl;
      for (const auto& l : it->lines()) {
        std::cout << l << std::endl;
      }
      std::cout << "Header end" << std::endl;
    }
  }
}

void LheWeightValidation::envelop(const std::vector<std::unique_ptr<TH1F>>& var, std::vector<MonitorElement*>& tmps) {
  if (var.empty())
    return;

  for (int b = 0; b < var.at(0)->GetNbinsX() + 2; b++) {
    float valU = var.at(0)->GetBinContent(b);
    float valD = valU;

    if (valU == 0.)
      continue;

    for (unsigned v = 1; v < var.size(); v++) {
      if (var.at(v)->GetEntries() == 0.)
        continue;

      valU = std::max(valU, (float)var.at(v)->GetBinContent(b));
      valD = std::min(valD, (float)var.at(v)->GetBinContent(b));
    }
    tmps.at(1)->setBinContent(b, valU);
    tmps.at(2)->setBinContent(b, valD);
  }

  tmps.at(1)->setEntries(var.at(0)->GetEntries());
  tmps.at(2)->setEntries(var.at(0)->GetEntries());
  tmps.at(1)->getTH1()->Sumw2(true);
  tmps.at(2)->getTH1()->Sumw2(true);

  return;
}

void LheWeightValidation::pdfRMS(const std::vector<std::unique_ptr<TH1F>>& var, std::vector<MonitorElement*>& tmps) {
  if (var.empty())
    return;

  float denom = var.size();
  for (int b = 0; b < tmps.at(0)->getNbinsX() + 2; b++) {
    float valNom = tmps.at(0)->getBinContent(b);
    float rmsSq = 0.;
    if (valNom == 0.)
      continue;

    for (unsigned v = 0; v < var.size(); v++) {
      if (var.at(v)->GetEntries() == 0.)
        continue;

      float dev = (float)var.at(v)->GetBinContent(b) - valNom;
      rmsSq += dev * dev;
    }

    float rms = std::sqrt(rmsSq / denom);
    float rmsup = valNom + rms;
    float rmsdn = valNom - rms;
    tmps.at(5)->setBinContent(b, rmsup);
    tmps.at(6)->setBinContent(b, rmsdn);
  }

  tmps.at(5)->setEntries(tmps.at(0)->getTH1F()->GetEntries());
  tmps.at(6)->setEntries(tmps.at(0)->getTH1F()->GetEntries());
  tmps.at(5)->getTH1()->Sumw2(true);
  tmps.at(6)->getTH1()->Sumw2(true);

  return;
}
