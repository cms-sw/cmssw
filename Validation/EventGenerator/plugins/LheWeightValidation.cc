#include "Validation/EventGenerator/interface/LheWeightValidation.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"
#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"

using namespace edm;

LheWeightValidation::LheWeightValidation(const edm::ParameterSet& iPSet) {
  lheLabel_ = iPSet.getParameter<edm::InputTag>("lheProduct"),
  genParticleToken = consumes<reco::GenParticleCollection>(iPSet.getParameter<edm::InputTag>("genParticles"));
  lheEvtToken = consumes<LHEEventProduct>(lheLabel_);
  lheRunToken = consumes<LHERunInfoProduct, edm::InRun>(lheLabel_);
  genJetToken = consumes<reco::GenJetCollection>(iPSet.getParameter<edm::InputTag>("genJets"));
  dumpLHEheader = iPSet.getParameter<bool>("dumpLHEheader");
  nScaleVar = iPSet.getParameter<int>("nScaleVar");
  idxPdfStart = iPSet.getParameter<int>("idxPdfStart");
  idxPdfEnd = iPSet.getParameter<int>("idxPdfEnd");
  leadLepPtRange = iPSet.getParameter<double>("leadLepPtRange");
  leadLepPtNbin = iPSet.getParameter<int>("leadLepPtNbin");
  leadLepPtCut = iPSet.getParameter<double>("leadLepPtCut");
  lepEtaCut = iPSet.getParameter<double>("lepEtaCut");
  rapidityRange = iPSet.getParameter<double>("rapidityRange");
  rapidityNbin = iPSet.getParameter<int>("rapidityNbin");
  jetPtCut = iPSet.getParameter<double>("jetPtCut");
  jetEtaCut = iPSet.getParameter<double>("jetEtaCut");
  nJetsNbin = iPSet.getParameter<int>("nJetsNbin");
  jetPtRange = iPSet.getParameter<double>("jetPtRange");
  jetPtNbin = iPSet.getParameter<int>("jetPtNbin");

  nPdfVar = idxPdfEnd - idxPdfStart + 1;
}

LheWeightValidation::~LheWeightValidation() {}

void LheWeightValidation::bookHistograms(DQMStore::IBooker& i, edm::Run const& iRun, edm::EventSetup const&) {
  // check LHE product exists
  edm::Handle<LHERunInfoProduct> lheInfo;
  iRun.getByLabel(lheLabel_, lheInfo);

  if (!lheInfo.isValid())
    return;

  ///Setting the DQM top directories
  std::string folderName = "Generator/LHEWeight";
  dqm = new DQMHelper(&i);
  i.setCurrentFolder(folderName);

  // Number of analyzed events
  nEvt = dqm->book1dHisto("nEvt", "n analyzed Events", 1, 0., 1., "bin", "Number of Events");
  nlogWgt = dqm->book1dHisto("nlogWgt", "Log10(n weights)", 100, 0., 5., "log_{10}(nWgts)", "Number of Events");
  wgtVal = dqm->book1dHisto("wgtVal", "weights", 100, -1.5, 3., "weight", "Number of Weights");

  bookTemplates(leadLepPtScaleVar,
                leadLepPtPdfVar,
                leadLepPtTemp,
                "leadLepPt",
                "leading lepton Pt",
                leadLepPtNbin,
                0.,
                leadLepPtRange,
                "Pt_{l} (GeV)",
                "Number of Events");
  bookTemplates(leadLepEtaScaleVar,
                leadLepEtaPdfVar,
                leadLepEtaTemp,
                "leadLepEta",
                "leading lepton #eta",
                rapidityNbin,
                -rapidityRange,
                rapidityRange,
                "#eta_{l}",
                "Number of Events");
  bookTemplates(jetMultScaleVar,
                jetMultPdfVar,
                jetMultTemp,
                "JetMultiplicity",
                "Gen jet multiplicity",
                nJetsNbin,
                0,
                nJetsNbin,
                "n",
                "Number of Events");
  bookTemplates(leadJetPtScaleVar,
                leadJetPtPdfVar,
                leadJetPtTemp,
                "leadJetPt",
                "leading Gen jet Pt",
                jetPtNbin,
                0.,
                jetPtRange,
                "Pt_{j} (GeV)",
                "Number of Events");
  bookTemplates(leadJetEtaScaleVar,
                leadJetEtaPdfVar,
                leadJetEtaTemp,
                "leadJetEta",
                "leading Gen jet #eta",
                rapidityNbin,
                -rapidityRange,
                rapidityRange,
                "#eta_{j}",
                "Number of Events");

  return;
}

void LheWeightValidation::bookTemplates(std::vector<std::unique_ptr<TH1F>>& scaleVar,
                                        std::vector<std::unique_ptr<TH1F>>& pdfVar,
                                        std::vector<MonitorElement*>& tmps,
                                        std::string name,
                                        std::string title,
                                        int nbin,
                                        float low,
                                        float high,
                                        std::string xtitle,
                                        std::string ytitle) {
  tmps.push_back(dqm->book1dHisto(name, title, nbin, low, high, xtitle, ytitle));
  tmps.push_back(dqm->book1dHisto(name + "ScaleUp", title + " scale up", nbin, low, high, xtitle, ytitle));
  tmps.at(1)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(name + "ScaleDn", title + " scale down", nbin, low, high, xtitle, ytitle));
  tmps.at(2)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(
      name + "ScaleUp_ratio", "Ratio of " + title + " scale upper envelop / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(3)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(
      name + "ScaleDn_ratio", "Ratio of " + title + " scale lower envelop / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(4)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(name + "PdfUp", title + " PDF upper RMS", nbin, low, high, xtitle, ytitle));
  tmps.at(5)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(name + "PdfDn", title + " PDF lower RMS", nbin, low, high, xtitle, ytitle));
  tmps.at(6)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(
      name + "PdfUp_ratio", "Ratio of " + title + " PDF upper RMS / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(7)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(
      name + "PdfDn_ratio", "Ratio of " + title + " PDF lower RMS / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(8)->setEfficiencyFlag();

  for (int idx = 0; idx < nScaleVar; idx++) {
    scaleVar.push_back(
        std::make_unique<TH1F>(std::string(name + "Scale" + std::to_string(idx)).c_str(),
                               std::string(";" + std::string(xtitle) + ";" + std::string(ytitle)).c_str(),
                               nbin,
                               low,
                               high));
    scaleVar.at(idx)->Sumw2();
  }

  for (int idx = 0; idx < nPdfVar; idx++) {
    pdfVar.push_back(std::make_unique<TH1F>(std::string(name + "Pdf" + std::to_string(idx)).c_str(),
                                            std::string(";" + std::string(xtitle) + ";" + std::string(ytitle)).c_str(),
                                            nbin,
                                            low,
                                            high));
    pdfVar.at(idx)->Sumw2();
  }
}  // to get ratio plots correctly - need to modify PostProcessor_cff.py as well!

void LheWeightValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void LheWeightValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<LHEEventProduct> lheEvt;

  if (!lheEvtToken.isUninitialized())
    iEvent.getByToken(lheEvtToken, lheEvt);

  if (!lheEvt.isValid())
    return;  // do nothing if there is no LHE product

  orgWgt = lheEvt->originalXWGTUP();
  weight = orgWgt / std::abs(orgWgt);
  weights = lheEvt->weights();

  nEvt->Fill(0.5, weight);
  nlogWgt->Fill(std::log10(lheEvt->weights().size()), weight);

  for (unsigned idx = 0; idx < lheEvt->weights().size(); idx++)
    wgtVal->Fill(weights[idx].wgt / orgWgt);

  edm::Handle<reco::GenParticleCollection> ptcls;
  iEvent.getByToken(genParticleToken, ptcls);
  edm::Handle<reco::GenJetCollection> genjets;
  iEvent.getByToken(genJetToken, genjets);

  std::vector<reco::GenParticleRef> leptons;

  for (unsigned iptc = 0; iptc < ptcls->size(); iptc++) {
    reco::GenParticleRef ptc(ptcls, iptc);
    if (ptc->status() == 1 && (std::abs(ptc->pdgId()) == 11 || std::abs(ptc->pdgId()) == 13)) {
      if (ptc->pt() > leadLepPtCut && std::abs(ptc->eta()) < lepEtaCut)
        leptons.push_back(ptc);
    }
  }

  std::sort(leptons.begin(), leptons.end(), HepMCValidationHelper::sortByPtRef<reco::GenParticleRef>);

  if (!leptons.empty()) {
    reco::GenParticleRef leadLep = leptons.at(0);
    fillTemplates(leadLepPtScaleVar, leadLepPtPdfVar, leadLepPtTemp, leadLep->pt());
    fillTemplates(leadLepEtaScaleVar, leadLepEtaPdfVar, leadLepEtaTemp, leadLep->eta());
  }

  std::vector<reco::GenJetRef> genjetVec;

  for (unsigned igj = 0; igj < genjets->size(); igj++) {
    reco::GenJetRef genjet(genjets, igj);

    if (genjet->pt() > jetPtCut && std::abs(genjet->eta()) < jetEtaCut)
      genjetVec.push_back(genjet);
  }

  fillTemplates(jetMultScaleVar, jetMultPdfVar, jetMultTemp, (float)genjetVec.size());

  if (!genjetVec.empty()) {
    std::sort(genjetVec.begin(), genjetVec.end(), HepMCValidationHelper::sortByPtRef<reco::GenJetRef>);

    auto leadJet = genjetVec.at(0);
    fillTemplates(leadJetPtScaleVar, leadJetPtPdfVar, leadJetPtTemp, leadJet->pt());
    fillTemplates(leadJetEtaScaleVar, leadJetEtaPdfVar, leadJetEtaTemp, leadJet->eta());
  }
}  //analyze

void LheWeightValidation::fillTemplates(std::vector<std::unique_ptr<TH1F>>& scaleVar,
                                        std::vector<std::unique_ptr<TH1F>>& pdfVar,
                                        std::vector<MonitorElement*>& tmps,
                                        float obs) {
  tmps.at(0)->Fill(obs, weight);

  if (static_cast<int>(weights.size()) >= nScaleVar) {
    for (int iWgt = 0; iWgt < nScaleVar; iWgt++)
      scaleVar.at(iWgt)->Fill(obs, weights[iWgt].wgt / orgWgt);
  }

  if (static_cast<int>(weights.size()) >= idxPdfEnd) {
    for (int iWgt = 0; iWgt < nPdfVar; iWgt++)
      pdfVar.at(iWgt)->Fill(obs, weights[idxPdfStart + iWgt].wgt / orgWgt);
  }
}

void LheWeightValidation::dqmEndRun(const edm::Run& iRun, const edm::EventSetup& c) {
  if (lheRunToken.isUninitialized())
    return;

  edm::Handle<LHERunInfoProduct> lheInfo;
  iRun.getByToken(lheRunToken, lheInfo);

  if (!lheInfo.isValid())
    return;

  envelop(leadLepPtScaleVar, leadLepPtTemp);
  pdfRMS(leadLepPtPdfVar, leadLepPtTemp);
  envelop(leadLepEtaScaleVar, leadLepEtaTemp);
  pdfRMS(leadLepEtaPdfVar, leadLepEtaTemp);
  envelop(jetMultScaleVar, jetMultTemp);
  pdfRMS(jetMultPdfVar, jetMultTemp);
  envelop(leadJetPtScaleVar, leadJetPtTemp);
  pdfRMS(leadJetPtPdfVar, leadJetPtTemp);
  envelop(leadJetEtaScaleVar, leadJetEtaTemp);
  pdfRMS(leadJetEtaPdfVar, leadJetEtaTemp);

  if (dumpLHEheader) {
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
