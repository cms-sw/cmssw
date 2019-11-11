#include "Validation/EventGenerator/interface/LheWeightValidation.h"
#include "TLorentzVector.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"
#include "Validation/EventGenerator/interface/GenPtcValidationHelper.h"

#include "TH1F.h"
#include "TString.h"
#include <algorithm>
#include <cmath>
using namespace edm;

LheWeightValidation::LheWeightValidation(const edm::ParameterSet& iPSet)
{
  genParticleToken = consumes<reco::GenParticleCollection>(iPSet.getParameter<edm::InputTag>("genParticles"));
  lheEvtToken = consumes<LHEEventProduct>(iPSet.getParameter<edm::InputTag>("lheProduct"));
  lheRunToken = consumes<LHERunInfoProduct,edm::InRun>(iPSet.getParameter<edm::InputTag>("lheProduct"));
  genJetToken = consumes<reco::GenJetCollection>(iPSet.getParameter<edm::InputTag>("genJets"));
  dumpLHEheader = iPSet.getParameter<bool>("dumpLHEheader");
  nScaleVar = iPSet.getParameter<int>("nScaleVar");
  idxPdfStart = iPSet.getParameter<int>("idxPdfStart");
  idxPdfEnd = iPSet.getParameter<int>("idxPdfEnd");
  leadLepPtRange = iPSet.getParameter<double>("leadLepPtRange");
  leadLepPtNbin = iPSet.getParameter<int>("leadLepPtNbin");
  leadLepPtCut = iPSet.getParameter<double>("leadLepPtCut");
  subLeadLepPtCut = iPSet.getParameter<double>("subLeadLepPtCut");
  lepEtaCut = iPSet.getParameter<double>("lepEtaCut");
  FSRdRCut = iPSet.getParameter<double>("FSRdRCut");
  ZptRange = iPSet.getParameter<double>("ZptRange");
  ZptNbin = iPSet.getParameter<int>("ZptNbin");
  ZmassLow = iPSet.getParameter<double>("ZmassLow");
  ZmassHigh = iPSet.getParameter<double>("ZmassHigh");
  ZmassNbin = iPSet.getParameter<int>("ZmassNbin");
  rapidityRange = iPSet.getParameter<double>("rapidityRange");
  rapidityNbin = iPSet.getParameter<int>("rapidityNbin");
  jetPtCut = iPSet.getParameter<double>("jetPtCut");
  jetEtaCut = iPSet.getParameter<double>("jetEtaCut");
  nJetsNbin = iPSet.getParameter<int>("nJetsNbin");
  jetPtRange = iPSet.getParameter<double>("jetPtRange");
  jetPtNbin = iPSet.getParameter<int>("jetPtNbin");

  nPdfVar = idxPdfEnd - idxPdfStart + 1; // from 1973 to 2075 (103 variations of 292200), total 1080, start idx 1001
}

LheWeightValidation::~LheWeightValidation() {}

void LheWeightValidation::bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) {
  ///Setting the DQM top directories
  std::string folderName = "Generator/LHEWeight";
  dqm = new DQMHelper(&i);
  i.setCurrentFolder(folderName);

  // Number of analyzed events
  nEvt = dqm->book1dHisto("nEvt", "n analyzed Events", 1, 0., 1., "bin", "Number of Events");
  nlogWgt = dqm->book1dHisto("nlogWgt","Log10(n weights)",100,0.,5.,"log_{10}(nWgts)","Number of Events");
  wgtVal = dqm->book1dHisto("wgtVal","weights",100,-1.5,3.,"weight","Number of Weigths");

  bookTemplates(leadLepPtScaleVar,leadLepPtPdfVar,leadLepPtTemp,"leadLepPt","leading lepton Pt",leadLepPtNbin,0.,leadLepPtRange,"Pt_{l} (GeV)","Number of Events");
  bookTemplates(leadLepEtaScaleVar,leadLepEtaPdfVar,leadLepEtaTemp,"leadLepEta","leading lepton #eta",rapidityNbin,-rapidityRange,rapidityRange,"#eta_{l}","Number of Events");
  bookTemplates(ZptScaleVar,ZptPdfVar,ZptTemp,"Zpt","Z Pt",ZptNbin,0.,ZptRange,"Pt_{Z} (GeV)","Number of Events");
  bookTemplates(ZmassScaleVar,ZmassPdfVar,ZmassTemp,"Zmass","Z mass",ZmassNbin,ZmassLow,ZmassHigh,"M_{Z} (GeV)","Numberr of Events");
  bookTemplates(ZrapidityScaleVar,ZrapidityPdfVar,ZrapidityTemp,"Zrapidity","Z rapidity",rapidityNbin,-rapidityRange,rapidityRange,"Y_{Z}","Number of Events");
  bookTemplates(jetMultScaleVar,jetMultPdfVar,jetMultTemp,"JetMultiplicity","Gen jet multiplicity",nJetsNbin,0,nJetsNbin,"n","Number of Events");
  bookTemplates(leadJetPtScaleVar,leadJetPtPdfVar,leadJetPtTemp,"leadJetPt","leading Gen jet Pt",jetPtNbin,0.,jetPtRange,"Pt_{j} (GeV)","Number of Events");
  bookTemplates(leadJetEtaScaleVar,leadJetEtaPdfVar,leadJetEtaTemp,"leadJetEta","leading Gen jet #eta",rapidityNbin,-rapidityRange,rapidityRange,"#eta_{j}","Number of Events");

  return;
}

void LheWeightValidation::bookTemplates(std::vector<TH1F*>& scaleVar, std::vector<TH1F*>& pdfVar, std::vector<MonitorElement*>& tmps,
  std::string name, std::string title, int nbin, float low, float high, std::string xtitle, std::string ytitle) {

  tmps.push_back(dqm->book1dHisto(name,title,nbin,low,high,xtitle,ytitle));
  tmps.push_back(dqm->book1dHisto(name+"ScaleUp",title+" scale up",nbin,low,high,xtitle,ytitle)); tmps.at(1)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(name+"ScaleDn",title+" scale down",nbin,low,high,xtitle,ytitle)); tmps.at(2)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(name+"ScaleUp_ratio","Ratio of "+title+" scale upper envelop / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(3)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(name+"ScaleDn_ratio","Ratio of "+title+" scale lower envelop / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(4)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(name+"PdfUp",title+" PDF upper RMS",nbin,low,high,xtitle,ytitle)); tmps.at(5)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(name+"PdfDn",title+" PDF lower RMS",nbin,low,high,xtitle,ytitle)); tmps.at(6)->getTH1()->Sumw2(false);
  tmps.push_back(dqm->book1dHisto(name+"PdfUp_ratio","Ratio of "+title+" PDF upper RMS / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(7)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(name+"PdfDn_ratio","Ratio of "+title+" PDF lower RMS / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(8)->setEfficiencyFlag();

  for (int idx = 0; idx < nScaleVar; idx++) {
    scaleVar.push_back(new TH1F((TString)name+"Scale"+(TString)std::to_string(idx),";"+(TString)xtitle+";"+(TString)ytitle,nbin,low,high)); scaleVar.at(idx)->Sumw2();
  }

  for (int idx = 0; idx < nPdfVar; idx++) {
    pdfVar.push_back(new TH1F((TString)name+"Pdf"+(TString)std::to_string(idx),";"+(TString)xtitle+";"+(TString)ytitle,nbin,low,high)); pdfVar.at(idx)->Sumw2();
  }
} // to get ratio plots correctly - need to modify PostProcessor_cff.py as well!

void LheWeightValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {
  c.getData(fPDGTable);
}

void LheWeightValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<LHEEventProduct> lheEvt;
  iEvent.getByToken(lheEvtToken,lheEvt);
  weight = 1.;
  orgWgt = lheEvt->originalXWGTUP();
  weights = lheEvt->weights();

  nEvt->Fill(0.5, weight);
  nlogWgt->Fill(std::log10(lheEvt->weights().size()), weight);
  for (unsigned idx = 0; idx < lheEvt->weights().size(); idx++) {
    wgtVal->Fill(weights[idx].wgt/orgWgt, weight);
  }

  edm::Handle<reco::GenParticleCollection> ptcls;
  iEvent.getByToken(genParticleToken,ptcls);
  edm::Handle<reco::GenJetCollection> genjets;
  iEvent.getByToken(genJetToken,genjets);

  std::vector<reco::GenParticleRef> leptons;
  std::vector<reco::GenParticleRef> dileptons;
  std::vector<reco::GenParticle> FSRphotons;

  for (unsigned iptc = 0; iptc < ptcls->size(); iptc++) {
    reco::GenParticleRef ptc(ptcls,iptc);
    if (GenPtcValidationHelper::isFinalStateLepton(ptc)) {
      if ( ptc->pt() > subLeadLepPtCut && std::abs(ptc->eta()) < lepEtaCut ) leptons.push_back(ptc);
    }
  }

  std::sort(leptons.begin(), leptons.end(), GenPtcValidationHelper::sortByPt<reco::GenParticleRef>);

  if ( leptons.size() > 0 && leptons.at(0)->pt() > leadLepPtCut ) {
    reco::GenParticleRef leadLep = leptons.at(0);
    fillTemplates(leadLepPtScaleVar,leadLepPtPdfVar,leadLepPtTemp,leadLep->pt());
    fillTemplates(leadLepEtaScaleVar,leadLepEtaPdfVar,leadLepEtaTemp,leadLep->eta());

    if (leptons.size() > 1) {
      reco::GenParticleRef subLeadLep = leptons.at(1);
      dileptons.push_back(leadLep); dileptons.push_back(subLeadLep);
      GenPtcValidationHelper::findFSRPhotons(dileptons,*ptcls,FSRdRCut,FSRphotons);

      math::XYZTLorentzVector leadLepMom(leadLep->px(),leadLep->py(),leadLep->pz(),leadLep->p());
      math::XYZTLorentzVector subLeadLepMom(subLeadLep->px(),subLeadLep->py(),subLeadLep->pz(),subLeadLep->p());

      auto dilepFSRmom = leadLepMom + subLeadLepMom;
      for (unsigned ifsr = 0; ifsr < FSRphotons.size(); ifsr++) {
        auto fsr = FSRphotons.at(ifsr);
        math::XYZTLorentzVector FSRmom(fsr.px(),fsr.py(),fsr.pz(),fsr.p());
        dilepFSRmom += FSRmom;
      }

      fillTemplates(ZptScaleVar,ZptPdfVar,ZptTemp,dilepFSRmom.Pt());
      fillTemplates(ZmassScaleVar,ZmassPdfVar,ZmassTemp,dilepFSRmom.M());
      fillTemplates(ZrapidityScaleVar,ZrapidityPdfVar,ZrapidityTemp,dilepFSRmom.Rapidity());

    }
  }

  std::vector<reco::GenJetRef> genjetVec;

  for (unsigned igj = 0; igj < genjets->size(); igj++) {
    reco::GenJetRef genjet(genjets,igj);

    if ( genjet->pt() > jetPtCut && std::abs(genjet->eta()) < jetEtaCut ) {
      genjetVec.push_back(genjet);
    }
  }

  fillTemplates(jetMultScaleVar,jetMultPdfVar,jetMultTemp,(float)genjetVec.size());

  if (genjetVec.size() > 0) {
    std::sort(genjetVec.begin(), genjetVec.end(), GenPtcValidationHelper::sortByPt<reco::GenJetRef>);

    auto leadJet = genjetVec.at(0);
    fillTemplates(leadJetPtScaleVar,leadJetPtPdfVar,leadJetPtTemp,leadJet->pt());
    fillTemplates(leadJetEtaScaleVar,leadJetEtaPdfVar,leadJetEtaTemp,leadJet->eta());
  }
}  //analyze

void LheWeightValidation::fillTemplates(std::vector<TH1F*>& scaleVar, std::vector<TH1F*>& pdfVar, std::vector<MonitorElement*>& tmps, float obs) {
  tmps.at(0)->Fill(obs,weight);
  for (int iWgt = 0; iWgt < nScaleVar; iWgt++) {
    scaleVar.at(iWgt)->Fill(obs,weights[iWgt].wgt/orgWgt);
  }
  for (int iWgt = 0; iWgt < nPdfVar; iWgt++) {
    pdfVar.at(iWgt)->Fill(obs,weights[idxPdfStart+iWgt].wgt/orgWgt);
  }
}

void LheWeightValidation::dqmEndRun(const edm::Run& r, const edm::EventSetup& c) {
  envelop(leadLepPtScaleVar,leadLepPtTemp); pdfRMS(leadLepPtPdfVar,leadLepPtTemp);
  envelop(leadLepEtaScaleVar,leadLepEtaTemp); pdfRMS(leadLepEtaPdfVar,leadLepEtaTemp);
  envelop(ZptScaleVar,ZptTemp); pdfRMS(ZptPdfVar,ZptTemp);
  envelop(ZmassScaleVar,ZmassTemp); pdfRMS(ZmassPdfVar,ZmassTemp);
  envelop(ZrapidityScaleVar,ZrapidityTemp); pdfRMS(ZrapidityPdfVar,ZrapidityTemp);
  envelop(jetMultScaleVar,jetMultTemp); pdfRMS(jetMultPdfVar,jetMultTemp);
  envelop(leadJetPtScaleVar,leadJetPtTemp); pdfRMS(leadJetPtPdfVar,leadJetPtTemp);
  envelop(leadJetEtaScaleVar,leadJetEtaTemp); pdfRMS(leadJetEtaPdfVar,leadJetEtaTemp);

  edm::Handle<LHERunInfoProduct> run;
  r.getByToken(lheRunToken,run);

  if (dumpLHEheader) {
    for (auto it = run->headers_begin(); it != run->headers_end(); it++) {
      std::cout << "Header start" << std::endl;
      std::cout << "Tag: " << it->tag() << std::endl;
      for (const auto& l : it->lines()) {
        std::cout << l << std::endl;
      }
      std::cout << "Header end" << std::endl;
    }
  }
}

void LheWeightValidation::envelop(const std::vector<TH1F*>& var, std::vector<MonitorElement*>& tmps) {
  if ( var.size() < 1 ) return;
  for (int b = 0; b < var.at(0)->GetNbinsX()+2; b++) {
    float valU = var.at(0)->GetBinContent(b);
    float valD = valU;
    if (valU==0.) continue;
    for (unsigned v = 1; v < var.size(); v++) {
      valU = std::max( valU, (float)var.at(v)->GetBinContent(b) );
      valD = std::min( valD, (float)var.at(v)->GetBinContent(b) );
    }
    tmps.at(1)->setBinContent(b,valU);
    tmps.at(2)->setBinContent(b,valD);
  }
  tmps.at(1)->setEntries(var.at(0)->GetEntries()); tmps.at(2)->setEntries(var.at(0)->GetEntries());
  tmps.at(1)->getTH1()->Sumw2(true); tmps.at(2)->getTH1()->Sumw2(true);
  tmps.at(1)->update(); tmps.at(2)->update();
  return;
}

void LheWeightValidation::pdfRMS(const std::vector<TH1F*>& var, std::vector<MonitorElement*>& tmps) {
  if ( var.size() < 1 ) return;
  float denom = var.size();
  for (int b = 0; b < tmps.at(0)->getNbinsX()+2; b++) {
    float valNom = tmps.at(0)->getBinContent(b);
    float rmsSq = 0.;
    if (valNom==0.) continue;
    for (unsigned v = 0; v < var.size(); v++) {
      float dev = (float)var.at(v)->GetBinContent(b) - valNom;
      rmsSq += dev*dev;
    }
    float rms = std::sqrt(rmsSq/denom);
    float rmsup = valNom+rms; float rmsdn = valNom-rms;
    tmps.at(5)->setBinContent(b,rmsup);
    tmps.at(6)->setBinContent(b,rmsdn);
  }
  tmps.at(5)->setEntries(tmps.at(0)->getTH1F()->GetEntries()); tmps.at(6)->setEntries(tmps.at(0)->getTH1F()->GetEntries());
  tmps.at(5)->getTH1()->Sumw2(true); tmps.at(6)->getTH1()->Sumw2(true);
  tmps.at(5)->update(); tmps.at(6)->update();
  return;
}
