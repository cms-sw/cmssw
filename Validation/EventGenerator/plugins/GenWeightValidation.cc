#include "Validation/EventGenerator/interface/GenWeightValidation.h"
#include <iostream>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/DQMHelper.h"
#include "Validation/EventGenerator/interface/GenPtcValidationHelper.h"
using namespace edm;

GenWeightValidation::GenWeightValidation(const edm::ParameterSet& iPSet)
: wmanager_(iPSet, consumesCollector())
{
  genParticleToken = consumes<reco::GenParticleCollection>(iPSet.getParameter<edm::InputTag>("genParticles"));
  genJetToken = consumes<reco::GenJetCollection>(iPSet.getParameter<edm::InputTag>("genJets"));
  idxGenEvtInfo = iPSet.getParameter<int>("whichGenEventInfo");
  idxFSRup = iPSet.getParameter<int>("idxFSRup");
  idxFSRdown = iPSet.getParameter<int>("idxFSRdown");
  idxISRup = iPSet.getParameter<int>("idxISRup");
  idxISRdown = iPSet.getParameter<int>("idxISRdown");
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

  std::vector<int> idxs = {idxFSRup,idxFSRdown,idxISRup,idxISRdown};
  std::sort(idxs.begin(), idxs.end(), std::greater<int>());
  idxMax = idxs.at(0);
}

GenWeightValidation::~GenWeightValidation() {}

void GenWeightValidation::bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) {
  ///Setting the DQM top directories
  std::string folderName = "Generator/GenWeight";
  dqm = new DQMHelper(&i);
  i.setCurrentFolder(folderName);

  // Number of analyzed events
  nEvt = dqm->book1dHisto("nEvt", "n analyzed Events", 1, 0., 1., "bin", "Number of Events");
  nlogWgt = dqm->book1dHisto("nlogWgt","Log10(n weights)",100,0.,3.,"log_{10}(nWgts)","Number of Events");
  wgtVal = dqm->book1dHisto("wgtVal","weights",100,-1.5,3.,"weight","Number of Weigths");
  bookTemplates(leadLepPtTemp,"leadLepPt","leading lepton Pt",leadLepPtNbin,0.,leadLepPtRange,"Pt_{l} (GeV)","Number of Events");
  bookTemplates(leadLepEtaTemp,"leadLepEta","leading lepton #eta",rapidityNbin,-rapidityRange,rapidityRange,"#eta_{l}","Number of Events");
  bookTemplates(ZptTemp,"Zpt","Z Pt",ZptNbin,0.,ZptRange,"Pt_{Z} (GeV)","Number of Events");
  bookTemplates(ZmassTemp,"Zmass","Z mass",ZmassNbin,ZmassLow,ZmassHigh,"M_{Z} (GeV)","Numberr of Events");
  bookTemplates(ZrapidityTemp,"Zrapidity","Z rapidity",rapidityNbin,-rapidityRange,rapidityRange,"Y_{Z}","Number of Events");
  bookTemplates(jetMultTemp,"JetMultiplicity","Gen jet multiplicity",nJetsNbin,0,nJetsNbin,"n","Number of Events");
  bookTemplates(leadJetPtTemp,"leadJetPt","leading Gen jet Pt",jetPtNbin,0.,jetPtRange,"Pt_{j} (GeV)","Number of Events");
  bookTemplates(leadJetEtaTemp,"leadJetEta","leading Gen jet #eta",rapidityNbin,-rapidityRange,rapidityRange,"#eta_{j}","Number of Events");

  return;
}

void GenWeightValidation::bookTemplates(std::vector<MonitorElement*>& tmps, std::string name, std::string title, int nbin, float low, float high, std::string xtitle, std::string ytitle) {
  tmps.push_back(dqm->book1dHisto(name,title,nbin,low,high,xtitle,ytitle));
  tmps.push_back(dqm->book1dHisto(name+"FSRup",title+" FSR up",nbin,low,high,xtitle,ytitle));
  tmps.push_back(dqm->book1dHisto(name+"FSRdn",title+" FSR down",nbin,low,high,xtitle,ytitle));
  tmps.push_back(dqm->book1dHisto(name+"FSRup_ratio","Ratio of "+title+" FSR up / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(3)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(name+"FSRdn_ratio","Ratio of "+title+" FSR down / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(4)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(name+"ISRup",title+" ISR up",nbin,low,high,xtitle,ytitle));
  tmps.push_back(dqm->book1dHisto(name+"ISRdn",title+" ISR down",nbin,low,high,xtitle,ytitle));
  tmps.push_back(dqm->book1dHisto(name+"ISRup_ratio","Ratio of "+title+" ISR up / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(7)->setEfficiencyFlag();
  tmps.push_back(dqm->book1dHisto(name+"ISRdn_ratio","Ratio of "+title+" ISR down / Nominal",nbin,low,high,xtitle,ytitle)); tmps.at(8)->setEfficiencyFlag();
} // to get ratio plots correctly - need to modify PostProcessor_cff.py as well!

void GenWeightValidation::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) { c.getData(fPDGTable); }

void GenWeightValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  weight = 1.;
  weights = wmanager_.weightsCollection(iEvent);
  if ((int)weights.at(idxGenEvtInfo).size() <= idxMax) {
    std::cout << "Given GenEventInfo weight idx is larger than stored number of GenEventInfo weights. Skipping DQM GenEventInfo weight validation" << std::endl;
    return;
  }

  nEvt->Fill(0.5, weight);
  nlogWgt->Fill(std::log10(weights.at(idxGenEvtInfo).size()), weight);
  for (unsigned idx = 0; idx < weights.at(idxGenEvtInfo).size(); idx++) {
    if ( weights.at(idxGenEvtInfo).size() < 2 ) continue;
    wgtVal->Fill(weights.at(idxGenEvtInfo)[idx]/weights.at(idxGenEvtInfo)[1], weight);
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
    fillTemplates(leadLepPtTemp,leadLep->pt());
    fillTemplates(leadLepEtaTemp,leadLep->eta());

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

      fillTemplates(ZptTemp,dilepFSRmom.Pt());
      fillTemplates(ZmassTemp,dilepFSRmom.M());
      fillTemplates(ZrapidityTemp,dilepFSRmom.Rapidity());

    }
  }

  std::vector<reco::GenJetRef> genjetVec;

  for (unsigned igj = 0; igj < genjets->size(); igj++) {
    reco::GenJetRef genjet(genjets,igj);

    if ( genjet->pt() > jetPtCut && std::abs(genjet->eta()) < jetEtaCut ) {
      genjetVec.push_back(genjet);
    }
  }

  fillTemplates(jetMultTemp,(float)genjetVec.size());

  if (genjetVec.size() > 0) {
    std::sort(genjetVec.begin(), genjetVec.end(), GenPtcValidationHelper::sortByPt<reco::GenJetRef>);

    auto leadJet = genjetVec.at(0);
    fillTemplates(leadJetPtTemp,leadJet->pt());
    fillTemplates(leadJetEtaTemp,leadJet->eta());
  }
}  //analyze

void GenWeightValidation::fillTemplates(std::vector<MonitorElement*>& tmps, float obs) {
  tmps.at(0)->Fill(obs,weight);
  tmps.at(1)->Fill(obs,weights.at(idxGenEvtInfo)[idxFSRup]/weights.at(idxGenEvtInfo)[1]);
  tmps.at(2)->Fill(obs,weights.at(idxGenEvtInfo)[idxFSRdown]/weights.at(idxGenEvtInfo)[1]);
  tmps.at(5)->Fill(obs,weights.at(idxGenEvtInfo)[idxISRup]/weights.at(idxGenEvtInfo)[1]);
  tmps.at(6)->Fill(obs,weights.at(idxGenEvtInfo)[idxISRdown]/weights.at(idxGenEvtInfo)[1]);
}

void GenWeightValidation::dqmEndRun(const edm::Run& r, const edm::EventSetup& c) {

}
