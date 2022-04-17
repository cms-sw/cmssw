#include "Validation/EventGenerator/interface/GenWeightValidation.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "Validation/EventGenerator/interface/HepMCValidationHelper.h"

using namespace edm;

GenWeightValidation::GenWeightValidation(const edm::ParameterSet& iPSet)
    : wmanager_(iPSet, consumesCollector()),
      genParticleToken_(consumes<reco::GenParticleCollection>(iPSet.getParameter<edm::InputTag>("genParticles"))),
      genJetToken_(consumes<reco::GenJetCollection>(iPSet.getParameter<edm::InputTag>("genJets"))),
      idxGenEvtInfo_(iPSet.getParameter<int>("whichGenEventInfo")),
      idxFSRup_(iPSet.getParameter<int>("idxFSRup")),
      idxFSRdown_(iPSet.getParameter<int>("idxFSRdown")),
      idxISRup_(iPSet.getParameter<int>("idxISRup")),
      idxISRdown_(iPSet.getParameter<int>("idxISRdown")),
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
      jetPtRange_(iPSet.getParameter<double>("jetPtRange")) {
  std::vector<int> idxs = {idxFSRup_, idxFSRdown_, idxISRup_, idxISRdown_};
  std::sort(idxs.begin(), idxs.end(), std::greater<int>());
  idxMax_ = idxs.at(0);
}

void GenWeightValidation::bookHistograms(DQMStore::IBooker& iBook, edm::Run const&, edm::EventSetup const&) {
  ///Setting the DQM top directories
  std::string folderName = "Generator/GenWeight";
  DQMHelper aDqmHelper(&iBook);
  iBook.setCurrentFolder(folderName);

  // Number of analyzed events
  nEvt_ = aDqmHelper.book1dHisto("nEvt", "n analyzed Events", 1, 0., 1., "bin", "Number of Events");
  nlogWgt_ = aDqmHelper.book1dHisto("nlogWgt", "Log10(n weights)", 100, 0., 3., "log_{10}(nWgts)", "Number of Events");
  wgtVal_ = aDqmHelper.book1dHisto("wgtVal", "weights", 100, -1.5, 3., "weight", "Number of Weights");
  bookTemplates(aDqmHelper,
                leadLepPtTemp_,
                "leadLepPt",
                "leading lepton Pt",
                leadLepPtNbin_,
                0.,
                leadLepPtRange_,
                "Pt_{l} (GeV)",
                "Number of Events");
  bookTemplates(aDqmHelper,
                leadLepEtaTemp_,
                "leadLepEta",
                "leading lepton #eta",
                rapidityNbin_,
                -rapidityRange_,
                rapidityRange_,
                "#eta_{l}",
                "Number of Events");
  bookTemplates(aDqmHelper,
                jetMultTemp_,
                "JetMultiplicity",
                "Gen jet multiplicity",
                nJetsNbin_,
                0,
                nJetsNbin_,
                "n",
                "Number of Events");
  bookTemplates(aDqmHelper,
                leadJetPtTemp_,
                "leadJetPt",
                "leading Gen jet Pt",
                jetPtNbin_,
                0.,
                jetPtRange_,
                "Pt_{j} (GeV)",
                "Number of Events");
  bookTemplates(aDqmHelper,
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

void GenWeightValidation::bookTemplates(DQMHelper& aDqmHelper,
                                        std::vector<MonitorElement*>& tmps,
                                        const std::string& name,
                                        const std::string& title,
                                        int nbin,
                                        float low,
                                        float high,
                                        const std::string& xtitle,
                                        const std::string& ytitle) {
  tmps.push_back(aDqmHelper.book1dHisto(name, title, nbin, low, high, xtitle, ytitle));
  tmps.push_back(aDqmHelper.book1dHisto(name + "FSRup", title + " FSR up", nbin, low, high, xtitle, ytitle));
  tmps.push_back(aDqmHelper.book1dHisto(name + "FSRdn", title + " FSR down", nbin, low, high, xtitle, ytitle));
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "FSRup_ratio", "Ratio of " + title + " FSR up / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(3)->setEfficiencyFlag();
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "FSRdn_ratio", "Ratio of " + title + " FSR down / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(4)->setEfficiencyFlag();
  tmps.push_back(aDqmHelper.book1dHisto(name + "ISRup", title + " ISR up", nbin, low, high, xtitle, ytitle));
  tmps.push_back(aDqmHelper.book1dHisto(name + "ISRdn", title + " ISR down", nbin, low, high, xtitle, ytitle));
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "ISRup_ratio", "Ratio of " + title + " ISR up / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(7)->setEfficiencyFlag();
  tmps.push_back(aDqmHelper.book1dHisto(
      name + "ISRdn_ratio", "Ratio of " + title + " ISR down / Nominal", nbin, low, high, xtitle, ytitle));
  tmps.at(8)->setEfficiencyFlag();
}  // to get ratio plots correctly - need to modify PostProcessor_cff.py as well!

void GenWeightValidation::dqmBeginRun(const edm::Run&, const edm::EventSetup&) {}

void GenWeightValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  weights_ = wmanager_.weightsCollection(iEvent);

  unsigned weightsSize = weights_.at(idxGenEvtInfo_).size();
  if (weightsSize < 2)
    return;  // no baseline weight in GenEventInfo

  weight_ = weights_.at(idxGenEvtInfo_)[0] / std::abs(weights_.at(idxGenEvtInfo_)[0]);
  nEvt_->Fill(0.5, weight_);
  nlogWgt_->Fill(std::log10(weightsSize), weight_);

  for (unsigned idx = 0; idx < weightsSize; idx++)
    wgtVal_->Fill(weights_.at(idxGenEvtInfo_)[idx] / weights_.at(idxGenEvtInfo_)[1], weight_);

  if ((int)weightsSize <= idxMax_)
    return;  // no PS weights in GenEventInfo

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
    fillTemplates(leadLepPtTemp_, leadLep->pt());
    fillTemplates(leadLepEtaTemp_, leadLep->eta());
  }

  std::vector<reco::GenJetRef> genjetVec;

  for (unsigned igj = 0; igj < genjets->size(); igj++) {
    reco::GenJetRef genjet(genjets, igj);

    if (genjet->pt() > jetPtCut_ && std::abs(genjet->eta()) < jetEtaCut_)
      genjetVec.push_back(genjet);
  }

  fillTemplates(jetMultTemp_, (float)genjetVec.size());

  if (!genjetVec.empty()) {
    std::sort(genjetVec.begin(), genjetVec.end(), HepMCValidationHelper::sortByPtRef<reco::GenJetRef>);

    auto leadJet = genjetVec.at(0);
    fillTemplates(leadJetPtTemp_, leadJet->pt());
    fillTemplates(leadJetEtaTemp_, leadJet->eta());
  }
}  //analyze

void GenWeightValidation::fillTemplates(std::vector<MonitorElement*>& tmps, float obs) {
  tmps.at(0)->Fill(obs, weight_);
  tmps.at(1)->Fill(obs, weights_.at(idxGenEvtInfo_)[idxFSRup_] / weights_.at(idxGenEvtInfo_)[1]);
  tmps.at(2)->Fill(obs, weights_.at(idxGenEvtInfo_)[idxFSRdown_] / weights_.at(idxGenEvtInfo_)[1]);
  tmps.at(5)->Fill(obs, weights_.at(idxGenEvtInfo_)[idxISRup_] / weights_.at(idxGenEvtInfo_)[1]);
  tmps.at(6)->Fill(obs, weights_.at(idxGenEvtInfo_)[idxISRdown_] / weights_.at(idxGenEvtInfo_)[1]);
}
