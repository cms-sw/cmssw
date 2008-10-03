#include <algorithm>

#include "TMath.h"

#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"
#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVATrainer.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepSignalSelEval.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include <DataFormats/PatCandidates/interface/Muon.h>
#include "DataFormats/PatCandidates/interface/MET.h"

#include <DataFormats/PatCandidates/interface/Lepton.h>

TtSemiLepSignalSelMVATrainer::TtSemiLepSignalSelMVATrainer(const edm::ParameterSet& cfg):
  leptons_   (cfg.getParameter<edm::InputTag>("leptons")),
  jets_      (cfg.getParameter<edm::InputTag>("jets")),
  //matching_  (cfg.getParameter<edm::InputTag>("matching")),
  METs_      (cfg.getParameter<edm::InputTag>("METs")),
  nJetsMax_  (cfg.getParameter<int>("nJetsMax")),
  lepChannel_(cfg.getParameter<int>("lepChannel"))
{
}

TtSemiLepSignalSelMVATrainer::~TtSemiLepSignalSelMVATrainer()
{
}

void
TtSemiLepSignalSelMVATrainer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  mvaComputer.update<TtSemiLepSignalSelMVARcd>("trainer", setup, "ttSemiLepSignalSelMVA");

  // can occur in the last iteration when the 
  // MVATrainer is about to save the result
  if(!mvaComputer) return;

  edm::Handle<double> h_csa07wgt;
  evt.getByLabel ("csa07EventWeightProducer","weight", h_csa07wgt);
  if(!h_csa07wgt.isValid()) return;
  float csa07wgt = *(h_csa07wgt);

  edm::Handle<int> h_csa07pid;
  evt.getByLabel ("csa07EventWeightProducer","AlpgenProcessID", h_csa07pid);
  if(!h_csa07pid.isValid()) return;
  int csa07pid = *(h_csa07pid);

  //std::cout<<"pid: "<<csa07pid<<std::endl;

  if((csa07pid<3000||csa07pid>=3010)&&(csa07pid<1000||csa07pid>=1010)) return;
  //else std::cout<<"event found: "<<csa07pid<<std::endl;

  //if(csa07pid>=3000&&csa07pid<3010) {
  //  edm::Handle<TtGenEvent> genEvt;
  //  evt.getByLabel("genEvt", genEvt);
  //}

  edm::Handle<edm::View<pat::MET> > MET_handle;
  evt.getByLabel(METs_,MET_handle);
  if(!MET_handle.isValid()) return;
  const edm::View<pat::MET> MET = *MET_handle;

  edm::Handle< edm::View<pat::Muon> > lepton_handle; 
  evt.getByLabel(leptons_, lepton_handle);
  if(!lepton_handle.isValid()) return;
  const edm::View<pat::Muon>& leptons = *lepton_handle;
  int nleptons = 0;
  for(edm::View<pat::Muon>::const_iterator it = leptons.begin(); it!=leptons.end(); it++) {
    if(it->pt()>30 && fabs(it->eta())<2.1) nleptons++;
  }

  // skip events with no appropriate lepton candidate in
  if(nleptons!=1) return;
  if(leptons.begin()->caloIso()>=1) return;
  if(leptons.begin()->trackIso()>=3) return;

  math::XYZTLorentzVector lepton = leptons.begin()->p4();

  if(lepton.Pt()<=30) return;
  if(fabs(lepton.Eta())>=2.1) return;
   
  edm::Handle< std::vector<pat::Jet> > jet_handle;
  evt.getByLabel(jets_, jet_handle);
  if(!jet_handle.isValid()) return;
  const std::vector<pat::Jet> jets = *jet_handle;
  //std::sort(jets.begin(),jets.end(),JetETComparison);
  if(jets.begin()->et()<=65.) return;

  double dRmin = 9999.;
  std::vector<pat::Jet> seljets;
  for(std::vector<pat::Jet>::const_iterator it = jets.begin(); it != jets.end(); it++) {
    if(it->et()>20. && fabs(it->eta())<2.4) {
      math::XYZTLorentzVector tv = it->p4();
      double tmpdR = TMath::Sqrt((tv.Eta()-lepton.Eta())*(tv.Eta()-lepton.Eta())
	    		        +(tv.Phi()-lepton.Phi())*(tv.Phi()-lepton.Phi()));
      if(tmpdR<dRmin) dRmin = tmpdR;
      seljets.push_back(*it);
    }
  }
  if(dRmin<=0.3) return;

  // skip events with less jets than partons
  unsigned int nPartons = 4;
  if(seljets.size()<nPartons /*|| leptons.begin()->pt()>1000 seljets[0].pt()<=65*/) return;

  if(!(leptons.size()==0)) {

    TtSemiLepSignalSel selection(seljets,lepton,MET,nJetsMax_);

    if(csa07pid>=3000 && csa07pid<3010) {
      edm::Handle<TtGenEvent> genEvt;
      evt.getByLabel("genEvt", genEvt);
      if(genEvt->isSemiLeptonic() && genEvt->semiLeptonicChannel() == lepChannel_) {
        std::cout<<"a ttbar event"<<std::endl;
        evaluateTtSemiLepSignalSel(mvaComputer, selection, csa07wgt, true, true);
      }
    }
    else if(csa07pid>=1000&&csa07pid<1010) {
      std::cout<<"a W+jets event"<<std::endl;
      evaluateTtSemiLepSignalSel(mvaComputer, selection, csa07wgt, true, false);
    }
  }
  else return;

}

// implement the plugins for the trainer
// -> defines TtSemiLepSignalSelMVAContainerSaveCondDB
// -> defines TtSemiLepSignalSelMVASaveFile
// -> defines TtSemiLepSignalSelMVATrainerLooper
MVA_TRAINER_IMPLEMENT(TtSemiLepSignalSelMVA);
