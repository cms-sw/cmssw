#include <algorithm>

#include "TMath.h"

#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"
#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVATrainer.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepSignalSelEval.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

TtSemiLepSignalSelMVATrainer::TtSemiLepSignalSelMVATrainer(const edm::ParameterSet& cfg):
  leptons_   (cfg.getParameter<edm::InputTag>("leptons")),
  jets_      (cfg.getParameter<edm::InputTag>("jets")),
  METs_      (cfg.getParameter<edm::InputTag>("METs")),
  maxNJets_  (cfg.getParameter<int>("maxNJets")),
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

  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);

  edm::Handle<edm::View<pat::MET> > MET_handle;
  evt.getByLabel(METs_,MET_handle);
  const edm::View<pat::MET> MET = *MET_handle;

  edm::Handle< edm::View<reco::RecoCandidate> > lepton_handle; 
  evt.getByLabel(leptons_, lepton_handle);
  const edm::View<reco::RecoCandidate>& leptons = *lepton_handle;

  edm::Handle< std::vector<pat::Jet> > jet_handle;
  evt.getByLabel(jets_, jet_handle);
  const std::vector<pat::Jet> jets = *jet_handle;
  //std::sort(jets.begin(),jets.end(),JetETComparison);
   
  // skip events with no appropriate lepton candidate in
  if( lepton_handle->empty() ) return;

  math::XYZTLorentzVector lepton = leptons.begin()->p4();

  // skip events with less jets than partons
  unsigned int nPartons = 4;
  if( jets.size() < nPartons ) return;

  if(!(leptons.size()==0)) {

    TtSemiLepSignalSel selection(jets,lepton,MET,maxNJets_);

    if(genEvt->isSemiLeptonic() && genEvt->semiLeptonicChannel() == lepChannel_) {
      evaluateTtSemiLepSignalSel(mvaComputer, selection, true, true);
    }
    else if(genEvt->isFullHadronic()||genEvt->isFullLeptonic()) {
      evaluateTtSemiLepSignalSel(mvaComputer, selection, true, false);
    }

  }

}

// implement the plugins for the trainer
// -> defines TtSemiLepSignalSelMVAContainerSaveCondDB
// -> defines TtSemiLepSignalSelMVASaveFile
// -> defines TtSemiLepSignalSelMVATrainerLooper
MVA_TRAINER_IMPLEMENT(TtSemiLepSignalSelMVA);
