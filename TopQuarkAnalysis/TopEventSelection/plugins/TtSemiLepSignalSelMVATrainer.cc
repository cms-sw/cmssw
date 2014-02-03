#include "TMath.h"
#include <algorithm>

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "TopQuarkAnalysis/TopEventSelection/plugins/TtSemiLepSignalSelMVATrainer.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLepSignalSelEval.h"

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"


TtSemiLepSignalSelMVATrainer::TtSemiLepSignalSelMVATrainer(const edm::ParameterSet& cfg):
  muons_     (cfg.getParameter<edm::InputTag>("muons")),
  electrons_ (cfg.getParameter<edm::InputTag>("elecs")),
  jets_      (cfg.getParameter<edm::InputTag>("jets")),
  METs_      (cfg.getParameter<edm::InputTag>("mets")),
  lepChannel_(cfg.getParameter<int>("lepChannel")),
  whatData_  (cfg.getParameter<int>("whatData")),
  maxEv_     (cfg.getParameter<int>("maxEv"))
{
}

TtSemiLepSignalSelMVATrainer::~TtSemiLepSignalSelMVATrainer()
{
}

void
TtSemiLepSignalSelMVATrainer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  //communication with CMSSW CondDB
  mvaComputer.update<TtSemiLepSignalSelMVARcd>("trainer", setup, "traintreeSaver");

  // can occur in the last iteration when the 
  // MVATrainer is about to save the result
  if(!mvaComputer) return;


  //make your preselection here!!
  //the following code is for the default example  
  edm::Handle<edm::View<pat::MET> > MET_handle;
  evt.getByLabel(METs_,MET_handle);
  if(!MET_handle.isValid()) return;
  const edm::View<pat::MET> MET = *MET_handle;

  edm::Handle< std::vector<pat::Jet> > jet_handle;
  evt.getByLabel(jets_, jet_handle);
  if(!jet_handle.isValid()) return;
  const std::vector<pat::Jet> jets = *jet_handle;
  unsigned int nJets = 0;
  std::vector<pat::Jet> seljets;
  //std::cout<<"number of jets: "<<jets.size()<<std::endl;
  for(std::vector<pat::Jet>::const_iterator it = jets.begin(); it != jets.end(); it++) {
    //std::cout<<"Jet Et: "<<it->et()<<" Eta: "<<fabs(it->eta())<<std::endl;
    if(!(pat::Flags::test(*it, pat::Flags::Overlap::Electrons))) continue;
    if(it->et()>30. && fabs(it->eta())<2.4) {
      seljets.push_back(*it);
      nJets++;
    }
  }
  //std::cout<<"selected Jets: "<<nJets<<std::endl;
  if(nJets<4) return;

  //sort by Pt
  sort(seljets.begin(),seljets.end(),JetwithHigherPt());
 
  edm::Handle< edm::View<pat::Muon> > muon_handle; 
  evt.getByLabel(muons_, muon_handle);
  if(!muon_handle.isValid()) return;
  const edm::View<pat::Muon> muons = *muon_handle;
  int nmuons = 0;
  std::vector<pat::Muon> selMuons;
  for(edm::View<pat::Muon>::const_iterator it = muons.begin(); it!=muons.end(); it++) {
    reco::TrackRef gltr = it->track(); // global track
    reco::TrackRef trtr = it->innerTrack(); // tracker track
    if(it->pt()>30 && fabs(it->eta())<2.1 && (it->pt()/(it->pt()+it->trackIso()+it->caloIso()))>0.95 && it->isGlobalMuon()){      
      if(gltr.isNull()) continue;  //temporary problems with dead trackrefs
      if((gltr->chi2()/gltr->ndof())<10 && trtr->numberOfValidHits()>=11) {
     
        double dRmin = 9999.;
        for(std::vector<pat::Jet>::const_iterator ajet = seljets.begin(); ajet != seljets.end(); ajet++) {
          math::XYZTLorentzVector jet = ajet->p4();
          math::XYZTLorentzVector muon = it->p4();
          double tmpdR = DeltaR(muon,jet);
          if(tmpdR<dRmin) dRmin = tmpdR;
        }
        reco::TrackRef trtr = it->track(); // tracker track
        if(dRmin>0.3) {   //temporary problems with muon isolation
          nmuons++;
          selMuons.push_back(*it);
        }
      }
    }
  }
  //std::cout<<"selected Muons: "<<nleptons<<std::endl;
  if(nmuons!=1) return;
  
  edm::Handle< edm::View<pat::Electron> > electron_handle; 
  evt.getByLabel(electrons_, electron_handle);
  if(!electron_handle.isValid()) return;
  const edm::View<pat::Electron> electrons = *electron_handle;
  int nelectrons = 0;
  for(edm::View<pat::Electron>::const_iterator it = electrons.begin(); it!=electrons.end(); it++) {
    if(it->pt()>30 && fabs(it->eta())<2.4 && (it->pt()/(it->pt()+it->trackIso()+it->caloIso()))>0.95 && it->isElectronIDAvailable("eidTight"))
    { 
      if(it->electronID("eidTight")==1) nelectrons++;
    }
  }
  if(nelectrons>0) return;
  //end of the preselection

  
  math::XYZTLorentzVector muon = selMuons.begin()->p4();

  //count the number of selected events
  selEv++;
  //skip event if enough events are already selected
  if(selEv>maxEv_ && maxEv_!=-1) return;

  //calculation of InputVariables
  //see TopQuarkAnalysis/TopTools/interface/TtSemiLepSignalSel.h
  //                             /src/TtSemiLepSignalSel.cc
  //all objects i.e. jets, muons, electrons... which are needed for the calculation
  //of the input-variables have to be passed to this class

  TtSemiLepSignalSel selection(seljets,muon,MET);

  //this is only needed for the default example
  edm::Handle<TtGenEvent> genEvt;
  evt.getByLabel("genEvt", genEvt);

  double weight = 1.0; //standard no weight, i.e. weight=1.0, set this to the corresponding weight if 
                       //different weights for different events are available  
  if(whatData_==-1) {  //your training-file contains both, signal and background events
    bool isSignal;
    isSignal = true;//true for signal, false for background this has to be derived in some way
    evaluateTtSemiLepSignalSel(mvaComputer, selection, weight, true, isSignal);
  }
  else {
   
    if(whatData_==1){ //your tree contains only signal events
      //if needed do a special signal selection here
      //the following code is for the default example
      if(genEvt->isSemiLeptonic() && genEvt->semiLeptonicChannel() == lepChannel_) {
        //std::cout<<"a tt_semlep muon event"<<std::endl;
        evaluateTtSemiLepSignalSel(mvaComputer, selection, weight, true, true);
      }
      else selEv--;
    }
    else if(whatData_==0){
      //std::cout<<"a Wjets event"<<std::endl;
      evaluateTtSemiLepSignalSel(mvaComputer, selection, weight, true, false);
    }
    else std::cout<<"Config File Error!! Please check <whatData> in TtSemiLepSignalSelMVATrainer.cfi";
  }
}

void TtSemiLepSignalSelMVATrainer::beginJob(){
  selEv = 0;
  if(whatData_!=-1 && whatData_!=0 && whatData_!=1){
    std::cout<<"Config File Error!! Please check <whatData> in TtSemiLepSignalSelMVATrainer.cfi"<<std::endl;;
    return;
  }
}

double TtSemiLepSignalSelMVATrainer::DeltaPhi(const math::XYZTLorentzVector& v1,const  math::XYZTLorentzVector& v2)
{
  double dPhi = fabs(v1.Phi() - v2.Phi());
  if (dPhi > TMath::Pi()) dPhi =  2*TMath::Pi() - dPhi;
  return dPhi;
}

double TtSemiLepSignalSelMVATrainer::DeltaR(const math::XYZTLorentzVector& v1,const math::XYZTLorentzVector& v2)
{
  double dPhi = DeltaPhi(v1,v2);
  double dR = TMath::Sqrt((v1.Eta()-v2.Eta())*(v1.Eta()-v2.Eta())+dPhi*dPhi);
  return dR;
}

// implement the plugins for the trainer
// -> defines TtSemiLepSignalSelMVAContainerSaveCondDB
// -> defines TtSemiLepSignalSelMVASaveFile
// -> defines TtSemiLepSignalSelMVATrainerLooper
MVA_TRAINER_IMPLEMENT(TtSemiLepSignalSelMVA);
