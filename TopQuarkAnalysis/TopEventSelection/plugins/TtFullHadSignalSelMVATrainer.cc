#include "TMath.h"
#include <algorithm>
#include <utility>

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "PhysicsTools/MVATrainer/interface/HelperMacros.h"

#include "TopQuarkAnalysis/TopEventSelection/plugins/TtFullHadSignalSelMVATrainer.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtFullHadSignalSelEval.h"

#include "DataFormats/PatCandidates/interface/Flags.h"


TtFullHadSignalSelMVATrainer::TtFullHadSignalSelMVATrainer(const edm::ParameterSet& cfg):
  jetsToken_      (consumes< std::vector<pat::Jet> >(cfg.getParameter<edm::InputTag>("jets"))),
  genEvtToken_      (consumes<TtGenEvent>(edm::InputTag("genEvt"))),
  whatData_  (cfg.getParameter<int>("whatData")),
  maxEv_     (cfg.getParameter<int>("maxEv")),
  weight_    (cfg.getParameter<double>("weight"))
{
}

TtFullHadSignalSelMVATrainer::~TtFullHadSignalSelMVATrainer()
{
}

void
TtFullHadSignalSelMVATrainer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
{
  //communication with CMSSW CondDB
  mvaComputer.update<TtFullHadSignalSelMVARcd>("trainer", setup, "traintreeSaver");

  // can occur in the last iteration when the
  // MVATrainer is about to save the result
  if(!mvaComputer) return;

  // get the jets out of the event
  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByToken(jetsToken_, jets);

  //count the number of selected events
  selEv++;
  //skip event if enough events are already selected
  if(selEv>maxEv_ && maxEv_!=-1) return;

  //calculation of InputVariables
  //see TopQuarkAnalysis/TopTools/interface/TtFullHadSignalSel.h
  //                             /src/TtFullHadSignalSel.cc
  //all objects, jets, which are needed for the calculation
  //of the input-variables have to be passed to this class

  TtFullHadSignalSel selection(*jets);

  if(whatData_==-1) {
    //your training-file contains both, signal and background events

    edm::Handle<TtGenEvent> genEvt;
    evt.getByToken(genEvtToken_, genEvt);

    bool isSignal = false;
    if(genEvt->isTtBar()){
      if(genEvt->isFullHadronic()) isSignal = true;
    }
    evaluateTtFullHadSignalSel(mvaComputer, selection, weight_, true, isSignal);
  }
  else {

    if(whatData_==1){
      //your tree contains only signal events

      evaluateTtFullHadSignalSel(mvaComputer, selection, weight_, true, true);
    }
    else if(whatData_==0){
      //your tree contains only background events

      evaluateTtFullHadSignalSel(mvaComputer, selection, weight_, true, false);
    }
    else std::cout<<"Config File Error!! Please check <whatData> in TtFullHadSignalSelMVATrainer_cfi";
  }
}

void TtFullHadSignalSelMVATrainer::beginJob(){
  selEv = 0;
  if(whatData_!=-1 && whatData_!=0 && whatData_!=1){
    std::cout<<"Config File Error!! Please check <whatData> in TtFullHadSignalSelMVATrainer_cfi"<<std::endl;;
    return;
  }
}

// implement the plugins for the trainer
// -> defines TtFullHadSignalSelMVAContainerSaveCondDB
// -> defines TtFullHadSignalSelMVASaveFile
// -> defines TtFullHadSignalSelMVATrainerLooper
MVA_TRAINER_IMPLEMENT(TtFullHadSignalSelMVA);
