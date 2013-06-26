//
// $Id: TtSemiEvtSolutionMaker.cc,v 1.43 2010/03/30 14:04:44 snaumann Exp $
//

#include "TopQuarkAnalysis/TopEventProducers/interface/TtSemiEvtSolutionMaker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiSimpleBestJetComb.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombObservables.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLRJetCombCalc.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLRSignalSelObservables.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLRSignalSelCalc.h"

#include <memory>


/// constructor
TtSemiEvtSolutionMaker::TtSemiEvtSolutionMaker(const edm::ParameterSet & iConfig) {
  // configurables
  electronSrc_     = iConfig.getParameter<edm::InputTag>    ("electronSource");
  muonSrc_         = iConfig.getParameter<edm::InputTag>    ("muonSource");
  metSrc_          = iConfig.getParameter<edm::InputTag>    ("metSource");
  jetSrc_          = iConfig.getParameter<edm::InputTag>    ("jetSource");
  leptonFlavour_   = iConfig.getParameter<std::string>      ("leptonFlavour");
  jetCorrScheme_   = iConfig.getParameter<int>              ("jetCorrectionScheme");
  nrCombJets_      = iConfig.getParameter<unsigned int>     ("nrCombJets");
  doKinFit_        = iConfig.getParameter<bool>             ("doKinFit");
  addLRSignalSel_  = iConfig.getParameter<bool>             ("addLRSignalSel");
  lrSignalSelObs_  = iConfig.getParameter<std::vector<int> >("lrSignalSelObs");
  lrSignalSelFile_ = iConfig.getParameter<std::string>      ("lrSignalSelFile");
  addLRJetComb_    = iConfig.getParameter<bool>             ("addLRJetComb");
  lrJetCombObs_    = iConfig.getParameter<std::vector<int> >("lrJetCombObs");
  lrJetCombFile_   = iConfig.getParameter<std::string>      ("lrJetCombFile");
  maxNrIter_       = iConfig.getParameter<int>              ("maxNrIter");
  maxDeltaS_       = iConfig.getParameter<double>           ("maxDeltaS");
  maxF_            = iConfig.getParameter<double>           ("maxF");
  jetParam_        = iConfig.getParameter<int>              ("jetParametrisation");
  lepParam_        = iConfig.getParameter<int>              ("lepParametrisation");
  metParam_        = iConfig.getParameter<int>              ("metParametrisation");
  constraints_     = iConfig.getParameter<std::vector<unsigned> >("constraints");
  matchToGenEvt_   = iConfig.getParameter<bool>             ("matchToGenEvt");
  matchingAlgo_    = iConfig.getParameter<int>              ("matchingAlgorithm");
  useMaxDist_      = iConfig.getParameter<bool>             ("useMaximalDistance");
  useDeltaR_       = iConfig.getParameter<bool>             ("useDeltaR");
  maxDist_         = iConfig.getParameter<double>           ("maximalDistance");

  // define kinfitter
  if(doKinFit_){
    myKinFitter = new TtSemiLepKinFitter(param(jetParam_), param(lepParam_), param(metParam_), maxNrIter_, maxDeltaS_, maxF_, constraints(constraints_));
  }

  // define jet combinations related calculators
  mySimpleBestJetComb                    = new TtSemiSimpleBestJetComb();
  myLRSignalSelObservables               = new TtSemiLRSignalSelObservables();
  myLRJetCombObservables                 = new TtSemiLRJetCombObservables();
  myLRJetCombObservables -> jetSource(jetSrc_);
  if (addLRJetComb_)   myLRJetCombCalc   = new TtSemiLRJetCombCalc(edm::FileInPath(lrJetCombFile_).fullPath(), lrJetCombObs_);

  // instantiate signal selection calculator
  if (addLRSignalSel_) myLRSignalSelCalc = new TtSemiLRSignalSelCalc(edm::FileInPath(lrSignalSelFile_).fullPath(), lrSignalSelObs_);

  // define what will be produced
  produces<std::vector<TtSemiEvtSolution> >();
}


/// destructor
TtSemiEvtSolutionMaker::~TtSemiEvtSolutionMaker() {
  if (doKinFit_)      delete myKinFitter;
  delete mySimpleBestJetComb;
  delete myLRSignalSelObservables;
  delete myLRJetCombObservables;
  if(addLRSignalSel_) delete myLRSignalSelCalc;
  if(addLRJetComb_)   delete myLRJetCombCalc;
}


void TtSemiEvtSolutionMaker::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {

  //
  //  TopObject Selection
  //

  // select lepton (the TtLepton vectors are, for the moment, sorted on pT)
  bool leptonFound = false;
  edm::Handle<std::vector<pat::Muon> > muons;
  if(leptonFlavour_ == "muon"){
    iEvent.getByLabel(muonSrc_, muons);
    if (muons->size() > 0) leptonFound = true;
  }
  edm::Handle<std::vector<pat::Electron> > electrons;
  if(leptonFlavour_ == "electron"){
    iEvent.getByLabel(electronSrc_, electrons);
    if (electrons->size() > 0) leptonFound = true;
  }  

  // select MET (TopMET vector is sorted on ET)
  bool metFound = false;
  edm::Handle<std::vector<pat::MET> > mets;
  iEvent.getByLabel(metSrc_, mets);
  if (mets->size() > 0) metFound = true;

  // select Jets
  bool jetsFound = false;
  edm::Handle<std::vector<pat::Jet> > jets;
  iEvent.getByLabel(jetSrc_, jets);
  if (jets->size() >= 4) jetsFound = true;

  //
  // Build Event solutions according to the ambiguity in the jet combination
  //
  std::vector<TtSemiEvtSolution> * evtsols = new std::vector<TtSemiEvtSolution>();
  if(leptonFound && metFound && jetsFound){
    // protect against reading beyond array boundaries
    unsigned int nrCombJets = nrCombJets_; // do not overwrite nrCombJets_
    if (jets->size() < nrCombJets) nrCombJets = jets->size();
    // loop over all jets
    for (unsigned int p=0; p<nrCombJets; p++) {
      for (unsigned int q=0; q<nrCombJets; q++) {
        for (unsigned int bh=0; bh<nrCombJets; bh++) {
          if (q>p && !(bh==p || bh==q)) {
            for (unsigned int bl=0; bl<nrCombJets; bl++) {
              if (!(bl==p || bl==q || bl==bh)) {
                TtSemiEvtSolution asol;
                asol.setJetCorrectionScheme(jetCorrScheme_);
                if(leptonFlavour_ == "muon")     asol.setMuon(muons, 0);
                if(leptonFlavour_ == "electron") asol.setElectron(electrons, 0);
                asol.setNeutrino(mets, 0);
                asol.setHadp(jets, p);
                asol.setHadq(jets, q);
                asol.setHadb(jets, bh);
                asol.setLepb(jets, bl);
                if (doKinFit_) {
                  asol = myKinFitter->addKinFitInfo(&asol);
                  // just to keep a record in the event (drop? -> present in provenance anyway...)
                  asol.setJetParametrisation(jetParam_);
                  asol.setLeptonParametrisation(lepParam_);
                  asol.setNeutrinoParametrisation(metParam_);
                }
		if(matchToGenEvt_){
		  edm::Handle<TtGenEvent> genEvt;
		  iEvent.getByLabel ("genEvt",genEvt); 
		  if (genEvt->numberOfBQuarks() == 2 &&  // FIXME: in rare cases W->bc decay, resulting in a wrong filled genEvt leading to a segmentation fault 
		      genEvt->numberOfLeptons() == 1) {  // FIXME: temporary solution to avoid crash in JetPartonMatching for non semi-leptonic events
		    asol.setGenEvt(genEvt);   
		  
		  }
		}
                // these lines calculate the observables to be used in the TtSemiSignalSelection LR
                (*myLRSignalSelObservables)(asol, *jets);

                // if asked for, calculate with these observable values the LRvalue and 
                // (depending on the configuration) probability this event is signal
                // FIXME: DO WE NEED TO DO THIS FOR EACH SOLUTION??? (S.L.20/8/07)
                if(addLRSignalSel_) (*myLRSignalSelCalc)(asol);

                // these lines calculate the observables to be used in the TtSemiJetCombination LR
                //(*myLRJetCombObservables)(asol);
		
		(*myLRJetCombObservables)(asol, iEvent);
                
		// if asked for, calculate with these observable values the LRvalue and 
                // (depending on the configuration) probability a jet combination is correct
                if(addLRJetComb_) (*myLRJetCombCalc)(asol);

                //std::cout<<"SignalSelLRval = "<<asol.getLRSignalEvtLRval()<<"  JetCombProb = "<<asol.getLRSignalEvtProb()<<std::endl;
                //std::cout<<"JetCombLRval = "<<asol.getLRJetCombLRval()<<"  JetCombProb = "<<asol.getLRJetCombProb()<<std::endl;

                // fill solution to vector
		asol.setupHyp();
                evtsols->push_back(asol);
              } 
            }
          }
        } 
      }
    }

    // if asked for, match the event solutions to the gen Event
    if(matchToGenEvt_){
      int bestSolution = -999; 
      int bestSolutionChangeWQ = -999;
      edm::Handle<TtGenEvent> genEvt;
      iEvent.getByLabel ("genEvt",genEvt); 
      if (genEvt->numberOfBQuarks() == 2 &&   // FIXME: in rare cases W->bc decay, resulting in a wrong filled genEvt leading to a segmentation fault
          genEvt->numberOfLeptons() == 1) {   // FIXME: temporary solution to avoid crash in JetPartonMatching for non semi-leptonic events
	std::vector<const reco::Candidate*> quarks;
        const reco::Candidate & genp  = *(genEvt->hadronicDecayQuark());
        const reco::Candidate & genq  = *(genEvt->hadronicDecayQuarkBar());
        const reco::Candidate & genbh = *(genEvt->hadronicDecayB());
        const reco::Candidate & genbl = *(genEvt->leptonicDecayB());
        quarks.push_back( &genp );
        quarks.push_back( &genq );
        quarks.push_back( &genbh );
        quarks.push_back( &genbl );
	std::vector<const reco::Candidate*> recjets;  
        for(size_t s=0; s<evtsols->size(); s++) {
          recjets.clear();
          const reco::Candidate & jetp  = (*evtsols)[s].getRecHadp();
          const reco::Candidate & jetq  = (*evtsols)[s].getRecHadq();
          const reco::Candidate & jetbh = (*evtsols)[s].getRecHadb();
          const reco::Candidate & jetbl = (*evtsols)[s].getRecLepb();
          recjets.push_back( &jetp );
          recjets.push_back( &jetq );
          recjets.push_back( &jetbh );
          recjets.push_back( &jetbl );
          JetPartonMatching aMatch(quarks, recjets, matchingAlgo_, useMaxDist_, useDeltaR_, maxDist_);   
          (*evtsols)[s].setGenEvt(genEvt);   
          (*evtsols)[s].setMCBestSumAngles(aMatch.getSumDistances());
          (*evtsols)[s].setMCBestAngleHadp(aMatch.getDistanceForParton(0));
          (*evtsols)[s].setMCBestAngleHadq(aMatch.getDistanceForParton(1));
          (*evtsols)[s].setMCBestAngleHadb(aMatch.getDistanceForParton(2));
          (*evtsols)[s].setMCBestAngleLepb(aMatch.getDistanceForParton(3));
          if(aMatch.getMatchForParton(2) == 2 && aMatch.getMatchForParton(3) == 3){
            if(aMatch.getMatchForParton(0) == 0 && aMatch.getMatchForParton(1) == 1) {
              bestSolution = s;
              bestSolutionChangeWQ = 0;
            } else if(aMatch.getMatchForParton(0) == 1 && aMatch.getMatchForParton(1) == 0) {
              bestSolution = s;
              bestSolutionChangeWQ = 1;
            }
          }
        }
      }
      for(size_t s=0; s<evtsols->size(); s++) {
        (*evtsols)[s].setMCBestJetComb(bestSolution);
        (*evtsols)[s].setMCChangeWQ(bestSolutionChangeWQ);     
      }
    }

    // add TtSemiSimpleBestJetComb to solutions
    int simpleBestJetComb = (*mySimpleBestJetComb)(*evtsols);
    for(size_t s=0; s<evtsols->size(); s++) (*evtsols)[s].setSimpleBestJetComb(simpleBestJetComb);

    // choose the best jet combination according to LR value
    if (addLRJetComb_ && evtsols->size()>0) {
      float bestLRVal = -1000000;
      int bestSol = (*evtsols)[0].getLRBestJetComb(); // duplicate the default
      for(size_t s=0; s<evtsols->size(); s++) {
        if ((*evtsols)[s].getLRJetCombLRval() > bestLRVal) {
          bestLRVal = (*evtsols)[s].getLRJetCombLRval();
          bestSol = s;
        }
      }
      for(size_t s=0; s<evtsols->size(); s++) {
        (*evtsols)[s].setLRBestJetComb(bestSol);
      }
    }

    //store the vector of solutions to the event     
    std::auto_ptr<std::vector<TtSemiEvtSolution> > pOut(evtsols);
    iEvent.put(pOut);

  } else {

    /*
    std::cout<<"No calibrated solutions built, because:  ";
    if(jets->size()<4)      					  std::cout<<"nr sel jets < 4"<<std::endl;
    if(leptonFlavour_ == "muon" && muons->size() == 0)    	  std::cout<<"no good muon candidate"<<std::endl;
    if(leptonFlavour_ == "electron" && electrons->size() == 0)   std::cout<<"no good electron candidate"<<std::endl;
    if(mets->size() == 0)    					  std::cout<<"no MET reconstruction"<<std::endl;
    */
//    TtSemiEvtSolution asol;
//    evtsols->push_back(asol);
    std::auto_ptr<std::vector<TtSemiEvtSolution> > pOut(evtsols);
    iEvent.put(pOut);
  }
}

TtSemiLepKinFitter::Param TtSemiEvtSolutionMaker::param(unsigned val) 
{
  TtSemiLepKinFitter::Param result;
  switch(val){
  case TtSemiLepKinFitter::kEMom       : result=TtSemiLepKinFitter::kEMom;       break;
  case TtSemiLepKinFitter::kEtEtaPhi   : result=TtSemiLepKinFitter::kEtEtaPhi;   break;
  case TtSemiLepKinFitter::kEtThetaPhi : result=TtSemiLepKinFitter::kEtThetaPhi; break;
  default: 
    throw cms::Exception("WrongConfig") 
      << "Chosen jet parametrization is not supported: " << val << "\n";
    break;
  }
  return result;
} 

TtSemiLepKinFitter::Constraint TtSemiEvtSolutionMaker::constraint(unsigned val) 
{
  TtSemiLepKinFitter::Constraint result;
  switch(val){
  case TtSemiLepKinFitter::kWHadMass     : result=TtSemiLepKinFitter::kWHadMass;     break;
  case TtSemiLepKinFitter::kWLepMass     : result=TtSemiLepKinFitter::kWLepMass;     break;
  case TtSemiLepKinFitter::kTopHadMass   : result=TtSemiLepKinFitter::kTopHadMass;   break;
  case TtSemiLepKinFitter::kTopLepMass   : result=TtSemiLepKinFitter::kTopLepMass;   break;
  case TtSemiLepKinFitter::kNeutrinoMass : result=TtSemiLepKinFitter::kNeutrinoMass; break;
  default: 
    throw cms::Exception("WrongConfig") 
      << "Chosen fit constraint is not supported: " << val << "\n";
    break;
  }
  return result;
} 

std::vector<TtSemiLepKinFitter::Constraint> TtSemiEvtSolutionMaker::constraints(std::vector<unsigned>& val)
{
  std::vector<TtSemiLepKinFitter::Constraint> result;
  for(unsigned i=0; i<val.size(); ++i){
    result.push_back(constraint(val[i]));
  } 
  return result;
}

