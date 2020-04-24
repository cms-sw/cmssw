//
//

#include <memory>

#include "TopQuarkAnalysis/TopEventProducers/interface/StEvtSolutionMaker.h"

StEvtSolutionMaker::StEvtSolutionMaker(const edm::ParameterSet& iConfig)
{
  // configurables
  electronSrcToken_    = mayConsume<std::vector<pat::Electron> >(iConfig.getParameter<edm::InputTag>("electronSource"));
  muonSrcToken_        = mayConsume<std::vector<pat::Muon> >(iConfig.getParameter<edm::InputTag>("muonSource"));
  metSrcToken_         = consumes<std::vector<pat::MET> >(iConfig.getParameter<edm::InputTag>("metSource"));
  jetSrcToken_         = consumes<std::vector<pat::Jet> >(iConfig.getParameter<edm::InputTag>("jetSource"));
  genEvtSrcToken_      = mayConsume<StGenEvent>(edm::InputTag("genEvt"));
  leptonFlavour_  = iConfig.getParameter< std::string >("leptonFlavour");
  jetCorrScheme_  = iConfig.getParameter<int>          ("jetCorrectionScheme");
  //jetInput_        = iConfig.getParameter< std::string > 	  ("jetInput");
  doKinFit_       = iConfig.getParameter< bool >       ("doKinFit");
  addLRJetComb_   = iConfig.getParameter< bool >       ("addLRJetComb");
  maxNrIter_      = iConfig.getParameter< int >        ("maxNrIter");
  maxDeltaS_      = iConfig.getParameter< double >     ("maxDeltaS");
  maxF_           = iConfig.getParameter< double >     ("maxF");
  jetParam_       = iConfig.getParameter<int>          ("jetParametrisation");
  lepParam_       = iConfig.getParameter<int>          ("lepParametrisation");
  metParam_       = iConfig.getParameter<int>          ("metParametrisation");
  constraints_    = iConfig.getParameter< std::vector<int> > ("constraints");
  matchToGenEvt_  = iConfig.getParameter< bool > 	("matchToGenEvt");

  // define kinfitter
  if(doKinFit_){
    myKinFitter = new StKinFitter(jetParam_, lepParam_, metParam_, maxNrIter_, maxDeltaS_, maxF_, constraints_);
  }
  // define what will be produced
  produces<std::vector<StEvtSolution> >();
}

StEvtSolutionMaker::~StEvtSolutionMaker()
{
  if (doKinFit_) delete myKinFitter;
}

void StEvtSolutionMaker::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //
  //  TopObject Selection
  //

  // select lepton (the TtLepton vectors are, for the moment, sorted on pT)
  bool leptonFound = false;
  edm::Handle<std::vector<pat::Muon> > muons;
  if(leptonFlavour_ == "muon"){
    iEvent.getByToken(muonSrcToken_, muons);
    if( !muons->empty() ) leptonFound = true;
  }
  edm::Handle<std::vector<pat::Electron> > electrons;
  if(leptonFlavour_ == "electron"){
    iEvent.getByToken(electronSrcToken_, electrons);
    if( !electrons->empty() ) leptonFound = true;
  }

  // select MET (TopMET vector is sorted on ET)
  bool metFound = false;
  edm::Handle<std::vector<pat::MET> > mets;
  iEvent.getByToken(metSrcToken_, mets);
  if( !mets->empty() ) metFound = true;

  // select Jets
  bool jetsFound = false;
  edm::Handle<std::vector<pat::Jet> > jets;
  iEvent.getByToken(jetSrcToken_, jets);
  unsigned int maxJets=2;//this has to become a custom-defined parameter (we may want 2 or 3 jets)
  if (jets->size() >= 2) jetsFound = true;

  std::vector<StEvtSolution> *evtsols = new std::vector<StEvtSolution>();
  if(leptonFound && metFound && jetsFound){
    std::cout<<"constructing solutions"<<std::endl;
    for (unsigned int b=0; b<maxJets; b++) {
      for (unsigned int l=0; l<maxJets; l++) {
	if(b!=l){  // to avoid double counting
	  StEvtSolution asol;
	  asol.setJetCorrectionScheme(jetCorrScheme_);
	  if(leptonFlavour_ == "muon")     asol.setMuon(muons, 0);
	  if(leptonFlavour_ == "electron") asol.setElectron(electrons, 0);
	  asol.setNeutrino(mets, 0);
	  asol.setBottom(jets, b);
	  asol.setLight(jets, l);

	  if(doKinFit_) asol = myKinFitter->addKinFitInfo(&asol);

	  /* to be adapted to ST (Andrea)
	     if(addLRJetComb_){
	     asol.setPtrueCombExist(jetCombProbs[m].getPTrueCombExist(&afitsol));
	     asol.setPtrueBJetSel(jetCombProbs[m].getPTrueBJetSel(&afitsol));
	     asol.setPtrueBhadrSel(jetCombProbs[m].getPTrueBhadrSel(&afitsol));
	     asol.setPtrueJetComb(afitsol.getPtrueCombExist()*afitsol.getPtrueBJetSel()*afitsol.getPtrueBhadrSel());
	     }
	  */
	  evtsols->push_back(asol);
	}
      }
    }

    // if asked for, match the event solutions to the gen Event
    if(matchToGenEvt_){
      /*
	edm::Handle<StGenEvent> genEvt;
	iEvent.getByToken (genEvtSrcToken_,genEvt);
	double bestSolDR = 9999.;
	int bestSol = 0;
	for(size_t s=0; s<evtsols->size(); s++) {
	(*evtsols)[s].setGenEvt(genEvt->particles());
	vector<double> bm = BestMatch((*evtsols)[s], false); //false to use DR, true SpaceAngles
	(*evtsols)[s].setSumDeltaRjp(bm[0]); // dRBB + dRLL
	(*evtsols)[s].setChangeBL((int) bm[1]); // has swapped or not
	(*evtsols)[s].setDeltaRB(bm[2]);
	(*evtsols)[s].setDeltaRL(bm[3]);
	if(bm[0]<bestSolDR) { bestSolDR =  bm[0]; bestSol = s; }
	}
	(*evtsols)[bestSol].setBestSol(true);
      */
    }

    //store the vector of solutions to the event
    std::unique_ptr<std::vector<StEvtSolution> > pOut(evtsols);
    iEvent.put(std::move(pOut));
  }
  else
    {

      std::cout<<"@@@ No calibrated solutions built, because:  " << std::endl;;
      if(jets->size()<maxJets)   				  std::cout<<"@ nr jets = " << jets->size() << " < " << maxJets <<std::endl;
      if(leptonFlavour_ == "muon" && !leptonFound)     	          std::cout<<"@ no good muon candidate"<<std::endl;
      if(leptonFlavour_ == "electron" && !leptonFound)             std::cout<<"@ no good electron candidate"<<std::endl;
      if(mets->empty())    					  std::cout<<"@ no MET reconstruction"<<std::endl;

      StEvtSolution asol;
      evtsols->push_back(asol);
      std::unique_ptr<std::vector<StEvtSolution> > pOut(evtsols);
      iEvent.put(std::move(pOut));
    }
}
