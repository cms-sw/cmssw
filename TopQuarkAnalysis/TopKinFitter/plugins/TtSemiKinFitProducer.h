#ifndef TtSemiKinFitProducer_h
#define TtSemiKinFitProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiKinFitter.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiEvtPartons.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"

template <typename LeptonCollection>
class TtSemiKinFitProducer : public edm::EDProducer {
  
 public:
  
  explicit TtSemiKinFitProducer(const edm::ParameterSet&);
  ~TtSemiKinFitProducer();
  
 private:

  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag jets_;
  edm::InputTag leps_;
  edm::InputTag mets_;

  edm::InputTag match_;
  bool useOnlyMatch_;

  unsigned maxNJets_;

  unsigned int maxNrIter_;
  double maxDeltaS_;
  double maxF_;
  unsigned int jetParam_;
  unsigned int lepParam_;
  unsigned int metParam_;
  std::vector<int> constraints_;

  TtSemiKinFitter* fitter;

};

template<typename LeptonCollection>
TtSemiKinFitProducer<LeptonCollection>::TtSemiKinFitProducer(const edm::ParameterSet& cfg):
  jets_        (cfg.getParameter<edm::InputTag>("jets")),
  leps_        (cfg.getParameter<edm::InputTag>("leps")),
  mets_        (cfg.getParameter<edm::InputTag>("mets")),
  match_       (cfg.getParameter<edm::InputTag>("match")),
  useOnlyMatch_(cfg.getParameter<bool>             ("useOnlyMatch"      )),
  maxNJets_    (cfg.getParameter<unsigned>         ("maxNJets"          )),
  maxNrIter_   (cfg.getParameter<unsigned>         ("maxNrIter"         )),
  maxDeltaS_   (cfg.getParameter<double>           ("maxDeltaS"         )),
  maxF_        (cfg.getParameter<double>           ("maxF"              )),
  jetParam_    (cfg.getParameter<unsigned>         ("jetParametrisation")),
  lepParam_    (cfg.getParameter<unsigned>         ("lepParametrisation")),
  metParam_    (cfg.getParameter<unsigned>         ("metParametrisation")),
  constraints_ (cfg.getParameter<std::vector<int> >("constraints"       ))
{
  fitter = new TtSemiKinFitter(jetParam_, lepParam_, metParam_, maxNrIter_, maxDeltaS_, maxF_, constraints_);

  produces< pat::Particle >("HadB");
  produces< pat::Particle >("HadP");
  produces< pat::Particle >("HadQ");
  produces< pat::Particle >("LepB");
  produces< pat::Particle >("LepL");
  produces< pat::Particle >("LepN");

  produces< std::vector<int> >();
  produces< double >("Chi2");
  produces< double >("Prob");
  produces< int >("Status");
}

template<typename LeptonCollection>
TtSemiKinFitProducer<LeptonCollection>::~TtSemiKinFitProducer()
{
  delete fitter;
}

template<typename LeptonCollection>
void TtSemiKinFitProducer<LeptonCollection>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr< pat::Particle > pHadB( new pat::Particle );
  std::auto_ptr< pat::Particle > pHadP( new pat::Particle );
  std::auto_ptr< pat::Particle > pHadQ( new pat::Particle );
  std::auto_ptr< pat::Particle > pLepB( new pat::Particle );
  std::auto_ptr< pat::Particle > pLepL( new pat::Particle );
  std::auto_ptr< pat::Particle > pLepN( new pat::Particle );

  std::auto_ptr< std::vector<int> > pCombi(new std::vector<int>);
  std::auto_ptr< double > pChi2( new double);
  std::auto_ptr< double > pProb( new double);
  std::auto_ptr< int > pStatus( new int);

  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  edm::Handle<std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  edm::Handle<LeptonCollection> leps;
  evt.getByLabel(leps_, leps);

  unsigned int nPartons = 4;

  edm::Handle<std::vector<int> > match;
  bool unvalidMatch = false;
  if(useOnlyMatch_) {
    evt.getByLabel(match_, match);
    // check if match is valid
    if(match->size()!=nPartons) unvalidMatch=true;
    else {
      for(unsigned int idx=0; idx<jets->size(); ++idx) {
	if(idx<0 || idx>=jets->size()) {
	  unvalidMatch=true;
	  break;
	}
      }
    }
  }

  // -----------------------------------------------------
  // skip events with no appropriate lepton candidate in
  // or empty MET or less jets than partons or unvalid match
  // -----------------------------------------------------

  if( leps->empty() || mets->empty() || jets->size()<nPartons || unvalidMatch ) {
    *pHadB=fitter->getFitHadb(); // these getters return empty objects here
    *pHadP=fitter->getFitHadp();
    *pHadQ=fitter->getFitHadq();
    *pLepB=fitter->getFitLepb();
    *pLepL=fitter->getFitLepl();
    *pLepN=fitter->getFitLepn();  
    evt.put(pHadB, "HadB");
    evt.put(pHadP, "HadP");
    evt.put(pHadQ, "HadQ");
    evt.put(pLepB, "LepB");
    evt.put(pLepL, "LepL");
    evt.put(pLepN, "LepN");  
    for(unsigned int i = 0; i < nPartons; ++i) 
      pCombi->push_back( -1 );
    evt.put(pCombi);
    *pChi2 = -1.;
    evt.put(pChi2, "Chi2");
    *pProb = -1.;
    evt.put(pProb, "Prob");
    *pStatus = -1;
    evt.put(pStatus, "Status");
    return;
  }

  // -----------------------------------------------------
  // analyze different jet combinations using the KinFitter
  // (or only a given jet combination if useOnlyMatch=true)
  // -----------------------------------------------------
  
  std::vector<int> jetIndices;
  if(!useOnlyMatch_) {
    for(unsigned int i=0; i<jets->size(); ++i){
      if(maxNJets_ >= nPartons && i == (unsigned int) maxNJets_) break;
      jetIndices.push_back(i);
    }
  }
  
  std::vector<int> combi;
  for(unsigned int i=0; i<nPartons; ++i) {
    if(useOnlyMatch_) combi.push_back( (*match)[i] );
    else combi.push_back(i);
  }

  pat::Particle bestHadb = pat::Particle();
  pat::Particle bestHadp = pat::Particle();
  pat::Particle bestHadq = pat::Particle();
  pat::Particle bestLepb = pat::Particle();
  pat::Particle bestLepl = pat::Particle();
  pat::Particle bestLepn = pat::Particle();
  
  double bestChi2 = -1.;
  std::vector<int> bestCombi;
  double bestProb = -1.;
  int bestStatus = -1;

  do{
    for(int cnt = 0; cnt < TMath::Factorial( combi.size() ); ++cnt){
      // take into account indistinguishability of the two jets from the hadr. W decay,
      // reduces combinatorics by a factor of 2
      if( combi[TtSemiEvtPartons::LightQ] < combi[TtSemiEvtPartons::LightQBar]
	 || useOnlyMatch_ ) {
	
	std::vector<pat::Jet> jetCombi;
	jetCombi.resize(nPartons);
	jetCombi[TtSemiEvtPartons::LightQ   ] = (*jets)[combi[TtSemiEvtPartons::LightQ   ]];
	jetCombi[TtSemiEvtPartons::LightQBar] = (*jets)[combi[TtSemiEvtPartons::LightQBar]];
	jetCombi[TtSemiEvtPartons::HadB     ] = (*jets)[combi[TtSemiEvtPartons::HadB     ]];
	jetCombi[TtSemiEvtPartons::LepB     ] = (*jets)[combi[TtSemiEvtPartons::LepB     ]];

	// do the kinematic fit
	int status = fitter->fit(jetCombi, (*leps)[0], (*mets)[0]);

	double chi2 = fitter->getS();
	// get details from the fitter if chi2 is the smallest found so far
	if(chi2 < bestChi2 || bestChi2 < 0) {
	  bestHadb = fitter->getFitHadb();
	  bestHadp = fitter->getFitHadp();
	  bestHadq = fitter->getFitHadq();
	  bestLepb = fitter->getFitLepb();
	  bestLepl = fitter->getFitLepl();
	  bestLepn = fitter->getFitLepn();
	  bestChi2 = chi2;
	  bestCombi = combi;
	  bestProb = fitter->getProb();
	  bestStatus = status;
	}

      }
      if(useOnlyMatch_) break; // don't go through combinatorics if useOnlyMatch was chosen
      next_permutation( combi.begin() , combi.end() );
    }
    if(useOnlyMatch_) break; // don't go through combinatorics if useOnlyMatch was chosen
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

  // -----------------------------------------------------
  // feed out result
  // -----------------------------------------------------

  // feed out particles that result from the kinematic fit
  *pHadB=bestHadb;
  *pHadP=bestHadp;
  *pHadQ=bestHadq;
  *pLepB=bestLepb;
  *pLepL=bestLepl;
  *pLepN=bestLepn;
  evt.put(pHadB, "HadB");
  evt.put(pHadP, "HadP");
  evt.put(pHadQ, "HadQ");
  evt.put(pLepB, "LepB");
  evt.put(pLepL, "LepL");
  evt.put(pLepN, "LepN");

  // feed out indices referring to the jet combination that gave the smallest chi2
  for(unsigned int i = 0; i < bestCombi.size(); ++i)
    pCombi->push_back( bestCombi[i] );
  evt.put(pCombi);

  // feed out chi2
  *pChi2=bestChi2;
  evt.put(pChi2, "Chi2");

  // feed out chi2 probability 
  *pProb=bestProb;
  evt.put(pProb, "Prob");

  // feed out status of the fitter
  *pStatus=bestStatus;
  evt.put(pStatus, "Status");
}

#endif
