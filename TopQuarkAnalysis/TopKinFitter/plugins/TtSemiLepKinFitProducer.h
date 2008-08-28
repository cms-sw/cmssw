#ifndef TtSemiLepKinFitProducer_h
#define TtSemiLepKinFitProducer_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/JetMCUtils/interface/combination.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopKinFitter/interface/TtSemiLepKinFitter.h"


template <typename LeptonCollection>
class TtSemiLepKinFitProducer : public edm::EDProducer {
  
 public:
  
  explicit TtSemiLepKinFitProducer(const edm::ParameterSet&);
  ~TtSemiLepKinFitProducer();
  
 private:
  // produce
  virtual void produce(edm::Event&, const edm::EventSetup&);

  // convert unsigned to Param
  TtSemiLepKinFitter::Param param(unsigned);
  // convert unsigned to Param
  TtSemiLepKinFitter::Constraint constraint(unsigned);
  // convert unsigned to Param
  std::vector<TtSemiLepKinFitter::Constraint> constraints(std::vector<unsigned>&);
  
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
  std::vector<unsigned> constraints_;

  TtSemiLepKinFitter* fitter;
};

template<typename LeptonCollection>
TtSemiLepKinFitProducer<LeptonCollection>::TtSemiLepKinFitProducer(const edm::ParameterSet& cfg):
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
  constraints_ (cfg.getParameter<std::vector<unsigned> >("constraints"  ))
{
  fitter = new TtSemiLepKinFitter(param(jetParam_), param(lepParam_), param(metParam_), maxNrIter_, maxDeltaS_, maxF_, constraints(constraints_));

  produces< std::vector<pat::Particle> >("Partons");
  produces< std::vector<pat::Particle> >("Leptons");
  produces< std::vector<pat::Particle> >("Neutrinos");

  produces< std::vector<int> >();
  produces< double >("Chi2");
  produces< double >("Prob");
  produces< int >("Status");
}

template<typename LeptonCollection>
TtSemiLepKinFitProducer<LeptonCollection>::~TtSemiLepKinFitProducer()
{
  delete fitter;
}

template<typename LeptonCollection>
void TtSemiLepKinFitProducer<LeptonCollection>::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr< std::vector<pat::Particle> > pPartons  ( new std::vector<pat::Particle> );
  std::auto_ptr< std::vector<pat::Particle> > pLeptons  ( new std::vector<pat::Particle> );
  std::auto_ptr< std::vector<pat::Particle> > pNeutrinos( new std::vector<pat::Particle> );

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
  pPartons->resize(nPartons);

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
    // the kinFit getters return empty objects here
    (*pPartons)[TtSemiLepEvtPartons::LightQ   ] = fitter->fittedHadP();
    (*pPartons)[TtSemiLepEvtPartons::LightQBar] = fitter->fittedHadQ();
    (*pPartons)[TtSemiLepEvtPartons::HadB     ] = fitter->fittedHadB();
    (*pPartons)[TtSemiLepEvtPartons::LepB     ] = fitter->fittedLepB();
    pLeptons  ->push_back( fitter->fittedLepton() );
    pNeutrinos->push_back( fitter->fittedNeutrino() );
    evt.put(pPartons,   "Partons");
    evt.put(pLeptons,   "Leptons");
    evt.put(pNeutrinos, "Neutrinos");
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
      if( combi[TtSemiLepEvtPartons::LightQ] < combi[TtSemiLepEvtPartons::LightQBar]
	 || useOnlyMatch_ ) {
	
	std::vector<pat::Jet> jetCombi;
	jetCombi.resize(nPartons);
	jetCombi[TtSemiLepEvtPartons::LightQ   ] = (*jets)[combi[TtSemiLepEvtPartons::LightQ   ]];
	jetCombi[TtSemiLepEvtPartons::LightQBar] = (*jets)[combi[TtSemiLepEvtPartons::LightQBar]];
	jetCombi[TtSemiLepEvtPartons::HadB     ] = (*jets)[combi[TtSemiLepEvtPartons::HadB     ]];
	jetCombi[TtSemiLepEvtPartons::LepB     ] = (*jets)[combi[TtSemiLepEvtPartons::LepB     ]];

	// do the kinematic fit
	int status = fitter->fit(jetCombi, (*leps)[0], (*mets)[0]);

	double chi2 = fitter->fitS();
	// get details from the fitter if chi2 is the smallest found so far
	if(chi2 < bestChi2 || bestChi2 < 0) {
	  bestHadb = fitter->fittedHadB();
	  bestHadp = fitter->fittedHadP();
	  bestHadq = fitter->fittedHadQ();
	  bestLepb = fitter->fittedLepB();
	  bestLepl = fitter->fittedLepton();
	  bestLepn = fitter->fittedNeutrino();
	  bestChi2 = chi2;
	  bestCombi= combi;
	  bestProb = fitter->fitProb();
	  bestStatus = status;
	}

      }
      if(useOnlyMatch_) break; // don't go through combinatorics if useOnlyMatch was chosen
      next_permutation( combi.begin(), combi.end() );
    }
    if(useOnlyMatch_) break; // don't go through combinatorics if useOnlyMatch was chosen
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));
  
  // -----------------------------------------------------
  // feed out result
  // -----------------------------------------------------
  
  // feed out particles that result from the kinematic fit
  (*pPartons)[TtSemiLepEvtPartons::LightQ   ] = bestHadp;
  (*pPartons)[TtSemiLepEvtPartons::LightQBar] = bestHadq;
  (*pPartons)[TtSemiLepEvtPartons::HadB     ] = bestHadb;
  (*pPartons)[TtSemiLepEvtPartons::LepB     ] = bestLepb;
  pLeptons  ->push_back( bestLepl );
  pNeutrinos->push_back( bestLepn );
  evt.put(pPartons,   "Partons");
  evt.put(pLeptons,   "Leptons");
  evt.put(pNeutrinos, "Neutrinos");
  
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
 
template<typename LeptonCollection>
TtSemiLepKinFitter::Param TtSemiLepKinFitProducer<LeptonCollection>::param(unsigned val) 
{
  TtSemiLepKinFitter::Param result;
  switch(val){
  case 0 : result=TtSemiLepKinFitter::kEMom;       break;
  case 1 : result=TtSemiLepKinFitter::kEtEtaPhi;   break;
  case 2 : result=TtSemiLepKinFitter::kEtThetaPhi; break;
  default: 
    throw cms::Exception("WrongConfig") 
      << "Chosen jet parametrization is not supported: " << val << "\n";
    break;
  }
  return result;
} 

template<typename LeptonCollection>
TtSemiLepKinFitter::Constraint TtSemiLepKinFitProducer<LeptonCollection>::constraint(unsigned val) 
{
  TtSemiLepKinFitter::Constraint result;
  switch(val){
  case 0 : result=TtSemiLepKinFitter::kWHadMass;     break;
  case 1 : result=TtSemiLepKinFitter::kWLepMass;     break;
  case 2 : result=TtSemiLepKinFitter::kTopHadMass;   break;
  case 3 : result=TtSemiLepKinFitter::kTopLepMass;   break;
  case 4 : result=TtSemiLepKinFitter::kNeutrinoMass; break;
  default: 
    throw cms::Exception("WrongConfig") 
      << "Chosen fit contraint is not supported: " << val << "\n";
    break;
  }
  return result;
} 

template<typename LeptonCollection>
std::vector<TtSemiLepKinFitter::Constraint> TtSemiLepKinFitProducer<LeptonCollection>::constraints(std::vector<unsigned>& val)
{
  std::vector<TtSemiLepKinFitter::Constraint> result;
  for(unsigned i=0; i<val.size(); ++i){
    result.push_back(constraint(val[i]));
  }
  return result; 
}

#endif
