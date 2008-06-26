#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiJetCombMVAComputer.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiJetCombEval.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

TtSemiJetCombMVAComputer::TtSemiJetCombMVAComputer(const edm::ParameterSet& cfg):
  leptons_ (cfg.getParameter<edm::InputTag>("leptons")),
  jets_    (cfg.getParameter<edm::InputTag>("jets")),
  nJetsMax_(cfg.getParameter<int>("nJetsMax"))
{
  produces< std::vector<int> >();
}

TtSemiJetCombMVAComputer::~TtSemiJetCombMVAComputer()
{
}

void
TtSemiJetCombMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  mvaComputer.update<TtSemiJetCombMVARcd>(setup, "ttSemiJetCombMVA");

  edm::Handle< edm::View<reco::RecoCandidate> > leptons; 
  evt.getByLabel(leptons_, leptons);

  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  std::auto_ptr< std::vector<int> > pOut(new std::vector<int>);

  unsigned int nPartons = 4;

  // skip events with no appropriate lepton candidate in
  // or less jets than partons
  if( leptons->empty() || jets->size() < nPartons ) {
    for(unsigned int i = 0; i < nPartons; ++i) 
      pOut->push_back( -1 );
    evt.put(pOut);
    return;
  }

  math::XYZTLorentzVector lepton = leptons->begin()->p4();

  // analyze jet combinations
  std::vector<int> jetIndices;
  for(unsigned int i=0; i<jets->size(); ++i){
    if(nJetsMax_ >= nPartons && i == (unsigned int) nJetsMax_) break;
    jetIndices.push_back(i);
  }
  
  std::vector<int> combi;
  for(unsigned int i=0; i<nPartons; ++i) 
    combi.push_back(i);
  
  double discrimMax =.0;
  std::vector<int> combiMax;

  do{
    for(int cnt = 0; cnt < TMath::Factorial( combi.size() ); ++cnt){
      if(combi[0] < combi[1]) {  // take into account indistinguishability 
	                         // of the two jets from the hadr. W decay,
	                         // reduces combinatorics by a factor of 2
	TtSemiJetComb jetComb(*jets, combi, lepton);

	// get discriminator here
	double discrim = evaluateTtSemiJetComb(mvaComputer, jetComb);
	if(discrim > discrimMax) {
	  discrimMax = discrim;
	  combiMax = combi;
	}
      }
      next_permutation( combi.begin() , combi.end() );
    }
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

  for(unsigned int i=0; i<combi.size(); ++i) 
    pOut->push_back( combi[i] );

  evt.put(pOut);
}

void 
TtSemiJetCombMVAComputer::beginJob(const edm::EventSetup&)
{
}

void 
TtSemiJetCombMVAComputer::endJob()
{
}

// implement the plugins for the computer container
// -> register TtSemiJetCombMVARcd
// -> define TtSemiJetCombMVAFileSource
MVA_COMPUTER_CONTAINER_IMPLEMENT(TtSemiJetCombMVA);
