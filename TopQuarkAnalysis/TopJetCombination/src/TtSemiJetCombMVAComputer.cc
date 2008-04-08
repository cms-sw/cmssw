#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiJetCombMVAComputer.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiJetCombEval.h"

TtSemiJetCombMVAComputer::TtSemiJetCombMVAComputer(const edm::ParameterSet& cfg):
  muons_     (cfg.getParameter<edm::InputTag>("muons")),
  jets_      (cfg.getParameter<edm::InputTag>("jets")),
  nJetsMax_  (cfg.getParameter<int>("nJetsMax")),
  discrimCut_(cfg.getParameter<double>("discrimCut")),
  histDir_   (cfg.getParameter<std::string>("hist_directory"))
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

  edm::Handle<TopMuonCollection> muons; 
  evt.getByLabel(muons_, muons);

  edm::Handle<TopJetCollection> topJets;
  evt.getByLabel(jets_, topJets);

  math::XYZTLorentzVector muon = (*(muons->begin())).p4();

  // analyze jet combinations
  std::vector<int> jetIndices;
  for(unsigned int i=0; i<topJets->size(); ++i){
    if(nJetsMax_ >= 4 && i == (unsigned int) nJetsMax_) break;
    jetIndices.push_back(i);
  }
  
  std::vector<int> combi;
  unsigned int combiSize = 4;
  for(unsigned int i=0; i<combiSize; ++i) 
    combi.push_back(i);
  
  double discrimMax =.0;
  std::vector<int> combiMax;
  TtSemiJetComb jetCombMax;
  
  do{
    for(int cnt=0; cnt<TMath::Factorial(combiSize); ++cnt){
      if(combi[0] < combi[1]) {  // take into account indistinguishability 
	                         // of the two jets from the hadr. W decay,
	                         // reduces combinatorics by a factor of 2
	TtSemiJetComb jetComb(*topJets, combi, muon);
	
	// get discriminator here
	double discrim = evaluateTtSemiJetComb(mvaComputer, jetComb);
	if(discrim > discrimMax) {
	  discrimMax = discrim;
	  combiMax = combi;
	  jetCombMax = jetComb;
	}
      }
      next_permutation( combi.begin() , combi.end() );
    }
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

  std::auto_ptr< std::vector<int> > pOut(new std::vector<int>);
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
