#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombMVAComputer.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepJetCombEval.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

TtSemiLepJetCombMVAComputer::TtSemiLepJetCombMVAComputer(const edm::ParameterSet& cfg):
  leptons_ (cfg.getParameter<edm::InputTag>("leptons")),
  jets_    (cfg.getParameter<edm::InputTag>("jets")),
  nJetsMax_(cfg.getParameter<int>("nJetsMax"))
{
  produces< std::vector<int> >();
  produces< std::string      >("Meth");
  produces< double           >("Disc");
}

TtSemiLepJetCombMVAComputer::~TtSemiLepJetCombMVAComputer()
{
}

void
TtSemiLepJetCombMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr< std::vector<int> > pOutCombi(new std::vector<int>);
  std::auto_ptr< std::string >      pOutMeth (new std::string);
  std::auto_ptr< double >           pOutDisc (new double);

  mvaComputer.update<TtSemiLepJetCombMVARcd>(setup, "ttSemiLepJetCombMVA");

  // read name of the last processor in the MVA calibration
  // (to be used as meta information)
  edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> calibContainer;
  setup.get<TtSemiLepJetCombMVARcd>().get( calibContainer );
  std::vector<PhysicsTools::Calibration::VarProcessor*> processors
    = (calibContainer->find("ttSemiLepJetCombMVA")).getProcessors();
  *pOutMeth = ( processors[ processors.size()-1 ] )->getInstanceName();
  evt.put(pOutMeth, "Meth");

  // get lepton and jets
  edm::Handle< edm::View<reco::RecoCandidate> > leptons; 
  evt.getByLabel(leptons_, leptons);

  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  unsigned int nPartons = 4;

  // skip events with no appropriate lepton candidate in
  // or less jets than partons
  if( leptons->empty() || jets->size() < nPartons ) {
    for(unsigned int i = 0; i < nPartons; ++i) 
      pOutCombi->push_back( -1 );
    evt.put(pOutCombi);
    *pOutDisc = 0.;
    evt.put(pOutDisc, "Disc");
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
      // take into account indistinguishability of the two jets from the hadr. W decay,
      // reduces combinatorics by a factor of 2
      if(combi[TtSemiLepEvtPartons::LightQ] < combi[TtSemiLepEvtPartons::LightQBar]) {

	TtSemiLepJetComb jetComb(*jets, combi, lepton);

	// get discriminator here
	double discrim = evaluateTtSemiLepJetComb(mvaComputer, jetComb);
	if(discrim > discrimMax) {
	  discrimMax = discrim;
	  combiMax = combi;
	}

      }
      next_permutation( combi.begin() , combi.end() );
    }
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

  // write result into the event
  for(unsigned int i = 0; i < combiMax.size(); ++i) 
    pOutCombi->push_back( combiMax[i] );
  evt.put(pOutCombi);

  *pOutDisc = discrimMax;
  evt.put(pOutDisc, "Disc");
}

void 
TtSemiLepJetCombMVAComputer::beginJob(const edm::EventSetup&)
{
}

void 
TtSemiLepJetCombMVAComputer::endJob()
{
}

// implement the plugins for the computer container
// -> register TtSemiLepJetCombMVARcd
// -> define TtSemiLepJetCombMVAFileSource
MVA_COMPUTER_CONTAINER_IMPLEMENT(TtSemiLepJetCombMVA);
