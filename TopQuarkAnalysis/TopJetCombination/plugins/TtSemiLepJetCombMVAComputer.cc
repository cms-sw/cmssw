#include "PhysicsTools/JetMCUtils/interface/combination.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetCombEval.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombMVAComputer.h"

TtSemiLepJetCombMVAComputer::TtSemiLepJetCombMVAComputer(const edm::ParameterSet& cfg):
  leps_    (cfg.getParameter<edm::InputTag>("leps")),
  jets_    (cfg.getParameter<edm::InputTag>("jets")),
  mets_    (cfg.getParameter<edm::InputTag>("mets")),
  maxNJets_(cfg.getParameter<int>("maxNJets")),
  maxNComb_(cfg.getParameter<int>("maxNComb"))
{
  produces<std::vector<std::vector<int> > >();
  produces<std::vector<double>            >("Discriminators");
  produces<std::string                    >("Method");
  produces<int                            >("NumberOfConsideredJets");
}

TtSemiLepJetCombMVAComputer::~TtSemiLepJetCombMVAComputer()
{
}

void
TtSemiLepJetCombMVAComputer::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr<std::vector<std::vector<int> > > pOut    (new std::vector<std::vector<int> >);
  std::auto_ptr<std::vector<double>            > pOutDisc(new std::vector<double>);
  std::auto_ptr<std::string                    > pOutMeth(new std::string);
  std::auto_ptr<int                            > pJetsConsidered(new int);

  mvaComputer.update<TtSemiLepJetCombMVARcd>(setup, "ttSemiLepJetCombMVA");

  // read name of the processor that provides the MVA discriminator
  // (to be used as meta information)
  edm::ESHandle<PhysicsTools::Calibration::MVAComputerContainer> calibContainer;
  setup.get<TtSemiLepJetCombMVARcd>().get( calibContainer );
  std::vector<PhysicsTools::Calibration::VarProcessor*> processors
    = (calibContainer->find("ttSemiLepJetCombMVA")).getProcessors();
  *pOutMeth = ( processors[ processors.size()-3 ] )->getInstanceName();
  evt.put(pOutMeth, "Method");

  // get lepton, jets and mets
  edm::Handle< edm::View<reco::RecoCandidate> > leptons; 
  evt.getByLabel(leps_, leptons);

  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  edm::Handle< std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  const unsigned int nPartons = 4;

  // skip events with no appropriate lepton candidate,
  // empty METs vector or less jets than partons
  if( leptons->empty() || mets->empty() || jets->size() < nPartons ) {
    std::vector<int> invalidCombi;
    for(unsigned int i = 0; i < nPartons; ++i) 
      invalidCombi.push_back( -1 );
    pOut->push_back( invalidCombi );
    evt.put(pOut);
    pOutDisc->push_back( 0. );
    evt.put(pOutDisc, "Discriminators");
    *pJetsConsidered = jets->size();
    evt.put(pJetsConsidered, "NumberOfConsideredJets");
    return;
  }

  const math::XYZTLorentzVector lepton = leptons->begin()->p4();

  const pat::MET *met = &(*mets)[0];

  // analyze jet combinations
  std::vector<int> jetIndices;
  for(unsigned int i=0; i<jets->size(); ++i){
    if(maxNJets_ >= (int) nPartons && maxNJets_ == (int) i) {
      *pJetsConsidered = i;
      break;
    }
    jetIndices.push_back(i);
  }
  
  std::vector<int> combi;
  for(unsigned int i=0; i<nPartons; ++i) 
    combi.push_back(i);

  typedef std::pair<double, std::vector<int> > discCombPair;
  std::list<discCombPair> discCombList;

  do{
    for(int cnt = 0; cnt < TMath::Factorial( combi.size() ); ++cnt){
      // take into account indistinguishability of the two jets from the hadr. W decay,
      // reduces combinatorics by a factor of 2
      if(combi[TtSemiLepEvtPartons::LightQ] < combi[TtSemiLepEvtPartons::LightQBar]) {

	TtSemiLepJetComb jetComb(*jets, combi, lepton, *met);

	// feed MVA input variables into a ValueList
	PhysicsTools::Variable::ValueList values;
	evaluateTtSemiLepJetComb(values, jetComb);

	// get discriminator from the MVAComputer
	double discrim = mvaComputer->eval( values );

	discCombList.push_back( std::make_pair(discrim, combi) );

      }
      next_permutation( combi.begin() , combi.end() );
    }
  }
  while(stdcomb::next_combination( jetIndices.begin(), jetIndices.end(), combi.begin(), combi.end() ));

  // sort results w.r.t. discriminator values
  discCombList.sort();

  // write result into the event
  // (starting with the JetComb having the highest discriminator value -> reverse iterator)
  unsigned int iDiscComb = 0;
  typedef std::list<discCombPair>::reverse_iterator discCombIterator;
  for(discCombIterator discCombPair = discCombList.rbegin(); discCombPair != discCombList.rend(); ++discCombPair) {
    if(maxNComb_ >= 1 && iDiscComb == (unsigned int) maxNComb_) break;
    pOut    ->push_back( discCombPair->second );
    pOutDisc->push_back( discCombPair->first  );
    iDiscComb++;
  }
  evt.put(pOut);
  evt.put(pOutDisc, "Discriminators");
  evt.put(pJetsConsidered, "NumberOfConsideredJets");
}

void 
TtSemiLepJetCombMVAComputer::beginJob()
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
