#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "TopQuarkAnalysis/TopTools/interface/MEzCalculator.h"

#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepHypothesis.h"

/// default constructor
TtSemiLepHypothesis::TtSemiLepHypothesis(const edm::ParameterSet& cfg):
  jets_(cfg.getParameter<edm::InputTag>("jets")),
  leps_(cfg.getParameter<edm::InputTag>("leps")),
  mets_(cfg.getParameter<edm::InputTag>("mets")),
  numberOfRealNeutrinoSolutions_(-1),
  lightQ_(0), lightQBar_(0), hadronicB_(0), 
  leptonicB_(0), neutrino_(0), lepton_(0)
{
  getMatch_ = false;
  if( cfg.exists("match") ) {
    getMatch_ = true;
    match_ = cfg.getParameter<edm::InputTag>("match");
  }
  if( cfg.exists("jetCorrectionLevel") ) {
    jetCorrectionLevel_ = cfg.getParameter<std::string>("jetCorrectionLevel");
  }
  produces<std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > >();
  produces<int>("Key");
  produces<int>("NumberOfRealNeutrinoSolutions");
}

/// default destructor
TtSemiLepHypothesis::~TtSemiLepHypothesis()
{
  if( lightQ_    ) delete lightQ_;
  if( lightQBar_ ) delete lightQBar_;
  if( hadronicB_ ) delete hadronicB_;
  if( leptonicB_ ) delete leptonicB_;
  if( neutrino_  ) delete neutrino_;
  if( lepton_    ) delete lepton_;
}

/// produce the event hypothesis as CompositeCandidate and Key
void
TtSemiLepHypothesis::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);
  
  edm::Handle<edm::View<reco::RecoCandidate> > leps;
  evt.getByLabel(leps_, leps);

  edm::Handle<std::vector<pat::MET> > mets;
  evt.getByLabel(mets_, mets);

  std::vector<std::vector<int> > matchVec;
  if( getMatch_ ) {
    edm::Handle<std::vector<std::vector<int> > > matchHandle;
    evt.getByLabel(match_, matchHandle);
    matchVec = *matchHandle;
  }
  else {
    std::vector<int> dummyMatch;
    for(unsigned int i = 0; i < 4; ++i) 
      dummyMatch.push_back( -1 );
    matchVec.push_back( dummyMatch );
  }

  // declare auto_ptr for products
  std::auto_ptr<std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > >
    pOut( new std::vector<std::pair<reco::CompositeCandidate, std::vector<int> > > );
  std::auto_ptr<int> pKey(new int);
  std::auto_ptr<int> pNeutrinoSolutions(new int);

  // go through given vector of jet combinations
  unsigned int idMatch = 0;
  typedef std::vector<std::vector<int> >::iterator MatchVecIterator;
  for(MatchVecIterator match = matchVec.begin(); match != matchVec.end(); ++match) {
    // reset pointers
    resetCandidates();
    // build hypothesis
    buildHypo(evt, leps, mets, jets, *match, idMatch++);
    pOut->push_back( std::make_pair(hypo(), *match) );
  }
  // feed out hyps and matches
  evt.put(pOut);

  // build and feed out key
  buildKey();
  *pKey=key();
  evt.put(pKey, "Key");

  // feed out number of real neutrino solutions
  *pNeutrinoSolutions=numberOfRealNeutrinoSolutions_;
  evt.put(pNeutrinoSolutions, "NumberOfRealNeutrinoSolutions");
}

/// reset candidate pointers before hypo build process
void
TtSemiLepHypothesis::resetCandidates()
{
  numberOfRealNeutrinoSolutions_ = -1;
  lightQ_    = 0;
  lightQBar_ = 0;
  hadronicB_ = 0;
  leptonicB_ = 0;
  neutrino_  = 0;
  lepton_    = 0;
}

/// return event hypothesis
reco::CompositeCandidate
TtSemiLepHypothesis::hypo()
{
  // check for sanity of the hypothesis
  if( !lightQ_ || !lightQBar_ || !hadronicB_ || 
      !leptonicB_ || !neutrino_ || !lepton_ )
    return reco::CompositeCandidate();
  
  // setup transient references
  reco::CompositeCandidate hyp, hadTop, hadW, lepTop, lepW;

  AddFourMomenta addFourMomenta;  
  // build up the top branch that decays leptonically
  lepW  .addDaughter(*lepton_,   TtSemiLepDaughter::Lep    );
  lepW  .addDaughter(*neutrino_, TtSemiLepDaughter::Nu     );
  addFourMomenta.set( lepW );
  lepTop.addDaughter( lepW,      TtSemiLepDaughter::LepW   );
  lepTop.addDaughter(*leptonicB_,TtSemiLepDaughter::LepB   );
  addFourMomenta.set( lepTop );
  
  // build up the top branch that decays hadronically
  hadW  .addDaughter(*lightQ_,   TtSemiLepDaughter::HadP   );
  hadW  .addDaughter(*lightQBar_,TtSemiLepDaughter::HadQ   );
  addFourMomenta.set( hadW );
  hadTop.addDaughter( hadW,      TtSemiLepDaughter::HadW   );
  hadTop.addDaughter(*hadronicB_,TtSemiLepDaughter::HadB   );
  addFourMomenta.set( hadTop );

  // build ttbar hypotheses
  hyp.addDaughter( lepTop,       TtSemiLepDaughter::LepTop );
  hyp.addDaughter( hadTop,       TtSemiLepDaughter::HadTop );
  addFourMomenta.set( hyp );

  return hyp;
}

/// determine lepton type of reco candidate and return a corresponding WDecay::LeptonType; the type is kNone if it is whether a muon nor an electron 
WDecay::LeptonType
TtSemiLepHypothesis::leptonType(const reco::RecoCandidate* cand)
{
  // check whetherwe are dealing with a reco muon or a reco electron
  WDecay::LeptonType type = WDecay::kNone;
  if( dynamic_cast<const reco::Muon*>(cand) ){
    type = WDecay::kMuon;
  } 
  else if( dynamic_cast<const reco::GsfElectron*>(cand) ){
    type = WDecay::kElec;
  }
  return type;
}

/// helper function to construct the proper correction level string for corresponding quarkType
std::string
TtSemiLepHypothesis::jetCorrectionLevel(const std::string& quarkType)
{
  // jetCorrectionLevel was not configured
  if(jetCorrectionLevel_.empty())
    throw cms::Exception("Configuration")
      << "Unconfigured jetCorrectionLevel. Please use an appropriate, non-empty string.\n";

  // quarkType is unknown
  if( !(quarkType=="wQuarkMix" ||
	quarkType=="udsQuark" ||
	quarkType=="cQuark" ||
	quarkType=="bQuark") )
    throw cms::Exception("Configuration")
      << quarkType << " is unknown as a quarkType for the jetCorrectionLevel.\n";

  // combine correction level; start with a ':' even if 
  // there is no flavor tag to be added, as it is needed
  // by setCandidate to disentangle the correction tag 
  // from a potential flavor tag, which can be empty
  std::string level=jetCorrectionLevel_+":";
  if( level=="had:" || level=="ue:" || level=="part:" ){
    if(quarkType=="wQuarkMix"){level+="wMix";}
    if(quarkType=="udsQuark" ){level+="uds"; }
    if(quarkType=="cQuark"   ){level+="c";   }
    if(quarkType=="bQuark"   ){level+="b";   }
  }
  return level;
}

/// use one object in a jet collection to set a ShallowClonePtrCandidate with proper jet corrections
void 
TtSemiLepHypothesis::setCandidate(const edm::Handle<std::vector<pat::Jet> >& handle, const int& idx,
				  reco::ShallowClonePtrCandidate*& clone, const std::string& correctionLevel)
{
  edm::Ptr<pat::Jet> ptr = edm::Ptr<pat::Jet>(handle, idx);
  // disentangle the correction from the potential flavor tag 
  // by the separating ':'; the flavor tag can be empty though
  std::string step   = correctionLevel.substr(0,correctionLevel.find(":"));
  std::string flavor = correctionLevel.substr(1+correctionLevel.find(":"));
  float corrFactor = 1.;
  if(flavor=="wMix")
    corrFactor = 0.75*ptr->corrFactor(step, "uds") + 0.25*ptr->corrFactor(step, "c");
  else
    corrFactor = ptr->corrFactor(step, flavor);
  clone = new reco::ShallowClonePtrCandidate( ptr, ptr->charge(), ptr->p4()*corrFactor, ptr->vertex() );
}

/// set neutrino, using mW = 80.4 to calculate the neutrino pz
void TtSemiLepHypothesis::setNeutrino(const edm::Handle<std::vector<pat::MET> >& met,
				      const edm::Handle<edm::View<reco::RecoCandidate> >& leps, const int& idx, const int& type)
{
  edm::Ptr<pat::MET> ptr = edm::Ptr<pat::MET>(met, 0);
  MEzCalculator mez;
  mez.SetMET( *(met->begin()) );
  if( leptonType( &(leps->front()) ) == WDecay::kMuon )
    mez.SetLepton( (*leps)[idx], true );
  else if( leptonType( &(leps->front()) ) == WDecay::kElec )
    mez.SetLepton( (*leps)[idx], false );
  else
    throw cms::Exception("UnimplementedFeature") << "Type of lepton given together with MET for solving neutrino kinematics is neither muon nor electron.\n";
  double pz = mez.Calculate(type);
  numberOfRealNeutrinoSolutions_ = mez.IsComplex() ? 0 : 2;
  const math::XYZTLorentzVector p4( ptr->px(), ptr->py(), pz, sqrt(ptr->px()*ptr->px() + ptr->py()*ptr->py() + pz*pz) );
  neutrino_ = new reco::ShallowClonePtrCandidate( ptr, ptr->charge(), p4, ptr->vertex() );
}
