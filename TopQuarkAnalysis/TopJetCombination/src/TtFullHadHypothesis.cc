#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtFullHadHypothesis.h"

/// default constructor
TtFullHadHypothesis::TtFullHadHypothesis(const edm::ParameterSet& cfg):
  jets_(cfg.getParameter<edm::InputTag>("jets")),
  lightQ_(0), lightQBar_(0), b_(0), bBar_(0), lightP_(0), lightPBar_(0)
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
}

/// default destructor
TtFullHadHypothesis::~TtFullHadHypothesis()
{
  if( lightQ_    ) delete lightQ_;
  if( lightQBar_ ) delete lightQBar_;
  if( b_         ) delete b_;
  if( bBar_      ) delete bBar_;
  if( lightP_    ) delete lightP_;
  if( lightPBar_ ) delete lightPBar_;
}

/// produce the event hypothesis as CompositeCandidate and Key
void
TtFullHadHypothesis::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  edm::Handle<std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);
  
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

  // go through given vector of jet combinations
  unsigned int idMatch = 0;
  typedef std::vector<std::vector<int> >::iterator MatchVecIterator;
  for(MatchVecIterator match = matchVec.begin(); match != matchVec.end(); ++match) {
    // reset pointers
    resetCandidates();
    // build hypothesis
    buildHypo(evt, jets, *match, idMatch++);
    pOut->push_back( std::make_pair(hypo(), *match) );
  }
  // feed out hyps and matches
  evt.put(pOut);

  // build and feed out key
  buildKey();
  *pKey=key();
  evt.put(pKey, "Key");
}

/// reset candidate pointers before hypo build process
void
TtFullHadHypothesis::resetCandidates()
{
  lightQ_    = 0;
  lightQBar_ = 0;
  b_         = 0;
  bBar_      = 0;
  lightP_    = 0;
  lightPBar_ = 0;
}

/// return event hypothesis
reco::CompositeCandidate
TtFullHadHypothesis::hypo()
{
  // check for sanity of the hypothesis
  if( !lightQ_ || !lightQBar_ || !b_ || 
      !bBar_ || !lightP_ || !lightPBar_ )
    return reco::CompositeCandidate();
  
  // setup transient references
  reco::CompositeCandidate hyp, top, w, topBar, wBar;

  AddFourMomenta addFourMomenta;  
  // build up the top bar branch
  wBar  .addDaughter(*lightP_,    TtFullHadDaughter::LightP    );
  wBar  .addDaughter(*lightPBar_, TtFullHadDaughter::LightPBar );
  addFourMomenta.set( wBar );
  topBar.addDaughter( wBar,  TtFullHadDaughter::WMinus );
  topBar.addDaughter(*bBar_, TtFullHadDaughter::BBar   );
  addFourMomenta.set( topBar );
  
  // build up the top branch that decays hadronically
  w  .addDaughter(*lightQ_,    TtFullHadDaughter::LightQ    );
  w  .addDaughter(*lightQBar_, TtFullHadDaughter::LightQBar );
  addFourMomenta.set( w );
  top.addDaughter( w,  TtFullHadDaughter::WPlus );
  top.addDaughter(*b_, TtFullHadDaughter::B     );
  addFourMomenta.set( top );

  // build ttbar hypotheses
  hyp.addDaughter( topBar, TtFullHadDaughter::TopBar );
  hyp.addDaughter( top,    TtFullHadDaughter::Top    );
  addFourMomenta.set( hyp );

  return hyp;
}

/// helper function to construct the proper correction level string for corresponding quarkType
std::string
TtFullHadHypothesis::jetCorrectionLevel(const std::string& quarkType)
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
  if( level=="L5Flavor:" || level=="L6UE:" || level=="L7Parton:" ){
    if(quarkType=="wQuarkMix"){level+="wMix";}
    if(quarkType=="udsQuark" ){level+="uds";}
    if(quarkType=="cQuark"   ){level+="charm";}
    if(quarkType=="bQuark"   ){level+="bottom";}
  }
  else{
    level+="none";
  }
  return level;
}

/// use one object in a jet collection to set a ShallowClonePtrCandidate with proper jet corrections
void 
TtFullHadHypothesis::setCandidate(const edm::Handle<std::vector<pat::Jet> >& handle, const int& idx,
				  reco::ShallowClonePtrCandidate*& clone, const std::string& correctionLevel)
{
  edm::Ptr<pat::Jet> ptr = edm::Ptr<pat::Jet>(handle, idx);
  // disentangle the correction from the potential flavor tag 
  // by the separating ':'; the flavor tag can be empty though
  std::string step   = correctionLevel.substr(0,correctionLevel.find(":"));
  std::string flavor = correctionLevel.substr(1+correctionLevel.find(":"));
  float corrFactor = 1.;
  if(flavor=="wMix")
    corrFactor = 0.75*ptr->jecFactor(step, "uds") + 0.25*ptr->jecFactor(step, "charm");
  else
    corrFactor = ptr->jecFactor(step, flavor);
  clone = new reco::ShallowClonePtrCandidate( ptr, ptr->charge(), ptr->p4()*corrFactor, ptr->vertex() );
}
