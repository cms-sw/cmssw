#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepJetCombMaxSumPtWMass.h"

#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

TtSemiLepJetCombMaxSumPtWMass::TtSemiLepJetCombMaxSumPtWMass(const edm::ParameterSet& cfg):
  jets_             (cfg.getParameter<edm::InputTag>("jets"             )),
  leps_             (cfg.getParameter<edm::InputTag>("leps"             )),
  maxNJets_         (cfg.getParameter<int>          ("maxNJets"         )),
  wMass_            (cfg.getParameter<double>       ("wMass"            )),
  useBTagging_      (cfg.getParameter<bool>         ("useBTagging"      )),
  bTagAlgorithm_    (cfg.getParameter<std::string>  ("bTagAlgorithm"    )),
  minBDiscBJets_    (cfg.getParameter<double>       ("minBDiscBJets"    )),
  maxBDiscLightJets_(cfg.getParameter<double>       ("maxBDiscLightJets"))
{
  if(maxNJets_<4 && maxNJets_!=-1)
    throw cms::Exception("WrongConfig") 
      << "Parameter maxNJets can not be set to " << maxNJets_ << ". \n"
      << "It has to be larger than 4 or can be set to -1 to take all jets.";

  produces<std::vector<std::vector<int> > >();
  produces<int>("NumberOfConsideredJets");
}

TtSemiLepJetCombMaxSumPtWMass::~TtSemiLepJetCombMaxSumPtWMass()
{
}

void
TtSemiLepJetCombMaxSumPtWMass::produce(edm::Event& evt, const edm::EventSetup& setup)
{
  std::auto_ptr<std::vector<std::vector<int> > > pOut(new std::vector<std::vector<int> >);
  std::auto_ptr<int> pJetsConsidered(new int);

  std::vector<int> match;
  for(unsigned int i = 0; i < 4; ++i) 
    match.push_back( -1 );

  // get jets
  edm::Handle< std::vector<pat::Jet> > jets;
  evt.getByLabel(jets_, jets);

  // get leptons
  edm::Handle< edm::View<reco::RecoCandidate> > leps; 
  evt.getByLabel(leps_, leps);


  // skip events without lepton candidate or less than 4 jets or no MET
  if(leps->empty() || jets->size() < 4){
    pOut->push_back( match );
    evt.put(pOut);
    *pJetsConsidered = jets->size();
    evt.put(pJetsConsidered, "NumberOfConsideredJets");
    return;
  }

  unsigned maxNJets = maxNJets_;
  if(maxNJets_ == -1 || (int)jets->size() < maxNJets_) maxNJets = jets->size();
  *pJetsConsidered = maxNJets;
  evt.put(pJetsConsidered, "NumberOfConsideredJets");

  std::vector<bool> isBJet;
  std::vector<bool> isLJet;
  int cntBJets = 0;
  if(useBTagging_) {
    for(unsigned int idx=0; idx<maxNJets; ++idx) {
      isBJet.push_back( ((*jets)[idx].bDiscriminator(bTagAlgorithm_) > minBDiscBJets_    ) );
      isLJet.push_back( ((*jets)[idx].bDiscriminator(bTagAlgorithm_) < maxBDiscLightJets_) );
      if((*jets)[idx].bDiscriminator(bTagAlgorithm_) > minBDiscBJets_    )cntBJets++;
    }
  }

  // -----------------------------------------------------
  // associate those jets with maximum pt of the vectorial 
  // sum to the hadronic decay chain
  // -----------------------------------------------------
  double maxPt=-1.;
  std::vector<int> maxPtIndices;
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);
  for(unsigned idx=0; idx<maxNJets; ++idx){
    if(useBTagging_ && (!isLJet[idx] || (cntBJets<=2 && isBJet[idx]))) continue;
    for(unsigned jdx=(idx+1); jdx<maxNJets; ++jdx){
      if(jdx==idx || (useBTagging_ && (!isLJet[jdx] || (cntBJets<=2 && isBJet[jdx]) || (cntBJets==3 && isBJet[idx] && isBJet[jdx])))) continue;
      for(unsigned kdx=0; kdx<maxNJets; ++kdx){
	if(kdx==idx || kdx==jdx || (useBTagging_ && !isBJet[kdx])) continue;
	reco::Particle::LorentzVector sum = 
	  (*jets)[idx].p4()+
	  (*jets)[jdx].p4()+
	  (*jets)[kdx].p4();
	if( maxPt<0. || maxPt<sum.pt() ){
	  maxPt=sum.pt();
	  maxPtIndices.clear();
	  maxPtIndices.push_back(idx);
	  maxPtIndices.push_back(jdx);
	  maxPtIndices.push_back(kdx);
	}
      }
    }
  }

  // -----------------------------------------------------
  // associate those jets that get closest to the W mass
  // with their invariant mass to the W boson
  // -----------------------------------------------------
  double wDist =-1.;
  std::vector<int> closestToWMassIndices;
  closestToWMassIndices.push_back(-1);
  closestToWMassIndices.push_back(-1);
  if( isValid(maxPtIndices[0], jets) && isValid(maxPtIndices[1], jets) && isValid(maxPtIndices[2], jets)) {
    for(unsigned idx=0; idx<maxPtIndices.size(); ++idx){  
      for(unsigned jdx=0; jdx<maxPtIndices.size(); ++jdx){  
	if( jdx==idx || maxPtIndices[idx]>maxPtIndices[jdx] || (useBTagging_ && (!isLJet[maxPtIndices[idx]] || !isLJet[maxPtIndices[jdx]] || (cntBJets<=2 && isBJet[maxPtIndices[idx]]) || (cntBJets<=2 && isBJet[maxPtIndices[jdx]]) || (cntBJets==3 && isBJet[maxPtIndices[idx]] && isBJet[maxPtIndices[jdx]])))) continue;
	reco::Particle::LorentzVector sum = 
	  (*jets)[maxPtIndices[idx]].p4()+
	  (*jets)[maxPtIndices[jdx]].p4();
	if( wDist<0. || wDist>fabs(sum.mass()-wMass_) ){
	  wDist=fabs(sum.mass()-wMass_);
	  closestToWMassIndices.clear();
	  closestToWMassIndices.push_back(maxPtIndices[idx]);
	  closestToWMassIndices.push_back(maxPtIndices[jdx]);
	}
      }
    }
  }
  int hadB=-1;
  for(unsigned idx=0; idx<maxPtIndices.size(); ++idx){
    // if this idx is not yet contained in the list of W mass candidates...
    if( std::find( closestToWMassIndices.begin(), closestToWMassIndices.end(), maxPtIndices[idx]) == closestToWMassIndices.end() ){
      hadB = maxPtIndices[idx];
      break; // there should be no other cadidates!
    }
  }

  // -----------------------------------------------------
  // associate the remaining jet with maximum pt of the   
  // vectorial sum with the leading lepton with the 
  // leptonic decay chain
  // -----------------------------------------------------
  maxPt=-1.;
  int lepB=-1;
  for(unsigned idx=0; idx<maxNJets; ++idx){
    if(useBTagging_ && !isBJet[idx]) continue;
    // make sure it's not used up already from the hadronic decay chain
    if( std::find(maxPtIndices.begin(), maxPtIndices.end(), idx) == maxPtIndices.end() ){
      reco::Particle::LorentzVector sum = 
	(*jets)[idx].p4()+(*leps)[ 0 ].p4();
      if( maxPt<0. || maxPt<sum.pt() ){
	maxPt=sum.pt();
	lepB=idx;
      }
    }
  }

  match[TtSemiLepEvtPartons::LightQ   ] = closestToWMassIndices[0];
  match[TtSemiLepEvtPartons::LightQBar] = closestToWMassIndices[1];
  match[TtSemiLepEvtPartons::HadB     ] = hadB;
  match[TtSemiLepEvtPartons::LepB     ] = lepB;

  pOut->push_back( match );
  evt.put(pOut);
}
