#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypWMassMaxSumPt.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"

TtSemiLepHypWMassMaxSumPt::TtSemiLepHypWMassMaxSumPt(const edm::ParameterSet& cfg):
  TtSemiLepHypothesis( cfg ),  
  maxNJets_(cfg.getParameter<int>   ("maxNJets")),
  wMass_   (cfg.getParameter<double>("wMass"   ))
{
  if(maxNJets_<4 && maxNJets_!=-1)
    throw cms::Exception("WrongConfig") 
      << "Parameter maxNJets can not be set to " << maxNJets_ << ". \n"
      << "It has to be larger than 4 or can be set to -1 to take all jets.";
}

TtSemiLepHypWMassMaxSumPt::~TtSemiLepHypWMassMaxSumPt() { }

void
TtSemiLepHypWMassMaxSumPt::buildHypo(edm::Event& evt,
				     const edm::Handle<edm::View<reco::RecoCandidate> >& leps,
				     const edm::Handle<std::vector<pat::MET> >& mets, 
				     const edm::Handle<std::vector<pat::Jet> >& jets, 
				     std::vector<int>& match, const unsigned int iComb)
{
  if(leps->empty() || mets->empty() || jets->size()<4){
    // create empty hypothesis
    return;
  }

  unsigned maxNJets = maxNJets_;
  if(maxNJets_ == -1) maxNJets = jets->size();

  match.clear();
  for(unsigned int i=0; i<4; ++i)
    match.push_back(-1);

  // -----------------------------------------------------
  // associate those jets that get closest to the W mass
  // with their invariant mass to the hadronic W boson
  // -----------------------------------------------------
  double wDist =-1.;
  std::vector<unsigned> closestToWMassIndices;
  for(unsigned idx=0; idx<maxNJets; ++idx){  
    for(unsigned jdx=(idx+1); jdx<maxNJets; ++jdx){
      reco::Particle::LorentzVector sum = 
	(*jets)[idx].p4()+
	(*jets)[jdx].p4();
      if( wDist<0. || wDist>fabs(sum.mass()-wMass_) ){
	wDist=fabs(sum.mass()-wMass_);
	closestToWMassIndices.clear();
	closestToWMassIndices.push_back(idx);
	closestToWMassIndices.push_back(jdx);
      }
    }
  }

  // -----------------------------------------------------
  // associate those jets with maximum pt of the vectorial 
  // sum to the hadronic decay chain
  // -----------------------------------------------------
  double maxPt=-1.;
  unsigned hadB=0;
  for(unsigned idx=0; idx<maxNJets; ++idx){
    // make sure it's not used up already from the hadronic W
    if( idx!=closestToWMassIndices[0] && idx!=closestToWMassIndices[1] ){
      reco::Particle::LorentzVector sum = 
	(*jets)[closestToWMassIndices[0]].p4()+
	(*jets)[closestToWMassIndices[1]].p4()+
	(*jets)[idx].p4();
      if( maxPt<0. || maxPt<sum.pt() ){
	maxPt=sum.pt();
	hadB=idx;
      }
    }
  }
  
  // -----------------------------------------------------
  // associate the remaining jet with maximum pt of the   
  // vectorial sum with the leading lepton with the 
  // leptonic b quark
  // -----------------------------------------------------
  maxPt=-1.;
  unsigned lepB=0;
  for(unsigned idx=0; idx<maxNJets; ++idx){
    // make sure it's not used up already from the hadronic decay chain
    if( idx!=closestToWMassIndices[0] && idx!=closestToWMassIndices[1] && idx!=hadB) {
      reco::Particle::LorentzVector sum = 
	(*jets)[idx].p4()+(*leps)[ 0 ].p4();
      if( maxPt<0. || maxPt<sum.pt() ){
	maxPt=sum.pt();
	lepB=idx;
      }
    }
  }

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if( isValid(closestToWMassIndices[0], jets) ){
    setCandidate(jets, closestToWMassIndices[0], lightQ_);
    match[TtSemiLepEvtPartons::LightQ] = closestToWMassIndices[0];
  }

  if( isValid(closestToWMassIndices[1], jets) ){
    setCandidate(jets, closestToWMassIndices[1], lightQBar_);
    match[TtSemiLepEvtPartons::LightQBar] = closestToWMassIndices[1];
  }
  
  if( isValid(hadB, jets) ){
    setCandidate(jets, hadB, hadronicB_);
    match[TtSemiLepEvtPartons::HadB] = hadB;
  }

  if( isValid(lepB, jets) ){
    setCandidate(jets, lepB, leptonicB_);
    match[TtSemiLepEvtPartons::LepB] = lepB;
  }

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  setCandidate(leps, 0, lepton_);
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  setCandidate(mets, 0, neutrino_);
}
