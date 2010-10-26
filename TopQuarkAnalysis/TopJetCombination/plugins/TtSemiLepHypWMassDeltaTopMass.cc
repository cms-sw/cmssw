#include "AnalysisDataFormats/TopObjects/interface/TtSemiLepEvtPartons.h"
#include "TopQuarkAnalysis/TopJetCombination/plugins/TtSemiLepHypWMassDeltaTopMass.h"

TtSemiLepHypWMassDeltaTopMass::TtSemiLepHypWMassDeltaTopMass(const edm::ParameterSet& cfg):
  TtSemiLepHypothesis( cfg ),  
  maxNJets_            (cfg.getParameter<int>        ("maxNJets"            )),
  wMass_               (cfg.getParameter<double>     ("wMass"               )),
  useBTagging_         (cfg.getParameter<bool>       ("useBTagging"         )),
  bTagAlgorithm_       (cfg.getParameter<std::string>("bTagAlgorithm"       )),
  minBDiscBJets_       (cfg.getParameter<double>     ("minBDiscBJets"       )),
  maxBDiscLightJets_   (cfg.getParameter<double>     ("maxBDiscLightJets"   )),
  neutrinoSolutionType_(cfg.getParameter<int>        ("neutrinoSolutionType"))
{
  if(maxNJets_<4 && maxNJets_!=-1)
    throw cms::Exception("WrongConfig") 
      << "Parameter maxNJets can not be set to " << maxNJets_ << ". \n"
      << "It has to be larger than 4 or can be set to -1 to take all jets.";
}

TtSemiLepHypWMassDeltaTopMass::~TtSemiLepHypWMassDeltaTopMass() { }

void
TtSemiLepHypWMassDeltaTopMass::buildHypo(edm::Event& evt,
				     const edm::Handle<edm::View<reco::RecoCandidate> >& leps,
				     const edm::Handle<std::vector<pat::MET> >& mets, 
				     const edm::Handle<std::vector<pat::Jet> >& jets, 
				     std::vector<int>& match, const unsigned int iComb)
{
  if(leps->empty() || mets->empty() || jets->size()<4){
    // create empty hypothesis
    return;
  }

  int maxNJets = maxNJets_;
  if(maxNJets_ == -1 || (int)jets->size() < maxNJets_) maxNJets = jets->size();

  std::vector<bool> isBJet;
  std::vector<bool> isLJet;
  int cntBJets = 0;
  if(useBTagging_) {
    for(int idx=0; idx<maxNJets; ++idx) {
      isBJet.push_back( ((*jets)[idx].bDiscriminator(bTagAlgorithm_) > minBDiscBJets_    ) );
      isLJet.push_back( ((*jets)[idx].bDiscriminator(bTagAlgorithm_) < maxBDiscLightJets_) );
      if((*jets)[idx].bDiscriminator(bTagAlgorithm_) > minBDiscBJets_    )cntBJets++;
    }
  }

  match.clear();
  for(unsigned int i=0; i<5; ++i)
    match.push_back(-1);

  // -----------------------------------------------------
  // add lepton
  // -----------------------------------------------------
  setCandidate(leps, 0, lepton_);
  match[TtSemiLepEvtPartons::Lepton] = 0;
  
  // -----------------------------------------------------
  // add neutrino
  // -----------------------------------------------------
  if(neutrinoSolutionType_ == -1)
    setCandidate(mets, 0, neutrino_);
  else
    setNeutrino(mets, leps, 0, neutrinoSolutionType_);

  // -----------------------------------------------------
  // associate those jets that get closest to the W mass
  // with their invariant mass to the hadronic W boson
  // -----------------------------------------------------
  double wDist =-1.;
  std::vector<int> closestToWMassIndices;
  closestToWMassIndices.push_back(-1);
  closestToWMassIndices.push_back(-1);
  for(int idx=0; idx<maxNJets; ++idx){
    if(useBTagging_ && (!isLJet[idx] || (cntBJets<=2 && isBJet[idx]))) continue;
    for(int jdx=(idx+1); jdx<maxNJets; ++jdx){
      if(useBTagging_ && (!isLJet[jdx] || (cntBJets<=2 && isBJet[jdx]) || (cntBJets==3 && isBJet[idx] && isBJet[jdx]))) continue;
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
  // associate those jets to the hadronic and the leptonic
  // b quark that minimize the difference between
  // hadronic and leptonic top mass
  // -----------------------------------------------------
  double deltaTop=-1.;
  int hadB=-1;
  int lepB=-1;
  if( isValid(closestToWMassIndices[0], jets) && isValid(closestToWMassIndices[1], jets)) {
    reco::Particle::LorentzVector lepW = neutrino_->p4() + lepton_->p4();
    reco::Particle::LorentzVector hadW =
      (*jets)[closestToWMassIndices[0]].p4()+
      (*jets)[closestToWMassIndices[1]].p4();
    // find hadronic b candidate
    for(int idx=0; idx<maxNJets; ++idx){
      if(useBTagging_ && !isBJet[idx]) continue;
      // make sure it's not used up already from the hadronic W
      if( idx!=closestToWMassIndices[0] && idx!=closestToWMassIndices[1] ){
	reco::Particle::LorentzVector hadTop = hadW + (*jets)[idx].p4();
	// find leptonic b candidate
	for(int jdx=0; jdx<maxNJets; ++jdx){
	  if(useBTagging_ && !isBJet[jdx]) continue;
	  // make sure it's not used up already from the hadronic branch
	  if( jdx!=closestToWMassIndices[0] && jdx!=closestToWMassIndices[1] && jdx!=idx ){
	    reco::Particle::LorentzVector lepTop = lepW + (*jets)[jdx].p4();
	    if( deltaTop<0. || deltaTop<fabs(hadTop.mass()-lepTop.mass()) ){
	      deltaTop=fabs(hadTop.mass()-lepTop.mass());
	      hadB=idx;
	      lepB=jdx;
	    }
	  }
	}
      }
    }
  }

  // -----------------------------------------------------
  // add jets
  // -----------------------------------------------------
  if( isValid(closestToWMassIndices[0], jets) ){
    setCandidate(jets, closestToWMassIndices[0], lightQ_, jetCorrectionLevel("wQuarkMix"));
    match[TtSemiLepEvtPartons::LightQ] = closestToWMassIndices[0];
  }

  if( isValid(closestToWMassIndices[1], jets) ){
    setCandidate(jets, closestToWMassIndices[1], lightQBar_, jetCorrectionLevel("wQuarkMix"));
    match[TtSemiLepEvtPartons::LightQBar] = closestToWMassIndices[1];
  }
  
  if( isValid(hadB, jets) ){
    setCandidate(jets, hadB, hadronicB_, jetCorrectionLevel("bQuark"));
    match[TtSemiLepEvtPartons::HadB] = hadB;
  }

  if( isValid(lepB, jets) ){
    setCandidate(jets, lepB, leptonicB_, jetCorrectionLevel("bQuark"));
    match[TtSemiLepEvtPartons::LepB] = lepB;
  }

}
