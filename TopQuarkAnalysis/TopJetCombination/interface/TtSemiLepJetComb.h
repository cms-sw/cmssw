#ifndef TtSemiLepJetComb_h
#define TtSemiLepJetComb_h

#include <vector>
#include <string>

#include "TMath.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

namespace JetComb{
  /// distinguish between hadronic and leptonic 
  /// decay chain of the top
  enum DecayType {kHad, kLep};
  /// supported std single variable types
  enum VarType   {kMass, kPt, kEta, kPhi, kTheta};
  /// supported comparison types 
  enum CompType  {kDeltaM, kDeltaR, kDeltaPhi, kDeltaTheta};
  /// b-tagging algorithms
  enum BTagAlgo  {kTrackCountHighEff, kTrackCountHighPur, kSoftMuon, kSoftMuonByPt, kSofMuonByIP3d, 
	          kSoftElec, kBProbability, kProbability, kSimpleSecondVtx, kCombSecondVtx, kCombSecondVtxMVA};
  /// operators for combining variables
  enum Operator  {kAdd, kMult};
}

/**
   \class   TtSemiLepJetComb TtSemiLepJetComb.h "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetComb.h"

   \brief   Common calculator class to keep multivariate analysis variables
            for jet combinations in semi-leptonic ttbar decays

   This class is currently used by TtSemiLepJetCombEval.h, which is included in both the
   TtSemiLepJetCombMVAComputer and the TtSemiLepJetCombMVATrainer.
*/

class TtSemiLepJetComb {

 public:
  
  /// emtpy constructor
  TtSemiLepJetComb();
  /// default constructor
  TtSemiLepJetComb(const std::vector<pat::Jet>&, const std::vector<int>&, const math::XYZTLorentzVector&, const pat::MET&);
  /// default destructor
  ~TtSemiLepJetComb();

  /// top quark candidate variable
  double topVar(JetComb::DecayType decay, JetComb::VarType var) const;
  /// W boson candidate variable
  double wBosonVar(JetComb::DecayType decay, JetComb::VarType var) const;
  /// b quark candidate variable
  double bQuarkVar(JetComb::DecayType decay, JetComb::VarType var) const;
  /// light quark candidate variable
  double lightQVar(bool qbar, JetComb::VarType var) const;
  /// lepton candidate variable
  double leptonVar(JetComb::VarType var) const;
  /// neutrino candidate variable
  double neutrinoVar(JetComb::VarType var) const;

  /// comparison between the two top candidates
  double compareHadTopLepTop(JetComb::CompType comp) const;
  /// comparison between the two W candidates
  double compareHadWLepW(JetComb::CompType comp) const;
  /// comparison between the two b candidates
  double compareHadBLepB(JetComb::CompType comp) const;
  /// comparison between the lightQ and the lightQBar candidate  
  double compareLightQuarks(JetComb::CompType comp) const;
  /// comparison between the lepton and the neutrino candidate 
  double compareLeptonNeutrino(JetComb::CompType comp) const;
  /// comparison between the top and the W candidate
  double compareTopW(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const;
  /// comparison between the top and the b candidate
  double compareTopB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const;
  /// comparison between the W and the b candidate
  double compareWB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const;
  /// comparison between the top and the lepton candidate
  double compareTopLepton(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the top and the neutrino candidate
  double compareTopNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the W and the lepton candidate
  double compareWLepton(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the W and the neutrino candidate
  double compareWNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the b and the lepton candidate
  double compareBLepton(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the b and the neutrino candidate
  double compareBNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const;
 
  /// pt of the hadronic top candidate relative to pt of the 
  /// sum of all other reconstruction possibilities (returns 
  /// values between 0 and 1)
  double relativePtHadronicTop() const;
  /// scalar sum of the pt of the b candidates relative to 
  /// scalar sum of the pt of the light quark candidates
  double bOverLightQPt() const;

  /// b-tag value of a b candidate
  double bTag(JetComb::DecayType decay, JetComb::BTagAlgo algo) const { return bTag(bQuark(decay), algo); };
  /// combined b-tag values of the two b candidates
  double combinedBTags(JetComb::BTagAlgo algo, JetComb::Operator op) const;
  /// combined b-tag values of the two light quark candidates
  double combinedBTagsForLightQuarks(JetComb::BTagAlgo algo, JetComb::Operator op) const;

  /// add an arbitary user defined variable with given key and value
  double addUserVar(std::string key, double value) { return userVariables_[key]=value;};
  /// receive user defined variable value with given key
  double userVar(const std::string& key) const {
    return (userVariables_.find(key)!=userVariables_.end() ? userVariables_.find(key)->second : -9999.);};

 private:

  /// reconstruct mother candidates from final state candidates
  void deduceMothers();
  /// b-tag value of one of the 4 jets
  double bTag(const pat::Jet& jet, JetComb::BTagAlgo algo) const;
  /// light quark candidate variable with default on q and not qbar
  double lightQVar(JetComb::VarType var) const { return lightQVar(false, var); };
  /// return lightQ or lightQBar candidate depending on argument
  const pat::Jet& lightQ(bool qbar=false) const { return (qbar ? hadQBarJet_ : hadQJet_); }
  /// return leptonic or hadronic b candidate depending on argument
  const pat::Jet& bQuark(JetComb::DecayType decay) const { return (decay==JetComb::kHad ? hadBJet_ : lepBJet_); }
  /// return leptonic or hadronic W candidate depending on argument
  const math::XYZTLorentzVector& wBoson(JetComb::DecayType decay) const { return (decay==JetComb::kHad ? hadW_ : lepW_); }
  /// return leptonic or hadronic top candidate depending on argument
  const math::XYZTLorentzVector& top(JetComb::DecayType decay) const { return (decay==JetComb::kHad ? hadTop_ : lepTop_); }  

 private:

  /// lightQ jet
  pat::Jet hadQJet_;
  /// lightQBar jet
  pat::Jet hadQBarJet_;
  /// hadronic b jet
  pat::Jet hadBJet_;
  /// leptonic b jet
  pat::Jet lepBJet_;
  /// neutrino candidate
  pat::MET neutrino_;
  /// lepton 4-vector
  math::XYZTLorentzVector lepton_;
  /// hadronic top 4-vector
  math::XYZTLorentzVector hadTop_;
  /// hadronic W 4-vector
  math::XYZTLorentzVector hadW_;
  /// leptonic top 4-vector
  math::XYZTLorentzVector lepTop_;
  /// leptonic W 4-vector
  math::XYZTLorentzVector lepW_;
  /// map for user defined variables
  std::map<std::string, double> userVariables_;
};

#endif
