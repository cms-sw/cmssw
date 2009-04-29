#ifndef TtSemiLepJetComb_h
#define TtSemiLepJetComb_h

#include <vector>
#include <string>

#include "TMath.h"
#include "Math/VectorUtil.h"
#include "DataFormats/Candidate/interface/Candidate.h"

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
  enum BTagAlgo  {kTrkCntHighEff, kTrkCntHighPur, kSoftMuon,
		  kSimpSecondVtx, kCombSecondVtx, kImpactParaMVA};
  /// operators for combining variables
  enum Operator  {kAdd, kMult};
}

// ----------------------------------------------------------------------
// common calculator class to keep multivariate analysis variables for
// jet combinations in semi leptonic ttbar decays
// ----------------------------------------------------------------------

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

  /// b-tag value of the b candidate
  double bTag(JetComb::DecayType decay, JetComb::BTagAlgo algo) const;
  /// combined b-tag values of the two b candidates
  double combinedBTags(JetComb::BTagAlgo algo, JetComb::Operator op) const;

  /// add an arbitary user defined variable with given key and value
  double addUserVar(std::string key, double value) { return userVariables_[key]=value;};
  /// receive user defined variable value with given key
  double userVar(const std::string& key) const {
    return (userVariables_.find(key)!=userVariables_.end() ? userVariables_.find(key)->second : -9999.);};

 private:

  /// reconstruct mother candidates from final state candidates
  void deduceMothers();
  /// sum of btag from leptonic and hadronic b candidate
  /// with corresponding btag label
  double summedBTag(const std::string& label) const;
  /// ligh quark candidate variable with default on q and not qbar
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

inline double TtSemiLepJetComb::topVar(JetComb::DecayType decay, JetComb::VarType var) const 
{
  switch(var){
  case JetComb::kMass  : return top(decay).mass();
  case JetComb::kPt    : return top(decay).pt();
  case JetComb::kEta   : return top(decay).eta();
  case JetComb::kPhi   : return top(decay).phi();
  case JetComb::kTheta : return top(decay).theta();
  };
  return -9999.;
}

inline double TtSemiLepJetComb::wBosonVar(JetComb::DecayType decay, JetComb::VarType var) const 
{
  switch(var){
  case JetComb::kMass  : return wBoson(decay).mass();
  case JetComb::kPt    : return wBoson(decay).pt();
  case JetComb::kEta   : return wBoson(decay).eta();
  case JetComb::kPhi   : return wBoson(decay).phi();
  case JetComb::kTheta : return wBoson(decay).theta();
  };
  return -9999.;
}

inline double TtSemiLepJetComb::bQuarkVar(JetComb::DecayType decay, JetComb::VarType var) const 
{
  switch(var){
  case JetComb::kMass  : return -9999.;
  case JetComb::kPt    : return bQuark(decay).p4().pt();
  case JetComb::kEta   : return bQuark(decay).p4().eta();
  case JetComb::kPhi   : return bQuark(decay).p4().phi();
  case JetComb::kTheta : return bQuark(decay).p4().theta();
  };
  return -9999.;
}

inline double TtSemiLepJetComb::lightQVar(bool qbar, JetComb::VarType var) const 
{
  switch(var){
  case JetComb::kMass  : return -9999.;
  case JetComb::kPt    : return lightQ(qbar).p4().pt();
  case JetComb::kEta   : return lightQ(qbar).p4().eta();
  case JetComb::kPhi   : return lightQ(qbar).p4().phi();
  case JetComb::kTheta : return lightQ(qbar).p4().theta();
  };
  return -9999.;
}

inline double TtSemiLepJetComb::leptonVar(JetComb::VarType var) const 
{
  switch(var){
  case JetComb::kMass  : return -9999.;
  case JetComb::kPt    : return lepton_.pt();
  case JetComb::kEta   : return lepton_.eta();
  case JetComb::kPhi   : return lepton_.phi();
  case JetComb::kTheta : return lepton_.theta();
  };
  return -9999.;
}

inline double TtSemiLepJetComb::neutrinoVar(JetComb::VarType var) const 
{
  switch(var){
  case JetComb::kMass  : return -9999.;
  case JetComb::kPt    : return neutrino_.p4().pt();
  case JetComb::kEta   : return neutrino_.p4().eta();
  case JetComb::kPhi   : return neutrino_.p4().phi();
  case JetComb::kTheta : return neutrino_.p4().theta();
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareHadTopLepTop(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return top(JetComb::kHad).mass() - top(JetComb::kLep).mass();  
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(JetComb::kHad), top(JetComb::kLep));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(JetComb::kHad), top(JetComb::kLep));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(JetComb::kHad), top(JetComb::kLep));
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareHadWLepW(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return wBoson(JetComb::kHad).mass() - wBoson(JetComb::kLep).mass();
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(JetComb::kHad), wBoson(JetComb::kLep));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(JetComb::kHad), wBoson(JetComb::kLep));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(JetComb::kHad), wBoson(JetComb::kLep));
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareHadBLepB(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (bQuark(JetComb::kHad).p4(), bQuark(JetComb::kLep).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(bQuark(JetComb::kHad).p4(), bQuark(JetComb::kLep).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (bQuark(JetComb::kHad).p4(), bQuark(JetComb::kLep).p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareLightQuarks(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (lightQ().p4(), lightQ(true).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(lightQ().p4(), lightQ(true).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (lightQ().p4(), lightQ(true).p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareLeptonNeutrino(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (lepton_, neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(lepton_, neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (lepton_, neutrino_.p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareTopW(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return top(dec1).mass() - wBoson(dec2).mass();
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(dec1), wBoson(dec2));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(dec1), wBoson(dec2));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(dec1), wBoson(dec2));
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareTopB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(dec1), bQuark(dec2).p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareWB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(dec1), bQuark(dec2).p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareTopLepton(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(decay), lepton_);
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(decay), lepton_);
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(decay), lepton_);
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareTopNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(decay), neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(decay), neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(decay), neutrino_.p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareWLepton(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(decay), lepton_);
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(decay), lepton_);
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(decay), lepton_);
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareWNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(decay), neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(decay), neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(decay), neutrino_.p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareBLepton(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (bQuark(decay).p4(), lepton_);
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(bQuark(decay).p4(), lepton_);
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (bQuark(decay).p4(), lepton_);
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareBNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (bQuark(decay).p4(), neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(bQuark(decay).p4(), neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (bQuark(decay).p4(), neutrino_.p4());
  };
  return -9999.;
}

inline double TtSemiLepJetComb::relativePtHadronicTop() const
{
  return top(JetComb::kHad).pt()/(top(JetComb::kHad).pt()                                               +
			 (lightQ()    .p4()+lightQ(true).p4()+bQuark(JetComb::kLep).p4()).pt()          +
			 (lightQ()    .p4()+bQuark(JetComb::kHad).p4()+bQuark(JetComb::kLep).p4()).pt() +
			 (bQuark(JetComb::kHad).p4()+lightQ(true).p4()+bQuark(JetComb::kLep).p4()).pt()
			 ); 
}

inline double TtSemiLepJetComb::bOverLightQPt() const 
{
  return (bQuark(JetComb::kHad).p4().pt()+bQuark(JetComb::kLep).p4().pt())/(lightQ().p4().pt()+lightQ(true).p4().pt());  
}

inline double TtSemiLepJetComb::bTag(JetComb::DecayType decay, JetComb::BTagAlgo algo) const
{
  switch(algo){
  case JetComb::kTrkCntHighEff : return bQuark(decay).bDiscriminator("trackCountingHighEffBJetTags"   );
  case JetComb::kTrkCntHighPur : return bQuark(decay).bDiscriminator("trackCountingHighPurBJetTags"   );
  case JetComb::kSoftMuon      : return bQuark(decay).bDiscriminator("softMuonBJetTags"               );
  case JetComb::kSimpSecondVtx : return bQuark(decay).bDiscriminator("simpleSecondaryVertexBJetTags"  );
  case JetComb::kCombSecondVtx : return bQuark(decay).bDiscriminator("combinedSecondaryVertexBJetTags");
  case JetComb::kImpactParaMVA : return bQuark(decay).bDiscriminator("impactParameterMVABJetTags"     );
  };
  return -9999.;
}

inline double TtSemiLepJetComb::combinedBTags(JetComb::BTagAlgo algo, JetComb::Operator op) const
{
  switch(op){
  case JetComb::kAdd  : return bTag(JetComb::kHad, algo) + bTag(JetComb::kLep, algo);
  case JetComb::kMult : return bTag(JetComb::kHad, algo) * bTag(JetComb::kLep, algo);
  };
  return -9999.;
}

#endif
