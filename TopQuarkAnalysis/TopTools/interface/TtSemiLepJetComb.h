#ifndef TtSemiLepJetComb_h
#define TtSemiLepJetComb_h

#include <vector>
#include <string>

#include "TMath.h"
#include "Math/VectorUtil.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"


namespace JetComb{
  /// distinguish between hadronic and leptonic 
  /// decay chain of the top
  enum DecayType {kHad, kLep};
  /// supported std single variable types
  enum VarType   {kMass, kPt, kEta, kPhi, kTheta};
  /// supported comparison types 
  enum CompType  {kDeltaM, kDeltaR, kDeltaPhi, kDeltaTheta};
}

// ----------------------------------------------------------------------
// common calculator class to keep likelihood
// variables for jet combinations in the 
// semi leptonic ttbar decays
// ----------------------------------------------------------------------

class TtSemiLepJetComb {

 public:
  
  /// emtpy constructor
  TtSemiLepJetComb();
  /// default constructir
  TtSemiLepJetComb(const std::vector<pat::Jet>&, const std::vector<int>, const math::XYZTLorentzVector&, const pat::MET&);
  /// default destructor
  ~TtSemiLepJetComb();

  /// single top candidate variable
  double topVar(JetComb::DecayType decay, JetComb::VarType var) const;
  /// single W candidate variable
  double wBosonVar(JetComb::DecayType decay, JetComb::VarType var) const;
  /// single b candidate variable
  double bQuarkVar(JetComb::DecayType decay, JetComb::VarType var) const;
  /// ligh quark candidate variable
  double lightQVar(bool qbar, JetComb::VarType var) const;
  /// lepton candidate variable
  double leptonVar(JetComb::VarType var) const;
  /// neutrino candidate variable
  double neutrinoVar(JetComb::VarType var) const;

  /// comparison between the two top candidates
  double compareTopTop(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const;
  /// comparison between the top and the W candidate
  double compareTopW(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const;
  /// comparison between the top and the b candidate
  double compareTopB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const;
  /// comparison between the W and the b candidate
  double compareWB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const;
  /// comparison between the lightQ and the lightQBar candidate  
  double compareLightQuarks(JetComb::CompType comp) const;
  /// comparison between the top and the lepton candidate
  double compareTopLepton(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the top and the neutrino candidate
  double compareTopNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the b and the lepton candidate
  double compareBLepton(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the b and the neutrino candidate
  double compareBNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const;
  /// comparison between the lepton and the neutrino candidate 
  double compareLeptonNeutrino(JetComb::CompType comp) const;
 
  /// pt of the hadronic top candidate relative to pt of the 
  /// sum of all other reconstruction possibilities (returns 
  /// values between 0 and 1)
  double relativePtHadronicTop() const;
  /// scalar sum of the pt of the b candidates relative to 
  /// scalar sum of the pt of the light quark candidates
  double bOverLightQPt() const;

  /// summed btag for 'trackCountingHighEffBJetTags'
  double summedTrackCountingHighEff() const { return summedBTag("trackCountingHighEffBJetTags"); };
  /// summed btag for 'trackCountingHighPurBJetTags'
  double summedTrackCountingHighPur() const { return summedBTag("trackCountingHighPurBJetTags"); };
  /// summed btag for 'softMuonBJetTags'
  double summedSoftMuon() const { return summedBTag("softMuonBJetTags"); };
  /// summed btag for 'simpleSecondaryVertexBJetTags'
  double summedSimpleSecondaryVertex() const { return summedBTag("simpleSecondaryVertexBJetTags"); };
  /// summed btag for 'combinedSecondaryVertexBJetTags'
  double summedCombinedSecondaryVertex() const { return summedBTag("combinedSecondaryVertexBJetTags"); };
  /// summed btag for 'impactParameterMVABJetTags'
  double summedImpactParameterMVA() const { return summedBTag("impactParameterMVABJetTags"); };

  /// add an arbitary user defined variable with given key and value
  double addUserVar(std::string key, double value) { return userVariables_[key]=value;};
  /// receive user defined variable value with given key
  double userVar(const std::string& key) const { return (userVariables_.find(key)!=userVariables_.end() ? userVariables_.find(key)->second : -9999.);};

 private:

  /// reconstruct mother candidates from final state canddiates
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

inline double TtSemiLepJetComb::compareTopTop(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return top(dec1).mass()-top(dec2).mass();  
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(dec1), top(dec2));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(dec1), top(dec2));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(dec1), top(dec2));
  };
  return -9999.;
}

inline double TtSemiLepJetComb::compareTopW(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(dec1), wBoson(dec2));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(dec1), wBoson(dec2));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(dec1), wBoson(dec2));
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

inline double TtSemiLepJetComb::summedBTag(const std::string& label) const
{
  return bQuark(JetComb::kHad).bDiscriminator(label.c_str())+bQuark(JetComb::kLep).bDiscriminator(label.c_str());
}

#endif
