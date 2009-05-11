#include "AnalysisDataFormats/TopObjects/interface/TtEventPartons.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetComb.h"

TtSemiLepJetComb::TtSemiLepJetComb()
{
}

TtSemiLepJetComb::TtSemiLepJetComb(const std::vector<pat::Jet>& jets, const std::vector<int>& combination,
				   const math::XYZTLorentzVector& lepton, const pat::MET& neutrino)
{ 
  // receive right jet association
  // from jet-parton matching 
  hadQJet_    = jets[ combination[TtSemiLepEvtPartons::LightQ   ] ];
  hadQBarJet_ = jets[ combination[TtSemiLepEvtPartons::LightQBar] ];
  hadBJet_    = jets[ combination[TtSemiLepEvtPartons::HadB     ] ];
  lepBJet_    = jets[ combination[TtSemiLepEvtPartons::LepB     ] ]; 
  lepton_     = lepton;
  neutrino_   = neutrino;
  // create mother candidates from 
  // final-state candidates
  deduceMothers();
}

TtSemiLepJetComb::~TtSemiLepJetComb() 
{
}

double TtSemiLepJetComb::topVar(JetComb::DecayType decay, JetComb::VarType var) const 
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

double TtSemiLepJetComb::wBosonVar(JetComb::DecayType decay, JetComb::VarType var) const 
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

double TtSemiLepJetComb::bQuarkVar(JetComb::DecayType decay, JetComb::VarType var) const 
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

double TtSemiLepJetComb::lightQVar(bool qbar, JetComb::VarType var) const 
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

double TtSemiLepJetComb::leptonVar(JetComb::VarType var) const 
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

double TtSemiLepJetComb::neutrinoVar(JetComb::VarType var) const 
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

double TtSemiLepJetComb::compareHadTopLepTop(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return top(JetComb::kHad).mass() - top(JetComb::kLep).mass();  
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(JetComb::kHad), top(JetComb::kLep));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(JetComb::kHad), top(JetComb::kLep));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(JetComb::kHad), top(JetComb::kLep));
  };
  return -9999.;
}

double TtSemiLepJetComb::compareHadWLepW(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return wBoson(JetComb::kHad).mass() - wBoson(JetComb::kLep).mass();
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(JetComb::kHad), wBoson(JetComb::kLep));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(JetComb::kHad), wBoson(JetComb::kLep));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(JetComb::kHad), wBoson(JetComb::kLep));
  };
  return -9999.;
}

double TtSemiLepJetComb::compareHadBLepB(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (bQuark(JetComb::kHad).p4(), bQuark(JetComb::kLep).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(bQuark(JetComb::kHad).p4(), bQuark(JetComb::kLep).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (bQuark(JetComb::kHad).p4(), bQuark(JetComb::kLep).p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::compareLightQuarks(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (lightQ().p4(), lightQ(true).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(lightQ().p4(), lightQ(true).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (lightQ().p4(), lightQ(true).p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::compareLeptonNeutrino(JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (lepton_, neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(lepton_, neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (lepton_, neutrino_.p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::compareTopW(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return top(dec1).mass() - wBoson(dec2).mass();
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(dec1), wBoson(dec2));
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(dec1), wBoson(dec2));
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(dec1), wBoson(dec2));
  };
  return -9999.;
}

double TtSemiLepJetComb::compareTopB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(dec1), bQuark(dec2).p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::compareWB(JetComb::DecayType dec1, JetComb::DecayType dec2, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(dec1), bQuark(dec2).p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(dec1), bQuark(dec2).p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::compareTopLepton(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(decay), lepton_);
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(decay), lepton_);
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(decay), lepton_);
  };
  return -9999.;
}

double TtSemiLepJetComb::compareTopNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (top(decay), neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(top(decay), neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (top(decay), neutrino_.p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::compareWLepton(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(decay), lepton_);
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(decay), lepton_);
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(decay), lepton_);
  };
  return -9999.;
}

double TtSemiLepJetComb::compareWNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (wBoson(decay), neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(wBoson(decay), neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (wBoson(decay), neutrino_.p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::compareBLepton(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (bQuark(decay).p4(), lepton_);
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(bQuark(decay).p4(), lepton_);
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (bQuark(decay).p4(), lepton_);
  };
  return -9999.;
}

double TtSemiLepJetComb::compareBNeutrino(JetComb::DecayType decay, JetComb::CompType comp) const 
{
  switch(comp){
  case JetComb::kDeltaM     : return -9999.;
  case JetComb::kDeltaR     : return ROOT::Math::VectorUtil::DeltaR  (bQuark(decay).p4(), neutrino_.p4());
  case JetComb::kDeltaPhi   : return ROOT::Math::VectorUtil::DeltaPhi(bQuark(decay).p4(), neutrino_.p4());
  case JetComb::kDeltaTheta : return ROOT::Math::VectorUtil::Angle   (bQuark(decay).p4(), neutrino_.p4());
  };
  return -9999.;
}

double TtSemiLepJetComb::relativePtHadronicTop() const
{
  return top(JetComb::kHad).pt()/(top(JetComb::kHad).pt()                                               +
			 (lightQ()    .p4()+lightQ(true).p4()+bQuark(JetComb::kLep).p4()).pt()          +
			 (lightQ()    .p4()+bQuark(JetComb::kHad).p4()+bQuark(JetComb::kLep).p4()).pt() +
			 (bQuark(JetComb::kHad).p4()+lightQ(true).p4()+bQuark(JetComb::kLep).p4()).pt()
			 ); 
}

double TtSemiLepJetComb::bOverLightQPt() const 
{
  return (bQuark(JetComb::kHad).p4().pt()+bQuark(JetComb::kLep).p4().pt())/(lightQ().p4().pt()+lightQ(true).p4().pt());  
}

double TtSemiLepJetComb::combinedBTags(JetComb::BTagAlgo algo, JetComb::Operator op) const
{
  switch(op){
  case JetComb::kAdd  : return bTag(JetComb::kHad, algo) + bTag(JetComb::kLep, algo);
  case JetComb::kMult : return bTag(JetComb::kHad, algo) * bTag(JetComb::kLep, algo);
  };
  return -9999.;
}

double TtSemiLepJetComb::combinedBTagsForLightQuarks(JetComb::BTagAlgo algo, JetComb::Operator op) const
{
  switch(op){
  case JetComb::kAdd  : return bTag(lightQ(), algo) + bTag(lightQ(true), algo);
  case JetComb::kMult : return bTag(lightQ(), algo) * bTag(lightQ(true), algo);
  };
  return -9999.;
}

// ----------------------------------------------------------------------
// private methods
// ----------------------------------------------------------------------

void 
TtSemiLepJetComb::deduceMothers()
{
  hadW_   = hadQJet_.p4() + hadQBarJet_.p4();
  lepW_   = lepton_       + neutrino_  .p4();
  hadTop_ = hadW_         + hadBJet_   .p4();
  lepTop_ = lepW_         + lepBJet_   .p4();
}

double TtSemiLepJetComb::bTag(const pat::Jet& jet, JetComb::BTagAlgo algo) const
{
  switch(algo){
  case JetComb::kTrackCountHighEff : return jet.bDiscriminator("trackCountingHighEffBJetTags"      );
  case JetComb::kTrackCountHighPur : return jet.bDiscriminator("trackCountingHighPurBJetTags"      );
  case JetComb::kSoftMuon          : return jet.bDiscriminator("softMuonBJetTags"                  );
  case JetComb::kSoftMuonByPt      : return jet.bDiscriminator("softMuonByPtBJetTags"              );
  case JetComb::kSofMuonByIP3d     : return jet.bDiscriminator("softMuonByIP3dBJetTags"            );
  case JetComb::kSoftElec          : return jet.bDiscriminator("softElectronBJetTags"              );
  case JetComb::kProbability       : return jet.bDiscriminator("jetProbabilityBJetTags"            );
  case JetComb::kBProbability      : return jet.bDiscriminator("jetBProbabilityBJetTags"           );
  case JetComb::kSimpleSecondVtx   : return jet.bDiscriminator("simpleSecondaryVertexBJetTags"     );
  case JetComb::kCombSecondVtx     : return jet.bDiscriminator("combinedSecondaryVertexBJetTags"   );
  case JetComb::kCombSecondVtxMVA  : return jet.bDiscriminator("combinedSecondaryVertexMVABJetTags");
  };
  return -9999.;
}
