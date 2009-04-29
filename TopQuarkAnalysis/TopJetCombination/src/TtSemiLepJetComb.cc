#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
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

void 
TtSemiLepJetComb::deduceMothers()
{
  hadW_   = hadQJet_.p4() + hadQBarJet_.p4();
  lepW_   = lepton_       + neutrino_  .p4();
  hadTop_ = hadW_         + hadBJet_   .p4();
  lepTop_ = lepW_         + lepBJet_   .p4();
}
