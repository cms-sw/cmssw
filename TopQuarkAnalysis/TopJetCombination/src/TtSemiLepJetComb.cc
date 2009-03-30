#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/TtSemiLepJetComb.h"


TtSemiLepJetComb::TtSemiLepJetComb()
{
}

TtSemiLepJetComb::TtSemiLepJetComb(const std::vector<pat::Jet>& jets, const std::vector<int> cmb, const math::XYZTLorentzVector& lep, const pat::MET& neu)
{ 
  // receive right jet association
  // from jet parton matching 
  hadQJet_    = jets[cmb[TtSemiLepEvtPartons::LightQ   ]];
  hadQBarJet_ = jets[cmb[TtSemiLepEvtPartons::LightQBar]];
  hadBJet_    = jets[cmb[TtSemiLepEvtPartons::HadB     ]];
  lepBJet_    = jets[cmb[TtSemiLepEvtPartons::LepB     ]]; 
  lepton_     = lep;
  neutrino_   = neu;
  // create mother candidates from 
  // final state candidates
  deduceMothers();
}


TtSemiLepJetComb::~TtSemiLepJetComb() 
{
}

void 
TtSemiLepJetComb::deduceMothers()
{
  hadW_   = hadQJet_.p4() + hadQBarJet_.p4();
  lepW_   = lepton_ + neutrino_.p4();
  hadTop_ = hadW_ + hadBJet_.p4();
  lepTop_ = lepW_ + lepBJet_.p4();
}
