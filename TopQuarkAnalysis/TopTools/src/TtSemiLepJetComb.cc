#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepJetComb.h"
#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepEvtPartons.h"

TtSemiLepJetComb::TtSemiLepJetComb(){}

TtSemiLepJetComb::TtSemiLepJetComb(const std::vector<pat::Jet>& jets, const std::vector<int> cmb,
				   const math::XYZTLorentzVector& lep, const math::XYZTLorentzVector& neu)
{ 
  // receive right jet association
  // from jet parton matching
  hadQJet    = jets[cmb[TtSemiLepEvtPartons::LightQ   ]].p4();
  hadQBarJet = jets[cmb[TtSemiLepEvtPartons::LightQBar]].p4();
  hadBJet    = jets[cmb[TtSemiLepEvtPartons::HadB     ]].p4();
  lepBJet    = jets[cmb[TtSemiLepEvtPartons::LepB     ]].p4();
  lepton     = lep;
  neutrino   = neu;
  deduceMothers();
}

TtSemiLepJetComb::TtSemiLepJetComb(const std::vector<pat::Jet>& jets, const std::vector<int> cmb,
			     const math::XYZTLorentzVector& lep)
{ 
  // receive right jet association
  // from jet parton matching
  hadQJet    = jets[cmb[TtSemiLepEvtPartons::LightQ   ]].p4();
  hadQBarJet = jets[cmb[TtSemiLepEvtPartons::LightQBar]].p4();
  hadBJet    = jets[cmb[TtSemiLepEvtPartons::HadB     ]].p4();
  lepBJet    = jets[cmb[TtSemiLepEvtPartons::LepB     ]].p4();
  lepton     = lep;
  neutrino   = math::XYZTLorentzVector(0, 0, 0, 0);
  deduceMothers();
}

TtSemiLepJetComb::~TtSemiLepJetComb() 
{
}

void 
TtSemiLepJetComb::deduceMothers()
{
  hadW   = hadQJet + hadQBarJet;
  lepW   = lepton + neutrino;
  hadTop = hadW + hadBJet;
  lepTop = lepW + lepBJet;
}
