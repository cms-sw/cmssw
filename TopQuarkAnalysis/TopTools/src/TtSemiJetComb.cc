#include "TopQuarkAnalysis/TopTools/interface/TtSemiJetComb.h"
#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"

TtSemiJetComb::TtSemiJetComb(){}

TtSemiJetComb::TtSemiJetComb(const std::vector<pat::Jet>& jets, const std::vector<int> cmb,
			     const math::XYZTLorentzVector& lep, const math::XYZTLorentzVector& neu)
{ 
  // receive right jet association
  // from jet parton matching
  hadQJet    = jets[cmb[TtSemiEvtPartons::LightQ   ]].p4();
  hadQBarJet = jets[cmb[TtSemiEvtPartons::LightQBar]].p4();
  hadBJet    = jets[cmb[TtSemiEvtPartons::HadB     ]].p4();
  lepBJet    = jets[cmb[TtSemiEvtPartons::LepB     ]].p4();
  lepton     = lep;
  neutrino   = neu;
  deduceMothers();
}

TtSemiJetComb::TtSemiJetComb(const std::vector<pat::Jet>& jets, const std::vector<int> cmb,
			     const math::XYZTLorentzVector& lep)
{ 
  // receive right jet association
  // from jet parton matching
  hadQJet    = jets[cmb[TtSemiEvtPartons::LightQ   ]].p4();
  hadQBarJet = jets[cmb[TtSemiEvtPartons::LightQBar]].p4();
  hadBJet    = jets[cmb[TtSemiEvtPartons::HadB     ]].p4();
  lepBJet    = jets[cmb[TtSemiEvtPartons::LepB     ]].p4();
  lepton     = lep;
  neutrino   = math::XYZTLorentzVector(0, 0, 0, 0);
  deduceMothers();
}

TtSemiJetComb::~TtSemiJetComb() 
{
}

void 
TtSemiJetComb::deduceMothers()
{
  hadW   = hadQJet + hadQBarJet;
  lepW   = lepton + neutrino;
  hadTop = hadW + hadBJet;
  lepTop = lepW + lepBJet;
}
