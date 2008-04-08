#include "TopQuarkAnalysis/TopTools/interface/TtSemiJetComb.h"
#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"

TtSemiJetComb::TtSemiJetComb(){}

TtSemiJetComb::TtSemiJetComb(const std::vector<TopJet> &topJets, const std::vector<int> cmb,
			     const math::XYZTLorentzVector &lep)
{ 
  // receive right jet assiciation
  // from jet parton matching
  hadQJet = topJets[cmb[TtSemiEvtPartons::LightQ]].p4();
  hadQBarJet = topJets[cmb[TtSemiEvtPartons::LightQBar]].p4();
  hadBJet = topJets[cmb[TtSemiEvtPartons::B]].p4();
  lepBJet = topJets[cmb[TtSemiEvtPartons::BBar]].p4();
  lepton = lep;
  deduceMothers();
}

TtSemiJetComb::~TtSemiJetComb() {}

void 
TtSemiJetComb::deduceMothers()
{
  hadW = hadQJet + hadQBarJet;
  lepW = lepton;  // +MET...
  hadTop = hadW + hadBJet;
  lepTop = lepW + lepBJet;
}
