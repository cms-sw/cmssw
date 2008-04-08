#ifndef TtSemiJetComb_h
#define TtSemiJetComb_h

#include <vector>
#include "TMath.h"
#include "Math/VectorUtil.h"

#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"

class TtSemiJetComb {
  // common calculator class for likelihood
  // variables in semi leptonic ttbar decays
public:

  TtSemiJetComb();
  TtSemiJetComb(const std::vector<TopJet>&, const std::vector<int>, const math::XYZTLorentzVector&);
  ~TtSemiJetComb();

  double angleHadQQBar() const { return ROOT::Math::VectorUtil::Angle(hadQJet, hadQBarJet) * TMath::RadToDeg(); }
  double angleHadWHadB() const { return ROOT::Math::VectorUtil::Angle(hadW, hadBJet) * TMath::RadToDeg(); }
  double angleLeptonLepB() const { return ROOT::Math::VectorUtil::Angle(lepton, lepBJet) * TMath::RadToDeg(); }
  double angleTopTop() const { return ROOT::Math::VectorUtil::Angle(hadTop, lepTop) * TMath::RadToDeg(); }
  double massHadW() const { return hadW.M(); }
  double massHadTop() const { return hadTop.M(); }
  double massLepW() const { return lepW.M(); }
  double massLepTop() const { return lepTop.M(); }
  double deltaMTopTop() const { return hadTop.M() - lepTop.M(); }
  
private:

  math::XYZTLorentzVector hadQJet;
  math::XYZTLorentzVector hadQBarJet;
  math::XYZTLorentzVector hadBJet;
  math::XYZTLorentzVector lepBJet;
  math::XYZTLorentzVector lepton;
  math::XYZTLorentzVector hadW;
  math::XYZTLorentzVector lepW;
  math::XYZTLorentzVector hadTop;
  math::XYZTLorentzVector lepTop;

  void deduceMothers();
};

#endif
