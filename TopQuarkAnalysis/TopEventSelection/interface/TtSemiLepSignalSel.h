#ifndef TtSemiLepSignalSel_h
#define TtSemiLepSignalSel_h

#include <vector>
#include "TMath.h"
#include "Math/VectorUtil.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

class TtSemiLepSignalSel {
  // common calculator class for likelihood
  // variables in semi leptonic ttbar decays
public:

  TtSemiLepSignalSel();
  TtSemiLepSignalSel(const std::vector<pat::Jet>&, const math::XYZTLorentzVector&, const edm::View<pat::MET>&);
  ~TtSemiLepSignalSel();
  
  double sumEt() const { return var_sumEt; }
  double Et1() const { return var_Et1/var_sumEt; }
  double lepeta() const { return fabs(var_lepeta); }
  double MET() const { return var_MET; }
    
  double dphiMETlepton() const { return var_dphiMETlepton; }
  
  double detajet2jet3() const { return var_detajet2jet3; }
  double detajet3jet4() const { return var_detajet3jet4; }

  double mindijetmass() const { return var_mindijetmass/massalljets; }
  double maxdijetmass() const { return var_maxdijetmass/massalljets; }

  double mindRjetlepton() const { return var_mindRjetlepton; }
 
   
  double DeltaPhi(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&);
  double DeltaR(const math::XYZTLorentzVector&, const math::XYZTLorentzVector&);

private:
  
  double var_sumEt;
  double var_Et1;
  double var_lepeta;
  double var_MET;
  
  double var_dphiMETlepton; 
  
  double var_detajet2jet3;
  double var_detajet3jet4;

  double var_mindijetmass;
  double var_maxdijetmass;

  double var_mindRjetlepton;
  

  double massalljets;
};

#endif
