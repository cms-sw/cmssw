#ifndef TtFullHadSignalSel_h
#define TtFullHadSignalSel_h

#include <vector>
#include "TMath.h"
#include "Math/VectorUtil.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

class TtFullHadSignalSel {
  // common calculator class for likelihood
  // variables in fully hadronic ttbar decays
public:
  TtFullHadSignalSel();
  TtFullHadSignalSel(const std::vector<pat::Jet>&);
  ~TtFullHadSignalSel();

  double H() const { return H_; }
  double Ht() const { return Ht_; }
  double Ht123() const { return Ht123_; }
  double Ht3jet() const { return Ht3jet_; }
  double sqrt_s() const { return sqrt_s_; }
  double Et56() const { return Et56_; }
  double M3() const { return M3_; }

  double TCHE_Bjets() const { return TCHE_Bjets_; }
  double TCHP_Bjets() const { return TCHP_Bjets_; }
  double SSVHE_Bjets() const { return SSVHE_Bjets_; }
  double SSVHP_Bjets() const { return SSVHP_Bjets_; }
  double CSV_Bjets() const { return CSV_Bjets_; }
  double CSVMVA_Bjets() const { return CSVMVA_Bjets_; }
  double SM_Bjets() const { return SM_Bjets_; }

  double TCHE_Bjet(unsigned short i) const {
    return (TCHE_BJet_Discs_.size() >= i) ? TCHE_BJet_Discs_.at(TCHE_BJet_Discs_.size() - i) : -100.;
  }
  double TCHP_Bjet(unsigned short i) const {
    return (TCHP_BJet_Discs_.size() >= i) ? TCHP_BJet_Discs_.at(TCHP_BJet_Discs_.size() - i) : -100.;
  }
  double SSVHE_Bjet(unsigned short i) const {
    return (SSVHE_BJet_Discs_.size() >= i) ? SSVHE_BJet_Discs_.at(SSVHE_BJet_Discs_.size() - i) : -100.;
  }
  double SSVHP_Bjet(unsigned short i) const {
    return (SSVHP_BJet_Discs_.size() >= i) ? SSVHP_BJet_Discs_.at(SSVHP_BJet_Discs_.size() - i) : -100.;
  }
  double CSV_Bjet(unsigned short i) const {
    return (CSV_BJet_Discs_.size() >= i) ? CSV_BJet_Discs_.at(CSV_BJet_Discs_.size() - i) : -100.;
  }
  double CSVMVA_Bjet(unsigned short i) const {
    return (CSVMVA_BJet_Discs_.size() >= i) ? CSVMVA_BJet_Discs_.at(CSVMVA_BJet_Discs_.size() - i) : -100.;
  }
  double SM_Bjet(unsigned short i) const {
    return (SM_BJet_Discs_.size() >= i) ? SM_BJet_Discs_.at(SM_BJet_Discs_.size() - i) : -100.;
  }

  double pt(unsigned short i) const { return (pts_.size() >= i) ? pts_.at(i - 1) : -1.; }

  double EtSin2Theta(unsigned short i, bool boosted = false) const {
    return boosted                       ? ((EtStars_.size() >= i) ? EtStars_.at(i - 1) : -1.)
           : (EtSin2Thetas_.size() >= i) ? EtSin2Thetas_.at(i - 1)
                                         : -1.;
  }
  double theta(unsigned short i, bool boosted = false) const {
    return boosted                 ? ((thetaStars_.size() >= i) ? thetaStars_.at(i - 1) : -1.)
           : (thetas_.size() >= i) ? thetas_.at(i - 1)
                                   : -1.;
  }
  double sinTheta(unsigned short i, bool boosted = false) const {
    return boosted                 ? ((thetaStars_.size() >= i) ? sin(thetaStars_.at(i - 1)) : -1.)
           : (thetas_.size() >= i) ? sin(thetas_.at(i - 1))
                                   : -1.;
  }

  double EtSin2Theta3jet(bool boosted = false) const { return boosted ? EtStar3jet_ : EtSin2Theta3jet_; }
  double theta3jet(bool boosted = false) const { return boosted ? thetaStar3jet_ : theta3jet_; }
  double sinTheta3jet(bool boosted = false) const { return boosted ? sinThetaStar3jet_ : sinTheta3jet_; }

  double pti_ptj(unsigned short i, unsigned short j, bool norm = false) const {
    return (pts_.size() >= j) ? (norm ? (pt(i) - pt(j)) / (pt(i) + pt(j)) : (pt(i) - pt(j))) : -1.;
  }

  double jet_etaetaMoment(unsigned short i, bool noB = false) const {
    return noB ? ((etaetaMomentsNoB_.size() >= i) ? etaetaMomentsNoB_.at(etaetaMomentsNoB_.size() - i) : -100.)
           : (etaetaMoments_.size() >= i) ? etaetaMoments_.at(etaetaMoments_.size() - i)
                                          : -100.;
  }
  double jet_etaphiMoment(unsigned short i, bool noB = false) const {
    return noB ? ((etaphiMomentsNoB_.size() >= i) ? etaphiMomentsNoB_.at(etaphiMomentsNoB_.size() - i) : -100.)
           : (etaphiMoments_.size() >= i) ? etaphiMoments_.at(etaphiMoments_.size() - i)
                                          : -100.;
  }
  double jet_phiphiMoment(unsigned short i, bool noB = false) const {
    return noB ? ((phiphiMomentsNoB_.size() >= i) ? phiphiMomentsNoB_.at(phiphiMomentsNoB_.size() - i) : -100.)
           : (phiphiMoments_.size() >= i) ? phiphiMoments_.at(phiphiMoments_.size() - i)
                                          : -100.;
  }

  double jet_etaetaMomentMoment(unsigned short i) const {
    return (etaetaMomentsMoment_.size() >= i) ? etaetaMomentsMoment_.at(etaetaMomentsMoment_.size() - i) : -100.;
  }
  double jet_etaphiMomentMoment(unsigned short i) const {
    return (etaphiMomentsMoment_.size() >= i) ? etaphiMomentsMoment_.at(etaphiMomentsMoment_.size() - i) : -100.;
  }
  double jet_phiphiMomentMoment(unsigned short i) const {
    return (phiphiMomentsMoment_.size() >= i) ? phiphiMomentsMoment_.at(phiphiMomentsMoment_.size() - i) : -100.;
  }

  double jets_etaetaMoment(bool noB = false) const { return noB ? jets_etaetaMomentNoB_ : jets_etaetaMoment_; }
  double jets_etaphiMoment(bool noB = false) const { return noB ? jets_etaphiMomentNoB_ : jets_etaphiMoment_; }
  double jets_phiphiMoment(bool noB = false) const { return noB ? jets_phiphiMomentNoB_ : jets_phiphiMoment_; }

  double jet_etaetaMomentLogEt(unsigned short i) const {
    return (etaetaMomentsLogEt_.size() >= i) ? etaetaMomentsLogEt_.at(etaetaMomentsLogEt_.size() - i) : -100.;
  }
  double jet_etaphiMomentLogEt(unsigned short i) const {
    return (etaphiMomentsLogEt_.size() >= i) ? etaphiMomentsLogEt_.at(etaphiMomentsLogEt_.size() - i) : -100.;
  }
  double jet_phiphiMomentLogEt(unsigned short i) const {
    return (phiphiMomentsLogEt_.size() >= i) ? phiphiMomentsLogEt_.at(phiphiMomentsLogEt_.size() - i) : -100.;
  }

  double jet_etaetaMomentMomentLogEt(unsigned short i) const {
    return (etaetaMomentsMomentLogEt_.size() >= i) ? etaetaMomentsMomentLogEt_.at(etaetaMomentsMomentLogEt_.size() - i)
                                                   : -100.;
  }
  double jet_etaphiMomentMomentLogEt(unsigned short i) const {
    return (etaphiMomentsMomentLogEt_.size() >= i) ? etaphiMomentsMomentLogEt_.at(etaphiMomentsMomentLogEt_.size() - i)
                                                   : -100.;
  }
  double jet_phiphiMomentMomentLogEt(unsigned short i) const {
    return (phiphiMomentsMomentLogEt_.size() >= i) ? phiphiMomentsMomentLogEt_.at(phiphiMomentsMomentLogEt_.size() - i)
                                                   : -100.;
  }

  double jets_etaetaMomentLogEt() const { return jets_etaetaMomentLogEt_; }
  double jets_etaphiMomentLogEt() const { return jets_etaphiMomentLogEt_; }
  double jets_phiphiMomentLogEt() const { return jets_phiphiMomentLogEt_; }

  double aplanarity(bool allJets = false) const { return allJets ? aplanarityAll_ : aplanarity_; }
  double sphericity(bool allJets = false) const { return allJets ? sphericityAll_ : sphericity_; }
  double circularity(bool allJets = false) const { return allJets ? circularityAll_ : circularity_; }
  double isotropy(bool allJets = false) const { return allJets ? isotropyAll_ : isotropy_; }
  double C(bool allJets = false) const { return allJets ? CAll_ : C_; }
  double D(bool allJets = false) const { return allJets ? DAll_ : D_; }

  double aplanarityAllCMS() const { return aplanarityAllCMS_; }
  double sphericityAllCMS() const { return sphericityAllCMS_; }
  double circularityAllCMS() const { return circularityAllCMS_; }
  double isotropyAllCMS() const { return isotropyAllCMS_; }
  double CAllCMS() const { return CAllCMS_; }
  double DAllCMS() const { return DAllCMS_; }

  double centrality(bool alternative = false) const { return alternative ? (Ht_ / sqrt_s_) : (Ht_ / H_); }

  double thrust(bool inCMS = false) const { return inCMS ? thrustCMS_ : thrust_; }

  double dRMin(unsigned short i) const { return (dR_.size() >= i) ? dR_.at(i - 1) : -1.; }
  double dRMinMass(unsigned short i) const { return (dRMass_.size() >= i) ? dRMass_.at(i - 1) : -1.; }
  double dRMinAngle(unsigned short i) const { return (dRAngle_.size() >= i) ? dRAngle_.at(i - 1) : -1.; }

  double sumDR3JetMin(unsigned short i) const { return (dR3Jets_.size() >= i) ? dR3Jets_.at(i - 1) : -1.; }
  double sumDR3JetMinMass(unsigned short i) const { return (dR3JetsMass_.size() >= i) ? dR3JetsMass_.at(i - 1) : -1.; }
  double massDiffMWCands(unsigned short i) const {
    return (massDiffMWCands_.size() >= i) ? massDiffMWCands_.at(i - 1) : -1.;
  }

private:
  double H_;
  double Ht_;
  double Ht123_;
  double Ht3jet_;
  double sqrt_s_;
  double Et56_;
  double M3_;

  double TCHE_Bjets_;
  double TCHP_Bjets_;
  double SSVHE_Bjets_;
  double SSVHP_Bjets_;
  double CSV_Bjets_;
  double CSVMVA_Bjets_;
  double SM_Bjets_;

  std::vector<double> TCHE_BJet_Discs_;
  std::vector<double> TCHP_BJet_Discs_;
  std::vector<double> SSVHE_BJet_Discs_;
  std::vector<double> SSVHP_BJet_Discs_;
  std::vector<double> CSV_BJet_Discs_;
  std::vector<double> CSVMVA_BJet_Discs_;
  std::vector<double> SM_BJet_Discs_;

  std::vector<double> pts_;
  std::vector<double> EtSin2Thetas_;
  std::vector<double> thetas_;
  std::vector<double> thetaStars_;
  std::vector<double> EtStars_;

  double EtSin2Theta3jet_;
  double theta3jet_;
  double thetaStar3jet_;
  double sinTheta3jet_;
  double sinThetaStar3jet_;
  double EtStar3jet_;

  std::vector<double> etaetaMoments_;
  std::vector<double> etaphiMoments_;
  std::vector<double> phiphiMoments_;

  std::vector<double> etaetaMomentsMoment_;
  std::vector<double> etaphiMomentsMoment_;
  std::vector<double> phiphiMomentsMoment_;

  std::vector<double> etaetaMomentsLogEt_;
  std::vector<double> etaphiMomentsLogEt_;
  std::vector<double> phiphiMomentsLogEt_;

  std::vector<double> etaetaMomentsMomentLogEt_;
  std::vector<double> etaphiMomentsMomentLogEt_;
  std::vector<double> phiphiMomentsMomentLogEt_;

  std::vector<double> etaetaMomentsNoB_;
  std::vector<double> etaphiMomentsNoB_;
  std::vector<double> phiphiMomentsNoB_;

  double jets_etaetaMoment_;
  double jets_etaphiMoment_;
  double jets_phiphiMoment_;

  double jets_etaetaMomentLogEt_;
  double jets_etaphiMomentLogEt_;
  double jets_phiphiMomentLogEt_;

  double jets_etaetaMomentNoB_;
  double jets_etaphiMomentNoB_;
  double jets_phiphiMomentNoB_;

  double aplanarity_;
  double sphericity_;
  double circularity_;
  double isotropy_;
  double C_;
  double D_;

  double aplanarityAll_;
  double sphericityAll_;
  double circularityAll_;
  double isotropyAll_;
  double CAll_;
  double DAll_;

  double aplanarityAllCMS_;
  double sphericityAllCMS_;
  double circularityAllCMS_;
  double isotropyAllCMS_;
  double CAllCMS_;
  double DAllCMS_;

  double thrust_;
  double thrustCMS_;

  std::vector<double> dR_;
  std::vector<double> dRMass_;
  std::vector<double> dRAngle_;

  std::vector<double> dR3Jets_;
  std::vector<double> dR3JetsMass_;

  std::vector<double> massDiffMWCands_;
};

#endif
