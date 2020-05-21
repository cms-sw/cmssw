#ifndef TtFullHadSignalSelEval_h
#define TtFullHadSignalSelEval_h

#include "Math/VectorUtil.h"
#include "TMath.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "TopQuarkAnalysis/TopEventSelection/interface/TtFullHadSignalSel.h"

inline double evaluateTtFullHadSignalSel(PhysicsTools::MVAComputerCache& mvaComputer,
                                         const TtFullHadSignalSel& sigsel,
                                         double weight = 1.0,
                                         const bool isSignal = false) {
  std::vector<PhysicsTools::Variable::Value> values;

  values.emplace_back("H", sigsel.H());
  values.emplace_back("Ht", sigsel.Ht());
  values.emplace_back("Ht123", sigsel.Ht123());
  values.emplace_back("Ht3jet", sigsel.Ht3jet());
  values.emplace_back("sqrt_s", sigsel.sqrt_s());
  values.emplace_back("Et56", sigsel.Et56());
  values.emplace_back("M3", sigsel.M3());

  values.emplace_back("TCHE_Bjets", sigsel.TCHE_Bjets());
  values.emplace_back("TCHP_Bjets", sigsel.TCHP_Bjets());
  values.emplace_back("SSVHE_Bjets", sigsel.SSVHE_Bjets());
  values.emplace_back("SSVHP_Bjets", sigsel.SSVHP_Bjets());
  values.emplace_back("CSV_Bjets", sigsel.CSV_Bjets());
  values.emplace_back("CSVMVA_Bjets", sigsel.CSVMVA_Bjets());
  values.emplace_back("SM_Bjets", sigsel.SM_Bjets());

  values.emplace_back("TCHE_Bjet1", sigsel.TCHE_Bjet(1));
  values.emplace_back("TCHE_Bjet2", sigsel.TCHE_Bjet(2));
  values.emplace_back("TCHE_Bjet3", sigsel.TCHE_Bjet(3));
  values.emplace_back("TCHE_Bjet4", sigsel.TCHE_Bjet(4));
  values.emplace_back("TCHE_Bjet5", sigsel.TCHE_Bjet(5));
  values.emplace_back("TCHE_Bjet6", sigsel.TCHE_Bjet(6));
  values.emplace_back("TCHP_Bjet1", sigsel.TCHP_Bjet(1));
  values.emplace_back("TCHP_Bjet2", sigsel.TCHP_Bjet(2));
  values.emplace_back("TCHP_Bjet3", sigsel.TCHP_Bjet(3));
  values.emplace_back("TCHP_Bjet4", sigsel.TCHP_Bjet(4));
  values.emplace_back("TCHP_Bjet5", sigsel.TCHP_Bjet(5));
  values.emplace_back("TCHP_Bjet6", sigsel.TCHP_Bjet(6));
  values.emplace_back("SSVHE_Bjet1", sigsel.SSVHE_Bjet(1));
  values.emplace_back("SSVHE_Bjet2", sigsel.SSVHE_Bjet(2));
  values.emplace_back("SSVHE_Bjet3", sigsel.SSVHE_Bjet(3));
  values.emplace_back("SSVHE_Bjet4", sigsel.SSVHE_Bjet(4));
  values.emplace_back("SSVHE_Bjet5", sigsel.SSVHE_Bjet(5));
  values.emplace_back("SSVHE_Bjet6", sigsel.SSVHE_Bjet(6));
  values.emplace_back("SSVHP_Bjet1", sigsel.SSVHP_Bjet(1));
  values.emplace_back("SSVHP_Bjet2", sigsel.SSVHP_Bjet(2));
  values.emplace_back("SSVHP_Bjet3", sigsel.SSVHP_Bjet(3));
  values.emplace_back("SSVHP_Bjet4", sigsel.SSVHP_Bjet(4));
  values.emplace_back("SSVHP_Bjet5", sigsel.SSVHP_Bjet(5));
  values.emplace_back("SSVHP_Bjet6", sigsel.SSVHP_Bjet(6));
  values.emplace_back("CSV_Bjet1", sigsel.CSV_Bjet(1));
  values.emplace_back("CSV_Bjet2", sigsel.CSV_Bjet(2));
  values.emplace_back("CSV_Bjet3", sigsel.CSV_Bjet(3));
  values.emplace_back("CSV_Bjet4", sigsel.CSV_Bjet(4));
  values.emplace_back("CSV_Bjet5", sigsel.CSV_Bjet(5));
  values.emplace_back("CSV_Bjet6", sigsel.CSV_Bjet(6));
  values.emplace_back("CSVMVA_Bjet1", sigsel.CSVMVA_Bjet(1));
  values.emplace_back("CSVMVA_Bjet2", sigsel.CSVMVA_Bjet(2));
  values.emplace_back("CSVMVA_Bjet3", sigsel.CSVMVA_Bjet(3));
  values.emplace_back("CSVMVA_Bjet4", sigsel.CSVMVA_Bjet(4));
  values.emplace_back("CSVMVA_Bjet5", sigsel.CSVMVA_Bjet(5));
  values.emplace_back("CSVMVA_Bjet6", sigsel.CSVMVA_Bjet(6));
  values.emplace_back("SM_Bjet1", sigsel.SM_Bjet(1));
  values.emplace_back("SM_Bjet2", sigsel.SM_Bjet(2));
  values.emplace_back("SM_Bjet3", sigsel.SM_Bjet(3));
  values.emplace_back("SM_Bjet4", sigsel.SM_Bjet(4));
  values.emplace_back("SM_Bjet5", sigsel.SM_Bjet(5));
  values.emplace_back("SM_Bjet6", sigsel.SM_Bjet(6));

  values.emplace_back("pt1", sigsel.pt(1));
  values.emplace_back("pt2", sigsel.pt(2));
  values.emplace_back("pt3", sigsel.pt(3));
  values.emplace_back("pt4", sigsel.pt(4));
  values.emplace_back("pt5", sigsel.pt(5));
  values.emplace_back("pt6", sigsel.pt(6));

  values.emplace_back("EtSin2Theta1", sigsel.EtSin2Theta(1));
  values.emplace_back("EtSin2Theta2", sigsel.EtSin2Theta(2));
  values.emplace_back("EtSin2Theta3", sigsel.EtSin2Theta(3));
  values.emplace_back("EtSin2Theta4", sigsel.EtSin2Theta(4));
  values.emplace_back("EtSin2Theta5", sigsel.EtSin2Theta(5));
  values.emplace_back("EtSin2Theta6", sigsel.EtSin2Theta(6));

  values.emplace_back("EtSin2Theta3jet", sigsel.EtSin2Theta3jet());

  values.emplace_back("theta1", sigsel.theta(1));
  values.emplace_back("theta2", sigsel.theta(2));
  values.emplace_back("theta3", sigsel.theta(3));
  values.emplace_back("theta4", sigsel.theta(4));
  values.emplace_back("theta5", sigsel.theta(5));
  values.emplace_back("theta6", sigsel.theta(6));

  values.emplace_back("theta3jet", sigsel.theta3jet());

  values.emplace_back("sinTheta1", sigsel.sinTheta(1));
  values.emplace_back("sinTheta2", sigsel.sinTheta(2));
  values.emplace_back("sinTheta3", sigsel.sinTheta(3));
  values.emplace_back("sinTheta4", sigsel.sinTheta(4));
  values.emplace_back("sinTheta5", sigsel.sinTheta(5));
  values.emplace_back("sinTheta6", sigsel.sinTheta(6));

  values.emplace_back("sinTheta3jet", sigsel.sinTheta3jet());

  values.emplace_back("thetaStar1", sigsel.theta(1, true));
  values.emplace_back("thetaStar2", sigsel.theta(2, true));
  values.emplace_back("thetaStar3", sigsel.theta(3, true));
  values.emplace_back("thetaStar4", sigsel.theta(4, true));
  values.emplace_back("thetaStar5", sigsel.theta(5, true));
  values.emplace_back("thetaStar6", sigsel.theta(6, true));

  values.emplace_back("thetaStar3jet", sigsel.theta3jet(true));

  values.emplace_back("sinThetaStar1", sigsel.sinTheta(1, true));
  values.emplace_back("sinThetaStar2", sigsel.sinTheta(2, true));
  values.emplace_back("sinThetaStar3", sigsel.sinTheta(3, true));
  values.emplace_back("sinThetaStar4", sigsel.sinTheta(4, true));
  values.emplace_back("sinThetaStar5", sigsel.sinTheta(5, true));
  values.emplace_back("sinThetaStar6", sigsel.sinTheta(6, true));

  values.emplace_back("sinThetaStar3jet", sigsel.sinTheta3jet(true));

  values.emplace_back("EtStar1", sigsel.EtSin2Theta(1, true));
  values.emplace_back("EtStar2", sigsel.EtSin2Theta(2, true));
  values.emplace_back("EtStar3", sigsel.EtSin2Theta(3, true));
  values.emplace_back("EtStar4", sigsel.EtSin2Theta(4, true));
  values.emplace_back("EtStar5", sigsel.EtSin2Theta(5, true));
  values.emplace_back("EtStar6", sigsel.EtSin2Theta(6, true));

  values.emplace_back("EtStar3jet", sigsel.EtSin2Theta3jet(true));

  values.emplace_back("pt1_pt2", sigsel.pti_ptj(1, 2));
  values.emplace_back("pt1_pt3", sigsel.pti_ptj(1, 3));
  values.emplace_back("pt1_pt4", sigsel.pti_ptj(1, 4));
  values.emplace_back("pt1_pt5", sigsel.pti_ptj(1, 5));
  values.emplace_back("pt1_pt6", sigsel.pti_ptj(1, 6));
  values.emplace_back("pt2_pt3", sigsel.pti_ptj(2, 3));
  values.emplace_back("pt2_pt4", sigsel.pti_ptj(2, 4));
  values.emplace_back("pt2_pt5", sigsel.pti_ptj(2, 5));
  values.emplace_back("pt2_pt6", sigsel.pti_ptj(2, 6));
  values.emplace_back("pt3_pt4", sigsel.pti_ptj(3, 4));
  values.emplace_back("pt3_pt5", sigsel.pti_ptj(3, 5));
  values.emplace_back("pt3_pt6", sigsel.pti_ptj(3, 6));
  values.emplace_back("pt4_pt5", sigsel.pti_ptj(4, 5));
  values.emplace_back("pt4_pt6", sigsel.pti_ptj(4, 6));
  values.emplace_back("pt5_pt6", sigsel.pti_ptj(5, 6));

  values.emplace_back("pt1_pt2_norm", sigsel.pti_ptj(1, 2, true));
  values.emplace_back("pt1_pt3_norm", sigsel.pti_ptj(1, 3, true));
  values.emplace_back("pt1_pt4_norm", sigsel.pti_ptj(1, 4, true));
  values.emplace_back("pt1_pt5_norm", sigsel.pti_ptj(1, 5, true));
  values.emplace_back("pt1_pt6_norm", sigsel.pti_ptj(1, 6, true));
  values.emplace_back("pt2_pt3_norm", sigsel.pti_ptj(2, 3, true));
  values.emplace_back("pt2_pt4_norm", sigsel.pti_ptj(2, 4, true));
  values.emplace_back("pt2_pt5_norm", sigsel.pti_ptj(2, 5, true));
  values.emplace_back("pt2_pt6_norm", sigsel.pti_ptj(2, 6, true));
  values.emplace_back("pt3_pt4_norm", sigsel.pti_ptj(3, 4, true));
  values.emplace_back("pt3_pt5_norm", sigsel.pti_ptj(3, 5, true));
  values.emplace_back("pt3_pt6_norm", sigsel.pti_ptj(3, 6, true));
  values.emplace_back("pt4_pt5_norm", sigsel.pti_ptj(4, 5, true));
  values.emplace_back("pt4_pt6_norm", sigsel.pti_ptj(4, 6, true));
  values.emplace_back("pt5_pt6_norm", sigsel.pti_ptj(5, 6, true));

  values.emplace_back("jet1_etaetaMoment", sigsel.jet_etaetaMoment(1));
  values.emplace_back("jet2_etaetaMoment", sigsel.jet_etaetaMoment(2));
  values.emplace_back("jet3_etaetaMoment", sigsel.jet_etaetaMoment(3));
  values.emplace_back("jet4_etaetaMoment", sigsel.jet_etaetaMoment(4));
  values.emplace_back("jet5_etaetaMoment", sigsel.jet_etaetaMoment(5));
  values.emplace_back("jet6_etaetaMoment", sigsel.jet_etaetaMoment(6));
  values.emplace_back("jet1_etaphiMoment", sigsel.jet_etaphiMoment(1));
  values.emplace_back("jet2_etaphiMoment", sigsel.jet_etaphiMoment(2));
  values.emplace_back("jet3_etaphiMoment", sigsel.jet_etaphiMoment(3));
  values.emplace_back("jet4_etaphiMoment", sigsel.jet_etaphiMoment(4));
  values.emplace_back("jet5_etaphiMoment", sigsel.jet_etaphiMoment(5));
  values.emplace_back("jet6_etaphiMoment", sigsel.jet_etaphiMoment(6));
  values.emplace_back("jet1_phiphiMoment", sigsel.jet_phiphiMoment(1));
  values.emplace_back("jet2_phiphiMoment", sigsel.jet_phiphiMoment(2));
  values.emplace_back("jet3_phiphiMoment", sigsel.jet_phiphiMoment(3));
  values.emplace_back("jet4_phiphiMoment", sigsel.jet_phiphiMoment(4));
  values.emplace_back("jet5_phiphiMoment", sigsel.jet_phiphiMoment(5));
  values.emplace_back("jet6_phiphiMoment", sigsel.jet_phiphiMoment(6));

  values.emplace_back("jet1_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(1));
  values.emplace_back("jet2_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(2));
  values.emplace_back("jet3_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(3));
  values.emplace_back("jet4_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(4));
  values.emplace_back("jet5_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(5));
  values.emplace_back("jet6_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(6));
  values.emplace_back("jet1_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(1));
  values.emplace_back("jet2_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(2));
  values.emplace_back("jet3_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(3));
  values.emplace_back("jet4_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(4));
  values.emplace_back("jet5_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(5));
  values.emplace_back("jet6_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(6));
  values.emplace_back("jet1_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(1));
  values.emplace_back("jet2_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(2));
  values.emplace_back("jet3_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(3));
  values.emplace_back("jet4_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(4));
  values.emplace_back("jet5_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(5));
  values.emplace_back("jet6_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(6));

  values.emplace_back("jet1_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(1));
  values.emplace_back("jet2_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(2));
  values.emplace_back("jet3_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(3));
  values.emplace_back("jet4_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(4));
  values.emplace_back("jet5_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(5));
  values.emplace_back("jet6_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(6));
  values.emplace_back("jet1_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(1));
  values.emplace_back("jet2_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(2));
  values.emplace_back("jet3_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(3));
  values.emplace_back("jet4_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(4));
  values.emplace_back("jet5_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(5));
  values.emplace_back("jet6_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(6));
  values.emplace_back("jet1_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(1));
  values.emplace_back("jet2_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(2));
  values.emplace_back("jet3_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(3));
  values.emplace_back("jet4_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(4));
  values.emplace_back("jet5_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(5));
  values.emplace_back("jet6_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(6));

  values.emplace_back("jet1_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(1));
  values.emplace_back("jet2_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(2));
  values.emplace_back("jet3_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(3));
  values.emplace_back("jet4_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(4));
  values.emplace_back("jet5_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(5));
  values.emplace_back("jet6_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(6));
  values.emplace_back("jet1_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(1));
  values.emplace_back("jet2_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(2));
  values.emplace_back("jet3_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(3));
  values.emplace_back("jet4_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(4));
  values.emplace_back("jet5_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(5));
  values.emplace_back("jet6_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(6));
  values.emplace_back("jet1_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(1));
  values.emplace_back("jet2_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(2));
  values.emplace_back("jet3_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(3));
  values.emplace_back("jet4_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(4));
  values.emplace_back("jet5_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(5));
  values.emplace_back("jet6_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(6));

  values.emplace_back("jet1_etaetaMomentNoB", sigsel.jet_etaetaMoment(1, true));
  values.emplace_back("jet2_etaetaMomentNoB", sigsel.jet_etaetaMoment(2, true));
  values.emplace_back("jet3_etaetaMomentNoB", sigsel.jet_etaetaMoment(3, true));
  values.emplace_back("jet4_etaetaMomentNoB", sigsel.jet_etaetaMoment(4, true));
  values.emplace_back("jet5_etaetaMomentNoB", sigsel.jet_etaetaMoment(5, true));
  values.emplace_back("jet6_etaetaMomentNoB", sigsel.jet_etaetaMoment(6, true));
  values.emplace_back("jet1_etaphiMomentNoB", sigsel.jet_etaphiMoment(1, true));
  values.emplace_back("jet2_etaphiMomentNoB", sigsel.jet_etaphiMoment(2, true));
  values.emplace_back("jet3_etaphiMomentNoB", sigsel.jet_etaphiMoment(3, true));
  values.emplace_back("jet4_etaphiMomentNoB", sigsel.jet_etaphiMoment(4, true));
  values.emplace_back("jet5_etaphiMomentNoB", sigsel.jet_etaphiMoment(5, true));
  values.emplace_back("jet6_etaphiMomentNoB", sigsel.jet_etaphiMoment(6, true));
  values.emplace_back("jet1_phiphiMomentNoB", sigsel.jet_phiphiMoment(1, true));
  values.emplace_back("jet2_phiphiMomentNoB", sigsel.jet_phiphiMoment(2, true));
  values.emplace_back("jet3_phiphiMomentNoB", sigsel.jet_phiphiMoment(3, true));
  values.emplace_back("jet4_phiphiMomentNoB", sigsel.jet_phiphiMoment(4, true));
  values.emplace_back("jet5_phiphiMomentNoB", sigsel.jet_phiphiMoment(5, true));
  values.emplace_back("jet6_phiphiMomentNoB", sigsel.jet_phiphiMoment(6, true));

  values.emplace_back("jets_etaetaMoment", sigsel.jets_etaetaMoment());
  values.emplace_back("jets_etaphiMoment", sigsel.jets_etaphiMoment());
  values.emplace_back("jets_phiphiMoment", sigsel.jets_phiphiMoment());

  values.emplace_back("jets_etaetaMomentLogEt", sigsel.jets_etaetaMomentLogEt());
  values.emplace_back("jets_etaphiMomentLogEt", sigsel.jets_etaphiMomentLogEt());
  values.emplace_back("jets_phiphiMomentLogEt", sigsel.jets_phiphiMomentLogEt());

  values.emplace_back("jets_etaetaMomentNoB", sigsel.jets_etaetaMoment(true));
  values.emplace_back("jets_etaphiMomentNoB", sigsel.jets_etaphiMoment(true));
  values.emplace_back("jets_phiphiMomentNoB", sigsel.jets_phiphiMoment(true));

  values.emplace_back("aplanarity", sigsel.aplanarity());
  values.emplace_back("sphericity", sigsel.sphericity());
  values.emplace_back("circularity", sigsel.circularity());
  values.emplace_back("isotropy", sigsel.isotropy());
  values.emplace_back("C", sigsel.C());
  values.emplace_back("D", sigsel.D());
  values.emplace_back("centrality", sigsel.centrality());

  values.emplace_back("thrust", sigsel.thrust());
  values.emplace_back("thrustCMS", sigsel.thrust(true));

  values.emplace_back("aplanarityAll", sigsel.aplanarity(true));
  values.emplace_back("sphericityAll", sigsel.sphericity(true));
  values.emplace_back("circularityAll", sigsel.circularity(true));
  values.emplace_back("isotropyAll", sigsel.isotropy(true));
  values.emplace_back("CAll", sigsel.C(true));
  values.emplace_back("DAll", sigsel.D(true));
  values.emplace_back("centralityAlt", sigsel.centrality(true));

  values.emplace_back("aplanarityAllCMS", sigsel.aplanarityAllCMS());
  values.emplace_back("sphericityAllCMS", sigsel.sphericityAllCMS());
  values.emplace_back("circularityAllCMS", sigsel.circularityAllCMS());
  values.emplace_back("isotropyAllCMS", sigsel.isotropyAllCMS());
  values.emplace_back("CAllCMS", sigsel.CAllCMS());
  values.emplace_back("DAllCMS", sigsel.DAllCMS());

  values.emplace_back("dRMin1", sigsel.dRMin(1));
  values.emplace_back("dRMin2", sigsel.dRMin(2));
  values.emplace_back("dRMin3", sigsel.dRMin(3));
  values.emplace_back("dRMin4", sigsel.dRMin(4));
  values.emplace_back("dRMin5", sigsel.dRMin(5));
  values.emplace_back("dRMin6", sigsel.dRMin(6));
  values.emplace_back("dRMin7", sigsel.dRMin(7));
  values.emplace_back("dRMin8", sigsel.dRMin(8));
  values.emplace_back("dRMin9", sigsel.dRMin(9));
  values.emplace_back("dRMin10", sigsel.dRMin(10));
  values.emplace_back("dRMin11", sigsel.dRMin(11));
  values.emplace_back("dRMin12", sigsel.dRMin(12));
  values.emplace_back("dRMin13", sigsel.dRMin(13));
  values.emplace_back("dRMin14", sigsel.dRMin(14));
  values.emplace_back("dRMin15", sigsel.dRMin(15));

  values.emplace_back("dRMin1Mass", sigsel.dRMinMass(1));
  values.emplace_back("dRMin2Mass", sigsel.dRMinMass(2));
  values.emplace_back("dRMin3Mass", sigsel.dRMinMass(3));
  values.emplace_back("dRMin4Mass", sigsel.dRMinMass(4));
  values.emplace_back("dRMin5Mass", sigsel.dRMinMass(5));
  values.emplace_back("dRMin6Mass", sigsel.dRMinMass(6));
  values.emplace_back("dRMin7Mass", sigsel.dRMinMass(7));
  values.emplace_back("dRMin8Mass", sigsel.dRMinMass(8));
  values.emplace_back("dRMin9Mass", sigsel.dRMinMass(9));
  values.emplace_back("dRMin10Mass", sigsel.dRMinMass(10));
  values.emplace_back("dRMin11Mass", sigsel.dRMinMass(11));
  values.emplace_back("dRMin12Mass", sigsel.dRMinMass(12));
  values.emplace_back("dRMin13Mass", sigsel.dRMinMass(13));
  values.emplace_back("dRMin14Mass", sigsel.dRMinMass(14));
  values.emplace_back("dRMin15Mass", sigsel.dRMinMass(15));

  values.emplace_back("dRMin1Angle", sigsel.dRMinAngle(1));
  values.emplace_back("dRMin2Angle", sigsel.dRMinAngle(2));
  values.emplace_back("dRMin3Angle", sigsel.dRMinAngle(3));
  values.emplace_back("dRMin4Angle", sigsel.dRMinAngle(4));
  values.emplace_back("dRMin5Angle", sigsel.dRMinAngle(5));
  values.emplace_back("dRMin6Angle", sigsel.dRMinAngle(6));
  values.emplace_back("dRMin7Angle", sigsel.dRMinAngle(7));
  values.emplace_back("dRMin8Angle", sigsel.dRMinAngle(8));
  values.emplace_back("dRMin9Angle", sigsel.dRMinAngle(9));
  values.emplace_back("dRMin10Angle", sigsel.dRMinAngle(10));
  values.emplace_back("dRMin11Angle", sigsel.dRMinAngle(11));
  values.emplace_back("dRMin12Angle", sigsel.dRMinAngle(12));
  values.emplace_back("dRMin13Angle", sigsel.dRMinAngle(13));
  values.emplace_back("dRMin14Angle", sigsel.dRMinAngle(14));
  values.emplace_back("dRMin15Angle", sigsel.dRMinAngle(15));

  values.emplace_back("sumDR3JetMin1", sigsel.sumDR3JetMin(1));
  values.emplace_back("sumDR3JetMin2", sigsel.sumDR3JetMin(2));
  values.emplace_back("sumDR3JetMin3", sigsel.sumDR3JetMin(3));
  values.emplace_back("sumDR3JetMin4", sigsel.sumDR3JetMin(4));
  values.emplace_back("sumDR3JetMin5", sigsel.sumDR3JetMin(5));
  values.emplace_back("sumDR3JetMin6", sigsel.sumDR3JetMin(6));
  values.emplace_back("sumDR3JetMin7", sigsel.sumDR3JetMin(7));
  values.emplace_back("sumDR3JetMin8", sigsel.sumDR3JetMin(8));
  values.emplace_back("sumDR3JetMin9", sigsel.sumDR3JetMin(9));
  values.emplace_back("sumDR3JetMin10", sigsel.sumDR3JetMin(10));
  values.emplace_back("sumDR3JetMin11", sigsel.sumDR3JetMin(11));
  values.emplace_back("sumDR3JetMin12", sigsel.sumDR3JetMin(12));
  values.emplace_back("sumDR3JetMin13", sigsel.sumDR3JetMin(13));
  values.emplace_back("sumDR3JetMin14", sigsel.sumDR3JetMin(14));
  values.emplace_back("sumDR3JetMin15", sigsel.sumDR3JetMin(15));
  values.emplace_back("sumDR3JetMin16", sigsel.sumDR3JetMin(16));
  values.emplace_back("sumDR3JetMin17", sigsel.sumDR3JetMin(17));
  values.emplace_back("sumDR3JetMin18", sigsel.sumDR3JetMin(18));
  values.emplace_back("sumDR3JetMin19", sigsel.sumDR3JetMin(19));
  values.emplace_back("sumDR3JetMin20", sigsel.sumDR3JetMin(20));

  values.emplace_back("sumDR3JetMin1Mass", sigsel.sumDR3JetMinMass(1));
  values.emplace_back("sumDR3JetMin2Mass", sigsel.sumDR3JetMinMass(2));
  values.emplace_back("sumDR3JetMin3Mass", sigsel.sumDR3JetMinMass(3));
  values.emplace_back("sumDR3JetMin4Mass", sigsel.sumDR3JetMinMass(4));
  values.emplace_back("sumDR3JetMin5Mass", sigsel.sumDR3JetMinMass(5));
  values.emplace_back("sumDR3JetMin6Mass", sigsel.sumDR3JetMinMass(6));
  values.emplace_back("sumDR3JetMin7Mass", sigsel.sumDR3JetMinMass(7));
  values.emplace_back("sumDR3JetMin8Mass", sigsel.sumDR3JetMinMass(8));
  values.emplace_back("sumDR3JetMin9Mass", sigsel.sumDR3JetMinMass(9));
  values.emplace_back("sumDR3JetMin10Mass", sigsel.sumDR3JetMinMass(10));
  values.emplace_back("sumDR3JetMin11Mass", sigsel.sumDR3JetMinMass(11));
  values.emplace_back("sumDR3JetMin12Mass", sigsel.sumDR3JetMinMass(12));
  values.emplace_back("sumDR3JetMin13Mass", sigsel.sumDR3JetMinMass(13));
  values.emplace_back("sumDR3JetMin14Mass", sigsel.sumDR3JetMinMass(14));
  values.emplace_back("sumDR3JetMin15Mass", sigsel.sumDR3JetMinMass(15));
  values.emplace_back("sumDR3JetMin16Mass", sigsel.sumDR3JetMinMass(16));
  values.emplace_back("sumDR3JetMin17Mass", sigsel.sumDR3JetMinMass(17));
  values.emplace_back("sumDR3JetMin18Mass", sigsel.sumDR3JetMinMass(18));
  values.emplace_back("sumDR3JetMin19Mass", sigsel.sumDR3JetMinMass(19));
  values.emplace_back("sumDR3JetMin20Mass", sigsel.sumDR3JetMinMass(20));

  values.emplace_back("massDiffMWCands1", sigsel.massDiffMWCands(1));
  values.emplace_back("massDiffMWCands2", sigsel.massDiffMWCands(2));
  values.emplace_back("massDiffMWCands3", sigsel.massDiffMWCands(3));
  values.emplace_back("massDiffMWCands4", sigsel.massDiffMWCands(4));
  values.emplace_back("massDiffMWCands5", sigsel.massDiffMWCands(5));
  values.emplace_back("massDiffMWCands6", sigsel.massDiffMWCands(6));
  values.emplace_back("massDiffMWCands7", sigsel.massDiffMWCands(7));
  values.emplace_back("massDiffMWCands8", sigsel.massDiffMWCands(8));
  values.emplace_back("massDiffMWCands9", sigsel.massDiffMWCands(9));
  values.emplace_back("massDiffMWCands10", sigsel.massDiffMWCands(10));
  values.emplace_back("massDiffMWCands11", sigsel.massDiffMWCands(11));
  values.emplace_back("massDiffMWCands12", sigsel.massDiffMWCands(12));
  values.emplace_back("massDiffMWCands13", sigsel.massDiffMWCands(13));
  values.emplace_back("massDiffMWCands14", sigsel.massDiffMWCands(14));
  values.emplace_back("massDiffMWCands15", sigsel.massDiffMWCands(15));
  values.emplace_back("massDiffMWCands16", sigsel.massDiffMWCands(16));
  values.emplace_back("massDiffMWCands17", sigsel.massDiffMWCands(17));
  values.emplace_back("massDiffMWCands18", sigsel.massDiffMWCands(18));
  values.emplace_back("massDiffMWCands19", sigsel.massDiffMWCands(19));
  values.emplace_back("massDiffMWCands20", sigsel.massDiffMWCands(20));

  return mvaComputer->eval(values);
}

#endif
