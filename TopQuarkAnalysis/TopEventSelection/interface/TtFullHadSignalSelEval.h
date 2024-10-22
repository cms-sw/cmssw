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

  values.push_back(PhysicsTools::Variable::Value("H", sigsel.H()));
  values.push_back(PhysicsTools::Variable::Value("Ht", sigsel.Ht()));
  values.push_back(PhysicsTools::Variable::Value("Ht123", sigsel.Ht123()));
  values.push_back(PhysicsTools::Variable::Value("Ht3jet", sigsel.Ht3jet()));
  values.push_back(PhysicsTools::Variable::Value("sqrt_s", sigsel.sqrt_s()));
  values.push_back(PhysicsTools::Variable::Value("Et56", sigsel.Et56()));
  values.push_back(PhysicsTools::Variable::Value("M3", sigsel.M3()));

  values.push_back(PhysicsTools::Variable::Value("TCHE_Bjets", sigsel.TCHE_Bjets()));
  values.push_back(PhysicsTools::Variable::Value("TCHP_Bjets", sigsel.TCHP_Bjets()));
  values.push_back(PhysicsTools::Variable::Value("SSVHE_Bjets", sigsel.SSVHE_Bjets()));
  values.push_back(PhysicsTools::Variable::Value("SSVHP_Bjets", sigsel.SSVHP_Bjets()));
  values.push_back(PhysicsTools::Variable::Value("CSV_Bjets", sigsel.CSV_Bjets()));
  values.push_back(PhysicsTools::Variable::Value("CSVMVA_Bjets", sigsel.CSVMVA_Bjets()));
  values.push_back(PhysicsTools::Variable::Value("SM_Bjets", sigsel.SM_Bjets()));

  values.push_back(PhysicsTools::Variable::Value("TCHE_Bjet1", sigsel.TCHE_Bjet(1)));
  values.push_back(PhysicsTools::Variable::Value("TCHE_Bjet2", sigsel.TCHE_Bjet(2)));
  values.push_back(PhysicsTools::Variable::Value("TCHE_Bjet3", sigsel.TCHE_Bjet(3)));
  values.push_back(PhysicsTools::Variable::Value("TCHE_Bjet4", sigsel.TCHE_Bjet(4)));
  values.push_back(PhysicsTools::Variable::Value("TCHE_Bjet5", sigsel.TCHE_Bjet(5)));
  values.push_back(PhysicsTools::Variable::Value("TCHE_Bjet6", sigsel.TCHE_Bjet(6)));
  values.push_back(PhysicsTools::Variable::Value("TCHP_Bjet1", sigsel.TCHP_Bjet(1)));
  values.push_back(PhysicsTools::Variable::Value("TCHP_Bjet2", sigsel.TCHP_Bjet(2)));
  values.push_back(PhysicsTools::Variable::Value("TCHP_Bjet3", sigsel.TCHP_Bjet(3)));
  values.push_back(PhysicsTools::Variable::Value("TCHP_Bjet4", sigsel.TCHP_Bjet(4)));
  values.push_back(PhysicsTools::Variable::Value("TCHP_Bjet5", sigsel.TCHP_Bjet(5)));
  values.push_back(PhysicsTools::Variable::Value("TCHP_Bjet6", sigsel.TCHP_Bjet(6)));
  values.push_back(PhysicsTools::Variable::Value("SSVHE_Bjet1", sigsel.SSVHE_Bjet(1)));
  values.push_back(PhysicsTools::Variable::Value("SSVHE_Bjet2", sigsel.SSVHE_Bjet(2)));
  values.push_back(PhysicsTools::Variable::Value("SSVHE_Bjet3", sigsel.SSVHE_Bjet(3)));
  values.push_back(PhysicsTools::Variable::Value("SSVHE_Bjet4", sigsel.SSVHE_Bjet(4)));
  values.push_back(PhysicsTools::Variable::Value("SSVHE_Bjet5", sigsel.SSVHE_Bjet(5)));
  values.push_back(PhysicsTools::Variable::Value("SSVHE_Bjet6", sigsel.SSVHE_Bjet(6)));
  values.push_back(PhysicsTools::Variable::Value("SSVHP_Bjet1", sigsel.SSVHP_Bjet(1)));
  values.push_back(PhysicsTools::Variable::Value("SSVHP_Bjet2", sigsel.SSVHP_Bjet(2)));
  values.push_back(PhysicsTools::Variable::Value("SSVHP_Bjet3", sigsel.SSVHP_Bjet(3)));
  values.push_back(PhysicsTools::Variable::Value("SSVHP_Bjet4", sigsel.SSVHP_Bjet(4)));
  values.push_back(PhysicsTools::Variable::Value("SSVHP_Bjet5", sigsel.SSVHP_Bjet(5)));
  values.push_back(PhysicsTools::Variable::Value("SSVHP_Bjet6", sigsel.SSVHP_Bjet(6)));
  values.push_back(PhysicsTools::Variable::Value("CSV_Bjet1", sigsel.CSV_Bjet(1)));
  values.push_back(PhysicsTools::Variable::Value("CSV_Bjet2", sigsel.CSV_Bjet(2)));
  values.push_back(PhysicsTools::Variable::Value("CSV_Bjet3", sigsel.CSV_Bjet(3)));
  values.push_back(PhysicsTools::Variable::Value("CSV_Bjet4", sigsel.CSV_Bjet(4)));
  values.push_back(PhysicsTools::Variable::Value("CSV_Bjet5", sigsel.CSV_Bjet(5)));
  values.push_back(PhysicsTools::Variable::Value("CSV_Bjet6", sigsel.CSV_Bjet(6)));
  values.push_back(PhysicsTools::Variable::Value("CSVMVA_Bjet1", sigsel.CSVMVA_Bjet(1)));
  values.push_back(PhysicsTools::Variable::Value("CSVMVA_Bjet2", sigsel.CSVMVA_Bjet(2)));
  values.push_back(PhysicsTools::Variable::Value("CSVMVA_Bjet3", sigsel.CSVMVA_Bjet(3)));
  values.push_back(PhysicsTools::Variable::Value("CSVMVA_Bjet4", sigsel.CSVMVA_Bjet(4)));
  values.push_back(PhysicsTools::Variable::Value("CSVMVA_Bjet5", sigsel.CSVMVA_Bjet(5)));
  values.push_back(PhysicsTools::Variable::Value("CSVMVA_Bjet6", sigsel.CSVMVA_Bjet(6)));
  values.push_back(PhysicsTools::Variable::Value("SM_Bjet1", sigsel.SM_Bjet(1)));
  values.push_back(PhysicsTools::Variable::Value("SM_Bjet2", sigsel.SM_Bjet(2)));
  values.push_back(PhysicsTools::Variable::Value("SM_Bjet3", sigsel.SM_Bjet(3)));
  values.push_back(PhysicsTools::Variable::Value("SM_Bjet4", sigsel.SM_Bjet(4)));
  values.push_back(PhysicsTools::Variable::Value("SM_Bjet5", sigsel.SM_Bjet(5)));
  values.push_back(PhysicsTools::Variable::Value("SM_Bjet6", sigsel.SM_Bjet(6)));

  values.push_back(PhysicsTools::Variable::Value("pt1", sigsel.pt(1)));
  values.push_back(PhysicsTools::Variable::Value("pt2", sigsel.pt(2)));
  values.push_back(PhysicsTools::Variable::Value("pt3", sigsel.pt(3)));
  values.push_back(PhysicsTools::Variable::Value("pt4", sigsel.pt(4)));
  values.push_back(PhysicsTools::Variable::Value("pt5", sigsel.pt(5)));
  values.push_back(PhysicsTools::Variable::Value("pt6", sigsel.pt(6)));

  values.push_back(PhysicsTools::Variable::Value("EtSin2Theta1", sigsel.EtSin2Theta(1)));
  values.push_back(PhysicsTools::Variable::Value("EtSin2Theta2", sigsel.EtSin2Theta(2)));
  values.push_back(PhysicsTools::Variable::Value("EtSin2Theta3", sigsel.EtSin2Theta(3)));
  values.push_back(PhysicsTools::Variable::Value("EtSin2Theta4", sigsel.EtSin2Theta(4)));
  values.push_back(PhysicsTools::Variable::Value("EtSin2Theta5", sigsel.EtSin2Theta(5)));
  values.push_back(PhysicsTools::Variable::Value("EtSin2Theta6", sigsel.EtSin2Theta(6)));

  values.push_back(PhysicsTools::Variable::Value("EtSin2Theta3jet", sigsel.EtSin2Theta3jet()));

  values.push_back(PhysicsTools::Variable::Value("theta1", sigsel.theta(1)));
  values.push_back(PhysicsTools::Variable::Value("theta2", sigsel.theta(2)));
  values.push_back(PhysicsTools::Variable::Value("theta3", sigsel.theta(3)));
  values.push_back(PhysicsTools::Variable::Value("theta4", sigsel.theta(4)));
  values.push_back(PhysicsTools::Variable::Value("theta5", sigsel.theta(5)));
  values.push_back(PhysicsTools::Variable::Value("theta6", sigsel.theta(6)));

  values.push_back(PhysicsTools::Variable::Value("theta3jet", sigsel.theta3jet()));

  values.push_back(PhysicsTools::Variable::Value("sinTheta1", sigsel.sinTheta(1)));
  values.push_back(PhysicsTools::Variable::Value("sinTheta2", sigsel.sinTheta(2)));
  values.push_back(PhysicsTools::Variable::Value("sinTheta3", sigsel.sinTheta(3)));
  values.push_back(PhysicsTools::Variable::Value("sinTheta4", sigsel.sinTheta(4)));
  values.push_back(PhysicsTools::Variable::Value("sinTheta5", sigsel.sinTheta(5)));
  values.push_back(PhysicsTools::Variable::Value("sinTheta6", sigsel.sinTheta(6)));

  values.push_back(PhysicsTools::Variable::Value("sinTheta3jet", sigsel.sinTheta3jet()));

  values.push_back(PhysicsTools::Variable::Value("thetaStar1", sigsel.theta(1, true)));
  values.push_back(PhysicsTools::Variable::Value("thetaStar2", sigsel.theta(2, true)));
  values.push_back(PhysicsTools::Variable::Value("thetaStar3", sigsel.theta(3, true)));
  values.push_back(PhysicsTools::Variable::Value("thetaStar4", sigsel.theta(4, true)));
  values.push_back(PhysicsTools::Variable::Value("thetaStar5", sigsel.theta(5, true)));
  values.push_back(PhysicsTools::Variable::Value("thetaStar6", sigsel.theta(6, true)));

  values.push_back(PhysicsTools::Variable::Value("thetaStar3jet", sigsel.theta3jet(true)));

  values.push_back(PhysicsTools::Variable::Value("sinThetaStar1", sigsel.sinTheta(1, true)));
  values.push_back(PhysicsTools::Variable::Value("sinThetaStar2", sigsel.sinTheta(2, true)));
  values.push_back(PhysicsTools::Variable::Value("sinThetaStar3", sigsel.sinTheta(3, true)));
  values.push_back(PhysicsTools::Variable::Value("sinThetaStar4", sigsel.sinTheta(4, true)));
  values.push_back(PhysicsTools::Variable::Value("sinThetaStar5", sigsel.sinTheta(5, true)));
  values.push_back(PhysicsTools::Variable::Value("sinThetaStar6", sigsel.sinTheta(6, true)));

  values.push_back(PhysicsTools::Variable::Value("sinThetaStar3jet", sigsel.sinTheta3jet(true)));

  values.push_back(PhysicsTools::Variable::Value("EtStar1", sigsel.EtSin2Theta(1, true)));
  values.push_back(PhysicsTools::Variable::Value("EtStar2", sigsel.EtSin2Theta(2, true)));
  values.push_back(PhysicsTools::Variable::Value("EtStar3", sigsel.EtSin2Theta(3, true)));
  values.push_back(PhysicsTools::Variable::Value("EtStar4", sigsel.EtSin2Theta(4, true)));
  values.push_back(PhysicsTools::Variable::Value("EtStar5", sigsel.EtSin2Theta(5, true)));
  values.push_back(PhysicsTools::Variable::Value("EtStar6", sigsel.EtSin2Theta(6, true)));

  values.push_back(PhysicsTools::Variable::Value("EtStar3jet", sigsel.EtSin2Theta3jet(true)));

  values.push_back(PhysicsTools::Variable::Value("pt1_pt2", sigsel.pti_ptj(1, 2)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt3", sigsel.pti_ptj(1, 3)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt4", sigsel.pti_ptj(1, 4)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt5", sigsel.pti_ptj(1, 5)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt6", sigsel.pti_ptj(1, 6)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt3", sigsel.pti_ptj(2, 3)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt4", sigsel.pti_ptj(2, 4)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt5", sigsel.pti_ptj(2, 5)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt6", sigsel.pti_ptj(2, 6)));
  values.push_back(PhysicsTools::Variable::Value("pt3_pt4", sigsel.pti_ptj(3, 4)));
  values.push_back(PhysicsTools::Variable::Value("pt3_pt5", sigsel.pti_ptj(3, 5)));
  values.push_back(PhysicsTools::Variable::Value("pt3_pt6", sigsel.pti_ptj(3, 6)));
  values.push_back(PhysicsTools::Variable::Value("pt4_pt5", sigsel.pti_ptj(4, 5)));
  values.push_back(PhysicsTools::Variable::Value("pt4_pt6", sigsel.pti_ptj(4, 6)));
  values.push_back(PhysicsTools::Variable::Value("pt5_pt6", sigsel.pti_ptj(5, 6)));

  values.push_back(PhysicsTools::Variable::Value("pt1_pt2_norm", sigsel.pti_ptj(1, 2, true)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt3_norm", sigsel.pti_ptj(1, 3, true)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt4_norm", sigsel.pti_ptj(1, 4, true)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt5_norm", sigsel.pti_ptj(1, 5, true)));
  values.push_back(PhysicsTools::Variable::Value("pt1_pt6_norm", sigsel.pti_ptj(1, 6, true)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt3_norm", sigsel.pti_ptj(2, 3, true)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt4_norm", sigsel.pti_ptj(2, 4, true)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt5_norm", sigsel.pti_ptj(2, 5, true)));
  values.push_back(PhysicsTools::Variable::Value("pt2_pt6_norm", sigsel.pti_ptj(2, 6, true)));
  values.push_back(PhysicsTools::Variable::Value("pt3_pt4_norm", sigsel.pti_ptj(3, 4, true)));
  values.push_back(PhysicsTools::Variable::Value("pt3_pt5_norm", sigsel.pti_ptj(3, 5, true)));
  values.push_back(PhysicsTools::Variable::Value("pt3_pt6_norm", sigsel.pti_ptj(3, 6, true)));
  values.push_back(PhysicsTools::Variable::Value("pt4_pt5_norm", sigsel.pti_ptj(4, 5, true)));
  values.push_back(PhysicsTools::Variable::Value("pt4_pt6_norm", sigsel.pti_ptj(4, 6, true)));
  values.push_back(PhysicsTools::Variable::Value("pt5_pt6_norm", sigsel.pti_ptj(5, 6, true)));

  values.push_back(PhysicsTools::Variable::Value("jet1_etaetaMoment", sigsel.jet_etaetaMoment(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaetaMoment", sigsel.jet_etaetaMoment(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaetaMoment", sigsel.jet_etaetaMoment(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaetaMoment", sigsel.jet_etaetaMoment(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaetaMoment", sigsel.jet_etaetaMoment(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaetaMoment", sigsel.jet_etaetaMoment(6)));
  values.push_back(PhysicsTools::Variable::Value("jet1_etaphiMoment", sigsel.jet_etaphiMoment(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaphiMoment", sigsel.jet_etaphiMoment(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaphiMoment", sigsel.jet_etaphiMoment(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaphiMoment", sigsel.jet_etaphiMoment(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaphiMoment", sigsel.jet_etaphiMoment(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaphiMoment", sigsel.jet_etaphiMoment(6)));
  values.push_back(PhysicsTools::Variable::Value("jet1_phiphiMoment", sigsel.jet_phiphiMoment(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_phiphiMoment", sigsel.jet_phiphiMoment(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_phiphiMoment", sigsel.jet_phiphiMoment(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_phiphiMoment", sigsel.jet_phiphiMoment(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_phiphiMoment", sigsel.jet_phiphiMoment(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_phiphiMoment", sigsel.jet_phiphiMoment(6)));

  values.push_back(PhysicsTools::Variable::Value("jet1_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaetaMomentMoment", sigsel.jet_etaetaMomentMoment(6)));
  values.push_back(PhysicsTools::Variable::Value("jet1_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaphiMomentMoment", sigsel.jet_etaphiMomentMoment(6)));
  values.push_back(PhysicsTools::Variable::Value("jet1_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_phiphiMomentMoment", sigsel.jet_phiphiMomentMoment(6)));

  values.push_back(PhysicsTools::Variable::Value("jet1_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaetaMomentLogEt", sigsel.jet_etaetaMomentLogEt(6)));
  values.push_back(PhysicsTools::Variable::Value("jet1_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaphiMomentLogEt", sigsel.jet_etaphiMomentLogEt(6)));
  values.push_back(PhysicsTools::Variable::Value("jet1_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(1)));
  values.push_back(PhysicsTools::Variable::Value("jet2_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(2)));
  values.push_back(PhysicsTools::Variable::Value("jet3_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(3)));
  values.push_back(PhysicsTools::Variable::Value("jet4_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(4)));
  values.push_back(PhysicsTools::Variable::Value("jet5_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(5)));
  values.push_back(PhysicsTools::Variable::Value("jet6_phiphiMomentLogEt", sigsel.jet_phiphiMomentLogEt(6)));

  values.push_back(
      PhysicsTools::Variable::Value("jet1_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(1)));
  values.push_back(
      PhysicsTools::Variable::Value("jet2_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(2)));
  values.push_back(
      PhysicsTools::Variable::Value("jet3_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(3)));
  values.push_back(
      PhysicsTools::Variable::Value("jet4_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(4)));
  values.push_back(
      PhysicsTools::Variable::Value("jet5_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(5)));
  values.push_back(
      PhysicsTools::Variable::Value("jet6_etaetaMomentMomentLogEt", sigsel.jet_etaetaMomentMomentLogEt(6)));
  values.push_back(
      PhysicsTools::Variable::Value("jet1_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(1)));
  values.push_back(
      PhysicsTools::Variable::Value("jet2_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(2)));
  values.push_back(
      PhysicsTools::Variable::Value("jet3_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(3)));
  values.push_back(
      PhysicsTools::Variable::Value("jet4_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(4)));
  values.push_back(
      PhysicsTools::Variable::Value("jet5_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(5)));
  values.push_back(
      PhysicsTools::Variable::Value("jet6_etaphiMomentMomentLogEt", sigsel.jet_etaphiMomentMomentLogEt(6)));
  values.push_back(
      PhysicsTools::Variable::Value("jet1_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(1)));
  values.push_back(
      PhysicsTools::Variable::Value("jet2_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(2)));
  values.push_back(
      PhysicsTools::Variable::Value("jet3_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(3)));
  values.push_back(
      PhysicsTools::Variable::Value("jet4_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(4)));
  values.push_back(
      PhysicsTools::Variable::Value("jet5_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(5)));
  values.push_back(
      PhysicsTools::Variable::Value("jet6_phiphiMomentMomentLogEt", sigsel.jet_phiphiMomentMomentLogEt(6)));

  values.push_back(PhysicsTools::Variable::Value("jet1_etaetaMomentNoB", sigsel.jet_etaetaMoment(1, true)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaetaMomentNoB", sigsel.jet_etaetaMoment(2, true)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaetaMomentNoB", sigsel.jet_etaetaMoment(3, true)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaetaMomentNoB", sigsel.jet_etaetaMoment(4, true)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaetaMomentNoB", sigsel.jet_etaetaMoment(5, true)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaetaMomentNoB", sigsel.jet_etaetaMoment(6, true)));
  values.push_back(PhysicsTools::Variable::Value("jet1_etaphiMomentNoB", sigsel.jet_etaphiMoment(1, true)));
  values.push_back(PhysicsTools::Variable::Value("jet2_etaphiMomentNoB", sigsel.jet_etaphiMoment(2, true)));
  values.push_back(PhysicsTools::Variable::Value("jet3_etaphiMomentNoB", sigsel.jet_etaphiMoment(3, true)));
  values.push_back(PhysicsTools::Variable::Value("jet4_etaphiMomentNoB", sigsel.jet_etaphiMoment(4, true)));
  values.push_back(PhysicsTools::Variable::Value("jet5_etaphiMomentNoB", sigsel.jet_etaphiMoment(5, true)));
  values.push_back(PhysicsTools::Variable::Value("jet6_etaphiMomentNoB", sigsel.jet_etaphiMoment(6, true)));
  values.push_back(PhysicsTools::Variable::Value("jet1_phiphiMomentNoB", sigsel.jet_phiphiMoment(1, true)));
  values.push_back(PhysicsTools::Variable::Value("jet2_phiphiMomentNoB", sigsel.jet_phiphiMoment(2, true)));
  values.push_back(PhysicsTools::Variable::Value("jet3_phiphiMomentNoB", sigsel.jet_phiphiMoment(3, true)));
  values.push_back(PhysicsTools::Variable::Value("jet4_phiphiMomentNoB", sigsel.jet_phiphiMoment(4, true)));
  values.push_back(PhysicsTools::Variable::Value("jet5_phiphiMomentNoB", sigsel.jet_phiphiMoment(5, true)));
  values.push_back(PhysicsTools::Variable::Value("jet6_phiphiMomentNoB", sigsel.jet_phiphiMoment(6, true)));

  values.push_back(PhysicsTools::Variable::Value("jets_etaetaMoment", sigsel.jets_etaetaMoment()));
  values.push_back(PhysicsTools::Variable::Value("jets_etaphiMoment", sigsel.jets_etaphiMoment()));
  values.push_back(PhysicsTools::Variable::Value("jets_phiphiMoment", sigsel.jets_phiphiMoment()));

  values.push_back(PhysicsTools::Variable::Value("jets_etaetaMomentLogEt", sigsel.jets_etaetaMomentLogEt()));
  values.push_back(PhysicsTools::Variable::Value("jets_etaphiMomentLogEt", sigsel.jets_etaphiMomentLogEt()));
  values.push_back(PhysicsTools::Variable::Value("jets_phiphiMomentLogEt", sigsel.jets_phiphiMomentLogEt()));

  values.push_back(PhysicsTools::Variable::Value("jets_etaetaMomentNoB", sigsel.jets_etaetaMoment(true)));
  values.push_back(PhysicsTools::Variable::Value("jets_etaphiMomentNoB", sigsel.jets_etaphiMoment(true)));
  values.push_back(PhysicsTools::Variable::Value("jets_phiphiMomentNoB", sigsel.jets_phiphiMoment(true)));

  values.push_back(PhysicsTools::Variable::Value("aplanarity", sigsel.aplanarity()));
  values.push_back(PhysicsTools::Variable::Value("sphericity", sigsel.sphericity()));
  values.push_back(PhysicsTools::Variable::Value("circularity", sigsel.circularity()));
  values.push_back(PhysicsTools::Variable::Value("isotropy", sigsel.isotropy()));
  values.push_back(PhysicsTools::Variable::Value("C", sigsel.C()));
  values.push_back(PhysicsTools::Variable::Value("D", sigsel.D()));
  values.push_back(PhysicsTools::Variable::Value("centrality", sigsel.centrality()));

  values.push_back(PhysicsTools::Variable::Value("thrust", sigsel.thrust()));
  values.push_back(PhysicsTools::Variable::Value("thrustCMS", sigsel.thrust(true)));

  values.push_back(PhysicsTools::Variable::Value("aplanarityAll", sigsel.aplanarity(true)));
  values.push_back(PhysicsTools::Variable::Value("sphericityAll", sigsel.sphericity(true)));
  values.push_back(PhysicsTools::Variable::Value("circularityAll", sigsel.circularity(true)));
  values.push_back(PhysicsTools::Variable::Value("isotropyAll", sigsel.isotropy(true)));
  values.push_back(PhysicsTools::Variable::Value("CAll", sigsel.C(true)));
  values.push_back(PhysicsTools::Variable::Value("DAll", sigsel.D(true)));
  values.push_back(PhysicsTools::Variable::Value("centralityAlt", sigsel.centrality(true)));

  values.push_back(PhysicsTools::Variable::Value("aplanarityAllCMS", sigsel.aplanarityAllCMS()));
  values.push_back(PhysicsTools::Variable::Value("sphericityAllCMS", sigsel.sphericityAllCMS()));
  values.push_back(PhysicsTools::Variable::Value("circularityAllCMS", sigsel.circularityAllCMS()));
  values.push_back(PhysicsTools::Variable::Value("isotropyAllCMS", sigsel.isotropyAllCMS()));
  values.push_back(PhysicsTools::Variable::Value("CAllCMS", sigsel.CAllCMS()));
  values.push_back(PhysicsTools::Variable::Value("DAllCMS", sigsel.DAllCMS()));

  values.push_back(PhysicsTools::Variable::Value("dRMin1", sigsel.dRMin(1)));
  values.push_back(PhysicsTools::Variable::Value("dRMin2", sigsel.dRMin(2)));
  values.push_back(PhysicsTools::Variable::Value("dRMin3", sigsel.dRMin(3)));
  values.push_back(PhysicsTools::Variable::Value("dRMin4", sigsel.dRMin(4)));
  values.push_back(PhysicsTools::Variable::Value("dRMin5", sigsel.dRMin(5)));
  values.push_back(PhysicsTools::Variable::Value("dRMin6", sigsel.dRMin(6)));
  values.push_back(PhysicsTools::Variable::Value("dRMin7", sigsel.dRMin(7)));
  values.push_back(PhysicsTools::Variable::Value("dRMin8", sigsel.dRMin(8)));
  values.push_back(PhysicsTools::Variable::Value("dRMin9", sigsel.dRMin(9)));
  values.push_back(PhysicsTools::Variable::Value("dRMin10", sigsel.dRMin(10)));
  values.push_back(PhysicsTools::Variable::Value("dRMin11", sigsel.dRMin(11)));
  values.push_back(PhysicsTools::Variable::Value("dRMin12", sigsel.dRMin(12)));
  values.push_back(PhysicsTools::Variable::Value("dRMin13", sigsel.dRMin(13)));
  values.push_back(PhysicsTools::Variable::Value("dRMin14", sigsel.dRMin(14)));
  values.push_back(PhysicsTools::Variable::Value("dRMin15", sigsel.dRMin(15)));

  values.push_back(PhysicsTools::Variable::Value("dRMin1Mass", sigsel.dRMinMass(1)));
  values.push_back(PhysicsTools::Variable::Value("dRMin2Mass", sigsel.dRMinMass(2)));
  values.push_back(PhysicsTools::Variable::Value("dRMin3Mass", sigsel.dRMinMass(3)));
  values.push_back(PhysicsTools::Variable::Value("dRMin4Mass", sigsel.dRMinMass(4)));
  values.push_back(PhysicsTools::Variable::Value("dRMin5Mass", sigsel.dRMinMass(5)));
  values.push_back(PhysicsTools::Variable::Value("dRMin6Mass", sigsel.dRMinMass(6)));
  values.push_back(PhysicsTools::Variable::Value("dRMin7Mass", sigsel.dRMinMass(7)));
  values.push_back(PhysicsTools::Variable::Value("dRMin8Mass", sigsel.dRMinMass(8)));
  values.push_back(PhysicsTools::Variable::Value("dRMin9Mass", sigsel.dRMinMass(9)));
  values.push_back(PhysicsTools::Variable::Value("dRMin10Mass", sigsel.dRMinMass(10)));
  values.push_back(PhysicsTools::Variable::Value("dRMin11Mass", sigsel.dRMinMass(11)));
  values.push_back(PhysicsTools::Variable::Value("dRMin12Mass", sigsel.dRMinMass(12)));
  values.push_back(PhysicsTools::Variable::Value("dRMin13Mass", sigsel.dRMinMass(13)));
  values.push_back(PhysicsTools::Variable::Value("dRMin14Mass", sigsel.dRMinMass(14)));
  values.push_back(PhysicsTools::Variable::Value("dRMin15Mass", sigsel.dRMinMass(15)));

  values.push_back(PhysicsTools::Variable::Value("dRMin1Angle", sigsel.dRMinAngle(1)));
  values.push_back(PhysicsTools::Variable::Value("dRMin2Angle", sigsel.dRMinAngle(2)));
  values.push_back(PhysicsTools::Variable::Value("dRMin3Angle", sigsel.dRMinAngle(3)));
  values.push_back(PhysicsTools::Variable::Value("dRMin4Angle", sigsel.dRMinAngle(4)));
  values.push_back(PhysicsTools::Variable::Value("dRMin5Angle", sigsel.dRMinAngle(5)));
  values.push_back(PhysicsTools::Variable::Value("dRMin6Angle", sigsel.dRMinAngle(6)));
  values.push_back(PhysicsTools::Variable::Value("dRMin7Angle", sigsel.dRMinAngle(7)));
  values.push_back(PhysicsTools::Variable::Value("dRMin8Angle", sigsel.dRMinAngle(8)));
  values.push_back(PhysicsTools::Variable::Value("dRMin9Angle", sigsel.dRMinAngle(9)));
  values.push_back(PhysicsTools::Variable::Value("dRMin10Angle", sigsel.dRMinAngle(10)));
  values.push_back(PhysicsTools::Variable::Value("dRMin11Angle", sigsel.dRMinAngle(11)));
  values.push_back(PhysicsTools::Variable::Value("dRMin12Angle", sigsel.dRMinAngle(12)));
  values.push_back(PhysicsTools::Variable::Value("dRMin13Angle", sigsel.dRMinAngle(13)));
  values.push_back(PhysicsTools::Variable::Value("dRMin14Angle", sigsel.dRMinAngle(14)));
  values.push_back(PhysicsTools::Variable::Value("dRMin15Angle", sigsel.dRMinAngle(15)));

  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin1", sigsel.sumDR3JetMin(1)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin2", sigsel.sumDR3JetMin(2)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin3", sigsel.sumDR3JetMin(3)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin4", sigsel.sumDR3JetMin(4)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin5", sigsel.sumDR3JetMin(5)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin6", sigsel.sumDR3JetMin(6)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin7", sigsel.sumDR3JetMin(7)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin8", sigsel.sumDR3JetMin(8)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin9", sigsel.sumDR3JetMin(9)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin10", sigsel.sumDR3JetMin(10)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin11", sigsel.sumDR3JetMin(11)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin12", sigsel.sumDR3JetMin(12)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin13", sigsel.sumDR3JetMin(13)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin14", sigsel.sumDR3JetMin(14)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin15", sigsel.sumDR3JetMin(15)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin16", sigsel.sumDR3JetMin(16)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin17", sigsel.sumDR3JetMin(17)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin18", sigsel.sumDR3JetMin(18)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin19", sigsel.sumDR3JetMin(19)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin20", sigsel.sumDR3JetMin(20)));

  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin1Mass", sigsel.sumDR3JetMinMass(1)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin2Mass", sigsel.sumDR3JetMinMass(2)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin3Mass", sigsel.sumDR3JetMinMass(3)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin4Mass", sigsel.sumDR3JetMinMass(4)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin5Mass", sigsel.sumDR3JetMinMass(5)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin6Mass", sigsel.sumDR3JetMinMass(6)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin7Mass", sigsel.sumDR3JetMinMass(7)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin8Mass", sigsel.sumDR3JetMinMass(8)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin9Mass", sigsel.sumDR3JetMinMass(9)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin10Mass", sigsel.sumDR3JetMinMass(10)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin11Mass", sigsel.sumDR3JetMinMass(11)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin12Mass", sigsel.sumDR3JetMinMass(12)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin13Mass", sigsel.sumDR3JetMinMass(13)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin14Mass", sigsel.sumDR3JetMinMass(14)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin15Mass", sigsel.sumDR3JetMinMass(15)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin16Mass", sigsel.sumDR3JetMinMass(16)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin17Mass", sigsel.sumDR3JetMinMass(17)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin18Mass", sigsel.sumDR3JetMinMass(18)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin19Mass", sigsel.sumDR3JetMinMass(19)));
  values.push_back(PhysicsTools::Variable::Value("sumDR3JetMin20Mass", sigsel.sumDR3JetMinMass(20)));

  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands1", sigsel.massDiffMWCands(1)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands2", sigsel.massDiffMWCands(2)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands3", sigsel.massDiffMWCands(3)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands4", sigsel.massDiffMWCands(4)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands5", sigsel.massDiffMWCands(5)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands6", sigsel.massDiffMWCands(6)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands7", sigsel.massDiffMWCands(7)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands8", sigsel.massDiffMWCands(8)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands9", sigsel.massDiffMWCands(9)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands10", sigsel.massDiffMWCands(10)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands11", sigsel.massDiffMWCands(11)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands12", sigsel.massDiffMWCands(12)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands13", sigsel.massDiffMWCands(13)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands14", sigsel.massDiffMWCands(14)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands15", sigsel.massDiffMWCands(15)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands16", sigsel.massDiffMWCands(16)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands17", sigsel.massDiffMWCands(17)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands18", sigsel.massDiffMWCands(18)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands19", sigsel.massDiffMWCands(19)));
  values.push_back(PhysicsTools::Variable::Value("massDiffMWCands20", sigsel.massDiffMWCands(20)));

  return mvaComputer->eval(values);
}

#endif
