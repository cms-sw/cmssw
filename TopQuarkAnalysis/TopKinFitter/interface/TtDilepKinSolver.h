//based on a code by Jan Valenta
#ifndef TtDilepKinSolver_h
#define TtDilepKinSolver_h

#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"

#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TMath.h"

class TF2;

class TtDilepKinSolver {

 public:
  
  TtDilepKinSolver();
  TtDilepKinSolver(double,double,double, double xx = 0, double yy = 0);
  ~TtDilepKinSolver();
  
  inline void useWeightFromMC(bool useMC) { useMCforBest_ = useMC; }
  TtDilepEvtSolution addKinSolInfo(TtDilepEvtSolution * asol);
  
 private:
  void FindCoeff(const TLorentzVector al, 
		 const TLorentzVector l,
		 const TLorentzVector b_al,
		 const TLorentzVector b_l,
		 double mt, double mat, double pxboost, double pyboost,
		 double* q_coeff);
  void TopRec(const TLorentzVector al, 
	      const TLorentzVector l,
	      const TLorentzVector b_al,
	      const TLorentzVector b_l, double sol);
  double WeightSolfromMC();
  double WeightSolfromShape();
  int quartic(double* q_coeff, double* q_sol);
  int cubic(double* c_coeff, double* c_sol);
  
  //Utility Methods
  double sqr(double x) {return (x*x);}
  void SWAP(double& realone, double& realtwo);
    
 private:
  double topmass_begin;
  double topmass_end;
  double topmass_step;
  double pxmiss_, pymiss_;
  
  double mw, maw, mb, mab;
  
  double C;
  double D;
  double F;
  double pom;
  double k16;
  double k26;
  double k36;
  double k46;
  double k56;
  double k51;
  double k61;
  double m1;
  double m2;
  double m3;
  double n1;
  double n2;
  double n3;
  
  TLorentzVector LV_n, LV_n_, LV_t, LV_t_, LV_tt_t, LV_tt_t_;
  
  //provisional
  TLorentzVector genLV_n, genLV_n_;
  // flag to swith from WeightSolfromMC() to WeightSolfromShape()
  bool useMCforBest_;
  // Event shape
  TF2* EventShape_;
};


#endif
