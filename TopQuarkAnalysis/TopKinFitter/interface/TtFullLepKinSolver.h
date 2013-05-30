#ifndef TtFullLepKinSolver_h
#define TtFullLepKinSolver_h

#include "AnalysisDataFormats/TopObjects/interface/TtDilepEvtSolution.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "TLorentzVector.h"
#include "TMath.h"

class TF2;

/*
  \class   TtFullLepKinSolver TtFullLepKinSolver.h "TopQuarkAnalysis/TopKinFitter/interface/TtFullLepKinSolver.h"
  
  \brief   Class to calculate solutions for neutrino momenta in dileptonic ttbar events and related probability weights

  Class to calculate solutions for neutrino momenta in dileptonic ttbar events and related probability weights.
  A fourth-order polynomial in p_x(nu) is used with coefficients that are functions of the top-quark mass.
  If physical (non-imaginary) solutions are found, the neutrino momenta are compared to the expected neutrino spectrum
  (from simulation) to obtain a probability weight for each solution.
  This class is based on a code by Jan Valenta.
  
**/

class TtFullLepKinSolver {

 public:

  ///
  struct NeutrinoSolution {
    double weight;
    reco::LeafCandidate neutrino;
    reco::LeafCandidate neutrinoBar; 
  };

  /// default constructor
  TtFullLepKinSolver();
  /// constructor with parameters to configure the top-mass scan and the neutrino spectrum
  TtFullLepKinSolver(const double, const double, const double, const std::vector<double>&, const double=80.4, const double=4.8);
  /// destructor
  ~TtFullLepKinSolver();

  ///
  inline void useWeightFromMC(bool useMC) { useMCforBest_ = useMC; }
  ///
  TtDilepEvtSolution addKinSolInfo(TtDilepEvtSolution * asol); 
  ///
  void SetConstraints(const double xx=0, const double yy=0);
  ///
  NeutrinoSolution getNuSolution(const TLorentzVector& LV_l, 
                                 const TLorentzVector& LV_l_, 
			         const TLorentzVector& LV_b, 
			         const TLorentzVector& LV_b_); 			  
			     
 private:

  ///
  void FindCoeff(const TLorentzVector& al, 
		 const TLorentzVector& l,
		 const TLorentzVector& b_al,
		 const TLorentzVector& b_l,
		 const double mt, const double mat, const double pxboost, const double pyboost,
		 double* q_coeff);
  ///
  void TopRec(const TLorentzVector& al, 
	      const TLorentzVector& l,
	      const TLorentzVector& b_al,
	      const TLorentzVector& b_l, const double sol);
  ///
  double WeightSolfromMC() const;
  /// use the parametrized event shape to obtain the solution weight.
  double WeightSolfromShape() const;
  ///
  int quartic(double* q_coeff, double* q_sol) const;
  ///
  int cubic(const double* c_coeff, double* c_sol) const;
  ///
  double sqr(const double x) const {return (x*x);}
  ///
  void SWAP(double& realone, double& realtwo) const;
    
 private:
  ///
  const double topmass_begin;
  ///
  const double topmass_end;
  ///
  const double topmass_step;
  ///
  const double mw;
  ///
  const double mb;
  ///
  double pxmiss_, pymiss_;
  
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
  
  ///
  TLorentzVector LV_n, LV_n_, LV_t, LV_t_, LV_tt_t, LV_tt_t_;  
  /// provisional
  TLorentzVector genLV_n, genLV_n_;  
    
  /// flag to swith from WeightSolfromMC() to WeightSolfromShape()
  bool useMCforBest_;
  /// Event shape
  TF2* EventShape_;  
};


#endif
