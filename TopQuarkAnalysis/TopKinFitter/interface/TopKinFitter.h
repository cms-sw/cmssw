#ifndef TopKinFitter_h
#define TopKinFitter_h

#include "TMath.h"

#include "PhysicsTools/KinFitter/interface/TKinFitter.h"

/*
  \class   TopKinFitter TopKinFitter.h "TopQuarkAnalysis/TopKinFitter/interface/TopKinFitter.h"
  
  \brief   one line description to be added here...

  text to be added here...
  
**/

class TopKinFitter {
  
 public:
  
  /// supported parameterizations
  enum Param{ kEMom, kEtEtaPhi, kEtThetaPhi };

 public:
  /// default constructor
  explicit TopKinFitter(const int maxNrIter=200, const double maxDeltaS=5e-5, const double maxF=1e-4,
			const double mW=80.4, const double mTop=173.);
  /// default destructor
  ~TopKinFitter();

  /// return chi2 of fit (not normalized to degrees of freedom)
  double fitS()  const { return fitter_->getS(); };
  /// return number of used iterations
  int fitNrIter() const { return fitter_->getNbIter(); };
  /// return fit probability
  double fitProb() const { return TMath::Prob(fitter_->getS(), fitter_->getNDF()); };
  
 protected:
  /// convert Param to human readable form
  std::string param(const Param& param) const;
  
 protected:
  // kinematic fitter
  TKinFitter* fitter_;
  /// maximal allowed number of iterations to be used for the fit
  int maxNrIter_;
  /// maximal allowed chi2 (not normalized to degrees of freedom)
  double maxDeltaS_;
  /// maximal allowed distance from constraints
  double maxF_;
  /// W mass value used for constraints
  double mW_;
  /// top mass value used for constraints
  double mTop_;
};

#endif
