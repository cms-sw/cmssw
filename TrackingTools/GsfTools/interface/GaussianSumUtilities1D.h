#ifndef GaussianSumUtilities1D_h_
#define GaussianSumUtilities1D_h_

// #include "TROOT.h"

#include "TrackingTools/GsfTools/interface/SingleGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"

#include <vector>

/** Utility class for the analysis of one-dimensional Gaussian
 *  mixtures. The input state is assumed to exist for 
 *  the lifetime of this object.
 */

class GaussianSumUtilities1D {
private:
  enum ModeStatus { Valid, NotValid, NotComputed };

public:
  GaussianSumUtilities1D (const MultiGaussianState1D& state) :
    theState(state),
//     theStates(state.components()),
    theModeStatus(NotComputed) {} 
  ~GaussianSumUtilities1D () {}

  /// number of components
  inline unsigned int size () const {return components().size();}
  /// components
  inline const std::vector<SingleGaussianState1D>& components () const {
    return theState.components();
  }
  /// weight of a component
  inline double weight (unsigned int i) const {return components()[i].weight();}
  /// mean value of a component
  inline double mean (unsigned int i) const {return components()[i].mean();}
  /// standard deviation of a component
  inline double standardDeviation (unsigned int i) const {
//     return sqrt(components()[i].variance());
    return components()[i].standardDeviation();
  }
  /// variance of a component
  inline double variance (unsigned int i) const {return components()[i].variance();}
  /// pdf of a single component at x
  double pdf(unsigned int i, double x)  const;
  /// Quantile (i.e. x for a given value of the c.d.f.)
  double quantile (const double) const;
  /// mode status
  bool modeIsValid () const;
  /** Mode "state": mean = mode, variance = local variance at mode,
   *  weight chosen to have pdf(mode) equal to the one of the mixture */
  const SingleGaussianState1D& mode () const;
  /// value of the p.d.f.
  double pdf (double) const;
  /// value of the c.d.f.
  double cdf (const double&) const;
  /// first derivative of the p.d.f.
  double d1Pdf (const double&) const;
  /// second derivative of the p.d.f.
  double d2Pdf (const double&) const;
  /// third derivative of the p.d.f.
  double d3Pdf (const double&) const;
  /// ln(pdf)
  double lnPdf (const double&) const;
  /// first derivative of ln(pdf)
  double d1LnPdf (const double&) const;
  /// second derivative of ln(pdf)
  double d2LnPdf (const double&) const;

  /// combined weight
  double weight () const {
    return theState.weight();
  }
  /// combined mean
  double mean () const {
    return theState.mean();
  }
  /// combined covariance
  double variance () const {
    return theState.variance();
  }

private:
  /** Finds mode. Input: start value and typical scale. 
   *  Output: mode and pdf(mode). Return value is true on success.
   */
  bool findMode (double& mode, double& pdfAtMode, 
		 const double& xStart, const double& scale) const;
  /// Value of gaussian distribution
  static double gauss (double, double, double);
  /// Integrated value of gaussian distribution
  static double gaussInt (double, double, double);
  /// Mean value of combined state
  double combinedMean() const;
  /// calculation of mode
  void computeMode () const;
  /** Local variance from Hessian matrix. 
   *  Only valid if x corresponds to a (local) maximum! */
  double localVariance (double x) const;

  // the state of the mode finder
  struct FinderState {
    FinderState(){}
    FinderState(size_t n): pdfs(n){}
    double x;
    double y;
    double yd; // d1LnPdf
    double yd2; // d2LnPdf
    std::vector<double> pdfs;
  };

  // update tre state at x
  void update(FinderState & state, double x) const;

  /// pdf components
  std::vector<double> pdfComponents (const double&) const;
  /// pdf components
  void pdfComponents (double, std::vector<double> & ) const;
  /// value of the p.d.f. using the pdf components at the evaluation point
  static double pdf (double, const std::vector<double>&);
  /// first derivative of the p.d.f. using the pdf components at the evaluation point
  double d1Pdf (double, const std::vector<double>&) const;
  /// second derivative of the p.d.f. using the pdf components at the evaluation point
  double d2Pdf (double, const std::vector<double>&) const;
  /// third derivative of the p.d.f. using the pdf components at the evaluation point
  double d3Pdf (double, const std::vector<double>&) const;
  /// ln(pdf) using the pdf components at the evaluation point
  static double lnPdf (double, const std::vector<double>&);
  /// first derivative of ln(pdf) using the pdf components at the evaluation point
  double d1LnPdf (double, const std::vector<double>&) const;
  /// second derivative of ln(pdf) using the pdf components at the evaluation point
  double d2LnPdf (double, const std::vector<double>&) const;

private:
  const MultiGaussianState1D& theState;
//   std::vector<SingleGaussianState1D> theStates;

  mutable ModeStatus theModeStatus;
  mutable SingleGaussianState1D theMode;
//   mutable double theMode;
};
#endif
