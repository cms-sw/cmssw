#ifndef GsfBetheHeitlerUpdator_h_
#define GsfBetheHeitlerUpdator_h_

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"
#include "TrackingTools/GsfTracking/interface/Triplet.h"

#include <vector>
#include <iosfwd>
#include <string>

/** \class GsfBetheHeitlerUpdator
 *  Description of electron energy loss according to Bethe-Heitler
 *  as a sum of Gaussian components. The weights and parameters of the
 *  Gaussians as a function of x/X0 are parametrized as polynomials.
 *  The coefficients of these polynomials are read from a file at 
 * construction time.
 */

class GsfBetheHeitlerUpdator GCC11_FINAL: public GsfMaterialEffectsUpdator {

private:
  /** Helper class for construction & evaluation of a polynomial
   */
  class Polynomial {
  public:
    /// Default constructor (needed for construction of a vector)
    Polynomial () {}
    /** Constructor from a vector of coefficients
     *  (in decreasing order of powers of x)
     */
    Polynomial (const std::vector<double>& coefficients) :
      theCoeffs(coefficients) {}
    /// Evaluation of the polynomial 
    double operator() (const double& x) const {
      double sum(0.);
      for ( std::vector<double>::const_iterator i=theCoeffs.begin();
	    i!=theCoeffs.end(); i++ )  sum = x*sum + *i;
      return sum;
    }
  private:
    std::vector<double> theCoeffs;
  };

public:
  enum CorrectionFlag { NoCorrection=0, MeanCorrection=1, FullCorrection=2 };

public:
  virtual GsfBetheHeitlerUpdator* clone() const
  {
    return new GsfBetheHeitlerUpdator(*this);
  }
  
public:
  /// constructor with explicit filename and correction flag
  GsfBetheHeitlerUpdator (const std::string fileName, const int correctionFlag);

private:
  typedef std::vector< Triplet<double,double,double> > GSContainer;

  /// Computation: generates vectors of weights, means and standard deviations
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect[]) const;

private:

  /// Read parametrization from file
  void readParameters (const std::string);
  /// Read coefficients of one polynomial from file
  Polynomial readPolynomial (std::ifstream&,const int);

  /// Logistic function (needed for transformation of weight and mean)
  inline double logisticFunction (const double x) const {return 1./(1.+exp(-x));}
  /// First moment of the Bethe-Heitler distribution (in z=E/E0)
  inline double BetheHeitlerMean (const double rl) const
  {
    return exp(-rl);
  }
  /// Second moment of the Bethe-Heitler distribution (in z=E/E0)
  inline double BetheHeitlerVariance (const double rl) const
  {
    return exp(-rl*log(3.)/log(2.)) - exp(-2*rl);
  }
  /// Filling of mixture (in terms of z=E/E0)
  void getMixtureParameters (const double, GSContainer&) const;
  /// Correction for weight of component 1
  void correctWeights (GSContainer&) const;
  /// Correction for mean of component 1
  double correctedFirstMean (const double, const GSContainer&) const;
  /// Correction for variance of component 1
  double correctedFirstVar (const double,const GSContainer&) const;
  

private:
  int theNrComponents;                  /// number of components used for parameterisation
  int theTransformationCode;            /// values to be transformed by logistic / exp. function?
  int theCorrectionFlag;                /// correction of 1st or 1st&2nd moments

 std::vector<Polynomial> thePolyWeights;    /// parametrisation of weight for each component
 std::vector<Polynomial> thePolyMeans;      /// parametrisation of mean for each component
 std::vector<Polynomial> thePolyVars;       /// parametrisation of variance for each component

};

#endif
