#ifndef GsfBetheHeitlerUpdator_h_
#define GsfBetheHeitlerUpdator_h_

#include "TrackingTools/GsfTracking/interface/GsfMaterialEffectsUpdator.h"

#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"



#include "TrackingTools/GsfTracking/interface/Triplet.h"

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
  static constexpr int MaxSize=6;
  static constexpr int MaxOrder=6;

  /** Helper class for construction & evaluation of a polynomial
   */
  class Polynomial {
  public:
    /// Default constructor (needed for construction of a vector)
    Polynomial () {}
    /** Constructor from a vector of coefficients
     *  (in decreasing order of powers of x)
     */
    Polynomial (float coefficients[], int is) :
      m_size(is) {
      for (int i=0; i!=m_size; ++i)
	theCoeffs[i]=coefficients[i];
    }
    /// Evaluation of the polynomial 
    float operator() (float x) const {
      float sum=theCoeffs[0];
      for (int i=1; i!=m_size; ++i)
	sum = x*sum + theCoeffs[i];
      return sum;
    }
  private:
    float theCoeffs[MaxOrder] ={0};
    int m_size=0;
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
  typedef Triplet<float,float,float> GSContainer;

  /// Computation: generates vectors of weights, means and standard deviations
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection, Effect[]) const;

private:

  /// Read parametrization from file
  void readParameters (const std::string);
  /// Read coefficients of one polynomial from file
  Polynomial readPolynomial (std::ifstream&,const int);

 
  /// Filling of mixture (in terms of z=E/E0)
  void getMixtureParameters (const float, GSContainer[]) const;
  /// Correction for weight of component 1
  void correctWeights (GSContainer[]) const;
  /// Correction for mean of component 1
  float correctedFirstMean (const float, const GSContainer[]) const;
  /// Correction for variance of component 1
  float correctedFirstVar (const float,const GSContainer[]) const;
  

private:
  int theNrComponents;                  /// number of components used for parameterisation
  int theTransformationCode;            /// values to be transformed by logistic / exp. function?
  int theCorrectionFlag;                /// correction of 1st or 1st&2nd moments

  Polynomial thePolyWeights[MaxSize];    /// parametrisation of weight for each component
  Polynomial thePolyMeans[MaxSize];      /// parametrisation of mean for each componentP
  Polynomial thePolyVars[MaxSize];       /// parametrisation of variance for each component

};

#endif
