#ifndef GSUtilities_h
#define GSUtilities_h

/** Some utilities for analysing 1D Gaussian mixtures.
 * Copied from ORCA's EgammaGSUtilities.
 */

class GSUtilities {
public:
  /// constructor from arrays of weights, parameters and standard deviations
  GSUtilities (const unsigned nComp, const float* weights,
	       const float* parameters, const float* errors) :
    theNComp(nComp),
    theWeights(nullptr),
    theParameters(nullptr),
    theErrors(nullptr)
  {
    if ( theNComp ) {
      theWeights = new float[theNComp];
      theParameters = new float[theNComp];
      theErrors = new float[theNComp];
    }
    const float* wPtr1(weights);
    const float* pPtr1(parameters);
    const float* ePtr1(errors);
    float* wPtr2(theWeights);
    float* pPtr2(theParameters);
    float* ePtr2(theErrors);
    for ( unsigned i=0; i<theNComp; i++ ) {
      *(wPtr2++) = weights ? *(wPtr1++) : 1.;
      *(pPtr2++) = *(pPtr1++);
      *(ePtr2++) = *(ePtr1++);
    }
  } 
  ~GSUtilities () 
  {
    delete [] theWeights;
    delete [] theParameters;
    delete [] theErrors;
  }
  /** normalised integral from -inf to x
   *  (taking into account under- & overflows) 
   */
  float quantile (const float) const;
  /// mode
  float mode () const;
  /// value of the pdf
  double pdf (const double&) const;
  /// value of integral(pdf)
  double cdf (const double&) const;
  /// first derivative of pdf
  double dpdf1 (const double&) const;
  /// second derivative of pdf
  double dpdf2 (const double&) const;

  /// mean value of combined state
  double combinedMean() const;
  //  mean value of errors 
  double errorCombinedMean() const;
  //  error for the highest weight
  float errorHighestWeight() const;
  // max weight component - chiara
  float maxWeight() const;
  // mode error + some utilities functions
  float errorMode();
  float getMax(float);
  float getMin(float);

private:
  /// value of gaussian distribution
  double gauss (const double&, const double&, const double&) const;
  /// integrated value of gaussian distribution
  double gaussInt (const double&, const double&, const double&) const;
  /// mean value of combined state
  /// double combinedMean() const;
  /// mode from starting value
  double findMode (const double) const;

private:
  unsigned theNComp;
  float* theWeights;
  float* theParameters;
  float* theErrors;
};
#endif
