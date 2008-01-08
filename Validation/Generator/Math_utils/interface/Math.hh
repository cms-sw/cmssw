/***********************************
Math is a namespace that contains a number of randomly useful routines

Bruce Knuteson 2003
***********************************/

#ifndef __Math
#define __Math

#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include "VistaTools/Math_utils/interface/matrix.hh"

namespace Math
{

  /*******  Formatting routines  *******/

  /// Return a string with n spaces
  std::string spacePad(int n);

  /// Convert double into a string
  std::string ftoa(double x);

  /// Replace substring x of the string s with the substring y
  std::string replaceSubString(std::string s, std::string x, std::string y);

  /// Round x to absolute tolerance tol
  double toleranceRound(double x, double tol=1);

  /// Round x to nSigFigs signficant figures
  double sigFigRound(double x, int nSigFigs = 2);

  /// Just make x look nice
  double nice(double x, int addedprecision = 0);

  /// Transform a string s into a vector, with elements separated by <separator>
  std::vector<std::string> vectorizeString(std::string s, std::string separator);

  /// Load a matrix from a file, each row having unit weight
  void loadMatrixFromFile(std::string filename, std::vector<std::vector<double> >& events);

  /// Load a matrix from a file, each row having arbitary weight
  void loadMatrixFromFile(std::string filename, std::vector<std::vector<double> >& events, std::vector<double>& weights);

  // Get a vector<string> with a list of all files matching <pattern> in directory <dir>
  std::vector<std::string> getFilesInDirectory(std::string dir, std::string pattern="*");

  // Return a random temporary file name
  std::string getTmpFilename();
  

  /*******    Simple numerics    *******/

   /// Add two numbers in quadrature
  double addInQuadrature(double a, double b);

  /// Check to see whether x is NaN
  bool isNaNQ(double x);

  /// Return the difference in radians between phi1 and phi2,
  /// being careful with the cyclicity of the azimuthal angle
  double deltaphi(double phi1, double phi2);

  // Compute deltaR = sqrt(deltaEta^2 + deltaPhi^2)
  double deltaR(double phi1, double eta1, double phi2, double eta2);

  /// Check to see whether two doubles (or vectors of double) a and b are equal to within tol
  bool MPEquality(double a, double b, double tol=1.e-6);
  bool MPEquality(const std::vector<double>& a, const std::vector<double>& b, double tol=1.e-6);

  /// Given a vector<double> x, smear all the entries to make them different by at least tol.
  /// This is necessary for the purpose of getting paw to not choke when making bins.
  void makeDifferent(std::vector<double>& x, double tol=0.001);

  /// Compute the gamma function.  
  /// This function returns only returns correct values for integer and half-integer arguments!
  double gamma(double x);

  /// Compute the volume of the unit sphere in d dimensions
  double volumeOfUnitSphere(int d);

  /// Compute the Euclidean distance between two points a and b
  inline double distanceBetweenPoints(const std::vector<double> & a, const std::vector<double> & b);
  inline double distanceBetweenPoints(const std::vector<double> & a, const std::vector<double> & b, matrix & covarianceMatrixInv); ///< use distance metric given by covarianceMatrixInv
  inline double distanceSqdBetweenPoints(const double * a, const double * b, int n);

  /// Return the minimum (or maximum) of a and b
  inline double min(double a, double b);
  inline double max(double a, double b);

 /// Get the digits of n in an arbitrary base
  std::vector<int> getDigits(int n, int base, int size=0); ///< always return <size> digits, if <size> != 0
  std::vector<int> getDigits(int n, std::vector<int> base, int size=0);
  std::vector<int> integerNthRoot(int a, int n);
  std::vector<std::vector<int> > permutationSet(std::vector<int> q); ///< returns a vector<vector<int> > of size q.size() factorial by q.size()
  std::vector<std::vector<int> > permutationSet(int n);

  /** KeepNSmallest is a class for holding onto the N smallest values of a large set of numbers.  
      keepNSmallest.update(value, number) asks keepNSmallest to keep track of value and number 
      iff value is one of the N smallest numbers that keepNSmallest has seen so far.  
      keepNSmallest.getValues() returns a vector of the smallest values it has been given, ordered by increasing value.  
      keepNSmallest.getNumbers() returns a vector of the numbers corresponding to the smallest values it has been given, 
      ordered such that keepNSmallest.getNumbers()[i] is the number that corresponds to the value keepNSmallest.getValues()[i].  */

  class KeepNSmallest
  {
  public:
    KeepNSmallest(int N);
    void update(double value, int number);
    std::vector<int> getNumbers();
    std::vector<double> getValues();
  private:
    int N;
    std::vector<double> values;
    std::vector<int> numbers;
  };

   /** KeepWSmallest is a class for holding onto the smallest values of a large set of numbers; the total weight held will not exceed W.  
      keepNSmallest.update(value, number, weight) asks keepNSmallest to keep track of value and number 
      iff value is one of the smallest numbers that keepNSmallest has seen so far within its weight limit of W.
      keepNSmallest.getValues() returns a vector of the smallest values it has been given, ordered by increasing value.  
      keepNSmallest.getNumbers() returns a vector of the numbers corresponding to the smallest values it has been given, 
      ordered such that keepNSmallest.getNumbers()[i] is the number that corresponds to the value keepNSmallest.getValues()[i].  */

  class KeepWSmallest
  {
  public:
    KeepWSmallest(double W, int N=0);
    void update(double value, int number, double weight=1);
    std::vector<int> getNumbers();
    std::vector<double> getValues();
  private:
    double W;
    int N;
    std::vector< std::vector<double> > entries;
  };

  /// Solve the matrix equation A.x = B for x
  std::vector<double> linearEquationSolve(matrix A, std::vector<double> B);

  /// Calculate a reasonable choice for the kernel smoothing parameter h.
  /// Input:  
  ///    fractionalWeight:  the fractional weight of each Monte Carlo event
  ///               nvars:  the dimensionality of the space
  double calculateSmoothingParameter(double fractionalWeight, int nvars);

  /// Translate pseudorapidity to theta
  double eta2theta(double eta);
  /// Translate theta to pseudorapidity
  double theta2eta(double theta);

  /// Return +- x^n, negative if x is negative, positive if x is positive
  double sgnpow(double x, double n);

  /// Return i^j, with i and j both integers
  int intpow(int i, int j);

  /// Return the binomial error
  double binomialError(double p, int N);
  
  /*******  Statistics routines  *******/

  /// Compute the probability of background b +- deltab to fluctuate 
  /// up to or above the observed number of events N. If the b pulled from
  /// gaussian is big, use gaussian approximation of poisson. However,
  /// this approximation doesn't work well if d >> or << b.
  double probOfThisEffect(double b, int N, double deltab = 0, std::string opt=">=");

  /// Compute the probability of observing exactly N events from a background b,
  /// estimated with Monte Carlo events of weight wtMC 
  /// (and having an associated Monte Carlo statistical error)
  double poissonConvolutedWithPoissonIntegrated(double b, double wtMC, int N);

  /// Ditto in the case where the background is estimated using many (>10) Monte Carlo events,
  /// so that the Monte Carlo error on b is gaussian distributed with width deltab
  double poissonConvolutedWithGaussianIntegrated(double b, double deltab, int N);

  /// Compute a rough magnitude of discrepancy, faster and less acurate than using probOfThisEffect
  double roughMagnitudeOfDiscrepancy(int N, double b, double deltab);

  //returns the probability to pull n from a Poisson distribution of mean "mean".
  double poisson(int n, double mean);
  
// probability to observe d if bkg=b0+/-deltab. Convolute the poisson with gaussian. The gaussian represents the uncertainty in b0 by deltab. We pull b from that gaussian, and then we pull d from a poisson of mean b.
  double ppg(int d, double b0, double deltab);

  /// Compute a much more accurate magnitude of discrepancy. It comes in contrast to roughMagnitudeOfDiscrepancy. This should be more accurate than probOfThisEffect, especially if d >> b0 or d << b0.
  double accurateMagnitudeOfDiscrepancy(int d, double b0, double deltab);

  /// Compute the background b which has a probability effect of fluctuating up to or above the observed number of events N
  double bkgFromEffect(double effect, int N);

  /// Convert a standard deviation s into a probability
  double sigma2prob(double s);

  /// Convert a probability p into a standard deviation
  double prob2sigma(double p);

  /**  Pulling random numbers from distributions  **/

  /// Return a random double pulled from a gaussian with mean mu and standard deviation sigma
  double gasdev(double mu, double sigma);
  std::vector<double> randMultiGauss(const std::vector<double>& mu, const matrix& sigma);
  std::vector<double> randMultiGauss(const std::vector<double>& mu, const std::vector<std::vector<double> >& sigma);

  /// Return a random number pulled from an exponential with decay length lambda
  double expdev(double lambda);

  /// Return a random integer pulled from a poisson distribution with mean mu
  int poisson(double mu);

  /// Return a random integer pulled from a poisson distribution with mean mean convoluted with a gaussian to account for a systematic error
  int fluctuate(double mean, double systematicError);

  /**  Computing statistics from vectors  **/
  double computeSum(const std::vector<double>& x);
  double computeAverage(const std::vector<double>& x);
  long double computeAverage(std::vector<long double> x);
  double computeAverage(std::vector<int> x);
  double computeRMS(std::vector<double> x);
  long double computeRMS(std::vector<long double> x);

  /// Compute the "effective" number of Monte Carlo events in an ensemble of events with weights provided in a vector of doubles.
  /// (The idea here is that if there are n events with very large weight and m events
  /// with very small weight (say), then the effective number of events will be close to n)
  double effectiveNumberOfEvents(const std::vector<double> & wt);

  /// Compute the statistical error in an ensemble of n Monte Carlo events.
  /// epsilonWt is the "weight quantum", the largest weight in the Monte Carlo sample,
  /// and a lower bound on the amount by which everything is uncertain
  double computeMCerror(double sum, double n, double epsilonWt=0);
  double computeMCerror(const std::vector<double> & wt, double epsilonWt=0);

  /// Toss away a fraction alpha of the events, from the tails of the distributions
  void tossAwayTail(std::vector<std::vector<double> >& events, double alpha=0); 
 
 /// Compute the median of a multivariate sample
  std::vector<double> computeMedian(std::vector<std::vector<double> > events);

  /// Compute the lNorm covariance matrix of a multivariate sample, tossing away outliers (alpha)
  std::vector<std::vector<double> > computeCovarianceMatrix(std::vector<std::vector<double> > events, double alpha, int lNorm);

  /// Compute the lNorm correlation matrix of a multivariate sample, tossing away outliers (alpha)
  std::vector<std::vector<double> > computeCorrelationMatrix(std::vector<std::vector<double> > events, double alpha, int lNorm = 2);

   // Make a nice histogram range, taking positions in the form of vector<double> a,
  // and returning nbins, low edge and high edge of range
  void makeNiceHistogramRange(std::vector<double> a, int& nbins, double& lo, double& hi);
  void makeNiceHistogramRange(const std::vector<double>& bkgWeights,
			      const std::vector<double>& sigWeights,
			      const std::vector<std::vector<double> >& bkgEvents, 
			      const std::vector<std::vector<double> >& sigEvents, 
			      const std::vector<std::vector<double> >& dataEvents,
			      std::vector<std::vector<double> >& range, std::vector<int>& nbins 
			      );
  std::vector<double> putIntoBins(std::vector<double> binEdges, std::vector<double> points, std::vector<double> wt);
  std::vector<int> putIntoBins(std::vector<double> binEdges, std::vector<double> points);

 /**   Minimization routines  **/

  // A class that can be evaluated like a function, for passing to Math::minimize
  class FunctionObject
  {
  public:
    FunctionObject() {};
    virtual double operator()(const std::vector<double>& x) { return 0; };
    virtual ~FunctionObject() {}
  private:
  };

  /// Minimize the function funk (to tolerance tol) with initial parameters x having characteristic length scale dx
  /// dx is computed automatically if not specified
  double minimize(std::vector<double>& x, FunctionObject* funk, std::vector<double> dx = std::vector<double>(0), 
		  double tol = 1e-2, bool stoppable = false);

  /// Numerical recipies routines implementing a simplex algorithm for performing the minimization
  double amotry(matrix & p, std::vector<double> & y, std::vector<double> & psum, int ndim, FunctionObject* funk, int ihi, double fac);
  void amoeba(matrix & p, std::vector<double> & y, int ndim, FunctionObject* funk, double ftol=1e-6, int NMAX=10000, bool stoppable=false);

  /// Parallel vector sorting routines
  template<class T1, class T2> void parallelBubbleSort(std::vector<T1>& x, std::vector<T2>& y);
  template<class T1, class T2> void parallelQuickSort(std::vector<T1>& x, std::vector<T2>& y);
  template<class T1, class T2> void parallelReverseSort(std::vector<T1>& x, std::vector<T2>& y); ///< BubbleSort

  std::string system(std::string command);

  double innerProduct(std::vector<double> a, std::vector<double> b);
  std::vector<double> outerProduct(std::vector<double> a, std::vector<double> b);
  double norm(std::vector<double> a) ;
  std::vector<double> normalizeVector(std::vector<double> a);
  double planesAngle(std::vector<double> a1, std::vector<double> a2, std::vector<double> b1, std::vector<double> b2) ;
  int criticalD(double b, double pvalueTarget) ;
}

#include "VistaTools/Math_utils/interface/Math.ii"

#endif


