/****************************************************************************
*
* This is a part of TOTEM ofFileline software.
*
* Based on TMultiDimFet.cc (from ROOT) by Christian Holm Christense.
*
* Authors:
*   Hubert Niewiadomski
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "Riostream.h"
#include "SimTransport/TotemRPProtonTransportParametrization/interface/TMultiDimFet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"
#include "TH1.h"
#include "TH2.h"
#include "TROOT.h"
#include "TDecompChol.h"
#include "TDatime.h"
#include <iostream>
#include <map>

#define RADDEG (180. / TMath::Pi())
#define DEGRAD (TMath::Pi() / 180.)
#define HIST_XORIG 0
#define HIST_DORIG 1
#define HIST_XNORM 2
#define HIST_DSHIF 3
#define HIST_RX 4
#define HIST_RD 5
#define HIST_RTRAI 6
#define HIST_RTEST 7
#define PARAM_MAXSTUDY 1
#define PARAM_SEVERAL 2
#define PARAM_RELERR 3
#define PARAM_MAXTERMS 4

//____________________________________________________________________
//static void mdfHelper(int&, double*, double&, double*, int);

//____________________________________________________________________
ClassImp(TMultiDimFet);

//____________________________________________________________________
// Static instance. Used with mdfHelper and TMinuit
//TMultiDimFet* TMultiDimFet::fgInstance = nullptr;

//____________________________________________________________________
TMultiDimFet::TMultiDimFet() {
  // Empty CTOR. Do not use
  fMeanQuantity = 0;
  fMaxQuantity = 0;
  fMinQuantity = 0;
  fSumSqQuantity = 0;
  fSumSqAvgQuantity = 0;
  fPowerLimit = 1;

  fMaxAngle = 0;
  fMinAngle = 1;

  fNVariables = 0;
  fMaxVariables = 0;
  fMinVariables = 0;
  fSampleSize = 0;

  fMaxAngle = 0;
  fMinAngle = 0;

  fPolyType = kMonomials;
  fShowCorrelation = kFALSE;

  fIsUserFunction = kFALSE;

  fHistograms = nullptr;
  fHistogramMask = 0;

  //fFitter                  = nullptr;
  //fgInstance               = nullptr;
}

const TMultiDimFet &TMultiDimFet::operator=(const TMultiDimFet &in) {
  if (this == &in) {
    return in;
  }

  fMeanQuantity = in.fMeanQuantity;  // Mean of dependent quantity

  fMaxQuantity = 0.0;       //! Max value of dependent quantity
  fMinQuantity = 0.0;       //! Min value of dependent quantity
  fSumSqQuantity = 0.0;     //! SumSquare of dependent quantity
  fSumSqAvgQuantity = 0.0;  //! Sum of squares away from mean

  fNVariables = in.fNVariables;  // Number of independent variables

  fMaxVariables.ResizeTo(in.fMaxVariables.GetLwb(), in.fMaxVariables.GetUpb());
  fMaxVariables = in.fMaxVariables;  // max value of independent variables

  fMinVariables.ResizeTo(in.fMinVariables.GetLwb(), in.fMinVariables.GetUpb());
  fMinVariables = in.fMinVariables;  // min value of independent variables

  fSampleSize = 0;      //! Size of training sample
  fTestSampleSize = 0;  //! Size of test sample
  fMinAngle = 1;        //! Min angle for acepting new function
  fMaxAngle = 0.0;      //! Max angle for acepting new function

  fMaxTerms = in.fMaxTerms;  // Max terms expected in final expr.

  fMinRelativeError = 0.0;  //! Min relative error accepted

  fMaxPowers.clear();  //! [fNVariables] maximum powers
  fPowerLimit = 1;     //! Control parameter

  fMaxFunctions = in.fMaxFunctions;  // max number of functions

  fFunctionCodes.clear();  //! [fMaxFunctions] acceptance code
  fMaxStudy = 0;           //! max functions to study

  fMaxPowersFinal.clear();  //! [fNVariables] maximum powers from fit;

  fMaxFunctionsTimesNVariables = in.fMaxFunctionsTimesNVariables;  // fMaxFunctionsTimesNVariables
  fPowers = in.fPowers;

  fPowerIndex = in.fPowerIndex;  // [fMaxTerms] Index of accepted powers

  fMaxResidual = 0.0;    //! Max redsidual value
  fMinResidual = 0.0;    //! Min redsidual value
  fMaxResidualRow = 0;   //! Row giving max residual
  fMinResidualRow = 0;   //! Row giving min residual
  fSumSqResidual = 0.0;  //! Sum of Square residuals

  fNCoefficients = in.fNCoefficients;  // Dimension of model coefficients

  fCoefficients.ResizeTo(in.fCoefficients.GetLwb(), in.fCoefficients.GetUpb());
  fCoefficients = in.fCoefficients;  // Vector of the final coefficients

  fRMS = 0.0;                   //! Root mean square of fit
  fChi2 = 0.0;                  //! Chi square of fit
  fParameterisationCode = 0;    //! Exit code of parameterisation
  fError = 0.0;                 //! Error from parameterization
  fTestError = 0.0;             //! Error from test
  fPrecision = 0.0;             //! Relative precision of param
  fTestPrecision = 0.0;         //! Relative precision of test
  fCorrelationCoeff = 0.0;      //! Multi Correlation coefficient
  fTestCorrelationCoeff = 0.0;  //! Multi Correlation coefficient
  fHistograms = nullptr;        //! List of histograms
  fHistogramMask = 0;           //! Bit pattern of hisograms used
  //fFitter = nullptr;            //! Fit object (MINUIT)

  fPolyType = in.fPolyType;                // Type of polynomials to use
  fShowCorrelation = in.fShowCorrelation;  // print correlation matrix
  fIsUserFunction = in.fIsUserFunction;    // Flag for user defined function
  fIsVerbose = in.fIsVerbose;              //
  return in;
}

//____________________________________________________________________
TMultiDimFet::TMultiDimFet(Int_t dimension, EMDFPolyType type, Option_t *option)
    : TNamed("multidimfit", "Multi-dimensional fit object"),
      fQuantity(dimension),
      fSqError(dimension),
      fVariables(dimension * 100),
      fMeanVariables(dimension),
      fMaxVariables(dimension),
      fMinVariables(dimension) {
  // Constructor
  // Second argument is the type of polynomials to use in
  // parameterisation, one of:
  //      TMultiDimFet::kMonomials
  //      TMultiDimFet::kChebyshev
  //      TMultiDimFet::kLegendre
  //
  // Options:
  //   K      Compute (k)correlation matrix
  //   V      Be verbose
  //
  // Default is no options.
  //

  //fgInstance = this;

  fMeanQuantity = 0;
  fMaxQuantity = 0;
  fMinQuantity = 0;
  fSumSqQuantity = 0;
  fSumSqAvgQuantity = 0;
  fPowerLimit = 1;

  fMaxAngle = 0;
  fMinAngle = 1;

  fNVariables = dimension;
  fMaxVariables = 0;
  fMaxFunctionsTimesNVariables = fMaxFunctions * fNVariables;
  fMinVariables = 0;
  fSampleSize = 0;
  fTestSampleSize = 0;
  fMinRelativeError = 0.01;
  fError = 0;
  fTestError = 0;
  fPrecision = 0;
  fTestPrecision = 0;
  fParameterisationCode = 0;

  fPolyType = type;
  fShowCorrelation = kFALSE;
  fIsVerbose = kFALSE;

  TString opt = option;
  opt.ToLower();

  if (opt.Contains("k"))
    fShowCorrelation = kTRUE;
  if (opt.Contains("v"))
    fIsVerbose = kTRUE;

  fIsUserFunction = kFALSE;

  fHistograms = nullptr;
  fHistogramMask = 0;

  fMaxPowers.resize(dimension);
  fMaxPowersFinal.resize(dimension);
  //fFitter                 = nullptr;
}

//____________________________________________________________________
TMultiDimFet::~TMultiDimFet() {
  if (fHistograms)
    fHistograms->Clear("nodelete");
  delete fHistograms;
}

//____________________________________________________________________
void TMultiDimFet::AddRow(const Double_t *x, Double_t D, Double_t E) {
  // Add a row consisting of fNVariables independent variables, the
  // known, dependent quantity, and optionally, the square error in
  // the dependent quantity, to the training sample to be used for the
  // parameterization.
  // The mean of the variables and quantity is calculated on the fly,
  // as outlined in TPrincipal::AddRow.
  // This sample should be representive of the problem at hand.
  // Please note, that if no error is given Poisson statistics is
  // assumed and the square error is set to the value of dependent
  // quantity.  See also the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  if (!x)
    return;

  if (++fSampleSize == 1) {
    fMeanQuantity = D;
    fMaxQuantity = D;
    fMinQuantity = D;
  } else {
    fMeanQuantity *= 1 - 1. / Double_t(fSampleSize);
    fMeanQuantity += D / Double_t(fSampleSize);
    fSumSqQuantity += D * D;

    if (D >= fMaxQuantity)
      fMaxQuantity = D;
    if (D <= fMinQuantity)
      fMinQuantity = D;
  }

  // If the vector isn't big enough to hold the new data, then
  // expand the vector by half it's size.
  Int_t size = fQuantity.GetNrows();
  if (fSampleSize > size) {
    fQuantity.ResizeTo(size + size / 2);
    fSqError.ResizeTo(size + size / 2);
  }

  // Store the value
  fQuantity(fSampleSize - 1) = D;
  fSqError(fSampleSize - 1) = (E == 0 ? D : E);

  // Store data point in internal vector
  // If the vector isn't big enough to hold the new data, then
  // expand the vector by half it's size
  size = fVariables.GetNrows();
  if (fSampleSize * fNVariables > size)
    fVariables.ResizeTo(size + size / 2);

  // Increment the data point counter
  Int_t i, j;
  for (i = 0; i < fNVariables; i++) {
    if (fSampleSize == 1) {
      fMeanVariables(i) = x[i];
      fMaxVariables(i) = x[i];
      fMinVariables(i) = x[i];
    } else {
      fMeanVariables(i) *= 1 - 1. / Double_t(fSampleSize);
      fMeanVariables(i) += x[i] / Double_t(fSampleSize);

      // Update the maximum value for this component
      if (x[i] >= fMaxVariables(i))
        fMaxVariables(i) = x[i];

      // Update the minimum value for this component
      if (x[i] <= fMinVariables(i))
        fMinVariables(i) = x[i];
    }

    // Store the data.
    j = (fSampleSize - 1) * fNVariables + i;
    fVariables(j) = x[i];
  }
}

//____________________________________________________________________
void TMultiDimFet::AddTestRow(const Double_t *x, Double_t D, Double_t E) {
  // Add a row consisting of fNVariables independent variables, the
  // known, dependent quantity, and optionally, the square error in
  // the dependent quantity, to the test sample to be used for the
  // test of the parameterization.
  // This sample needn't be representive of the problem at hand.
  // Please note, that if no error is given Poisson statistics is
  // assumed and the square error is set to the value of dependent
  // quantity.  See also the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  if (fTestSampleSize++ == 0) {
    fTestQuantity.ResizeTo(fNVariables);
    fTestSqError.ResizeTo(fNVariables);
    fTestVariables.ResizeTo(fNVariables * 100);
  }

  // If the vector isn't big enough to hold the new data, then
  // expand the vector by half it's size.
  Int_t size = fTestQuantity.GetNrows();
  if (fTestSampleSize > size) {
    fTestQuantity.ResizeTo(size + size / 2);
    fTestSqError.ResizeTo(size + size / 2);
  }

  // Store the value
  fTestQuantity(fTestSampleSize - 1) = D;
  fTestSqError(fTestSampleSize - 1) = (E == 0 ? D : E);

  // Store data point in internal vector
  // If the vector isn't big enough to hold the new data, then
  // expand the vector by half it's size
  size = fTestVariables.GetNrows();
  if (fTestSampleSize * fNVariables > size)
    fTestVariables.ResizeTo(size + size / 2);

  // Increment the data point counter
  Int_t i, j;
  for (i = 0; i < fNVariables; i++) {
    j = fNVariables * (fTestSampleSize - 1) + i;
    fTestVariables(j) = x[i];

    if (x[i] > fMaxVariables(i))
      Warning("AddTestRow", "variable %d (row: %d) too large: %f > %f", i, fTestSampleSize, x[i], fMaxVariables(i));
    if (x[i] < fMinVariables(i))
      Warning("AddTestRow", "variable %d (row: %d) too small: %f < %f", i, fTestSampleSize, x[i], fMinVariables(i));
  }
}

//____________________________________________________________________
void TMultiDimFet::Clear(Option_t *option) {
  // Clear internal structures and variables
  Int_t i, j, n = fNVariables, m = fMaxFunctions;

  // Training sample, dependent quantity
  fQuantity.Zero();
  fSqError.Zero();
  fMeanQuantity = 0;
  fMaxQuantity = 0;
  fMinQuantity = 0;
  fSumSqQuantity = 0;
  fSumSqAvgQuantity = 0;

  // Training sample, independent variables
  fVariables.Zero();
  fNVariables = 0;
  fSampleSize = 0;
  fMeanVariables.Zero();
  fMaxVariables.Zero();
  fMinVariables.Zero();

  // Test sample
  fTestQuantity.Zero();
  fTestSqError.Zero();
  fTestVariables.Zero();
  fTestSampleSize = 0;

  // Functions
  fFunctions.Zero();
  fMaxFunctions = 0;
  fMaxStudy = 0;
  fMaxFunctionsTimesNVariables = 0;
  fOrthFunctions.Zero();
  fOrthFunctionNorms.Zero();

  // Control parameters
  fMinRelativeError = 0;
  fMinAngle = 0;
  fMaxAngle = 0;
  fMaxTerms = 0;

  // Powers
  for (i = 0; i < n; i++) {
    fMaxPowers[i] = 0;
    fMaxPowersFinal[i] = 0;
    for (j = 0; j < m; j++)
      fPowers[i * n + j] = 0;
  }
  fPowerLimit = 0;

  // Residuals
  fMaxResidual = 0;
  fMinResidual = 0;
  fMaxResidualRow = 0;
  fMinResidualRow = 0;
  fSumSqResidual = 0;

  // Fit
  fNCoefficients = 0;
  fOrthCoefficients = 0;
  fOrthCurvatureMatrix = 0;
  fRMS = 0;
  fCorrelationMatrix.Zero();
  fError = 0;
  fTestError = 0;
  fPrecision = 0;
  fTestPrecision = 0;

  // Coefficients
  fCoefficients.Zero();
  fCoefficientsRMS.Zero();
  fResiduals.Zero();
  fHistograms->Clear(option);

  // Options
  fPolyType = kMonomials;
  fShowCorrelation = kFALSE;
  fIsUserFunction = kFALSE;
}

//____________________________________________________________________
Double_t TMultiDimFet::Eval(const Double_t *x, const Double_t *coeff) const {
  // Evaluate parameterization at point x. Optional argument coeff is
  // a vector of coefficients for the parameterisation, fNCoefficients
  // elements long.
  //   int fMaxFunctionsTimesNVariables = fMaxFunctions * fNVariables;
  Double_t returnValue = fMeanQuantity;
  Double_t term = 0;
  Int_t i, j;

  for (i = 0; i < fNCoefficients; i++) {
    // Evaluate the ith term in the expansion
    term = (coeff ? coeff[i] : fCoefficients(i));
    for (j = 0; j < fNVariables; j++) {
      // Evaluate the factor (polynomial) in the j-th variable.
      Int_t p = fPowers[fPowerIndex[i] * fNVariables + j];
      Double_t y = 1 + 2. / (fMaxVariables(j) - fMinVariables(j)) * (x[j] - fMaxVariables(j));
      term *= EvalFactor(p, y);
    }
    // Add this term to the final result
    returnValue += term;
  }
  return returnValue;
}

void TMultiDimFet::ReducePolynomial(double error) {
  if (error == 0.0)
    return;
  else {
    ZeroDoubiousCoefficients(error);
  }
}

void TMultiDimFet::ZeroDoubiousCoefficients(double error) {
  typedef std::multimap<double, int> cmt;
  cmt m;

  for (int i = 0; i < fNCoefficients; i++) {
    m.insert(std::pair<double, int>(TMath::Abs(fCoefficients(i)), i));
  }

  double del_error_abs = 0;
  int deleted_terms_count = 0;

  for (cmt::iterator it = m.begin(); it != m.end() && del_error_abs < error; ++it) {
    if (TMath::Abs(it->first) + del_error_abs < error) {
      fCoefficients(it->second) = 0.0;
      del_error_abs = TMath::Abs(it->first) + del_error_abs;
      deleted_terms_count++;
    } else
      break;
  }

  int fNCoefficients_new = fNCoefficients - deleted_terms_count;
  TVectorD fCoefficients_new(fNCoefficients_new);
  std::vector<Int_t> fPowerIndex_new;

  int ind = 0;
  for (int i = 0; i < fNCoefficients; i++) {
    if (fCoefficients(i) != 0.0) {
      fCoefficients_new(ind) = fCoefficients(i);
      fPowerIndex_new.push_back(fPowerIndex[i]);
      ind++;
    }
  }
  fNCoefficients = fNCoefficients_new;
  fCoefficients.ResizeTo(fNCoefficients);
  fCoefficients = fCoefficients_new;
  fPowerIndex = fPowerIndex_new;
  edm::LogInfo("TMultiDimFet") << deleted_terms_count << " terms removed"
                               << "\n";
}

//____________________________________________________________________
Double_t TMultiDimFet::EvalControl(const Int_t *iv) {
  // PRIVATE METHOD:
  // Calculate the control parameter from the passed powers
  Double_t s = 0;
  Double_t epsilon = 1e-6;  // a small number
  for (Int_t i = 0; i < fNVariables; i++) {
    if (fMaxPowers[i] != 1)
      s += (epsilon + iv[i] - 1) / (epsilon + fMaxPowers[i] - 1);
  }
  return s;
}

//____________________________________________________________________
Double_t TMultiDimFet::EvalFactor(Int_t p, Double_t x) const {
  // PRIVATE METHOD:
  // Evaluate function with power p at variable value x
  Int_t i = 0;
  Double_t p1 = 1;
  Double_t p2 = 0;
  Double_t p3 = 0;
  Double_t r = 0;

  switch (p) {
    case 1:
      r = 1;
      break;
    case 2:
      r = x;
      break;
    default:
      p2 = x;
      for (i = 3; i <= p; i++) {
        p3 = p2 * x;
        if (fPolyType == kLegendre)
          p3 = ((2 * i - 3) * p2 * x - (i - 2) * p1) / (i - 1);
        else if (fPolyType == kChebyshev)
          p3 = 2 * x * p2 - p1;
        p1 = p2;
        p2 = p3;
      }
      r = p3;
  }

  return r;
}

//____________________________________________________________________
void TMultiDimFet::FindParameterization(double precision) {
  // Find the parameterization
  //
  // Options:
  //     None so far
  //
  // For detailed description of what this entails, please refer to the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  MakeNormalized();
  MakeCandidates();
  MakeParameterization();
  MakeCoefficients();
  ReducePolynomial(precision);
}

//____________________________________________________________________
/*
void TMultiDimFet::Fit(Option_t *option)
{
   // Try to fit the found parameterisation to the test sample.
   //
   // Options
   //     M     use Minuit to improve coefficients
   //
   // Also, refer to
   // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html

   //fgInstance = this;

   Int_t i, j;
   Double_t*      x    = new Double_t[fNVariables];
   Double_t  sumSqD    = 0;
   Double_t    sumD    = 0;
   Double_t  sumSqR    = 0;
   Double_t    sumR    = 0;

   // Calculate the residuals over the test sample
   for (i = 0; i < fTestSampleSize; i++) {
      for (j = 0; j < fNVariables; j++)
         x[j] = fTestVariables(i * fNVariables + j);
      Double_t res =  fTestQuantity(i) - Eval(x);
      sumD         += fTestQuantity(i);
      sumSqD       += fTestQuantity(i) * fTestQuantity(i);
      sumR         += res;
      sumSqR       += res * res;
      if (TESTBIT(fHistogramMask,HIST_RTEST))
         ((TH1D*)fHistograms->FindObject("res_test"))->Fill(res);
   }
   Double_t dAvg         = sumSqD - (sumD * sumD) / fTestSampleSize;
   Double_t rAvg         = sumSqR - (sumR * sumR) / fTestSampleSize;
   fTestCorrelationCoeff = (dAvg - rAvg) / dAvg;
   fTestError            = sumSqR;
   fTestPrecision        = sumSqR / sumSqD;

   TString opt(option);
   opt.ToLower();

   if (!opt.Contains("m"))
      MakeChi2();

   if (fNCoefficients * 50 > fTestSampleSize)
      Warning("Fit", "test sample is very small");

   if (!opt.Contains("m"))
      return;

   fFitter = TVirtualFitter::Fitter(nullptr,fNCoefficients);
   fFitter->SetFCN(mdfHelper);

   const Int_t  maxArgs = 16;
   Int_t           args = 1;
   Double_t*   arglist  = new Double_t[maxArgs];
   arglist[0]           = -1;
   fFitter->ExecuteCommand("SET PRINT",arglist,args);

   for (i = 0; i < fNCoefficients; i++) {
      Double_t startVal = fCoefficients(i);
      Double_t startErr = fCoefficientsRMS(i);
      fFitter->SetParameter(i, Form("coeff%02d",i),
         startVal, startErr, 0, 0);
   }

   args                 = 1;
   fFitter->ExecuteCommand("MIGRAD",arglist,args);

   for (i = 0; i < fNCoefficients; i++) {
      Double_t val = 0, err = 0, low = 0, high = 0;
      fFitter->GetParameter(i, Form("coeff%02d",i),
         val, err, low, high);
      fCoefficients(i)    = val;
      fCoefficientsRMS(i) = err;
   }
}
*/

//____________________________________________________________________
void TMultiDimFet::MakeCandidates() {
  // PRIVATE METHOD:
  // Create list of candidate functions for the parameterisation. See
  // also
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  Int_t i = 0;
  Int_t j = 0;
  Int_t k = 0;

  // The temporary array to store the powers in. We don't need to
  // initialize this array however.
  Int_t *powers = new Int_t[fNVariables * fMaxFunctions];

  // store of `control variables'
  Double_t *control = new Double_t[fMaxFunctions];

  // We've better initialize the variables
  Int_t *iv = new Int_t[fNVariables];
  for (i = 0; i < fNVariables; i++)
    iv[i] = 1;

  if (!fIsUserFunction) {
    // Number of funcs selected
    Int_t numberFunctions = 0;

    while (kTRUE) {
      // Get the control value for this function
      Double_t s = EvalControl(iv);

      if (s <= fPowerLimit) {
        // Call over-loadable method Select, as to allow the user to
        // interfere with the selection of functions.
        if (Select(iv)) {
          numberFunctions++;

          // If we've reached the user defined limit of how many
          // functions we can consider, break out of the loop
          if (numberFunctions > fMaxFunctions)
            break;

          // Store the control value, so we can sort array of powers
          // later on
          control[numberFunctions - 1] = Int_t(1.0e+6 * s);

          // Store the powers in powers array.
          for (i = 0; i < fNVariables; i++) {
            j = (numberFunctions - 1) * fNVariables + i;
            powers[j] = iv[i];
          }
        }  // if (Select())
      }    // if (s <= fPowerLimit)

      for (i = 0; i < fNVariables; i++)
        if (iv[i] < fMaxPowers[i])
          break;

      // If all variables have reached their maximum power, then we
      // break out of the loop
      if (i == fNVariables) {
        fMaxFunctions = numberFunctions;
        fMaxFunctionsTimesNVariables = fMaxFunctions * fNVariables;
        break;
      }

      // Next power in variable i
      iv[i]++;

      for (j = 0; j < i; j++)
        iv[j] = 1;
    }  // while (kTRUE)
  } else {
    // In case the user gave an explicit function
    for (i = 0; i < fMaxFunctions; i++) {
      // Copy the powers to working arrays
      for (j = 0; j < fNVariables; j++) {
        powers[i * fNVariables + j] = fPowers[i * fNVariables + j];
        iv[j] = fPowers[i * fNVariables + j];
      }

      control[i] = Int_t(1.0e+6 * EvalControl(iv));
    }
  }

  // Now we need to sort the powers according to least `control
  // variable'
  Int_t *order = new Int_t[fMaxFunctions];
  for (i = 0; i < fMaxFunctions; i++)
    order[i] = i;
  fPowers.resize(fMaxFunctions * fNVariables);

  for (i = 0; i < fMaxFunctions; i++) {
    Double_t x = control[i];
    Int_t l = order[i];
    k = i;

    for (j = i; j < fMaxFunctions; j++) {
      if (control[j] <= x) {
        x = control[j];
        l = order[j];
        k = j;
      }
    }

    if (k != i) {
      control[k] = control[i];
      control[i] = x;
      order[k] = order[i];
      order[i] = l;
    }
  }

  for (i = 0; i < fMaxFunctions; i++)
    for (j = 0; j < fNVariables; j++)
      fPowers[i * fNVariables + j] = powers[order[i] * fNVariables + j];

  delete[] control;
  delete[] powers;
  delete[] order;
  delete[] iv;
}

Double_t TMultiDimFet::MakeChi2(const Double_t *coeff) {
  // Calculate Chi square over either the test sample. The optional
  // argument coeff is a vector of coefficients to use in the
  // evaluation of the parameterisation. If coeff == 0, then the found
  // coefficients is used.
  // Used my MINUIT for fit (see TMultDimFit::Fit)
  fChi2 = 0;
  Int_t i, j;
  Double_t *x = new Double_t[fNVariables];
  for (i = 0; i < fTestSampleSize; i++) {
    // Get the stored point
    for (j = 0; j < fNVariables; j++)
      x[j] = fTestVariables(i * fNVariables + j);

    // Evaluate function. Scale to shifted values
    Double_t f = Eval(x, coeff);

    // Calculate contribution to Chic square
    fChi2 += 1. / TMath::Max(fTestSqError(i), 1e-20) * (fTestQuantity(i) - f) * (fTestQuantity(i) - f);
  }

  // Clean up
  delete[] x;

  return fChi2;
}

//____________________________________________________________________
void TMultiDimFet::MakeCode(const char *filename, Option_t *option) {
  // Generate the file <filename> with .C appended if argument doesn't
  // end in .cxx or .C. The contains the implementation of the
  // function:
  //
  //   Double_t <funcname>(Double_t *x)
  //
  // which does the same as TMultiDimFet::Eval. Please refer to this
  // method.
  //
  // Further, the static variables:
  //
  //     Int_t    gNVariables
  //     Int_t    gNCoefficients
  //     Double_t gDMean
  //     Double_t gXMean[]
  //     Double_t gXMin[]
  //     Double_t gXMax[]
  //     Double_t gCoefficient[]
  //     Int_t    gPower[]
  //
  // are initialized. The only ROOT header file needed is Rtypes.h
  //
  // See TMultiDimFet::MakeRealCode for a list of options

  TString outName(filename);
  if (!outName.EndsWith(".C") && !outName.EndsWith(".cxx"))
    outName += ".C";

  MakeRealCode(outName.Data(), "", option);
}

void TMultiDimFet::MakeCoefficientErrors() {
  // PRIVATE METHOD:
  // Compute the errors on the coefficients. For this to be done, the
  // curvature matrix of the non-orthogonal functions, is computed.
  Int_t i = 0;
  Int_t j = 0;
  Int_t k = 0;
  TVectorD iF(fSampleSize);
  TVectorD jF(fSampleSize);
  fCoefficientsRMS.ResizeTo(fNCoefficients);

  TMatrixDSym curvatureMatrix(fNCoefficients);

  // Build the curvature matrix
  for (i = 0; i < fNCoefficients; i++) {
    iF = TMatrixDRow(fFunctions, i);
    for (j = 0; j <= i; j++) {
      jF = TMatrixDRow(fFunctions, j);
      for (k = 0; k < fSampleSize; k++)
        curvatureMatrix(i, j) += 1 / TMath::Max(fSqError(k), 1e-20) * iF(k) * jF(k);
      curvatureMatrix(j, i) = curvatureMatrix(i, j);
    }
  }

  // Calculate Chi Square
  fChi2 = 0;
  for (i = 0; i < fSampleSize; i++) {
    Double_t f = 0;
    for (j = 0; j < fNCoefficients; j++)
      f += fCoefficients(j) * fFunctions(j, i);
    fChi2 += 1. / TMath::Max(fSqError(i), 1e-20) * (fQuantity(i) - f) * (fQuantity(i) - f);
  }

  // Invert the curvature matrix
  const TVectorD diag = TMatrixDDiag_const(curvatureMatrix);
  curvatureMatrix.NormByDiag(diag);

  TDecompChol chol(curvatureMatrix);
  if (!chol.Decompose())
    Error("MakeCoefficientErrors", "curvature matrix is singular");
  chol.Invert(curvatureMatrix);

  curvatureMatrix.NormByDiag(diag);

  for (i = 0; i < fNCoefficients; i++)
    fCoefficientsRMS(i) = TMath::Sqrt(curvatureMatrix(i, i));
}

//____________________________________________________________________
void TMultiDimFet::MakeCoefficients() {
  // PRIVATE METHOD:
  // Invert the model matrix B, and compute final coefficients. For a
  // more thorough discussion of what this means, please refer to the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  //
  // First we invert the lower triangle matrix fOrthCurvatureMatrix
  // and store the inverted matrix in the upper triangle.

  Int_t i = 0, j = 0;
  Int_t col = 0, row = 0;

  // Invert the B matrix
  for (col = 1; col < fNCoefficients; col++) {
    for (row = col - 1; row > -1; row--) {
      fOrthCurvatureMatrix(row, col) = 0;
      for (i = row; i <= col; i++)
        fOrthCurvatureMatrix(row, col) -= fOrthCurvatureMatrix(i, row) * fOrthCurvatureMatrix(i, col);
    }
  }

  // Compute the final coefficients
  fCoefficients.ResizeTo(fNCoefficients);

  for (i = 0; i < fNCoefficients; i++) {
    Double_t sum = 0;
    for (j = i; j < fNCoefficients; j++)
      sum += fOrthCurvatureMatrix(i, j) * fOrthCoefficients(j);
    fCoefficients(i) = sum;
  }

  // Compute the final residuals
  fResiduals.ResizeTo(fSampleSize);
  for (i = 0; i < fSampleSize; i++)
    fResiduals(i) = fQuantity(i);

  for (i = 0; i < fNCoefficients; i++)
    for (j = 0; j < fSampleSize; j++)
      fResiduals(j) -= fCoefficients(i) * fFunctions(i, j);

  // Compute the max and minimum, and squared sum of the evaluated
  // residuals
  fMinResidual = 10e10;
  fMaxResidual = -10e10;
  Double_t sqRes = 0;
  for (i = 0; i < fSampleSize; i++) {
    sqRes += fResiduals(i) * fResiduals(i);
    if (fResiduals(i) <= fMinResidual) {
      fMinResidual = fResiduals(i);
      fMinResidualRow = i;
    }
    if (fResiduals(i) >= fMaxResidual) {
      fMaxResidual = fResiduals(i);
      fMaxResidualRow = i;
    }
  }

  fCorrelationCoeff = fSumSqResidual / fSumSqAvgQuantity;
  fPrecision = TMath::Sqrt(sqRes / fSumSqQuantity);

  // If we use histograms, fill some more
  if (TESTBIT(fHistogramMask, HIST_RD) || TESTBIT(fHistogramMask, HIST_RTRAI) || TESTBIT(fHistogramMask, HIST_RX)) {
    for (i = 0; i < fSampleSize; i++) {
      if (TESTBIT(fHistogramMask, HIST_RD))
        ((TH2D *)fHistograms->FindObject("res_d"))->Fill(fQuantity(i), fResiduals(i));
      if (TESTBIT(fHistogramMask, HIST_RTRAI))
        ((TH1D *)fHistograms->FindObject("res_train"))->Fill(fResiduals(i));

      if (TESTBIT(fHistogramMask, HIST_RX))
        for (j = 0; j < fNVariables; j++)
          ((TH2D *)fHistograms->FindObject(Form("res_x_%d", j)))->Fill(fVariables(i * fNVariables + j), fResiduals(i));
    }
  }  // If histograms
}

//____________________________________________________________________
void TMultiDimFet::MakeCorrelation() {
  // PRIVATE METHOD:
  // Compute the correlation matrix
  if (!fShowCorrelation)
    return;

  fCorrelationMatrix.ResizeTo(fNVariables, fNVariables + 1);

  Double_t d2 = 0;
  Double_t ddotXi = 0;   // G.Q. needs to be reinitialized in the loop over i fNVariables
  Double_t xiNorm = 0;   // G.Q. needs to be reinitialized in the loop over i fNVariables
  Double_t xidotXj = 0;  // G.Q. needs to be reinitialized in the loop over j fNVariables
  Double_t xjNorm = 0;   // G.Q. needs to be reinitialized in the loop over j fNVariables

  Int_t i, j, k, l, m;  // G.Q. added m variable
  for (i = 0; i < fSampleSize; i++)
    d2 += fQuantity(i) * fQuantity(i);

  for (i = 0; i < fNVariables; i++) {
    ddotXi = 0.;  // G.Q. reinitialisation
    xiNorm = 0.;  // G.Q. reinitialisation
    for (j = 0; j < fSampleSize; j++) {
      // Index of sample j of variable i
      k = j * fNVariables + i;
      ddotXi += fQuantity(j) * (fVariables(k) - fMeanVariables(i));
      xiNorm += (fVariables(k) - fMeanVariables(i)) * (fVariables(k) - fMeanVariables(i));
    }
    fCorrelationMatrix(i, 0) = ddotXi / TMath::Sqrt(d2 * xiNorm);

    for (j = 0; j < i; j++) {
      xidotXj = 0.;  // G.Q. reinitialisation
      xjNorm = 0.;   // G.Q. reinitialisation
      for (k = 0; k < fSampleSize; k++) {
        // Index of sample j of variable i
        // l =  j * fNVariables + k;  // G.Q.
        l = k * fNVariables + j;  // G.Q.
        m = k * fNVariables + i;  // G.Q.
        // G.Q.        xidotXj += (fVariables(i) - fMeanVariables(i))
        // G.Q.          * (fVariables(l) - fMeanVariables(j));
        xidotXj +=
            (fVariables(m) - fMeanVariables(i)) * (fVariables(l) - fMeanVariables(j));  // G.Q. modified index for Xi
        xjNorm += (fVariables(l) - fMeanVariables(j)) * (fVariables(l) - fMeanVariables(j));
      }
      fCorrelationMatrix(i, j + 1) = xidotXj / TMath::Sqrt(xiNorm * xjNorm);
    }
  }
}

//____________________________________________________________________
Double_t TMultiDimFet::MakeGramSchmidt(Int_t function) {
  // PRIVATE METHOD:
  // Make Gram-Schmidt orthogonalisation. The class description gives
  // a thorough account of this algorithm, as well as
  // references. Please refer to the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html

  // calculate w_i, that is, evaluate the current function at data
  // point i
  Double_t f2 = 0;
  fOrthCoefficients(fNCoefficients) = 0;
  fOrthFunctionNorms(fNCoefficients) = 0;
  Int_t j = 0;
  Int_t k = 0;

  for (j = 0; j < fSampleSize; j++) {
    fFunctions(fNCoefficients, j) = 1;
    fOrthFunctions(fNCoefficients, j) = 0;
    // First, however, we need to calculate f_fNCoefficients
    for (k = 0; k < fNVariables; k++) {
      Int_t p = fPowers[function * fNVariables + k];
      Double_t x = fVariables(j * fNVariables + k);
      fFunctions(fNCoefficients, j) *= EvalFactor(p, x);
    }

    // Calculate f dot f in f2
    f2 += fFunctions(fNCoefficients, j) * fFunctions(fNCoefficients, j);
    // Assign to w_fNCoefficients f_fNCoefficients
    fOrthFunctions(fNCoefficients, j) = fFunctions(fNCoefficients, j);
  }

  // the first column of w is equal to f
  for (j = 0; j < fNCoefficients; j++) {
    Double_t fdw = 0;
    // Calculate (f_fNCoefficients dot w_j) / w_j^2
    for (k = 0; k < fSampleSize; k++) {
      fdw += fFunctions(fNCoefficients, k) * fOrthFunctions(j, k) / fOrthFunctionNorms(j);
    }

    fOrthCurvatureMatrix(fNCoefficients, j) = fdw;
    // and subtract it from the current value of w_ij
    for (k = 0; k < fSampleSize; k++)
      fOrthFunctions(fNCoefficients, k) -= fdw * fOrthFunctions(j, k);
  }

  for (j = 0; j < fSampleSize; j++) {
    // calculate squared length of w_fNCoefficients
    fOrthFunctionNorms(fNCoefficients) += fOrthFunctions(fNCoefficients, j) * fOrthFunctions(fNCoefficients, j);

    // calculate D dot w_fNCoefficients in A
    fOrthCoefficients(fNCoefficients) += fQuantity(j) * fOrthFunctions(fNCoefficients, j);
  }

  // First test, but only if didn't user specify
  if (!fIsUserFunction)
    if (TMath::Sqrt(fOrthFunctionNorms(fNCoefficients) / (f2 + 1e-10)) < TMath::Sin(fMinAngle * DEGRAD))
      return 0;

  // The result found by this code for the first residual is always
  // much less then the one found be MUDIFI. That's because it's
  // supposed to be. The cause is the improved precision of Double_t
  // over DOUBLE PRECISION!
  fOrthCurvatureMatrix(fNCoefficients, fNCoefficients) = 1;
  Double_t b = fOrthCoefficients(fNCoefficients);
  fOrthCoefficients(fNCoefficients) /= fOrthFunctionNorms(fNCoefficients);

  // Calculate the residual from including this fNCoefficients.
  Double_t dResidur = fOrthCoefficients(fNCoefficients) * b;

  return dResidur;
}

//____________________________________________________________________
void TMultiDimFet::MakeHistograms(Option_t *option) {
  // Make histograms of the result of the analysis. This message
  // should be sent after having read all data points, but before
  // finding the parameterization
  //
  // Options:
  //     A         All the below
  //     X         Original independent variables
  //     D         Original dependent variables
  //     N         Normalised independent variables
  //     S         Shifted dependent variables
  //     R1        Residuals versus normalised independent variables
  //     R2        Residuals versus dependent variable
  //     R3        Residuals computed on training sample
  //     R4        Residuals computed on test sample
  //
  // For a description of these quantities, refer to
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  TString opt(option);
  opt.ToLower();

  if (opt.Length() < 1)
    return;

  if (!fHistograms)
    fHistograms = new TList;

  // Counter variable
  Int_t i = 0;

  // Histogram of original variables
  if (opt.Contains("x") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_XORIG);
    for (i = 0; i < fNVariables; i++)
      if (!fHistograms->FindObject(Form("x_%d_orig", i)))
        fHistograms->Add(
            new TH1D(Form("x_%d_orig", i), Form("Original variable # %d", i), 100, fMinVariables(i), fMaxVariables(i)));
  }

  // Histogram of original dependent variable
  if (opt.Contains("d") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_DORIG);
    if (!fHistograms->FindObject("d_orig"))
      fHistograms->Add(new TH1D("d_orig", "Original Quantity", 100, fMinQuantity, fMaxQuantity));
  }

  // Histograms of normalized variables
  if (opt.Contains("n") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_XNORM);
    for (i = 0; i < fNVariables; i++)
      if (!fHistograms->FindObject(Form("x_%d_norm", i)))
        fHistograms->Add(new TH1D(Form("x_%d_norm", i), Form("Normalized variable # %d", i), 100, -1, 1));
  }

  // Histogram of shifted dependent variable
  if (opt.Contains("s") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_DSHIF);
    if (!fHistograms->FindObject("d_shifted"))
      fHistograms->Add(
          new TH1D("d_shifted", "Shifted Quantity", 100, fMinQuantity - fMeanQuantity, fMaxQuantity - fMeanQuantity));
  }

  // Residual from training sample versus independent variables
  if (opt.Contains("r1") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_RX);
    for (i = 0; i < fNVariables; i++)
      if (!fHistograms->FindObject(Form("res_x_%d", i)))
        fHistograms->Add(new TH2D(Form("res_x_%d", i),
                                  Form("Computed residual versus x_%d", i),
                                  100,
                                  -1,
                                  1,
                                  35,
                                  fMinQuantity - fMeanQuantity,
                                  fMaxQuantity - fMeanQuantity));
  }

  // Residual from training sample versus. dependent variable
  if (opt.Contains("r2") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_RD);
    if (!fHistograms->FindObject("res_d"))
      fHistograms->Add(new TH2D("res_d",
                                "Computed residuals vs Quantity",
                                100,
                                fMinQuantity - fMeanQuantity,
                                fMaxQuantity - fMeanQuantity,
                                35,
                                fMinQuantity - fMeanQuantity,
                                fMaxQuantity - fMeanQuantity));
  }

  // Residual from training sample
  if (opt.Contains("r3") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_RTRAI);
    if (!fHistograms->FindObject("res_train"))
      fHistograms->Add(new TH1D("res_train",
                                "Computed residuals over training sample",
                                100,
                                fMinQuantity - fMeanQuantity,
                                fMaxQuantity - fMeanQuantity));
  }
  if (opt.Contains("r4") || opt.Contains("a")) {
    SETBIT(fHistogramMask, HIST_RTEST);
    if (!fHistograms->FindObject("res_test"))
      fHistograms->Add(new TH1D("res_test",
                                "Distribution of residuals from test",
                                100,
                                fMinQuantity - fMeanQuantity,
                                fMaxQuantity - fMeanQuantity));
  }
}

//____________________________________________________________________
void TMultiDimFet::MakeMethod(const Char_t *classname, Option_t *option) {
  // Generate the file <classname>MDF.cxx which contains the
  // implementation of the method:
  //
  //   Double_t <classname>::MDF(Double_t *x)
  //
  // which does the same as  TMultiDimFet::Eval. Please refer to this
  // method.
  //
  // Further, the public static members:
  //
  //   Int_t    <classname>::fgNVariables
  //   Int_t    <classname>::fgNCoefficients
  //   Double_t <classname>::fgDMean
  //   Double_t <classname>::fgXMean[]       //[fgNVariables]
  //   Double_t <classname>::fgXMin[]        //[fgNVariables]
  //   Double_t <classname>::fgXMax[]        //[fgNVariables]
  //   Double_t <classname>::fgCoefficient[] //[fgNCoeffficents]
  //   Int_t    <classname>::fgPower[]       //[fgNCoeffficents*fgNVariables]
  //
  // are initialized, and assumed to exist. The class declaration is
  // assumed to be in <classname>.h and assumed to be provided by the
  // user.
  //
  // See TMultiDimFet::MakeRealCode for a list of options
  //
  // The minimal class definition is:
  //
  //   class <classname> {
  //   public:
  //     Int_t    <classname>::fgNVariables;     // Number of variables
  //     Int_t    <classname>::fgNCoefficients;  // Number of terms
  //     Double_t <classname>::fgDMean;          // Mean from training sample
  //     Double_t <classname>::fgXMean[];        // Mean from training sample
  //     Double_t <classname>::fgXMin[];         // Min from training sample
  //     Double_t <classname>::fgXMax[];         // Max from training sample
  //     Double_t <classname>::fgCoefficient[];  // Coefficients
  //     Int_t    <classname>::fgPower[];        // Function powers
  //
  //     Double_t Eval(Double_t *x);
  //   };
  //
  // Whether the method <classname>::Eval should be static or not, is
  // up to the user.

  MakeRealCode(Form("%sMDF.cxx", classname), classname, option);
}

//____________________________________________________________________
void TMultiDimFet::MakeNormalized() {
  // PRIVATE METHOD:
  // Normalize data to the interval [-1;1]. This is needed for the
  // classes method to work.

  Int_t i = 0;
  Int_t j = 0;
  Int_t k = 0;

  for (i = 0; i < fSampleSize; i++) {
    if (TESTBIT(fHistogramMask, HIST_DORIG))
      ((TH1D *)fHistograms->FindObject("d_orig"))->Fill(fQuantity(i));

    fQuantity(i) -= fMeanQuantity;
    fSumSqAvgQuantity += fQuantity(i) * fQuantity(i);

    if (TESTBIT(fHistogramMask, HIST_DSHIF))
      ((TH1D *)fHistograms->FindObject("d_shifted"))->Fill(fQuantity(i));

    for (j = 0; j < fNVariables; j++) {
      Double_t range = 1. / (fMaxVariables(j) - fMinVariables(j));
      k = i * fNVariables + j;

      // Fill histograms of original independent variables
      if (TESTBIT(fHistogramMask, HIST_XORIG))
        ((TH1D *)fHistograms->FindObject(Form("x_%d_orig", j)))->Fill(fVariables(k));

      // Normalise independent variables
      fVariables(k) = 1 + 2 * range * (fVariables(k) - fMaxVariables(j));

      // Fill histograms of normalised independent variables
      if (TESTBIT(fHistogramMask, HIST_XNORM))
        ((TH1D *)fHistograms->FindObject(Form("x_%d_norm", j)))->Fill(fVariables(k));
    }
  }
  // Shift min and max of dependent variable
  fMaxQuantity -= fMeanQuantity;
  fMinQuantity -= fMeanQuantity;

  // Shift mean of independent variables
  for (i = 0; i < fNVariables; i++) {
    Double_t range = 1. / (fMaxVariables(i) - fMinVariables(i));
    fMeanVariables(i) = 1 + 2 * range * (fMeanVariables(i) - fMaxVariables(i));
  }
}

//____________________________________________________________________
void TMultiDimFet::MakeParameterization() {
  // PRIVATE METHOD:
  // Find the parameterization over the training sample. A full account
  // of the algorithm is given in the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html

  Int_t i = -1;
  Int_t j = 0;
  Int_t k = 0;
  Int_t maxPass = 3;
  Int_t studied = 0;
  Double_t squareResidual = fSumSqAvgQuantity;
  fNCoefficients = 0;
  fSumSqResidual = fSumSqAvgQuantity;
  fFunctions.ResizeTo(fMaxTerms, fSampleSize);
  fOrthFunctions.ResizeTo(fMaxTerms, fSampleSize);
  fOrthFunctionNorms.ResizeTo(fMaxTerms);
  fOrthCoefficients.ResizeTo(fMaxTerms);
  fOrthCurvatureMatrix.ResizeTo(fMaxTerms, fMaxTerms);
  fFunctions = 1;

  fFunctionCodes.resize(fMaxFunctions);
  fPowerIndex.resize(fMaxTerms);
  Int_t l;
  for (l = 0; l < fMaxFunctions; l++)
    fFunctionCodes[l] = 0;
  for (l = 0; l < fMaxTerms; l++)
    fPowerIndex[l] = 0;

  if (fMaxAngle != 0)
    maxPass = 100;
  if (fIsUserFunction)
    maxPass = 1;

  // Loop over the number of functions we want to study.
  // increment inspection counter
  while (kTRUE) {
    // Reach user defined limit of studies
    if (studied++ >= fMaxStudy) {
      fParameterisationCode = PARAM_MAXSTUDY;
      break;
    }

    // Considered all functions several times
    if (k >= maxPass) {
      fParameterisationCode = PARAM_SEVERAL;
      break;
    }

    // increment function counter
    i++;

    // If we've reached the end of the functions, restart pass
    if (i == fMaxFunctions) {
      if (fMaxAngle != 0)
        fMaxAngle += (90 - fMaxAngle) / 2;
      i = 0;
      studied--;
      k++;
      continue;
    }
    if (studied == 1)
      fFunctionCodes[i] = 0;
    else if (fFunctionCodes[i] >= 2)
      continue;

    // Print a happy message
    if (fIsVerbose && studied == 1)
      edm::LogInfo("TMultiDimFet") << "Coeff   SumSqRes    Contrib   Angle      QM   Func"
                                   << "     Value        W^2  Powers"
                                   << "\n";

    // Make the Gram-Schmidt
    Double_t dResidur = MakeGramSchmidt(i);

    if (dResidur == 0) {
      // This function is no good!
      // First test is in MakeGramSchmidt
      fFunctionCodes[i] = 1;
      continue;
    }

    // If user specified function, assume she/he knows what he's doing
    if (!fIsUserFunction) {
      // Flag this function as considered
      fFunctionCodes[i] = 2;

      // Test if this function contributes to the fit
      if (!TestFunction(squareResidual, dResidur)) {
        fFunctionCodes[i] = 1;
        continue;
      }
    }

    // If we get to here, the function currently considered is
    // fNCoefficients, so we increment the counter
    // Flag this function as OK, and store and the number in the
    // index.
    fFunctionCodes[i] = 3;
    fPowerIndex[fNCoefficients] = i;
    fNCoefficients++;

    // We add the current contribution to the sum of square of
    // residuals;
    squareResidual -= dResidur;

    // Calculate control parameter from this function
    for (j = 0; j < fNVariables; j++) {
      if (fNCoefficients == 1 || fMaxPowersFinal[j] <= fPowers[i * fNVariables + j] - 1)
        fMaxPowersFinal[j] = fPowers[i * fNVariables + j] - 1;
    }
    Double_t s = EvalControl(&fPowers[i * fNVariables]);

    // Print the statistics about this function
    if (fIsVerbose) {
      edm::LogVerbatim("TMultiDimFet") << std::setw(5) << fNCoefficients << " " << std::setw(10) << std::setprecision(4)
                                       << squareResidual << " " << std::setw(10) << std::setprecision(4) << dResidur
                                       << " " << std::setw(7) << std::setprecision(3) << fMaxAngle << " "
                                       << std::setw(7) << std::setprecision(3) << s << " " << std::setw(5) << i << " "
                                       << std::setw(10) << std::setprecision(4) << fOrthCoefficients(fNCoefficients - 1)
                                       << " " << std::setw(10) << std::setprecision(4)
                                       << fOrthFunctionNorms(fNCoefficients - 1) << " " << std::flush;
      for (j = 0; j < fNVariables; j++)
        edm::LogInfo("TMultiDimFet") << " " << fPowers[i * fNVariables + j] - 1 << std::flush;
      edm::LogInfo("TMultiDimFet") << "\n";
    }

    if (fNCoefficients >= fMaxTerms /* && fIsVerbose */) {
      fParameterisationCode = PARAM_MAXTERMS;
      break;
    }

    Double_t err = TMath::Sqrt(TMath::Max(1e-20, squareResidual) / fSumSqAvgQuantity);
    if (err < fMinRelativeError) {
      fParameterisationCode = PARAM_RELERR;
      break;
    }
  }

  fError = TMath::Max(1e-20, squareResidual);
  fSumSqResidual -= fError;
  fRMS = TMath::Sqrt(fError / fSampleSize);
}

//____________________________________________________________________
void TMultiDimFet::MakeRealCode(const char *filename, const char *classname, Option_t *) {
  // PRIVATE METHOD:
  // This is the method that actually generates the code for the
  // evaluation the parameterization on some point.
  // It's called by TMultiDimFet::MakeCode and TMultiDimFet::MakeMethod.
  //
  // The options are: NONE so far
  Int_t i, j;

  Bool_t isMethod = (classname[0] == '\0' ? kFALSE : kTRUE);
  const char *prefix = (isMethod ? Form("%s::", classname) : "");
  const char *cv_qual = (isMethod ? "" : "static ");

  std::ofstream outFile(filename, std::ios::out | std::ios::trunc);
  if (!outFile) {
    Error("MakeRealCode", "couldn't open output file '%s'", filename);
    return;
  }

  if (fIsVerbose)
    edm::LogInfo("TMultiDimFet") << "Writing on file \"" << filename << "\" ... " << std::flush;
  //
  // Write header of file
  //
  // Emacs mode line ;-)
  outFile << "// -*- mode: c++ -*-"
          << "\n";
  // Info about creator
  outFile << "// "
          << "\n"
          << "// File " << filename << " generated by TMultiDimFet::MakeRealCode"
          << "\n";
  // Time stamp
  TDatime date;
  outFile << "// on " << date.AsString() << "\n";
  // ROOT version info
  outFile << "// ROOT version " << gROOT->GetVersion() << "\n"
          << "//"
          << "\n";
  // General information on the code
  outFile << "// This file contains the function "
          << "\n"
          << "//"
          << "\n"
          << "//    double  " << prefix << "MDF(double *x); "
          << "\n"
          << "//"
          << "\n"
          << "// For evaluating the parameterization obtained"
          << "\n"
          << "// from TMultiDimFet and the point x"
          << "\n"
          << "// "
          << "\n"
          << "// See TMultiDimFet class documentation for more "
          << "information "
          << "\n"
          << "// "
          << "\n";
  // Header files
  if (isMethod)
    // If these are methods, we need the class header
    outFile << "#include \"" << classname << ".h\""
            << "\n";

  //
  // Now for the data
  //
  outFile << "//"
          << "\n"
          << "// Static data variables"
          << "\n"
          << "//"
          << "\n";
  outFile << cv_qual << "int    " << prefix << "gNVariables    = " << fNVariables << ";"
          << "\n";
  outFile << cv_qual << "int    " << prefix << "gNCoefficients = " << fNCoefficients << ";"
          << "\n";
  outFile << cv_qual << "double " << prefix << "gDMean         = " << fMeanQuantity << ";"
          << "\n";

  // Assignment to mean vector.
  outFile << "// Assignment to mean vector."
          << "\n";
  outFile << cv_qual << "double " << prefix << "gXMean[] = {"
          << "\n";
  for (i = 0; i < fNVariables; i++)
    outFile << (i != 0 ? ", " : "  ") << fMeanVariables(i) << std::flush;
  outFile << " };"
          << "\n"
          << "\n";

  // Assignment to minimum vector.
  outFile << "// Assignment to minimum vector."
          << "\n";
  outFile << cv_qual << "double " << prefix << "gXMin[] = {"
          << "\n";
  for (i = 0; i < fNVariables; i++)
    outFile << (i != 0 ? ", " : "  ") << fMinVariables(i) << std::flush;
  outFile << " };"
          << "\n"
          << "\n";

  // Assignment to maximum vector.
  outFile << "// Assignment to maximum vector."
          << "\n";
  outFile << cv_qual << "double " << prefix << "gXMax[] = {"
          << "\n";
  for (i = 0; i < fNVariables; i++)
    outFile << (i != 0 ? ", " : "  ") << fMaxVariables(i) << std::flush;
  outFile << " };"
          << "\n"
          << "\n";

  // Assignment to coefficients vector.
  outFile << "// Assignment to coefficients vector."
          << "\n";
  outFile << cv_qual << "double " << prefix << "gCoefficient[] = {" << std::flush;
  for (i = 0; i < fNCoefficients; i++)
    outFile << (i != 0 ? "," : "") << "\n"
            << "  " << fCoefficients(i) << std::flush;
  outFile << "\n"
          << " };"
          << "\n"
          << "\n";

  // Assignment to powers vector.
  outFile << "// Assignment to powers vector."
          << "\n"
          << "// The powers are stored row-wise, that is"
          << "\n"
          << "//  p_ij = " << prefix << "gPower[i * NVariables + j];"
          << "\n";
  outFile << cv_qual << "int    " << prefix << "gPower[] = {" << std::flush;
  for (i = 0; i < fNCoefficients; i++) {
    for (j = 0; j < fNVariables; j++) {
      if (j != 0)
        outFile << std::flush << "  ";
      else
        outFile << "\n"
                << "  ";
      outFile << fPowers[fPowerIndex[i] * fNVariables + j]
              << (i == fNCoefficients - 1 && j == fNVariables - 1 ? "" : ",") << std::flush;
    }
  }
  outFile << "\n"
          << "};"
          << "\n"
          << "\n";

  //
  // Finally we reach the function itself
  //
  outFile << "// "
          << "\n"
          << "// The " << (isMethod ? "method " : "function ") << "  double " << prefix << "MDF(double *x)"
          << "\n"
          << "// "
          << "\n";
  outFile << "double " << prefix << "MDF(double *x) {"
          << "\n"
          << "  double returnValue = " << prefix << "gDMean;"
          << "\n"
          << "  int    i = 0, j = 0, k = 0;"
          << "\n"
          << "  for (i = 0; i < " << prefix << "gNCoefficients ; i++) {"
          << "\n"
          << "    // Evaluate the ith term in the expansion"
          << "\n"
          << "    double term = " << prefix << "gCoefficient[i];"
          << "\n"
          << "    for (j = 0; j < " << prefix << "gNVariables; j++) {"
          << "\n"
          << "      // Evaluate the polynomial in the jth variable."
          << "\n"
          << "      int power = " << prefix << "gPower[" << prefix << "gNVariables * i + j]; "
          << "\n"
          << "      double p1 = 1, p2 = 0, p3 = 0, r = 0;"
          << "\n"
          << "      double v =  1 + 2. / (" << prefix << "gXMax[j] - " << prefix << "gXMin[j]) * (x[j] - " << prefix
          << "gXMax[j]);"
          << "\n"
          << "      // what is the power to use!"
          << "\n"
          << "      switch(power) {"
          << "\n"
          << "      case 1: r = 1; break; "
          << "\n"
          << "      case 2: r = v; break; "
          << "\n"
          << "      default: "
          << "\n"
          << "        p2 = v; "
          << "\n"
          << "        for (k = 3; k <= power; k++) { "
          << "\n"
          << "          p3 = p2 * v;"
          << "\n";
  if (fPolyType == kLegendre)
    outFile << "          p3 = ((2 * i - 3) * p2 * v - (i - 2) * p1)"
            << " / (i - 1);"
            << "\n";
  if (fPolyType == kChebyshev)
    outFile << "          p3 = 2 * v * p2 - p1; "
            << "\n";
  outFile << "          p1 = p2; p2 = p3; "
          << "\n"
          << "        }"
          << "\n"
          << "        r = p3;"
          << "\n"
          << "      }"
          << "\n"
          << "      // multiply this term by the poly in the jth var"
          << "\n"
          << "      term *= r; "
          << "\n"
          << "    }"
          << "\n"
          << "    // Add this term to the final result"
          << "\n"
          << "    returnValue += term;"
          << "\n"
          << "  }"
          << "\n"
          << "  return returnValue;"
          << "\n"
          << "}"
          << "\n"
          << "\n";

  // EOF
  outFile << "// EOF for " << filename << "\n";

  // Close the file
  outFile.close();

  if (fIsVerbose)
    edm::LogInfo("TMultiDimFet") << "done"
                                 << "\n";
}

//____________________________________________________________________
void TMultiDimFet::Print(Option_t *option) const {
  // Print statistics etc.
  // Options are
  //   P        Parameters
  //   S        Statistics
  //   C        Coefficients
  //   R        Result of parameterisation
  //   F        Result of fit
  //   K        Correlation Matrix
  //   M        Pretty print formula
  //
  Int_t i = 0;
  Int_t j = 0;

  TString opt(option);
  opt.ToLower();

  if (opt.Contains("p")) {
    // Print basic parameters for this object
    edm::LogInfo("TMultiDimFet") << "User parameters:"
                                 << "\n"
                                 << "----------------"
                                 << "\n"
                                 << " Variables:                    " << fNVariables << "\n"
                                 << " Data points:                  " << fSampleSize << "\n"
                                 << " Max Terms:                    " << fMaxTerms << "\n"
                                 << " Power Limit Parameter:        " << fPowerLimit << "\n"
                                 << " Max functions:                " << fMaxFunctions << "\n"
                                 << " Max functions to study:       " << fMaxStudy << "\n"
                                 << " Max angle (optional):         " << fMaxAngle << "\n"
                                 << " Min angle:                    " << fMinAngle << "\n"
                                 << " Relative Error accepted:      " << fMinRelativeError << "\n"
                                 << " Maximum Powers:               " << std::flush;
    for (i = 0; i < fNVariables; i++)
      edm::LogInfo("TMultiDimFet") << " " << fMaxPowers[i] - 1 << std::flush;
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << "\n"
                                 << " Parameterisation will be done using " << std::flush;
    if (fPolyType == kChebyshev)
      edm::LogInfo("TMultiDimFet") << "Chebyshev polynomials"
                                   << "\n";
    else if (fPolyType == kLegendre)
      edm::LogInfo("TMultiDimFet") << "Legendre polynomials"
                                   << "\n";
    else
      edm::LogInfo("TMultiDimFet") << "Monomials"
                                   << "\n";
    edm::LogInfo("TMultiDimFet") << "\n";
  }

  if (opt.Contains("s")) {
    // Print statistics for read data
    edm::LogInfo("TMultiDimFet") << "Sample statistics:"
                                 << "\n"
                                 << "------------------"
                                 << "\n"
                                 << "                 D" << std::flush;
    for (i = 0; i < fNVariables; i++)
      edm::LogInfo("TMultiDimFet") << " " << std::setw(10) << i + 1 << std::flush;
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << " Max:   " << std::setw(10) << std::setprecision(7) << fMaxQuantity << std::flush;
    for (i = 0; i < fNVariables; i++)
      edm::LogInfo("TMultiDimFet") << " " << std::setw(10) << std::setprecision(4) << fMaxVariables(i) << std::flush;
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << " Min:   " << std::setw(10) << std::setprecision(7) << fMinQuantity << std::flush;
    for (i = 0; i < fNVariables; i++)
      edm::LogInfo("TMultiDimFet") << " " << std::setw(10) << std::setprecision(4) << fMinVariables(i) << std::flush;
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << " Mean:  " << std::setw(10) << std::setprecision(7) << fMeanQuantity << std::flush;
    for (i = 0; i < fNVariables; i++)
      edm::LogInfo("TMultiDimFet") << " " << std::setw(10) << std::setprecision(4) << fMeanVariables(i) << std::flush;
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << " Function Sum Squares:         " << fSumSqQuantity << "\n"
                                 << "\n";
  }

  if (opt.Contains("r")) {
    edm::LogInfo("TMultiDimFet") << "Results of Parameterisation:"
                                 << "\n"
                                 << "----------------------------"
                                 << "\n"
                                 << " Total reduction of square residuals    " << fSumSqResidual << "\n"
                                 << " Relative precision obtained:           " << fPrecision << "\n"
                                 << " Error obtained:                        " << fError << "\n"
                                 << " Multiple correlation coefficient:      " << fCorrelationCoeff << "\n"
                                 << " Reduced Chi square over sample:        " << fChi2 / (fSampleSize - fNCoefficients)
                                 << "\n"
                                 << " Maximum residual value:                " << fMaxResidual << "\n"
                                 << " Minimum residual value:                " << fMinResidual << "\n"
                                 << " Estimated root mean square:            " << fRMS << "\n"
                                 << " Maximum powers used:                   " << std::flush;
    for (j = 0; j < fNVariables; j++)
      edm::LogInfo("TMultiDimFet") << fMaxPowersFinal[j] << " " << std::flush;
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << " Function codes of candidate functions."
                                 << "\n"
                                 << "  1: considered,"
                                 << "  2: too little contribution,"
                                 << "  3: accepted." << std::flush;
    for (i = 0; i < fMaxFunctions; i++) {
      if (i % 60 == 0)
        edm::LogInfo("TMultiDimFet") << "\n"
                                     << " " << std::flush;
      else if (i % 10 == 0)
        edm::LogInfo("TMultiDimFet") << " " << std::flush;
      edm::LogInfo("TMultiDimFet") << fFunctionCodes[i];
    }
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << " Loop over candidates stopped because " << std::flush;
    switch (fParameterisationCode) {
      case PARAM_MAXSTUDY:
        edm::LogInfo("TMultiDimFet") << "max allowed studies reached"
                                     << "\n";
        break;
      case PARAM_SEVERAL:
        edm::LogInfo("TMultiDimFet") << "all candidates considered several times"
                                     << "\n";
        break;
      case PARAM_RELERR:
        edm::LogInfo("TMultiDimFet") << "wanted relative error obtained"
                                     << "\n";
        break;
      case PARAM_MAXTERMS:
        edm::LogInfo("TMultiDimFet") << "max number of terms reached"
                                     << "\n";
        break;
      default:
        edm::LogInfo("TMultiDimFet") << "some unknown reason"
                                     << "\n";
        break;
    }
    edm::LogInfo("TMultiDimFet") << "\n";
  }

  if (opt.Contains("f")) {
    edm::LogInfo("TMultiDimFet") << "Results of Fit:"
                                 << "\n"
                                 << "---------------"
                                 << "\n"
                                 << " Test sample size:                      " << fTestSampleSize << "\n"
                                 << " Multiple correlation coefficient:      " << fTestCorrelationCoeff << "\n"
                                 << " Relative precision obtained:           " << fTestPrecision << "\n"
                                 << " Error obtained:                        " << fTestError << "\n"
                                 << " Reduced Chi square over sample:        " << fChi2 / (fSampleSize - fNCoefficients)
                                 << "\n"
                                 << "\n";
    /*
      if (fFitter) {
         fFitter->PrintResults(1,1);
         edm::LogInfo("TMultiDimFet") << "\n";
      }
*/
  }

  if (opt.Contains("c")) {
    edm::LogInfo("TMultiDimFet") << "Coefficients:"
                                 << "\n"
                                 << "-------------"
                                 << "\n"
                                 << "   #         Value        Error   Powers"
                                 << "\n"
                                 << " ---------------------------------------"
                                 << "\n";
    for (i = 0; i < fNCoefficients; i++) {
      edm::LogInfo("TMultiDimFet") << " " << std::setw(3) << i << "  " << std::setw(12) << fCoefficients(i) << "  "
                                   << std::setw(12) << fCoefficientsRMS(i) << "  " << std::flush;
      for (j = 0; j < fNVariables; j++)
        edm::LogInfo("TMultiDimFet") << " " << std::setw(3) << fPowers[fPowerIndex[i] * fNVariables + j] - 1
                                     << std::flush;
      edm::LogInfo("TMultiDimFet") << "\n";
    }
    edm::LogInfo("TMultiDimFet") << "\n";
  }
  if (opt.Contains("k") && fCorrelationMatrix.IsValid()) {
    edm::LogInfo("TMultiDimFet") << "Correlation Matrix:"
                                 << "\n"
                                 << "-------------------";
    fCorrelationMatrix.Print();
  }

  if (opt.Contains("m")) {
    edm::LogInfo("TMultiDimFet") << std::setprecision(25);
    edm::LogInfo("TMultiDimFet") << "Parameterization:"
                                 << "\n"
                                 << "-----------------"
                                 << "\n"
                                 << "  Normalised variables: "
                                 << "\n";
    for (i = 0; i < fNVariables; i++)
      edm::LogInfo("TMultiDimFet") << "\ty" << i << "\t:= 1 + 2 * (x" << i << " - " << fMaxVariables(i) << ") / ("
                                   << fMaxVariables(i) << " - " << fMinVariables(i) << ")"
                                   << "\n";
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << "  f[";
    for (i = 0; i < fNVariables; i++) {
      edm::LogInfo("TMultiDimFet") << "y" << i;
      if (i != fNVariables - 1)
        edm::LogInfo("TMultiDimFet") << ", ";
    }
    edm::LogInfo("TMultiDimFet") << "] := ";
    for (Int_t i = 0; i < fNCoefficients; i++) {
      if (i != 0)
        edm::LogInfo("TMultiDimFet") << " " << (fCoefficients(i) < 0 ? "- " : "+ ") << TMath::Abs(fCoefficients(i));
      else
        edm::LogInfo("TMultiDimFet") << fCoefficients(i);
      for (Int_t j = 0; j < fNVariables; j++) {
        Int_t p = fPowers[fPowerIndex[i] * fNVariables + j];
        switch (p) {
          case 1:
            break;
          case 2:
            edm::LogInfo("TMultiDimFet") << " * y" << j;
            break;
          default:
            switch (fPolyType) {
              case kLegendre:
                edm::LogInfo("TMultiDimFet") << " * L" << p - 1 << "(y" << j << ")";
                break;
              case kChebyshev:
                edm::LogInfo("TMultiDimFet") << " * C" << p - 1 << "(y" << j << ")";
                break;
              default:
                edm::LogInfo("TMultiDimFet") << " * y" << j << "^" << p - 1;
                break;
            }
        }
      }
    }
    edm::LogInfo("TMultiDimFet") << "\n";
  }
}

//____________________________________________________________________
void TMultiDimFet::PrintPolynomialsSpecial(Option_t *option) const {
  //   M        Pretty print formula
  //
  Int_t i = 0;
  //   Int_t j = 0;

  TString opt(option);
  opt.ToLower();

  if (opt.Contains("m")) {
    edm::LogInfo("TMultiDimFet") << std::setprecision(25);
    edm::LogInfo("TMultiDimFet") << "Parameterization:"
                                 << "\n"
                                 << "-----------------"
                                 << "\n"
                                 << "  Normalised variables: "
                                 << "\n";
    for (i = 0; i < fNVariables; i++)
      edm::LogInfo("TMultiDimFet") << "\tdouble y" << i << "\t=1+2*(x" << i << "-" << fMaxVariables(i) << ")/("
                                   << fMaxVariables(i) << "-" << fMinVariables(i) << ");"
                                   << "\n";
    edm::LogInfo("TMultiDimFet") << "\n"
                                 << "  f[";
    for (i = 0; i < fNVariables; i++) {
      edm::LogInfo("TMultiDimFet") << "y" << i;
      if (i != fNVariables - 1)
        edm::LogInfo("TMultiDimFet") << ", ";
    }
    edm::LogInfo("TMultiDimFet") << "] := " << fMeanQuantity << " + ";
    for (Int_t i = 0; i < fNCoefficients; i++) {
      if (i != 0)
        edm::LogInfo("TMultiDimFet") << " " << (fCoefficients(i) < 0 ? "-" : "+") << TMath::Abs(fCoefficients(i));
      else
        edm::LogInfo("TMultiDimFet") << fCoefficients(i);
      for (Int_t j = 0; j < fNVariables; j++) {
        Int_t p = fPowers[fPowerIndex[i] * fNVariables + j];
        switch (p) {
          case 1:
            break;
          case 2:
            edm::LogInfo("TMultiDimFet") << "*y" << j;
            break;
          default:
            switch (fPolyType) {
              case kLegendre:
                edm::LogInfo("TMultiDimFet") << "*Leg(" << p - 1 << ",y" << j << ")";
                break;
              case kChebyshev:
                edm::LogInfo("TMultiDimFet") << "*C" << p - 1 << "(y" << j << ")";
                break;
              default:
                edm::LogInfo("TMultiDimFet") << "*y" << j << "**" << p - 1;
                break;
            }
        }
      }
      edm::LogInfo("TMultiDimFet") << "\n";
    }
    edm::LogInfo("TMultiDimFet") << "\n";
  }
}

//____________________________________________________________________
Bool_t TMultiDimFet::Select(const Int_t *) {
  // Selection method. User can override this method for specialized
  // selection of acceptable functions in fit. Default is to select
  // all. This message is sent during the build-up of the function
  // candidates table once for each set of powers in
  // variables. Notice, that the argument array contains the powers
  // PLUS ONE. For example, to De select the function
  //     f = x1^2 * x2^4 * x3^5,
  // this method should return kFALSE if given the argument
  //     { 3, 4, 6 }
  return kTRUE;
}

//____________________________________________________________________
void TMultiDimFet::SetMaxAngle(Double_t ang) {
  // Set the max angle (in degrees) between the initial data vector to
  // be fitted, and the new candidate function to be included in the
  // fit.  By default it is 0, which automatically chooses another
  // selection criteria. See also
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  if (ang >= 90 || ang < 0) {
    Warning("SetMaxAngle", "angle must be in [0,90)");
    return;
  }

  fMaxAngle = ang;
}

//____________________________________________________________________
void TMultiDimFet::SetMinAngle(Double_t ang) {
  // Set the min angle (in degrees) between a new candidate function
  // and the subspace spanned by the previously accepted
  // functions. See also
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  if (ang > 90 || ang <= 0) {
    Warning("SetMinAngle", "angle must be in [0,90)");
    return;
  }

  fMinAngle = ang;
}

//____________________________________________________________________
void TMultiDimFet::SetPowers(const Int_t *powers, Int_t terms) {
  // Define a user function. The input array must be of the form
  // (p11, ..., p1N, ... ,pL1, ..., pLN)
  // Where N is the dimension of the data sample, L is the number of
  // terms (given in terms) and the first number, labels the term, the
  // second the variable.  More information is given in the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  fIsUserFunction = kTRUE;
  fMaxFunctions = terms;
  fMaxTerms = terms;
  fMaxStudy = terms;
  fMaxFunctionsTimesNVariables = fMaxFunctions * fNVariables;
  fPowers.resize(fMaxFunctions * fNVariables);
  Int_t i, j;
  for (i = 0; i < fMaxFunctions; i++)
    for (j = 0; j < fNVariables; j++)
      fPowers[i * fNVariables + j] = powers[i * fNVariables + j] + 1;
}

//____________________________________________________________________
void TMultiDimFet::SetPowerLimit(Double_t limit) {
  // Set the user parameter for the function selection. The bigger the
  // limit, the more functions are used. The meaning of this variable
  // is defined in the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  fPowerLimit = limit;
}

//____________________________________________________________________
void TMultiDimFet::SetMaxPowers(const Int_t *powers) {
  // Set the maximum power to be considered in the fit for each
  // variable. See also
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  if (!powers)
    return;

  for (Int_t i = 0; i < fNVariables; i++)
    fMaxPowers[i] = powers[i] + 1;
}

//____________________________________________________________________
void TMultiDimFet::SetMinRelativeError(Double_t error) {
  // Set the acceptable relative error for when sum of square
  // residuals is considered minimized. For a full account, refer to
  // the
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html
  fMinRelativeError = error;
}

//____________________________________________________________________
Bool_t TMultiDimFet::TestFunction(Double_t squareResidual, Double_t dResidur) {
  // PRIVATE METHOD:
  // Test whether the currently considered function contributes to the
  // fit. See also
  // Begin_Html<a href="#TMultiDimFet:description">class description</a>End_Html

  if (fNCoefficients != 0) {
    // Now for the second test:
    if (fMaxAngle == 0) {
      // If the user hasn't supplied a max angle do the test as,
      if (dResidur < squareResidual / (fMaxTerms - fNCoefficients + 1 + 1E-10)) {
        return kFALSE;
      }
    } else {
      // If the user has provided a max angle, test if the calculated
      // angle is less then the max angle.
      if (TMath::Sqrt(dResidur / fSumSqAvgQuantity) < TMath::Cos(fMaxAngle * DEGRAD)) {
        return kFALSE;
      }
    }
  }
  // If we get here, the function is OK
  return kTRUE;
}

//____________________________________________________________________
//void mdfHelper(int& /*npar*/, double* /*divs*/, double& chi2,
//               double* coeffs, int /*flag*/)
/*{
   // Helper function for doing the minimisation of Chi2 using Minuit

   // Get pointer  to current TMultiDimFet object.
  // TMultiDimFet* mdf = TMultiDimFet::Instance();
   //chi2     = mdf->MakeChi2(coeffs);
}
*/
