#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include "Utilities/Timing/interface/TimingReport.h"
// #include "Utilities/Timing/interface/TimerStack.h"

// #include "TROOT.h"
// #include "TMath.h"
#include "Math/ProbFuncMathCore.h"
#include "Math/PdfFuncMathCore.h"
#include "Math/QuantFuncMathCore.h"

#include <iostream>
#include <cmath>

#include <map>
#include <functional>
#include <numeric>
#include <algorithm>


double GaussianSumUtilities1D::pdf(unsigned int i)  const {
  return weight(i)*gauss(x,mean(i),standardDeviation(i));
}


double
GaussianSumUtilities1D::quantile (const double q) const
{
  return ROOT::Math::gaussian_quantile(q,1.);
}


// double
// GaussianSumUtilities1D::quantile (const double q) const
// {
//   //
//   // mean and sigma of highest weight component
//   //
//   int iwMax(-1);
//   float wMax(0.);
//   for ( unsigned int i=0; i<size(); i++ ) {
//     if ( weight(i)>wMax ) {
//       iwMax = i;
//       wMax = weight(i);
//     }
//   }
//   //
//   // Find start values: begin with mean and
//   // go towards f(x)=0 until two points on
//   // either side are found (assumes monotonously
//   // increasing function)
//   //
//   double x1(mean(iwMax));
//   double y1(cdf(x1)-q);
//   double dx = y1>0. ? -standardDeviation(iwMax) : standardDeviation(iwMax);
//   double x2(x1+dx);
//   double y2(cdf(x2)-q);
//   while ( y1*y2>0. ) {
//     x1 = x2;
//     y1 = y2;
//     x2 += dx;
//     y2 = cdf(x2) - q;
//   }
//   //
//   // now use bisection to find final value
//   //
//   double x(0.);
//   while ( true ) {
//     // use linear interpolation
//     x = -(x2*y1-x1*y2)/(y2-y1);
//     double y = cdf(x) - q;
//     if ( fabs(y)<1.e-6 )  break;
//     if ( y*y1>0. ) {
//       y1 = y;
//       x1 = x;
//     }
//     else {
//       y2 = y;
//       x2 = x;
//     }
//   }
//   return x;
// }

bool
GaussianSumUtilities1D::modeIsValid () const
{
  if ( theModeStatus==NotComputed )  computeMode();
  return theModeStatus==Valid;
}

const SingleGaussianState1D&
GaussianSumUtilities1D::mode () const
{
  if ( theModeStatus==NotComputed )  computeMode();
//     std::cout << "Mode calculation failed!!" << std::endl;
  return theMode;
}

void
GaussianSumUtilities1D::computeMode () const
{
//   TimerStack tstack;
//   tstack.benchmark("GSU1D::benchmark",100000);
//   FastTimerStackPush(tstack,"GaussianSumUtilities1D::computeMode");
  theModeStatus = NotValid;
  //
  // Use means of individual components as start values.
  // Sort by value of pdf.
  //
  typedef std::multimap<double, int, std::greater<double> > StartMap;
  StartMap xStart;
  for ( unsigned int i=0; i<size(); i++ ) {
    xStart.insert(std::make_pair(pdf(mean(i)),i));
  }
  //
  // Now try with each start value
  //
  int iRes(-1);     // index of start component for current estimate
  double xRes(mean((*xStart.begin()).second)); // current estimate of mode
  double yRes(-1.); // pdf at current estimate of mode
//   std::pair<double,double> result(-1.,mean((*xStart.begin()).second));
  for ( StartMap::const_iterator i=xStart.begin(); i!=xStart.end(); i++ ) {
    //
    // Convergence radius for a single Gaussian = 1 sigma: don't try
    // start values within 1 sigma of the current solution
    //
    if ( theModeStatus==Valid &&
	 fabs(mean((*i).second)-mean(iRes))/standardDeviation(iRes)<1. )  continue;
    //
    // If a solution exists: drop as soon as the pdf at
    // start value drops to < 75% of maximum (try to avoid
    // unnecessary searches for the mode)
    //
    if ( theModeStatus==Valid && 
   	 (*i).first/(*xStart.begin()).first<0.75 )  break;
    //
    // Try to find mode
    //
    double x;
    double y;
    bool valid = findMode(x,y,mean((*i).second),standardDeviation((*i).second));
    //
    // consider only successful searches
    //
    if ( valid ) { //...
      //
      // update result only for significant changes in pdf(solution)
      //
      if ( yRes<0. || (y-yRes)/(y+yRes)>1.e-10 ) {
      iRes = (*i).second;               // store index
      theModeStatus = Valid;            // update status
      xRes = x;                         // current solution
      yRes = y;                         // and its pdf value
//       result = std::make_pair(y,x);     // store solution and pdf(solution)
      }
    } //...
  } 
  //
  // check (existance of) solution
  //
  if ( theModeStatus== Valid ) {
    //
    // Construct single Gaussian state with 
    //  mean = mode
    //  variance = local variance at mode
    //  weight such that the pdf's of the mixture and the
    //    single state are equal at the mode
    //
    double mode = xRes;
    double varMode = localVariance(mode);
    double wgtMode = pdf(mode)*sqrt(2*M_PI*varMode);
    theMode = SingleGaussianState1D(mode,varMode,wgtMode);
  }
  else {
    //
    // mode finding failed: set solution to highest component
    //  (alternative would be global mean / variance ..?)
    //
    edm::LogWarning("GaussianSumUtilities") << "1D mode calculation failed";
//     double x = mean();
//     double y = pdf(x);
//     result = std::make_pair(y,x);
//     theMode = SingleGaussianState1D(mean(),variance(),weight());
    //
    // look for component with highest value at mean
    //
    unsigned int icMax(0);
    double ySqMax(0.);
    for ( unsigned int ic=0; ic<size(); ++ic ) {
      double w = weight(ic);
      double ySq = w*w/variance(ic);
      if ( ic==0 || ySqMax ) {
	icMax = ic;
	ySqMax = ySq;
      }
    }
    theMode = SingleGaussianState1D(components()[icMax]);
  }
  
}

bool
GaussianSumUtilities1D::findMode (double& xMode, double& yMode,
				  const double& xStart,
				  const double& scale) const
{
  //
  // try with Newton on (lnPdf)'(x)
  //
  double x1(0.);
  double y1(0.);
  std::vector<double> pdfs(size());
  pdfComponents(xStart,pdfs);
  double x2(xStart);
  double y2(pdf(xStart,pdfs));
  double yd(d1LnPdf(xStart,pdfs));
  double yd2(d2LnPdf(xStart,pdfs));
  double xmin(xStart-1.*scale);
  double xmax(xStart+1.*scale);
  //
  // preset result
  //
  bool result(false);
  xMode = x2;
  yMode = y2;
  //
  // Iterate
  //
  int nLoop(0);
  while ( nLoop++<20 ) {
    if ( nLoop>1 && yd2<0. &&  
 	 ( fabs(yd*scale)<1.e-10 || fabs(y2-y1)/(y2+y1)<1.e-14 ) ) {
      result = true;
      break;
    }
    if ( fabs(yd2)<std::numeric_limits<float>::min() )  
      yd2 = yd2>0. ? std::numeric_limits<float>::min() : -std::numeric_limits<float>::min();
    double dx = -yd/yd2;
    x1 = x2;
    y1 = y2;
    x2 += dx;
    if ( yd2>0. && (x2<xmin||x2>xmax) )  return false;

    pdfComponents(x2, pdfs);
    y2 = pdf(x2,pdfs);
    yd = d1LnPdf(x2,pdfs);
    yd2 = d2LnPdf(x2,pdfs);
  }
  //
  // result
  //
  if ( result ) {
    xMode = x2;
    yMode = y2;
  }
  return result;
}

double
GaussianSumUtilities1D::pdf (double x) const
{
  double result(0.);
  size_t s=size();
  for ( unsigned int i=0; i<s; i++ )
    result += pdf(i);
  return result;
}

double
GaussianSumUtilities1D::cdf (const double& x) const
{
  double result(0.);
  size_t s=size();
  for ( unsigned int i=0; i<s; i++ )
    result += weight(i)*gaussInt(x,mean(i),standardDeviation(i));
  return result;
}

double
GaussianSumUtilities1D::d1Pdf (const double& x) const
{
  return d1Pdf(x,pdfComponents(x));
}

double
GaussianSumUtilities1D::d2Pdf (const double& x) const
{
  return d2Pdf(x,pdfComponents(x));
}

double
GaussianSumUtilities1D::d3Pdf (const double& x) const
{
  return d3Pdf(x,pdfComponents(x));
}

double
GaussianSumUtilities1D::lnPdf (const double& x) const
{
  return lnPdf(x,pdfComponents(x));
}

double
GaussianSumUtilities1D::d1LnPdf (const double& x) const
{
  return d1LnPdf(x,pdfComponents(x));
}

double
GaussianSumUtilities1D::d2LnPdf (const double& x) const
{
  return d2LnPdf(x,pdfComponents(x));
}

std::vector<double>
GaussianSumUtilities1D::pdfComponents (const double& x) const
{
  std::vector<double> result;
  result.reserve(size());
  for ( unsigned int i=0; i<size(); i++ )
    result.push_back(weight(i)*gauss(x,mean(i),standardDeviation(i)));
  return result;
}

void GaussianSumUtilities1D::pdfComponents (double x, std::vector<double> & result) const {
  size_t s = size();
  if (s!=result.size()) result.resize(s);
  for ( unsigned int i=0; i<s; i++ )
    result[i]=pdf(i);
}


double
GaussianSumUtilities1D::pdf (double, const std::vector<double>& pdfs)
{
  return std::accumulate(pdfs.begin(),pdfs.end(),0.);
}

double
GaussianSumUtilities1D::d1Pdf (double x, const std::vector<double>& pdfs) const
{
  double result(0.);
  size_t s=size();
  for ( unsigned int i=0; i<s; i++ ) {
    double dx = (x-mean(i))/standardDeviation(i);
    result += -pdfs[i]*dx/standardDeviation(i);
  }
  return result;
}

double
GaussianSumUtilities1D::d2Pdf (double x, const std::vector<double>& pdfs) const
{
  double result(0.);
  size_t s=size();
  for ( unsigned int i=0; i<s; i++ ) {
    double dx = (x-mean(i))/standardDeviation(i);
    result += pdfs[i]/standardDeviation(i)/standardDeviation(i)*(dx*dx-1);
  }
  return result;
}

double
GaussianSumUtilities1D::d3Pdf (double x, const std::vector<double>& pdfs) const
{
  double result(0.);
  size_t s=size();
  for ( unsigned int i=0; i<s; i++ ) {
    double dx = (x-mean(i))/standardDeviation(i);
    result += pdfs[i]/standardDeviation(i)/standardDeviation(i)/standardDeviation(i)*
      (-dx*dx+3)*dx;
  }
  return result;
}

double
GaussianSumUtilities1D::lnPdf (double x, const std::vector<double>& pdfs)
{
  double f(pdf(x,pdfs));
  double result(-std::numeric_limits<float>::max());
  if ( f>std::numeric_limits<double>::min() )  result = log(f);
  return result;
}

double
GaussianSumUtilities1D::d1LnPdf (double x, const std::vector<double>& pdfs) const
{

  double f = pdf(x,pdfs);
  double result(d1Pdf(x,pdfs));
  if ( f>std::numeric_limits<double>::min() )  result /= f;
  else  result = 0.;
  return result;
}

double
GaussianSumUtilities1D::d2LnPdf (double x, const std::vector<double>& pdfs) const
{

  double f = pdf(x,pdfs);
  double df = d1LnPdf(x,pdfs);
  double result(-df*df);
  if ( f>std::numeric_limits<double>::min() )  result += d2Pdf(x,pdfs)/f;
  return result;
}

double 
GaussianSumUtilities1D::gauss (double x, double mean, double sigma) const 
{
//   const double fNorm(1./sqrt(2*M_PI));
//   double result(0.);

//   double d((x-mean)/sigma);
//   if ( fabs(d)<20. )  result = exp(-d*d/2.);
//   result *= fNorm/sigma;
//   return result;
  return ROOT::Math::gaussian_pdf(x,sigma,mean);
}

double 
GaussianSumUtilities1D::gaussInt (double x, double mean, double sigma) const 
{
  return ROOT::Math::normal_cdf(x,sigma,mean);
}

double
GaussianSumUtilities1D::combinedMean () const
{
  double s0(0.);
  double s1(0.);
  for ( unsigned int i=0; i<size(); i++ ) {
    s0 += weight(i);
    s1 += weight(i)*mean(i);
  }
  return s1/s0;
}

double
GaussianSumUtilities1D::localVariance (const double& x) const
{
  double result = -pdf(x)/d2Pdf(x);
  // FIXME: wrong curvature seems to be non-existant but should add a proper recovery
  if ( result<0. )
    edm::LogWarning("GaussianSumUtilities") << "1D variance at mode < 0";    
  return result;
}
