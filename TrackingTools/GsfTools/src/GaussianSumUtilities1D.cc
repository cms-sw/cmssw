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


double GaussianSumUtilities1D::pdf(unsigned int i, double x)  const {
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
    //two many log warnings to actually be useful - comment out
    //edm::LogWarning("GaussianSumUtilities") << "1D mode calculation failed";
//     double x = mean();
//     double y = pdf(x);
//     result = std::make_pair(y,x);
//     theMode = SingleGaussianState1D(mean(),variance(),weight());
    //
    // look for component with highest value at mean
    //
    int icMax(0);
    double ySqMax(0.);
    int s = size();
    for (int ic=0; ic<s; ++ic ) {
      double w = weight(ic);
      double ySq = w*w/variance(ic);
      if ( ySq>ySqMax ) {
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
  FinderState  state(size());
  update(state,xStart);

  double xmin(xStart-1.*scale);
  double xmax(xStart+1.*scale);
  //
  // preset result
  //
  bool result(false);
  xMode = state.x;
  yMode = state.y;
  //
  // Iterate
  //
  int nLoop(0);
  while ( nLoop++<20 ) {
    if ( nLoop>1 && state.yd2<0. &&  
 	 ( fabs(state.yd*scale)<1.e-10 || fabs(state.y-y1)/(state.y+y1)<1.e-14 ) ) {
      result = true;
      break;
    }
    if ( fabs(state.yd2)<std::numeric_limits<float>::min() )  
      state.yd2 = state.yd2>0. ? std::numeric_limits<float>::min() : -std::numeric_limits<float>::min();
    double dx = -state.yd/state.yd2;
    x1 = state.x;
    y1 = state.y;
    double x2 = x1 + dx;
    if ( state.yd2>0. && (x2<xmin||x2>xmax) )  return false;
    update(state,x2);
  }
  //
  // result
  //
  if ( result ) {
    xMode = state.x;
    yMode = state.y;
  }
  return result;
}


void GaussianSumUtilities1D::update(FinderState & state, double x) const {
  state.x = x;

  pdfComponents(state.x, state.pdfs);
  state.y = pdf(state.x, state.pdfs);
  state.yd = 0;
  if (state.y>std::numeric_limits<double>::min()) state.yd= d1Pdf(state.x,state.pdfs)/state.y;
  state.yd2 = -state.yd*state.yd;
  if (state.y > std::numeric_limits<double>::min()) state.yd2 += d2Pdf(state.x,state.pdfs)/state.y;
}


double
GaussianSumUtilities1D::pdf (double x) const
{
  double result(0.);
  size_t s=size();
  for ( unsigned int i=0; i<s; i++ )
    result += pdf(i,x);
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

namespace {
  struct PDF {
  PDF(double ix) : x(ix){}
    double x;
    double operator()(SingleGaussianState1D const & sg) const {
       return sg.weight()*ROOT::Math::gaussian_pdf(x,sg.standardDeviation(),sg.mean());
    }
  };
}
void GaussianSumUtilities1D::pdfComponents (double x, std::vector<double> & result) const {
  size_t s = size();
  if (s!=result.size()) result.resize(s);  
  std::transform(components().begin(),components().end(),result.begin(),PDF(x));
}
/*
void GaussianSumUtilities1D::pdfComponents (double x, std::vector<double> & result) const {
   size_t s = size();
  if (s!=result.size()) result.resize(s);
  double* __restrict__ v = &result.front();
  SingleGaussianState1D const * __restrict__ sgv = &components().front();
  for ( unsigned int i=0; i<s; i++ )
    v[i]= sgv[i].weight()*gauss(x,sgv[i].mean(),sgv[i].standardDeviation());
//    result[i]=pdf(i,x);
}
*/

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
GaussianSumUtilities1D::gauss (double x, double mean, double sigma) 
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
GaussianSumUtilities1D::gaussInt (double x, double mean, double sigma) 
{
  return ROOT::Math::normal_cdf(x,sigma,mean);
}

double
GaussianSumUtilities1D::combinedMean () const
{
  double s0(0.);
  double s1(0.);
  int s = size();
  SingleGaussianState1D const * __restrict__ sgv = &components().front();
  for (int i=0; i<s; i++ )
    s0 += sgv[i].weight();
  for (int i=0; i<s; i++ )
    s1 += sgv[i].weight()*sgv[i].mean();
  return s1/s0;
}

double
GaussianSumUtilities1D::localVariance (double x) const
{
  std::vector<double> pdfs;
  pdfComponents(x,pdfs);
  double result = -pdf(x,pdfs)/d2Pdf(x,pdfs);
  // FIXME: wrong curvature seems to be non-existant but should add a proper recovery
  if ( result<0. )
    edm::LogWarning("GaussianSumUtilities") << "1D variance at mode < 0";    
  return result;
}
