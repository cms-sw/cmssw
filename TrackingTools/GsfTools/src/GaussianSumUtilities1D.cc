#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TROOT.h"
#include "TMath.h"

#include <iostream>
#include <cmath>

#include <map>
#include <functional>

// #define DBG_GS1D
// #define DRAW_GS1D
#ifdef DRAW_GS1D
#include "TCanvas.h"
#include "TGraph.h"
#include "TPolyMarker.h"
#endif

double
GaussianSumUtilities1D::quantile (const double q) const
{
  //
  // mean and sigma of highest weight component
  //
  int iwMax(-1);
  float wMax(0.);
  for ( unsigned int i=0; i<size(); i++ ) {
    if ( weight(i)>wMax ) {
      iwMax = i;
      wMax = weight(i);
    }
  }
  //
  // Find start values: begin with mean and
  // go towards f(x)=0 until two points on
  // either side are found (assumes monotonously
  // increasing function)
  //
  double x1(mean(iwMax));
  double y1(cdf(x1)-q);
  double dx = y1>0. ? -standardDeviation(iwMax) : standardDeviation(iwMax);
  double x2(x1+dx);
  double y2(cdf(x2)-q);
  while ( y1*y2>0. ) {
    x1 = x2;
    y1 = y2;
    x2 += dx;
    y2 = cdf(x2) - q;
  }
  //
  // now use bisection to find final value
  //
  double x(0.);
  while ( true ) {
    // use linear interpolation
    x = -(x2*y1-x1*y2)/(y2-y1);
    double y = cdf(x) - q;
    if ( fabs(y)<1.e-6 )  break;
    if ( y*y1>0. ) {
      y1 = y;
      x1 = x;
    }
    else {
      y2 = y;
      x2 = x;
    }
  }
  return x;
}

bool
GaussianSumUtilities1D::modeIsValid () const
{
  if ( theModeStatus==NotComputed )  computeMode();
  return theModeStatus==Valid;
}

double
GaussianSumUtilities1D::mode () const
{
  if ( theModeStatus==NotComputed )  computeMode();
  if ( !theModeStatus == Valid )  
    edm::LogWarning("GaussianSumUtilities") << "1D mode calculation failed";
//     std::cout << "Mode calculation failed!!" << std::endl;
  return theMode;
}

void
GaussianSumUtilities1D::computeMode () const
{
#ifdef DRAW_GS1D
  {
  gPad->Clear();
  std::cout << "gPad = " << gPad << std::endl;
  std::multimap<double,double> drawMap;
  const int npDraw(1024);
  double xpDraw[npDraw];
  double ypDraw[npDraw];
  for ( unsigned int i=0; i<size(); i++ ) {
    double ave = mean(i);
    double sig = standardDeviation(i);
    for ( double xi=-3.; xi<3.; ) {
      double x = ave + xi*sig;
      drawMap.insert(std::make_pair(x,pdf(x)));
      xi += 0.2;
    }
  }
  int np(0);
  double xMin(FLT_MAX);
  double xMax(-FLT_MAX);
  double yMax(-FLT_MAX);
  for ( std::multimap<double,double>::const_iterator im=drawMap.begin();
	im!=drawMap.end(); ++im,++np ) {
    if ( np>=1024 )  break;
    xpDraw[np] = (*im).first;
    ypDraw[np] = (*im).second;
    if ( xMin>(*im).first )  xMin = (*im).first;
    if ( xMax<(*im).first )  xMax = (*im).first;
    if ( yMax<(*im).second )  yMax = (*im).second;
  }
//   for ( int i=0; i<np; ++i )
//     std::cout << i << " " << xpDraw[i] << " " << ypDraw[i] << std::endl;
  gPad->DrawFrame(xMin,0.,xMax,1.05*yMax);  
  TGraph* g = new TGraph(np,xpDraw,ypDraw);
  g->SetLineWidth(2);
  g->Draw("C");
  TGraph* gc = new TGraph();
  gc->SetLineStyle(2);
  int np2(0);
  double xpDraw2[1024];
  double ypDraw2[1024];
  for ( unsigned int i=0; i<size(); i++ ) {
    double ave = mean(i);
    double sig = standardDeviation(i);
    SingleGaussianState1D sgs(ave,variance(i),weight(i));
    std::vector<SingleGaussianState1D> sgsv(1,sgs);
    MultiGaussianState1D mgs(sgsv);
    GaussianSumUtilities1D gsu(mgs);
    np2 = 0;
    for ( double xi=-3.; xi<3.; ) {
      double x = ave + xi*sig;
      xpDraw2[np2] = x;
      ypDraw2[np2] = gsu.pdf(x);
      ++np2;
      xi += 0.2;
    }
//    for ( int i=0; i<np2; ++i )
//      std::cout << i << " " << xpDraw2[i] << " " << ypDraw2[i] << std::endl;
//     std::cout << "np2 = " << np2 
// 	 << " ave = " << ave 
// 	 << " sig = " << sig 
// 	 << " weight = " << weight(i)
// 	 << " pdf = " << gsu.pdf(ave) << std::endl;
    if ( np2>0 )  gc->DrawGraph(np2,xpDraw2,ypDraw2);
//     break;
  }
  gPad->Modified();
  gPad->Update();
  }
#endif
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
#ifdef DRAW_GS1D
  int ind(0);
#endif
  int iRes(-1); // index of current estimate
  std::pair<double,double> result(-1.,mean((*xStart.begin()).second));
  for ( StartMap::const_iterator i=xStart.begin(); i!=xStart.end(); i++ ) {
#ifdef DRAW_GS1D
    double xp[2];
    double yp[2];
    TPolyMarker* g = new TPolyMarker();
    g->SetMarkerStyle(20+ind/4);
    g->SetMarkerColor(1+ind%4);
    ++ind;
#endif
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
      if ( result.first<0. || (y-result.first)/(y+result.first)>1.e-10 ) {
#ifdef DBG_GS1D
      if ( iRes>=0 ) 
	std::cout << "dxStart = " << (result.second-mean((*i).second))/standardDeviation(iRes) << std::endl;
#endif
      iRes = (*i).second;               // store index
      theModeStatus = Valid;            // update status
      result = std::make_pair(y,x);     // store solution and pdf(solution)
      }
    } //...
#ifdef DRAW_GS1D
    xp[0] = xp[1] = mean((*i).second);
    yp[0] = yp[1] = (*i).first;
    if ( valid ) {
      xp[1] = x;
      yp[1] = y;
    }
    g->DrawPolyMarker(2,xp,yp);
#endif
#ifdef DBG_GS1D
    ++ind; //...
#endif
  } 
  //
  // check (existance of) solution
  //
  if ( theModeStatus== Valid ) {
#ifdef DBG_GS1D
    std::cout << "Ratio of pdfs at  first / real maximum = " << iRes << " " << mean(iRes) << " "
	 << pdf(mean(iRes))/(*xStart.begin()).first << std::endl;
#endif
  }
  else {
    //
    // mode finding failed: set solution to total mean
    std::cout << "Mode finder failed!!!" << std::endl;
    double x = mean();
    double y = pdf(x);
    result = std::make_pair(y,x);
  }
#ifdef DRAW_GS1D
  gPad->Modified();
  gPad->Update();
#endif
  //
  // results
  //
  theMode = result.second;
#ifdef DRAW_GS1D
  {
    TGraph* gm = new TGraph();
    gm->SetLineColor(2);
    int np(0);
    double xp[1024];
    double yp[1024];
    double mode = theMode;
    double var = localVariance(mode);
    double sig = sqrt(var);
    SingleGaussianState1D sgs(mode,var,pdf(mode)*sqrt(2*TMath::Pi())*sig);
    std::vector<SingleGaussianState1D> sgsv(1,sgs);
    MultiGaussianState1D mgs(sgsv);
    GaussianSumUtilities1D gsu(mgs);
    for ( double xi=-3.; xi<3.; ) {
      double x = mode + xi*sig;
      xp[np] = x;
      yp[np] = gsu.pdf(x);
      ++np;
      xi += 0.2;
    }
    gm->DrawGraph(np,xp,yp);
    gPad->Modified();
    gPad->Update();
  }
#endif
  
}

// bool
// GaussianSumUtilities1D::findModeAlternative (double& xMode, double& yMode,
// 					     const double& xStart,
// 					     const double& scale) const
// {
//   //
//   // try with Newton on pdf'(x)
//   //
//   bool result(false);
//   double x1(0.);
//   double y1(0.);
//   double x2(xStart);
//   double y2(pdf(xStart));
//   double yd(d1Pdf(xStart));
//   double yd2(d2Pdf(xStart));
//   double xmin(xStart-1.5*scale);
//   double xmax(xStart+1.5*scale);
//   double vDyDy(yd*localVariance(xStart)*yd);
//   int nLoop(0);
//   while ( nLoop++<20 ) {
//     if ( vDyDy>-FLT_MIN && vDyDy<1.e-6 ) {
//       result = true;
//       break;
//     }
//     double dx = -yd/yd2;
//     x1 = x2;
//     y1 = y2;
//     // try to estimate validity of extrapolation:
//     // - use a limit on dy/dx equivalent to a 1% change in y over scale
//     // - convert it to a limit on the step size based on d3y/dx3
//     double dxmax = sqrt(fabs(2.*(1.e-2*y2/scale)/d3Pdf(x1)));
//     if ( yd2*scale*scale/y2<-1.e-4 ) {
//       bool limit1 = fabs(dx)>dxmax;
//       if ( limit1 ) {
// 	if ( dx<-dxmax )  dx = -dxmax;
// 	else if ( dx>dxmax )  dx = dxmax;
//       }
//     }
//     else {
//       // move dxmax
//       dx = yd>0 ? dxmax : -dxmax;
//     }
//     x2 += dx;
//     bool limit2 = x2<xmin || x2>xmax;
//     if ( x2 < xmin )  x2 = xmin;
//     else if ( x2 > xmax )  x2 = xmax;
//     y2 = pdf(x2);
//     yd2 = d2Pdf(x2);
//     if ( limit2 ) {
//       xMode = xStart;
//       yMode = pdf(xStart);
//       return false;
//     }
//     y2 = pdf(x2);
//     yd = d1Pdf(x2);
//     if ( fabs(yd2)<1.e-10 )  yd2 = yd2>0. ? 1.e-10 : -1.e-10;
//     vDyDy = yd*localVariance(x2)*yd;
//   }
//   xMode = x2;
//   yMode = y2;
//   return result;
// }

bool
GaussianSumUtilities1D::findMode (double& xMode, double& yMode,
				  const double& xStart,
				  const double& scale) const
{
  //
  // try with Newton on (lnPdf)'(x)
  //
  bool result(false);
  double x1(0.);
  double y1(0.);
  double x2(xStart);
  double y2(pdf(xStart));
  double yd(d1LnPdf(xStart));
  double yd2(d2LnPdf(xStart));
  double ydLin(d1Pdf(xStart));
//   double xmin(xStart-1.5*scale);
//   double xmax(xStart+1.5*scale);
  double xmin(xStart-1.*scale);
  double xmax(xStart+1.*scale);
  double vDyDy(ydLin*localVariance(xStart)*ydLin);
  int nLoop(0);
  while ( nLoop++<20 ) {
    if ( nLoop>1 && vDyDy>-FLT_MIN && 
	 (vDyDy<1.e-8 || fabs(y2-y1)/(y2+y1)<1.e-10) ) {
      result = true;
      break;
    }
    double dx = -yd/yd2;
    x1 = x2;
    y1 = y2;
//     // try to estimate validity of extrapolation:
//     // - use a limit on dy/dx equivalent to a 1% change in y over scale
//     // - convert it to a limit on the step size based on d3y/dx3
//     double dxmax = sqrt(fabs(2.*(0.001*y2/scale)/d3Pdf(x1)));
//     if ( vDyDy<-FLT_MIN ) {
//       bool limit1 = fabs(dx)>dxmax;
//       if ( limit1 ) {
//  	if ( dx<-dxmax )  dx = -dxmax;
//  	else if ( dx>dxmax )  dx = dxmax;
//       }
//     }
    x2 += dx;
//     if ( x2 < xmin )  x2 = xmin + 0.000001*scale;
//     else if ( x2 > xmax )  x2 = xmax - 0.000001*scale;
    if ( x2<xmin || x2>xmax ) {
      xMode = xStart;
      yMode = pdf(xStart);
      return false;
    }
    y2 = pdf(x2);
    yd2 = d2LnPdf(x2);
    yd = d1LnPdf(x2);
    ydLin = d1Pdf(x2);
    if ( fabs(yd2)<1.e-10 )  yd2 = yd2>0. ? 1.e-10 : -1.e-10;
    vDyDy = ydLin*localVariance(x2)*ydLin;
  }
  xMode = x2;
  yMode = y2;
#ifdef DBG_GS1D
  std::cout << "Started from " << xStart << " " << pdf(xStart)
	    << " ; ended at " << xMode << " " << yMode << " after " 
	    << nLoop << " iterations" << std::endl;
#endif
  return result;
}

double
GaussianSumUtilities1D::pdf (const double& x) const
{
  double result(0.);
  for ( unsigned int i=0; i<size(); i++ )
    result += weight(i)*gauss(x,mean(i),standardDeviation(i));
  return result;
}

double
GaussianSumUtilities1D::cdf (const double& x) const
{
  double result(0.);
  for ( unsigned int i=0; i<size(); i++ )
    result += weight(i)*gaussInt(x,mean(i),standardDeviation(i));
  return result;
}

double
GaussianSumUtilities1D::d1Pdf (const double& x) const
{
  double result(0.);
  for ( unsigned int i=0; i<size(); i++ ) {
    double dx = (x-mean(i))/standardDeviation(i);
    result += -weight(i)*dx/standardDeviation(i)*
      gauss(x,mean(i),standardDeviation(i));
  }
  return result;
}

double
GaussianSumUtilities1D::d2Pdf (const double& x) const
{
  double result(0.);
  for ( unsigned int i=0; i<size(); i++ ) {
    double dx = (x-mean(i))/standardDeviation(i);
    result += weight(i)/standardDeviation(i)/standardDeviation(i)*
      (dx*dx-1)*gauss(x,mean(i),standardDeviation(i));
  }
  return result;
}

double
GaussianSumUtilities1D::d3Pdf (const double& x) const
{
  double result(0.);
  for ( unsigned int i=0; i<size(); i++ ) {
    double dx = (x-mean(i))/standardDeviation(i);
    result += weight(i)/standardDeviation(i)/standardDeviation(i)/standardDeviation(i)*
      (-dx*dx+3)*dx*gauss(x,mean(i),standardDeviation(i));
  }
  return result;
}

double
GaussianSumUtilities1D::lnPdf (const double& x) const
{
  double f(pdf(x));
  double result(-FLT_MAX);
  if ( result>DBL_MIN )  result = log(f);
  return result;
}

double
GaussianSumUtilities1D::d1LnPdf (const double& x) const
{

  double f = pdf(x);
  double result(d1Pdf(x));
  if ( f>DBL_MIN )  result /= f;
  else  result = 0.;
  return result;
}

double
GaussianSumUtilities1D::d2LnPdf (const double& x) const
{

  double f = pdf(x);
  double df = d1LnPdf(x);
  double result(-df*df);
  if ( f>DBL_MIN )  result += d2Pdf(x)/f;
  return result;
}

double 
GaussianSumUtilities1D::gauss (const double& x, const double& mean,
		    const double& sigma) const 
{
  const double fNorm(1./sqrt(2*TMath::Pi()));
  double result(0.);

  double d((x-mean)/sigma);
  if ( fabs(d)<20. )  result = exp(-d*d/2.);
  result *= fNorm/sigma;
  return result;
}

double 
GaussianSumUtilities1D::gaussInt (const double& x, const double& mean,
		       const double& sigma) const 
{
  return TMath::Freq((x-mean)/sigma);
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
  return result;
}
