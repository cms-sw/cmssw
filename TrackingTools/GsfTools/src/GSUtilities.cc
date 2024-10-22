#include "TrackingTools/GsfTools/interface/GSUtilities.h"

#include <TMath.h>

// #include <iostream>
#include <cmath>

#include <map>
#include <functional>

float GSUtilities::quantile(const float q) const {
  float qq = q;
  if (q > 1)
    qq = 1;
  else if (q < 0)
    qq = 0;
  //
  // mean and sigma of highest weight component
  //
  int iwMax(-1);
  float wMax(0.);
  for (unsigned i = 0; i < theNComp; i++) {
    if (theWeights[i] > wMax) {
      iwMax = i;
      wMax = theWeights[i];
    }
  }
  //
  // Find start values: begin with mean and
  // go towards f(x)=0 until two points on
  // either side are found (assumes monotonously
  // increasing function)
  //
  double x1(theParameters[iwMax]);
  double y1(cdf(x1) - qq);
  double dx = y1 > 0. ? -theErrors[iwMax] : theErrors[iwMax];
  double x2(x1 + dx);
  double y2(cdf(x2) - qq);
  while (y1 * y2 > 0.) {
    x1 = x2;
    y1 = y2;
    x2 += dx;
    y2 = cdf(x2) - qq;
  }
  //   std::cout << "(" << x1 << "," << y1 << ") / ("
  //        << x2 << "," << y2 << ")" << std::endl;
  //
  // now use bisection to find final value
  //
  double x(0.);
  while (true) {
    // use linear interpolation
    x = -(x2 * y1 - x1 * y2) / (y2 - y1);
    double y = cdf(x) - qq;
    //     std::cout << x << " " << y << std::endl;
    if (fabs(y) < 1.e-6)
      break;
    if (y * y1 > 0.) {
      y1 = y;
      x1 = x;
    } else {
      y2 = y;
      x2 = x;
    }
  }
  return x;
}

float GSUtilities::errorHighestWeight() const {
  int iwMax(-1);
  float wMax(0.);
  for (unsigned i = 0; i < theNComp; i++) {
    if (theWeights[i] > wMax) {
      iwMax = i;
      wMax = theWeights[i];
    }
  }
  return theErrors[iwMax];
}

float GSUtilities::mode() const {
  //
  // start values = means of components
  //
  typedef std::multimap<double, double, std::greater<double> > StartMap;
  StartMap xStart;
  for (unsigned i = 0; i < theNComp; i++) {
    xStart.insert(std::pair<double, float>(pdf(theParameters[i]), theParameters[i]));
  }
  //
  // now try with each start value
  //
  typedef std::multimap<double, double, std::greater<double> > ResultMap;
  ResultMap xFound;
  for (StartMap::const_iterator i = xStart.begin(); i != xStart.end(); i++) {
    double x = findMode((*i).second);
    xFound.insert(std::pair<double, double>(pdf(x), x));
    //     std::cout << "Started at " << (*i).second
    // 	 << " , found " << x << std::endl;
  }
  //
  // results
  //
  //   for ( ResultMap::const_iterator i=xFound.begin();
  // 	i!=xFound.end(); i++ ) {
  //     std::cout << "pdf at " << (*i).second << " = " << (*i).first << std::endl;
  //   }
  return xFound.begin()->second;
}

double GSUtilities::findMode(const double xStart) const {
  //
  // try with Newton
  //
  double y1(0.);
  double x(xStart);
  double y2(pdf(xStart));
  double yd(dpdf1(xStart));
  int nLoop(0);
  if ((y1 + y2) < 10 * DBL_MIN)
    return xStart;
  while (nLoop++ < 20 && fabs(y2 - y1) / (y2 + y1) > 1.e-6) {
    //     std::cout << "dy = " << y2-y1 << std::endl;
    double yd2 = dpdf2(x);
    if (fabs(yd2) < 10 * DBL_MIN)
      return xStart;
    x -= yd / dpdf2(x);
    yd = dpdf1(x);
    y1 = y2;
    y2 = pdf(x);
    //     std::cout << "New x / yd = " << x << " / " << yd << std::endl;
  }
  if (nLoop >= 20)
    return xStart;
  return x;
}

double GSUtilities::pdf(const double& x) const {
  double result(0.);
  for (unsigned i = 0; i < theNComp; i++)
    result += theWeights[i] * gauss(x, theParameters[i], theErrors[i]);
  return result;
}

double GSUtilities::cdf(const double& x) const {
  double result(0.);
  for (unsigned i = 0; i < theNComp; i++)
    result += theWeights[i] * gaussInt(x, theParameters[i], theErrors[i]);
  return result;
}

double GSUtilities::dpdf1(const double& x) const {
  double result(0.);
  for (unsigned i = 0; i < theNComp; i++) {
    double dx = (x - theParameters[i]) / theErrors[i];
    result += -theWeights[i] * dx / theErrors[i] * gauss(x, theParameters[i], theErrors[i]);
  }
  return result;
}

double GSUtilities::dpdf2(const double& x) const {
  double result(0.);
  for (unsigned i = 0; i < theNComp; i++) {
    double dx = (x - theParameters[i]) / theErrors[i];
    result += theWeights[i] / theErrors[i] / theErrors[i] * (dx * dx - 1) * gauss(x, theParameters[i], theErrors[i]);
  }
  return result;
}

double GSUtilities::gauss(const double& x, const double& mean, const double& sigma) const {
  const double fNorm(1. / sqrt(2 * TMath::Pi()));
  double result(0.);

  double d((x - mean) / sigma);
  if (fabs(d) < 20.)
    result = exp(-d * d / 2.);
  result *= fNorm / sigma;
  return result;
}

double GSUtilities::gaussInt(const double& x, const double& mean, const double& sigma) const {
  return TMath::Freq((x - mean) / sigma);
}

double GSUtilities::combinedMean() const {
  double s0(0.);
  double s1(0.);
  for (unsigned i = 0; i < theNComp; i++) {
    s0 += theWeights[i];
    s1 += theWeights[i] * theParameters[i];
  }
  return s1 / s0;
}

double GSUtilities::errorCombinedMean() const {
  double s0(0.);
  for (unsigned i = 0; i < theNComp; i++) {
    s0 += theWeights[i];
  }
  return 1. / (sqrt(s0));
}

float GSUtilities::maxWeight() const {
  // Look for the highest weight component
  //
  //int iwMax(-1);
  float wMax(0.);
  for (unsigned i = 0; i < theNComp; i++) {
    if (theWeights[i] > wMax) {
      //iwMax = i;
      wMax = theWeights[i];
    }
  }
  return wMax;
}

float GSUtilities::errorMode() {
  float mod = mode();
  float min = getMin(mod);
  float max = getMax(mod);
  int nBins = 1000;
  float dx = (max - min) / (float)nBins;

  float x1 = mod, x2 = mod, x1f = mod, x2f = mod;
  float I = 0;
  int cnt = 0;
  while (I < .68) {
    x1 -= dx;
    x2 += dx;
    if (pdf(x1) >= pdf(x2)) {
      x1f = x1;
      x2 -= dx;
    } else {
      x2f = x2;
      x1 += dx;
    }
    I = cdf(x2f) - cdf(x1f);
    cnt++;
    // for crazy pdf's return a crazy value
    if (cnt > 2500)
      return 100000.;
  }
  return (x2f - x1f) / 2;
}

float GSUtilities::getMin(float x) {
  int cnt = 0;
  float dx;
  if (fabs(x) < 2)
    dx = .5;
  else
    dx = fabs(x) / 10;
  while (cdf(x) > .1 && cnt < 1000) {
    x -= dx;
    cnt++;
  }
  return x;
}

float GSUtilities::getMax(float x) {
  int cnt = 0;
  float dx;
  if (fabs(x) < 2)
    dx = .5;
  else
    dx = fabs(x) / 10;
  while (cdf(x) < .9 && cnt < 1000) {
    x += dx;
    cnt++;
  }
  return x;
}
