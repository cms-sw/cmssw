#include "ParabolaFit.h"
#include <iostream>
using namespace std;
template <class T>
T sqr(T t) {
  return t * t;
}

void ParabolaFit::addPoint(double x, double y) {
  hasWeights = false;
  addPoint(x, y, 1.);
}

void ParabolaFit::addPoint(double x, double y, double w) {
  hasValues = false;
  hasErrors = false;
  Point p = {x, y, w};
  points.push_back(p);
}

const ParabolaFit::Result& ParabolaFit::result(bool doErrors) const {
  if (hasErrors)
    return theResult;
  if (hasValues && !doErrors)
    return theResult;

  double F0, F1, F2, F3, F4, F0y, F1y, F2y;
  F0 = F1 = F2 = F3 = F4 = F0y = F1y = F2y = 0.;

  typedef vector<Point>::const_iterator IT;
  for (IT ip = points.begin(); ip != points.end(); ip++) {
    double pow;
    double x = ip->x;
    double y = ip->y;
    double w = ip->w;

    F0 += w;
    F0y += w * y;
    F1 += w * x;
    F1y += w * x * y;
    pow = x * x;
    F2 += w * pow;
    F2y += w * pow * y;  // x^2
    pow *= x;
    F3 += w * pow;  // x^3
    pow *= x;
    F4 += w * pow;  // x^4
  }

  Column cA = {F0, F1, F2};
  Column cB = {F1, F2, F3};
  Column cC = {F2, F3, F4};
  Column cY = {F0y, F1y, F2y};

  double det0 = det(cA, cB, cC);

  if (!hasFixedParC) {
    theResult.parA = det(cY, cB, cC) / det0;
    theResult.parB = det(cA, cY, cC) / det0;
    theResult.parC = det(cA, cB, cY) / det0;
  } else {
    Column cCY = {F0y - theResult.parC * F2, F1y - theResult.parC * F3, F2y - theResult.parC * F4};
    double det0C = det(cA, cB);
    theResult.parA = det(cCY, cB) / det0C;
    theResult.parB = det(cA, cCY) / det0C;
  }

  //  std::cout <<" result: A="<<theResult.parA<<" B="<<theResult.parB<<" C="<<theResult.parC<<endl;
  double vAA, vBB, vCC, vAB, vAC, vBC;
  vAA = vBB = vCC = vAB = vAC = vBC = 0.;

  hasValues = true;
  if (!doErrors)
    return theResult;

  if (!hasWeights && dof() > 0) {
    //     cout <<" CHI2: " << chi2() <<" DOF: " << dof() << endl;
    double scale_w = 1. / chi2() / dof();
    for (IT ip = points.begin(); ip != points.end(); ip++)
      ip->w *= scale_w;
    //     cout <<" CHI2: " << chi2() <<" DOF: " << dof() << endl;
  }

  for (IT ip = points.begin(); ip != points.end(); ip++) {
    double w = ip->w;
    Column cX = {1., ip->x, sqr(ip->x)};

    double dXBC = det(cX, cB, cC);
    double dAXC = det(cA, cX, cC);
    double dABX = det(cA, cB, cX);

    vAA += w * sqr(dXBC);
    vBB += w * sqr(dAXC);
    vCC += w * sqr(dABX);
    vAB += w * dXBC * dAXC;
    vAC += w * dXBC * dABX;
    vBC += w * dAXC * dABX;
  }

  theResult.varAA = vAA / sqr(det0);
  theResult.varBB = vBB / sqr(det0);
  theResult.varCC = vCC / sqr(det0);
  theResult.varAB = vAB / sqr(det0);
  theResult.varAC = vAC / sqr(det0);
  theResult.varBC = vBC / sqr(det0);

  hasErrors = true;
  return theResult;
}

double ParabolaFit::chi2() const {
  double mychi2 = 0.;
  for (vector<Point>::const_iterator ip = points.begin(); ip != points.end(); ip++) {
    mychi2 += ip->w * sqr(ip->y - fun(ip->x));
  }
  return mychi2;
}

double ParabolaFit::fun(double x) const { return theResult.parA + theResult.parB * x + theResult.parC * x * x; }

int ParabolaFit::dof() const {
  int nPar = 3;
  if (hasFixedParC)
    nPar--;
  int nDof = points.size() - nPar;
  return (nDof > 0) ? nDof : 0;
}

double ParabolaFit::det(const Column& c1, const Column& c2, const Column& c3) const {
  return c1.r1 * c2.r2 * c3.r3 + c2.r1 * c3.r2 * c1.r3 + c3.r1 * c1.r2 * c2.r3 - c1.r3 * c2.r2 * c3.r1 -
         c2.r3 * c3.r2 * c1.r1 - c3.r3 * c1.r2 * c2.r1;
}

double ParabolaFit::det(const Column& c1, const Column& c2) const { return c1.r1 * c2.r2 - c1.r2 * c2.r1; }
