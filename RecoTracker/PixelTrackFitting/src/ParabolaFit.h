#ifndef RecoTracker_PixelTrackFitting_src_ParabolaFit_h
#define RecoTracker_PixelTrackFitting_src_ParabolaFit_h

#include <vector>

/** The parabola fit 
    y = parA + parB * x + parC * x*x
    see: R.L. Gluckstern, NIM 24 (1963) 381 
 */

class ParabolaFit {
public:
  struct Result {
    double parA, parB, parC;
    double varAA, varBB, varCC, varAB, varAC, varBC;
  };
  ParabolaFit() : doErr(true), hasFixedParC(false), hasValues(false), hasErrors(false), hasWeights(true) {}

  void addPoint(double x, double y);
  void addPoint(double x, double y, double weight);

  void skipErrorCalculationByDefault() { doErr = false; }
  void fixParC(double val) {
    hasFixedParC = true;
    theResult.parC = val;
  }

  const Result& result(bool doErrors) const;

  double parA() const {
    if (!hasValues)
      result(doErr);
    return theResult.parA;
  }
  double parB() const {
    if (!hasValues)
      result(doErr);
    return theResult.parB;
  }
  double parC() const {
    if (!hasValues)
      result(doErr);
    return theResult.parC;
  }
  double varAA() const {
    if (!hasErrors)
      result(true);
    return theResult.varAA;
  }
  double varBB() const {
    if (!hasErrors)
      result(true);
    return theResult.varBB;
  }
  double varCC() const {
    if (!hasErrors)
      result(true);
    return theResult.varCC;
  }
  double varAB() const {
    if (!hasErrors)
      result(true);
    return theResult.varAB;
  }
  double varAC() const {
    if (!hasErrors)
      result(true);
    return theResult.varAC;
  }
  double varBC() const {
    if (!hasErrors)
      result(true);
    return theResult.varBC;
  }

  double chi2() const;
  int dof() const;

private:
  struct Column {
    double r1;
    double r2;
    double r3;
  };
  double det(const Column& c1, const Column& c2, const Column& c3) const;
  double det(const Column& c1, const Column& c2) const;

  double fun(double x) const;

private:
  struct Point {
    double x;
    double y;
    mutable double w;
  };
  std::vector<Point> points;
  bool doErr, hasFixedParC;
  mutable bool hasValues, hasErrors;
  bool hasWeights;
  mutable Result theResult;
};

#endif
