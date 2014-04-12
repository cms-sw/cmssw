// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPMShape_h
#define HcalSimAlgos_HcalSiPMShape_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include <vector>

class HcalSiPMShape : public CaloVShape {
public:

  HcalSiPMShape();
  HcalSiPMShape(const HcalSiPMShape & other);

  virtual ~HcalSiPMShape() {}

  virtual double operator() (double time) const;

  virtual double timeToRise() const {return 3.5;}

  static double gexp(double t, double A, double c, double t0, double s);
  static double gexpIndefIntegral(double t, double A, double c, double t0, 
				  double s);
  static double gexpIntegral0Inf(double A, double c, double t0, double s);

protected:
  virtual double analyticPulseShape(double t) const;
  void computeShape();

private:

  int nBins_;
  std::vector<double> nt_;

};

#endif //HcalSimAlgos_HcalSiPMShape_h
