// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPMShape_h
#define HcalSimAlgos_HcalSiPMShape_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"
#include <vector>

class HcalSiPMShape : public CaloVShape {
public:

  HcalSiPMShape(unsigned int signalShape=203);
  HcalSiPMShape(const HcalSiPMShape & other);

  virtual ~HcalSiPMShape() {}

  virtual double operator() (double time) const;

  virtual double timeToRise() const {return 0.0;}

protected:
  void computeShape(unsigned int signalShape);

private:

  int nBins_;
  std::vector<double> nt_;

};

#endif //HcalSimAlgos_HcalSiPMShape_h
