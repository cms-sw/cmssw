#ifndef UniformMomentumGenerator_H
#define UniformMomentumGenerator_H

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"

class UniformMomentumGenerator {
public:

  UniformMomentumGenerator( double pmin = 1.0, double pmax = 100.0) :
    thePmin(pmin), thePmax(pmax) {}

  Basic3DVector<double> operator()() const;

private:

  double thePmin;
  double thePmax;

};

#endif
