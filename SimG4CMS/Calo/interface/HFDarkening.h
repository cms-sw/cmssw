#ifndef SimG4CMS_HFDarkening_h
#define SimG4CMS_HFDarkening_h

#include <cmath>
#include <iostream>
#include <vector>

typedef std::vector<double> vecOfDoubles;

namespace edm {
  class ParameterSet;
}

class HFDarkening {
public:
  HFDarkening(const edm::ParameterSet& pset);
  ~HFDarkening();

  double dose(unsigned int layer, double radius);
  double int_lumi(double intlumi);
  double degradation(double mrad);

  //These constants are used in HcalSD.cc
  static const unsigned int numberOfZLayers = 33;
  static const unsigned int numberOfRLayers = 13;

  static const unsigned int lowZLimit = 1115;
  static const unsigned int upperZLimit = 1280;

private:
  double HFDoseLayerDarkeningPars[numberOfZLayers][numberOfRLayers];
  static const unsigned int _numberOfZLayers = numberOfZLayers;
  static const unsigned int _numberOfRLayers = numberOfRLayers;
};

#endif  // HFDarkening_h
