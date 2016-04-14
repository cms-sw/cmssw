#ifndef SimG4CMS_HFDarkening_h
#define SimG4CMS_HFDarkening_h

#include <cmath>
#include <iostream>
#include <vector>

typedef std::vector<double> vecOfDoubles;

namespace edm
{
  class ParameterSet;
}

class HFDarkening {

public:
  HFDarkening(const edm::ParameterSet& pset);
  ~HFDarkening();
    
  double dose(int layer, double radius);
  double int_lumi(double intlumi);
  double degradation(double mrad);
  
  static const unsigned int numberOfZLayers = 33;
  static const unsigned int numberOfRLayers = 13;
  
private:
  std::vector<double> HFDoseLayerDarkeningPars[numberOfZlayers][numberOfRlayers];
};

#endif // HFDarkening_h
