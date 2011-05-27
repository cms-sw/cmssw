#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloCachedShapeIntegrator.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main()
{
  HcalShape shape;
  CaloShapeIntegrator i1(&shape);
  CaloCachedShapeIntegrator i2(&shape);

  for(int t = -25; t < 256; ++t) 
  {
    double v1 = i1(t);
    double v2 = i2(t);
    if(v1 > 0.)
    {
      // should be identical, but allow roundoff
      assert( fabs(v1 - v2)/v1 < 0.0001 );
    }
  }
}
