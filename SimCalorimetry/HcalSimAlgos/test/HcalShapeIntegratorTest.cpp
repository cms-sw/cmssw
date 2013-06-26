#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloCachedShapeIntegrator.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main()
{
  HFShape shape;
  CaloShapeIntegrator i1(&shape);
  CaloCachedShapeIntegrator i2(&shape);
  float maxdiff = 0.;
  for(float t = -25; t < 256; t += 0.25) 
  {
    double v1 = i1(t);
    double v2 = i2(t);
    float diff = fabs(v1-v2);
    if(diff > maxdiff)
    {
      maxdiff = diff;
    }
  }
  std::cout << "Maximum discrepancy " << maxdiff << std::endl;
}
