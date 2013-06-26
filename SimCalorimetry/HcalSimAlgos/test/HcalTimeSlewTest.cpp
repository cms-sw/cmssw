#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloCachedShapeIntegrator.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include <cmath>
#include <iostream>

bool compare(CaloSamples & s1, CaloSamples & s2)
{
  bool result = true;
  for(int i = 0; i < 10; ++i) 
  {
    double v1 = s1[i];
    double v2 = s2[i];
    if(fabs(v1) <  0.00001) {
      if(fabs(v2) > 0.0001) {
        std::cout << "BAD VALUE " << v1 << " " << v2 << std::endl;
        result = false;
      }
    } else {
      if(fabs(v1-v2)/v1 > 0.001) {
        std::cout << "BAD VALUE " << v1 << " " << v2 << std::endl;
        result = false;
      }
    }
  }
  return result;
}


double integrate(CaloSamples & s) {
  double result = 0.;
  for(int i = 0; i < 10; ++i) {
    result += s[i];
  } 
  return result;
}

double centroid(CaloSamples & s) {
  double weightedSum = 0.;
  double sum = 0.;
  for(int i = 0; i < 10; ++i) {
    weightedSum += s[i]*i;
    sum += s[i];
  }
  return weightedSum / sum;
}

int main() {
  DetId detId;
  CaloSamples samples(detId, 10.);
  CaloSamples samples2(detId, 10.);
  CaloSamples expected(detId, 10.);
  CaloSamples expected2(detId, 10.);

  double values[10] = {2., 4.,    8., 20.,   32., 28.,   24.,  12., 8., 4.};
  double offset = 5.;
  double offset2 = 30.;
  double expect[10] = {1.6, 3.6, 7.2, 17.6, 29.6, 28.8, 24.8, 14.4, 8.8, 4.8};
  double expect2[10] = {0., 1.6, 3.6, 7.2, 17.6, 29.6, 28.8, 24.8, 14.4, 8.8};

  for(int i = 0; i < 10; ++i) {
    samples[i] = values[i];
    samples2[i] = values[i];
    expected[i] = expect[i];
    expected2[i] = expect2[i];
  }
  samples.offsetTime(offset);
  samples2.offsetTime(offset2);

  compare(samples, expected);
  compare(samples2, expected2);

  // test inching forward and inching back
  HcalShape hcalShape;
  hcalShape.setShape(101);
  CaloCachedShapeIntegrator intShape(&hcalShape);
  CaloSamples signalStep(detId, 10), signalJump(detId, 10), signalOrig(detId, 10);
  for(int i = 0; i < 10; ++i) {
    double value = intShape(i*25);
    signalStep[i] = value;
    signalOrig[i] = value;
    signalJump[i] = value;
  }

  for(int i = 0; i < 15; ++i) {
    signalStep.offsetTime(1);
  }
  // make sure that 15 steps of 1 ns = one step of 15 ns
  signalJump.offsetTime(15);
  //compare(signalStep, signalJump);
  for(int i = 0; i < 15; ++i) {
    signalStep.offsetTime(-1);
  }
  // compare to original
  //compare(signalStep, signalOrig);
  // see if it has the same integral 
  std::cout << "Integrate " << integrate(signalStep) << " " << integrate(signalJump) << " " << integrate(signalOrig) << std::endl;
  std::cout << "Centroid " << centroid(signalOrig)*25 << " " << centroid(signalJump)*25. << std::endl;
}

