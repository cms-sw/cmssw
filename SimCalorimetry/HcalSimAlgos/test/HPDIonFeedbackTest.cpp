#include "SimCalorimetry/HcalSimAlgos/src/HPDIonFeedbackSim.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Random/JamesRandom.h"
#include <iostream>

int main()
{
  edm::ParameterSet pset;
  HPDIonFeedbackSim feedbackSim(pset);
  CLHEP::HepJamesRandom engine;
  feedbackSim.setRandomEngine(engine);
  HcalDetId detId(HcalBarrel, 1, 1, 1);

  int nRuns = 1000;

  for(double e = 0; e < 100; e += 10)
  {
    double originalCharge = e*6;
    double chargeSum = 0;
    for(int i = 0; i < nRuns; ++i)
    {
      chargeSum += feedbackSim.getIonFeedback(detId, originalCharge, 0., true, false);
    }
    std::cout << "ENERGY " << e << " FACTOR " << chargeSum/nRuns/6/e*100 << "%" << std::endl;
  }
}
