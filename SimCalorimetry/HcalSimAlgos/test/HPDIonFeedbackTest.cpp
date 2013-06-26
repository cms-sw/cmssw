#include "SimCalorimetry/HcalSimAlgos/interface/HPDIonFeedbackSim.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Random/JamesRandom.h"
#include <iostream>

int main()
{
  edm::ParameterSet pset;

  HcalShapes* theShapes = new HcalShapes();
  HPDIonFeedbackSim* feedbackSim = new HPDIonFeedbackSim(pset, theShapes);

  //  HPDIonFeedbackSim feedbackSim(pset, theShapes);
  CLHEP::HepJamesRandom engine;
  feedbackSim->setRandomEngine(engine);
  HcalDetId detId(HcalBarrel, 1, 1, 1);

  int nRuns = 1000;

  for(double e = 0; e < 100; e += 10)
  {
    double originalCharge = e*6;
    double chargeSum = 0;
    for(int i = 0; i < nRuns; ++i)
    {
      chargeSum += feedbackSim->getIonFeedback(detId, originalCharge, 0., true, false);
    }
    if(e > 1.e-20)
    std::cout << "ENERGY " << e << " FACTOR " << chargeSum/nRuns/6/e*100 << "%" << std::endl;
  }

  // test thermal noise

  for(int i = 0; i < 100; ++i)
  {
    CaloSamples samples(detId, 10);
    feedbackSim->addThermalNoise(samples);
    if(samples[7] > 1.e-20) std::cout << samples << std::endl; 
  }
    
}
