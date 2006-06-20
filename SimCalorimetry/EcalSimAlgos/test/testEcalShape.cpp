#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShape.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<iostream>
#include<iomanip>

int main() {

  edm::MessageDrop::instance()->debugEnabled = false;

  EcalSimParameterMap parameterMap;

  EBDetId barrel(1,1);
  double thisPhase = parameterMap.simParameters(barrel).timePhase();
  EcalShape theShape(thisPhase);

  std::cout << "Parameters for the ECAL MGPA shape \n" << std::endl;

  std::cout << "Rising time for ECAL shape (timePhase) = " << parameterMap.simParameters(barrel).timePhase() << std::endl;
  std::cout << "Bin of maximum = " << parameterMap.simParameters(barrel).binOfMaximum() << std::endl;

  theShape.display();

  //double ToM = theShape.computeTimeOfMaximum();
  //double T0 = theShape.computeT0();
  //double risingTime = Tom - T0;

  //std::cout << "\n Maximum time from tabulated values = " << std::setprecision(2) << ToM << std::endl;
  //std::cout << "\n Tzero from tabulated values        = " << std::setprecision(2) << T0 << std::endl;
  //std::cout << "\n Rising time from tabulated values  = " << std::setprecision(2) << risingTime << std::endl;

  return 0;

} 
