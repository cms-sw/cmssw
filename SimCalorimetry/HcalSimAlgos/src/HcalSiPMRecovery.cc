#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMRecovery.h"

HcalSiPMRecovery::HcalSiPMRecovery(double recoveryTime) : 
  theRecoveryTime(recoveryTime), theIntegral(0) {
}

HcalSiPMRecovery::~HcalSiPMRecovery() {
}

int HcalSiPMRecovery::getIntegral(double time) {
  int recoveredPixels = 0;
  std::multimap<double, int>::iterator past;
  for (past = theHistory.begin(); 
       past != theHistory.lower_bound(time-theRecoveryTime); ++past) { 
    recoveredPixels += past->second;
  }
  theHistory.erase(theHistory.begin(), past);
  theIntegral -= recoveredPixels;
  return theIntegral;
}

void HcalSiPMRecovery::addToHistory(double time, int pixels) {
  theHistory.insert(std::pair<double, int>(time,pixels));
  theIntegral += pixels;
}

void HcalSiPMRecovery::clearHistory() {
  theHistory.clear();
  theIntegral = 0;
}
