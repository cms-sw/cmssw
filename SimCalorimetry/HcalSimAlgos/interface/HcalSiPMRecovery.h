// -*- C++ -*-

#ifndef HcalSimAlgos_HcalSiPMRecovery_h
#define HcalSimAlgos_HcalSiPMRecovery_h

#include <map>

class HcalSiPMRecovery {
public:
  HcalSiPMRecovery(double recoveryTime = 250.);

  ~HcalSiPMRecovery();

  int getIntegral(double time);
  void addToHistory(double time, int pixels);
  void clearHistory();
  void setRecoveryTime(double recoveryTime) { theRecoveryTime = recoveryTime; }

protected:

  double theRecoveryTime;
  int theIntegral;
  std::multimap<double, int> theHistory;
};

#endif //HcalSimAlgos_HcalSiPMRecovery_h
