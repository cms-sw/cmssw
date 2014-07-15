// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPM_h
#define HcalSimAlgos_HcalSiPM_h

/**

  \class HcalSiPM

  \brief A general implementation for the response of a SiPM.

*/

#include <vector>
#include <algorithm>

namespace CLHEP {
  class HepRandomEngine;
}

class HcalSiPM {
 public:
  HcalSiPM(int nCells = 1, double tau = 15.);

  virtual ~HcalSiPM();

  void resetSiPM() { std::fill(theSiPM.begin(), theSiPM.end(), -999.); }
  virtual int hitCells(CLHEP::HepRandomEngine*, unsigned int photons, unsigned int integral = 0) const;
  virtual double hitCells(CLHEP::HepRandomEngine* , unsigned int pes, double tempDiff = 0.,
			  double photonTime = 0.);


  virtual double totalCharge() const { return totalCharge(theLastHitTime); }
  virtual double totalCharge(double time) const;
  // virtual void recoverForTime(double time, double dt = 0.);

  int getNCells() const { return theCellCount; }
  double getTau() const { return 1.0/theTauInv; }
  double getCrossTalk() const { return theCrossTalk; }
  double getTempDep() const { return theTempDep; }

  void setNCells(int nCells);
  void setTau(double tau) {theTauInv=1.0/tau;}
  void setCrossTalk(double xtalk);
  void setTemperatureDependence(double tempDep);

 protected:

  // void expRecover(double dt);

  double cellCharge(double deltaTime) const;

  unsigned int theCellCount;
  std::vector< double > theSiPM;
  double theTauInv;
  double theCrossTalk;
  double theTempDep;
  double theLastHitTime;
};

#endif //HcalSimAlgos_HcalSiPM_h
