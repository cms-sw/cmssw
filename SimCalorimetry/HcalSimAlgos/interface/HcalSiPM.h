// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPM_h
#define HcalSimAlgos_HcalSiPM_h

/**

  \class HcalSiPM

  \brief A general implementation for the response of a SiPM.

*/
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "CalibCalorimetry/HcalAlgos/interface/HcalSiPMnonlinearity.h"

namespace CLHEP {
  class HepRandomEngine;
}

class HcalSiPM {
 public:
  HcalSiPM(int nCells = 1, double tau = 15.);

  virtual ~HcalSiPM();

  void resetSiPM() { std::fill(theSiPM.begin(), theSiPM.end(), -999.); }
  virtual double hitCells(CLHEP::HepRandomEngine* , unsigned int pes, double tempDiff = 0.,
			  double photonTime = 0.);


  virtual double totalCharge() const { return totalCharge(theLastHitTime); }
  virtual double totalCharge(double time) const;

  int    getNCells()      const { return theCellCount; }
  double getTau()         const { return theTau; }
  double getCrossTalk()   const { return theCrossTalk; }
  double getTempDep()     const { return theTempDep; }

  void setNCells(int nCells);
  void setTau(double tau);
  void setCrossTalk(double xtalk); //  Borel-Tanner "lambda"
  void setTemperatureDependence(double tempDep);
  void setSaturationPars(const std::vector<float>& pars);

 protected:

  typedef std::pair<unsigned int, std::vector<double> > cdfpair;
  typedef std::unordered_map< unsigned int, cdfpair > cdfmap;

  // void expRecover(double dt);

  double cellCharge(double deltaTime) const;
  unsigned int addCrossTalkCells(CLHEP::HepRandomEngine* engine, unsigned int in_pes);

  //numerical random generation from Borel-Tanner distribution
  double Borel(unsigned int n, double lambda, unsigned int k);
  const cdfpair& BorelCDF(unsigned int k);

  unsigned int theCellCount;
  std::vector< double > theSiPM;
  double theTau;
  double theTauInv;
  double theCrossTalk;
  double theTempDep;
  double theLastHitTime;

  HcalSiPMnonlinearity *nonlin;

  cdfmap borelcdfs;
};

#endif //HcalSimAlgos_HcalSiPM_h
