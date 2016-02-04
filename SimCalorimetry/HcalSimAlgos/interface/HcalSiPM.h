// -*- C++ -*-
#ifndef HcalSimAlgos_HcalSiPM_h
#define HcalSimAlgos_HcalSiPM_h

/**

  \class HcalSiPM

  \brief A general implementation for the response of a SiPM.

*/

#include <vector>

#include "CLHEP/Random/RandGaussQ.h"

class HcalSiPM {
 public:
  HcalSiPM(int nCells = 1);

  virtual ~HcalSiPM();

  virtual int hitCells(int photons, int integral = 0) const;

  int getNCells() const { return theCellCount; }
  void setNCells(int nCells);
  void initRandomEngine(CLHEP::HepRandomEngine& engine);

 protected:
  virtual double errOnX(double x, double prehit = 0.) const;
  void getBeforeAndAfter(double val, int& before, int& after, 
			 const std::vector<double>& vec) const;

  int theCellCount;
  mutable CLHEP::RandGaussQ *theRndGauss;

  std::vector< double > theXSamples;
  std::vector< double > thePrehitSamples;
  std::vector< std::vector< double > > theErrSamples;

  void defaultErrInit();

};

#endif //HcalSimAlgos_HcalSiPM_h
