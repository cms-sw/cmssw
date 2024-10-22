#ifndef SimG4CMS_Calo_HcalQie_H
#define SimG4CMS_Calo_HcalQie_H
///////////////////////////////////////////////////////////////////////////////
// File: HcalQie.h
// Qie simulation for HCal hits
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloHit/interface/CaloHit.h"

#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

class HcalQie {
public:
  HcalQie(edm::ParameterSet const& p);
  virtual ~HcalQie();

  std::vector<int> getCode(int, const std::vector<CaloHit>&, CLHEP::HepRandomEngine*);
  double getEnergy(const std::vector<int>&);

private:
  std::vector<double> shape();
  std::vector<int> code();
  std::vector<double> charge();
  std::vector<double> weight(int binofmax, int mode, int npre, int numbucket);
  double codeToQ(int ic);
  int getCode(double charge);
  double getShape(double time);

private:
  std::vector<double> shape_;
  std::vector<int> code_;
  std::vector<double> charge_;
  int binOfMax, signalBuckets, preSamples, numOfBuckets;
  std::vector<double> weight_;
  double sigma, qToPE, eDepPerPE, baseline;
  int bmin_, bmax_;
  double phase_, rescale_;
};

#endif
