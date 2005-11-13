///////////////////////////////////////////////////////////////////////////////
// File: HcalQie.h
// Qie simulation for HCal hits
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalQie_H
#define HcalQie_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimG4CMS/Calo/interface/CaloHit.h"

#include <vector>

class HcalQie {

public:

  HcalQie(edm::ParameterSet const & p);
  virtual ~HcalQie();

  std::vector<int>     getCode(int, std::vector<CaloHit>);
  double               getEnergy(std::vector<int>);

private:
  
  std::vector<double>  shape();
  std::vector<int>     code();
  std::vector<double>  charge();
  std::vector<double>  weight(int binofmax, int mode, int npre, int numbucket);
  double               codeToQ(int ic);
  int                  getCode(double charge);
  double               getShape(double time);

private:

  int                  verbosity;
  std::vector<double>  shape_;
  std::vector<int>     code_;
  std::vector<double>  charge_;
  int                  binOfMax, signalBuckets, preSamples, numOfBuckets;
  std::vector<double>  weight_;
  double               sigma, qToPE, eDepPerPE, baseline;
  int                  bmin_, bmax_;
  double               phase_, rescale_;

};

#endif
