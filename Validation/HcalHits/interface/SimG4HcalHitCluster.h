///////////////////////////////////////////////////////////////////////////////
// File: SimG4HcalHitCluster.h
// Cluster class for analysis in SimG4HcalValidation
///////////////////////////////////////////////////////////////////////////////
#ifndef Validation_HcalHits_SimG4HcalHitCluster_H
#define Validation_HcalHits_SimG4HcalHitCluster_H

#include "SimDataFormats/CaloHit/interface/CaloHit.h"
#include <iostream>
#include <vector>

class SimG4HcalHitCluster {
public:
  SimG4HcalHitCluster();
  virtual ~SimG4HcalHitCluster();

  double e() const { return ec; }
  double eta() const { return etac; }
  double phi() const { return phic; }
  std::vector<CaloHit> *getHits() { return &hitsc; }

  bool operator<(const SimG4HcalHitCluster &cluster) const;
  SimG4HcalHitCluster &operator+=(const CaloHit &hit);

  double collectEcalEnergyR();

private:
  double my_cosh(float eta) { return 0.5 * (exp(eta) + exp(-eta)); }
  double my_sinh(float eta) { return 0.5 * (exp(eta) - exp(-eta)); }

  double ec, etac, phic;
  std::vector<CaloHit> hitsc;
};

std::ostream &operator<<(std::ostream &, const SimG4HcalHitCluster &);

#endif
