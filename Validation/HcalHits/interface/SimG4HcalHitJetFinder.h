///////////////////////////////////////////////////////////////////////////////
// File: SimG4HcalHitJetFinder.h
// Jet finder class for analysis in SimG4HcalValidation
///////////////////////////////////////////////////////////////////////////////
#ifndef Validation_HcalHits_SimG4HcalHitJetFinder_H
#define Validation_HcalHits_SimG4HcalHitJetFinder_H

#include "SimDataFormats/CaloHit/interface/CaloHit.h"
#include "Validation/HcalHits/interface/SimG4HcalHitCluster.h"

#include <vector>

class SimG4HcalHitJetFinder {
public:
  SimG4HcalHitJetFinder(double cone = 0.5);
  virtual ~SimG4HcalHitJetFinder();

  void setCone(double);
  void setInput(std::vector<CaloHit> *);
  std::vector<SimG4HcalHitCluster> *getClusters(bool);
  double rDist(const SimG4HcalHitCluster *, const CaloHit *) const;
  double rDist(const double, const double, const double, const double) const;

private:
  double jetcone;
  std::vector<CaloHit> input;
  std::vector<SimG4HcalHitCluster> clusvector;
};

#endif
