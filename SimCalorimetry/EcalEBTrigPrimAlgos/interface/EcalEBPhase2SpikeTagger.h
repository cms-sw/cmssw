#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTagger_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTagger_h

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <vector>

/** \class EcalEBPhase2SpikeTagger
   Tags spikes on a channel basis
*/

class EcalEBPhase2SpikeTagger {
public:
  EcalEBPhase2SpikeTagger(edm::ConsumesCollector &cc, bool debug) : debug_(debug), badXStatus_(nullptr) {};
  virtual ~EcalEBPhase2SpikeTagger() {};

  virtual bool process(const std::vector<int> &linInput) = 0;
  virtual void getRecords(edm::EventSetup const &setup) = 0;
  virtual void setParameters(EBDetId id, const EcalTPGCrystalStatus *ecaltpBadX) = 0;

protected:
  bool debug_;
  const EcalTPGCrystalStatusCode *badXStatus_;
};

#endif
