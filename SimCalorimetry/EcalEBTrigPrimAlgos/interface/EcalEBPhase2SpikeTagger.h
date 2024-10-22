#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTagger_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTagger_h

#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGPedestals.h"

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include <vector>

/** \class EcalEBPhase2SpikeTagger
   Tags spikes on a channel basis
*/

class EcalEBPhase2SpikeTagger {
private:
  bool debug_;
  const EcalLiteDTUPedestals *peds_;
  const EcalEBPhase2TPGLinearizationConstant *linConsts_;
  const EcalTPGCrystalStatusCode *badXStatus_;

public:
  EcalEBPhase2SpikeTagger(bool debug);
  virtual ~EcalEBPhase2SpikeTagger();

  bool process(const std::vector<int> &linInput);
  void setParameters(EBDetId id,
                     const EcalLiteDTUPedestalsMap *peds,
                     const EcalEBPhase2TPGLinearizationConstMap *ecaltplin,
                     const EcalTPGCrystalStatus *ecaltpBadX);
};

#endif
