#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2Linearizer_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2Linearizer_h

#include "DataFormats/EcalDigi/interface/EcalLiteDTUSample.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGPedestals.h"

#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"

#include <vector>

/** \class EcalEBPhase2Linearizer
\author L. Lutton, N. Marinelli - Univ. of Notre Dame
 Description: forPhase II 
 Performs the linearization of signal from Catia+LiteDTU 
*/

class EcalEBPhase2Linearizer {
private:
  bool debug_;
  int uncorrectedSample_;
  int gainID_;
  uint base_;
  uint mult_;
  uint shift_;
  int strip_;
  bool init_;
  float gainDivideByTen_ = 0.1;
  std::vector<uint> coeffs_;
  uint coeff_;
  //I2C Stuff. Would eventually get from outside linearizer (e.g., database)
  //Would also be different for each crystal
  uint I2CSub_;

  const EcalLiteDTUPedestals *peds_;
  const EcalEBPhase2TPGLinearizationConstant *linConsts_;
  const EcalTPGCrystalStatusCode *badXStatus_;

  std::vector<const EcalTPGCrystalStatusCode *> vectorbadXStatus_;

  int setInput(const EcalLiteDTUSample &RawSam);

  int doOutput();

public:
  EcalEBPhase2Linearizer(bool debug);
  virtual ~EcalEBPhase2Linearizer();

  void process(const EBDigiCollectionPh2::Digi &df, std::vector<int> &output_percry);
  void setParameters(EBDetId id,
                     const EcalLiteDTUPedestalsMap *peds,
                     const EcalEBPhase2TPGLinearizationConstMap *ecaltplin,
                     const EcalTPGCrystalStatus *ecaltpBadX);
};

#endif
