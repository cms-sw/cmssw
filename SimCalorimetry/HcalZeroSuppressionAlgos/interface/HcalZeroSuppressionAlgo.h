#ifndef SimCalorimetry_HcalZeroSuppressionAlgos_HCALZEROSUPPESSIONALGO_H
#define SimCalorimetry_HcalZeroSuppressionAlgos_HCALZEROSUPPESSIONALGO_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

/** \class HcalZeroSuppessionAlgo
  *  
  * $Date: 2007/07/12 19:33:49 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalZeroSuppressionAlgo {
public:
  enum ZSMode { zs_SingleChannel=0, zs_TriggerTowerOR=1, zs_AllDepthsOR=2 };
  void suppress(const HBHEDigiCollection& input, HBHEDigiCollection& output);
  void suppress(const HODigiCollection& input, HODigiCollection& output);
  void suppress(const HFDigiCollection& input, HFDigiCollection& output);
  virtual bool shouldKeep(const HBHEDataFrame& digi) const = 0;
  virtual bool shouldKeep(const HODataFrame& digi) const = 0;
  virtual bool shouldKeep(const HFDataFrame& digi) const = 0;
protected:
  HcalZeroSuppressionAlgo(ZSMode mode);
private:
  ZSMode m_mode;
};

#endif
