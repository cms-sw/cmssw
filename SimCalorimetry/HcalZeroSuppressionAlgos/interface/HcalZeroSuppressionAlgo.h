#ifndef SimCalorimetry_HcalZeroSuppressionAlgos_HCALZEROSUPPESSIONALGO_H
#define SimCalorimetry_HcalZeroSuppressionAlgos_HCALZEROSUPPESSIONALGO_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

/** \class HcalZeroSuppessionAlgo
  *  
  * $Date: 2007/09/24 15:28:01 $
  * $Revision: 1.2 $
  * \author J. Mans - Minnesota
  */

class HcalDbService;

class HcalZeroSuppressionAlgo {
public:
  enum ZSMode { zs_SingleChannel=0, zs_TriggerTowerOR=1, zs_AllDepthsOR=2 };
  void suppress(const HBHEDigiCollection& input, HBHEDigiCollection& output);
  void suppress(const HODigiCollection& input, HODigiCollection& output);
  void suppress(const HFDigiCollection& input, HFDigiCollection& output);
  virtual bool shouldKeep(const HBHEDataFrame& digi) const = 0;
  virtual bool shouldKeep(const HODataFrame& digi) const = 0;
  virtual bool shouldKeep(const HFDataFrame& digi) const = 0;
  void setDbService(const HcalDbService* db) { m_dbService=db; }
  void clearDbService() { m_dbService=0; }
  //  template <class DIGI> bool keepMe(const DIGI& inp, int threshold);

protected:
  HcalZeroSuppressionAlgo(ZSMode mode);
  const HcalDbService* m_dbService;
  
private:
  ZSMode m_mode;
};

#endif
