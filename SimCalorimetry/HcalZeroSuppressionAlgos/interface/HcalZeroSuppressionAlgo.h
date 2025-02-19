#ifndef SimCalorimetry_HcalZeroSuppressionAlgos_HCALZEROSUPPESSIONALGO_H
#define SimCalorimetry_HcalZeroSuppressionAlgos_HCALZEROSUPPESSIONALGO_H 1

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

/** \class HcalZeroSuppessionAlgo
  *  
  * $Date: 2009/10/31 00:02:39 $
  * $Revision: 1.4 $
  * \author J. Mans - Minnesota
  */

class HcalDbService;

class HcalZeroSuppressionAlgo {
public:
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
  HcalZeroSuppressionAlgo(bool markAndPass);
  const HcalDbService* m_dbService;
  
private:
  bool m_markAndPass;
};

#endif
