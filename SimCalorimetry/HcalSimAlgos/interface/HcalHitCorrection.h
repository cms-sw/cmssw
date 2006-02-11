#ifndef HcalSimAlgos_HcalHitCorrection_h
#define HcalSimAlgos_HcalHitCorrection_h

/** Applies a correction for time slewing
    Makes bigger signals come at a delayed time
  
 \Author Rick Wilkinson
 */

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
class HcalSimParameterMap;

class HcalHitCorrection : public CaloVHitCorrection
{
public:
  HcalHitCorrection(const HcalSimParameterMap * parameterMap);

  virtual void correct(PCaloHit & hit) const;

private:

  const HcalSimParameterMap * theParameterMap;

};

#endif

