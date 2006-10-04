#ifndef HcalSimAlgos_HcalHitCorrection_h
#define HcalSimAlgos_HcalHitCorrection_h

/** Applies a correction for time slewing
    Makes bigger signals come at a delayed time
  
 \Author Rick Wilkinson
 */

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include <map>
class HcalSimParameterMap;

class HcalHitCorrection : public CaloVHitCorrection
{
public:
  typedef std::map<HcalDetId, double> ChargeSumsByChannel;

  HcalHitCorrection(const HcalSimParameterMap * parameterMap);
  virtual ~HcalHitCorrection() {}

  void fillChargeSums(MixCollection<PCaloHit> & hits);

  void clear();

  /// how much charge we expect from this hit
  double charge(const PCaloHit & hit) const;

  /// how much delay this hit will get
  double delay(const PCaloHit & hit) const;

  /// applies the delay to the hit
  virtual void correct(PCaloHit & hit) const;

  /// which time bin the peak of the signal will fall in
  int timeBin(const PCaloHit & hit) const;

  /// simple average approximation
  double timeOfFlight(const HcalDetId & id) const;

private:

  const HcalSimParameterMap * theParameterMap;

  ChargeSumsByChannel theChargeSumsForTimeBin[10];

};

#endif

