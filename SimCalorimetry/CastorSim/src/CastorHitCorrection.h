#ifndef CastorSim_CastorHitCorrection_h
#define CastorSim_CastorHitCorrection_h

/** Applies a correction for time slewing
    Makes bigger signals come at a delayed time
  
 */

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include <map>
#include <vector>

class CastorHitCorrection : public CaloVHitCorrection
{
public:
  typedef std::map<DetId, double> ChargeSumsByChannel;

  CastorHitCorrection(const CaloVSimParameterMap * parameterMap);
  virtual ~CastorHitCorrection() {}

  void fillChargeSums(MixCollection<PCaloHit> & hits);

  void fillChargeSums(const std::vector<PCaloHit> & hits);

  void clear();

  /// how much charge we expect from this hit
  double charge(const PCaloHit & hit) const;

  /// how much delay this hit will get
  virtual double delay(const PCaloHit & hit) const;

  /// which time bin the peak of the signal will fall in
  int timeBin(const PCaloHit & hit) const;

  /// simple average approximation
  double timeOfFlight(const DetId & id) const;

private:

  const CaloVSimParameterMap * theParameterMap;

  ChargeSumsByChannel theChargeSumsForTimeBin[10];

};

#endif

