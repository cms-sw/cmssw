#ifndef HcalSimAlgos_HcalHitCorrection_h
#define HcalSimAlgos_HcalHitCorrection_h

/** Applies a correction for time slewing
    Makes bigger signals come at a delayed time
  
 \Author Rick Wilkinson
 */

#include "SimCalorimetry/CaloSimAlgos/interface/CaloVHitCorrection.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "CLHEP/Random/RandGaussQ.h"
#include <map>
#include <vector>

class HcalHitCorrection : public CaloVHitCorrection
{
public:
  typedef std::map<DetId, double> ChargeSumsByChannel;

  HcalHitCorrection(const CaloVSimParameterMap * parameterMap);
  virtual ~HcalHitCorrection() {}

  void fillChargeSums(MixCollection<PCaloHit> & hits);
  void fillChargeSums(std::vector<PCaloHit> & hits);

  void clear();

  /// how much charge we expect from this hit
  double charge(const PCaloHit & hit) const;

  /// how much delay this hit will get
  virtual double delay(const PCaloHit & hit) const;

  /// which time bin the peak of the signal will fall in
  int timeBin(const PCaloHit & hit) const;

  /// simple average approximation
  double timeOfFlight(const DetId & id) const;

  void setRandomEngine(CLHEP::HepRandomEngine & engine);

private:

  const CaloVSimParameterMap * theParameterMap;

  ChargeSumsByChannel theChargeSumsForTimeBin[10];

  CLHEP::RandGaussQ* theRandGaussQ;

};

#endif

