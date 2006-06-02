#ifndef EcalSelectiveReadoutSuppressor_h
#define EcalSelectiveReadoutSuppressor_h

#include <vector>
#include "boost/multi_array.hpp"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalSelectiveReadout;


class EcalSelectiveReadoutSuppressor 
{
public:
  /// default parameters

  
  EcalSelectiveReadoutSuppressor();
  EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params);

  enum {BARREL, ENDCAP};

  /// the mapping of which cell goes with which trigger tower
  void setTriggerMap(const EcalTrigTowerConstituentsMap * map);

  void setGeometry(const CaloGeometry * caloGeometry);
  
  void run(const EcalTrigPrimDigiCollection & trigPrims,
           EBDigiCollection & barrelDigis,
           EEDigiCollection & endcapDigis);

  void run(const EcalTrigPrimDigiCollection & trigPrims,
           const EBDigiCollection & barrelDigis,
           const EEDigiCollection & endcapDigis,
           EBDigiCollection & selectedBarrelDigis,
           EEDigiCollection & selectedEndcapDigis);
 
 private:
  /// helpers for constructors
  /** When a trigger tower (TT) is classified
   * as 'center', the TTs in the area (deltaEta+1)x(deltaPhi+1) 
   * around the 'center' are classified as 'neighbour'.
   *
   * The thresholds are the Low Et and High Et 
   * threshold for selective readout trigger tower classification
   */
  void initTowerThresholds(double lowThreshold, double highThreshold, int deltaEta, int deltaPhi);
  void initCellThresholds(double barrelLowInterest, double endcapLowInterest);


  /// three methods I don't know how to implement
  double energy(const EBDataFrame & barrelDigi) const;
  double energy(const EEDataFrame & endcapDigi) const;
  double Et(const EcalTriggerPrimitiveDigi & trigPrim) const;

  double threshold(const EBDetId & detId) const;
  double threshold(const EEDetId & detId) const;

  void setTriggerTowers(const EcalTrigPrimDigiCollection & trigPrims);
  /** Number of endcap, obviously two.
   */
  const static size_t nEndcaps = 2;

  /** Number of eta trigger tower divisions in one endcap.
   */
  const static size_t nEndcapTriggerTowersInEta = 11;
  
  /** Number of eta trigger tower divisions in the barrel.
   */
  const static size_t nBarrelTriggerTowersInEta = 34;
  
  /** Number of eta divisions in trigger towers for the whole ECAL
   */
  const static size_t nTriggerTowersInEta = 2*nEndcapTriggerTowersInEta+nBarrelTriggerTowersInEta;
  
  /** Number of phi divisions in trigger towers.
   */
  const static size_t nTriggerTowersInPhi = 72;


  /** Help class to comput selective readout flags. 
   */
  EcalSelectiveReadout* ecalSelectiveReadout;

  const EcalTrigTowerConstituentsMap * theTriggerMap;
  
  /** Switch for applying zero suppresion on channel Et instead of on channel
   * E. Default is false.
   */
   bool zeroSuppressOnEt;
   
  /** Trigger tower Et's: see setTriggerTower()
   */
  float triggerEt[nTriggerTowersInEta][nTriggerTowersInPhi];


  /** Zero suppresion threshold for the ECAL.
   * First index: 0 for barrel, 1 for endcap
   * 2nd index: channel interest (see EcalSelectiveReadout::towerInterest_t
   */
  double zsThreshold[2][4];
  
};

#endif
