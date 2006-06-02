#ifndef EcalSelectiveReadoutSuppressor_h
#define EcalSelectiveReadoutSuppressor_h

#include <vector>
#include "boost/multi_array.hpp"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EcalSelectiveReadout;


class EcalSelectiveReadoutSuppressor 
{
public:
  /// default parameters

  
  EcalSelectiveReadoutSuppressor();
  EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params);
  //  bool accepts(const EBDataFrame& frame) const;
  //bool accepts(const EEDataFrame& frame) const;


  /** Help method to retrieve the trigger tower Et's from the trigger
   * primitives. Values are put in the triggerEt array.
   */
  void setTriggerTowersMap(const CaloSubdetectorGeometry * endcapGeometry,
                           const CaloSubdetectorGeometry * triggerGeometry);
  
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

  /** Sets endcap trigger tower map.
   */
  void setTriggerTowersMap();


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
  
  /** Switch for applying zero suppresion on channel Et instead of on channel
   * E. Default is false.
   */
   bool zeroSuppressOnEt;
   
  /** Trigger tower Et's: see setTriggerTower()
   */
  float triggerEt[nTriggerTowersInEta][nTriggerTowersInPhi];


  /** Array type definition for the endcap crystal->TT map. TT stands for
   * trigger tower.
   * <P>First index: 0 for z<0 endcap, 1 for z>0 endcap<BR>
   * 2nd, 3rd index: x- and y-indices of the crystal<BR>
   * 4th index: 0 to get phi TT index, 1 to get eta TT index
   */
  typedef boost::multi_array<int,4> tower_t;
  
  /** crystal->TT map: see tower_t.
   */
  tower_t tower;
  
  /** Zero suppresion threshold for the ECAL.
   * First index: 0 for barrel, 1 for endcap
   * 2nd index: channel interest (see EcalSelectiveReadout::towerInterest_t
   */
  double zsThreshold[2][4];
  
};

#endif
