#ifndef EcalSelectiveReadoutSuppressor_h
#define EcalSelectiveReadoutSuppressor_h

#include <vector>
#include "boost/multi_array.hpp"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"


class EcalSelectiveReadoutSuppressor 
{
public:
  /// default parameters

  
  //  EcalSelectiveReadoutSuppressor();
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

  //for debugging
  EcalSelectiveReadout* getEcalSelectiveReadout(){return ecalSelectiveReadout;}
  
 private:

  /** Returns true if a digi passes the zero suppression.
   * @param frame, data frame (aka digi). Must be of type EEDataFrame or
   * EBDataFrame.
   * @para zero suppression threshold.
   */
  template<class T>
  bool accept(const T& frame, float threshold);
  
  /// helpers for constructors
  /** When a trigger tower (TT) is classified
   * as 'center', the TTs in the area (deltaEta+1)x(deltaPhi+1) 
   * around the 'center' are classified as 'neighbour'.
   *
   * The thresholds are the Low Et and High Et 
   * threshold for selective readout trigger tower classification
   */
  void initTowerThresholds(double lowThreshold, double highThreshold, int deltaEta, int deltaPhi);
  void initCellThresholds(double barrelLowInterest, double endcapLowInterest,
			  double barrelHighInterest, double endcapHighInterest);


  /// three methods I don't know how to implement
  double energy(const EBDataFrame & barrelDigi) const;
  double energy(const EEDataFrame & endcapDigi) const;
  double Et(const EcalTriggerPrimitiveDigi & trigPrim) const;

  /** Gets the integer weights used by the zero suppression
   * FIR filter.
   *<P><U>Weight definitions:</U>
   *<UL>
   *<LI>Uncalibrated normalized weights are defined as such that when applied
   * to the average pulse with the highest sample normalized to 1, the
   * result is 1.
   *<LI>Calibrated weights are defined for each crystal, as uncalibrated
   * normalized weights multiplied by an intercalibration constant which
   * is expected to be between 0.6 and 1.4
   *<LI>FIR weights are defined for each crystal as the closest signed integers
   * to 2**10 times the calibrated weigths. The absolute value of these
   * weights should not be greater than (2**12-1).
   *</UL>
   * If a FIR weights exceeds the (2**12-1) absolute value limit, its
   * absolute value is replaced by (2**12-1).
   */
  std::vector<int> getFIRWeigths();
  
  double threshold(const EBDetId & detId) const;
  double threshold(const EEDetId & detId) const;

  void setTtFlags(const EcalTrigPrimDigiCollection & trigPrims);
  
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
     
  /** Trigger tower flags: see setTtFlags()
   */
  EcalSelectiveReadout::ttFlag_t ttFlags[nTriggerTowersInEta][nTriggerTowersInPhi];

  /** Time position of the first sample to use in zero suppession FIR
   * filter. Numbering starts at 0.
   */
  int firstFIRSample;
  
  /** Weights of zero suppression FIR filter
   */
  std::vector<int> firWeights;

  /** Energy->ADC factor used to interpret the zero suppression thresholds
   * for EB
   */
  double ebMeV2ADC;

  /** Energy->ADC factor used to interpret the zero suppression thresholds
   * for EE.
   */
  double eeMeV2ADC;
  
  /** Deep of DCC zero suppression FIR filter (number of taps), in principal 6.
   */
  int nFIRTaps;

  /** DCC zero suppression FIR filter uncalibrated normalized weigths
   */
  std::vector<double> weights;
  
  /** Zero suppresion threshold for the ECAL.
   * First index: 0 for barrel, 1 for endcap
   * 2nd index: channel interest (see EcalSelectiveReadout::towerInterest_t
   */
  double zsThreshold[2][4];
};

#endif
