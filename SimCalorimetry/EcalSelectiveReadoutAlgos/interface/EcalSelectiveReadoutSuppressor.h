#ifndef EcalSelectiveReadoutSuppressor_h
#define EcalSelectiveReadoutSuppressor_h

#include <vector>
#include "boost/multi_array.hpp"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "FWCore/Framework/interface/EventSetup.h"


class EcalSelectiveReadoutSuppressor 
{
public:
  /// default parameters

  
  //  EcalSelectiveReadoutSuppressor();
  EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params);

  enum {BARREL, ENDCAP};

  static int getFIRTapCount(){ return nFIRTaps;}
  
  /// the mapping of which cell goes with which trigger tower
  void setTriggerMap(const EcalTrigTowerConstituentsMap * map);

  void setGeometry(const CaloGeometry * caloGeometry);
  
  void run(const edm::EventSetup& eventSetup,
	   const EcalTrigPrimDigiCollection & trigPrims,
           EBDigiCollection & barrelDigis,
           EEDigiCollection & endcapDigis);

  void run(const edm::EventSetup& eventSetup,
	   const EcalTrigPrimDigiCollection & trigPrims,
           const EBDigiCollection & barrelDigis,
           const EEDigiCollection & endcapDigis,
           EBDigiCollection & selectedBarrelDigis,
           EEDigiCollection & selectedEndcapDigis);

  //for debugging
  EcalSelectiveReadout* getEcalSelectiveReadout(){return ecalSelectiveReadout;}
  /// three methods I don't know how to implement
  int accumulate(const EcalDataFrame & frame,  bool & gain12saturated);
  double energy(const EcalDataFrame & frame);

  
 private:

  /** Returns true if a digi passes the zero suppression.
   * @param frame, data frame (aka digi). T must be an EEDataFrame
   * or an EBDataFrame 
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

  /**Transforms CMSSW eta ECAL crystal indices to indices starting at 0
   * to use for c-array or vector.
   * @param iEta CMSSW eta index (numbering -85...-1,1...85)
   * @return index in numbering from 0 to 169
   */
  int iEta2cIndex(int iEta) const{
    return (iEta<0)?iEta+85:iEta+84;
  }
  
  /**Transforms CMSSW phi ECAL crystal indices to indices starting at 0
   * to use for c-array or vector.
   * @param iPhi CMSSW phi index (numbering 1...360)
   * @return index in numbering 0...359
   */
  int iPhi2cIndex(int iPhi) const{
    return iPhi-1;
  }

  /**Transforms CMSSW eta ECAL TT indices to indices starting at 0
   * to use for c-array or vector.
   * @param iEta CMSSW eta index (numbering -28...-1,28...56)
   * @return index in numbering from 0 to 55
   */
  int iTTEta2cIndex(int iEta) const{
    return (iEta<0)?iEta+28:iEta+27;
  }
  
  /**Transforms CMSSW phi ECAL crystal indices to indices starting at 0
   * to use for c-array or vector.
   * @param iPhi CMSSW phi index (numbering 1...72)
   * @return index in numbering 0...71
   */
  int iTTPhi2cIndex(int iPhi) const{
    return iPhi-1;
  }

  
  double threshold(const EBDetId & detId) const;
  double threshold(const EEDetId & detId) const;

  void setTtFlags(const edm::EventSetup& eventSetup,
		  const EBDigiCollection& ebDigis,
		  const EEDigiCollection& eeDigis);
  
  void setTtFlags(const EcalTrigPrimDigiCollection & trigPrims);

  template<class T>
  double frame2Energy(const T& frame, int timeOffset = 0) const;

  
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
  double ebGeV2ADC;

  /** Energy->ADC factor used to interpret the zero suppression thresholds
   * for EE.
   */
  double eeGeV2ADC;
  
  /** Depth of DCC zero suppression FIR filter (number of taps),
   * in principal 6.
   */
  static const int nFIRTaps;

  /** DCC zero suppression FIR filter uncalibrated normalized weigths
   */
  std::vector<double> weights;
  
  /** Zero suppresion threshold for the ECAL.
   * First index: 0 for barrel, 1 for endcap
   * 2nd index: channel interest (see EcalSelectiveReadout::towerInterest_t
   */
  double zsThreshold[2][4];


  /** Switch for trigger primitive simulation module bypass debug mode.
   */
  bool trigPrimBypass_;


  /** When in trigger primitive simulation module bypass debug mode,
   * switch to enable Peak finder effect simulation
   */
  bool trigPrimBypassWithPeakFinder_; 
  
  /** Low TT Et threshold for trigger primitive simulation module bypass
   * debug mode.
   */
  double trigPrimBypassLTH_;

  /** Low TT Et threshold for trigger primitive simulation module bypass
   * debug mode.
   */
  double trigPrimBypassHTH_;
};
#endif
