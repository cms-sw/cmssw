#ifndef EcalSelectiveReadoutSuppressor_h
#define EcalSelectiveReadoutSuppressor_h

#include <vector>
#include "boost/multi_array.hpp"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/EcalSelectiveReadoutAlgos/src/EcalSelectiveReadout.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"

#include <memory>

class EcalSelectiveReadoutSuppressor{
public:
  /** Construtor.
   * @param params configuration from python file
   * @param settings configuration from condition DB
   */
  EcalSelectiveReadoutSuppressor(const edm::ParameterSet & params, const EcalSRSettings* settings);
  
  enum {BARREL, ENDCAP};

  /** Gets number of weights supported by the zero suppression filter
   * @return number of weights
   */
  static int getFIRTapCount(){ return nFIRTaps;}
  
  /** Set the mapping of which cell goes with which trigger tower
   * @param map the trigger tower map
   */
  void setTriggerMap(const EcalTrigTowerConstituentsMap * map);

  /** Set the ECAL electronics mapping
   * @param map the ECAL electronics map
   */
  void setElecMap(const EcalElectronicsMapping * map);

  
  /** Sets the geometry of the calorimeters
   */
  void setGeometry(const CaloGeometry * caloGeometry);

  /** Runs the selective readout(SR) algorithm.
   * @deprecated use the other run methode instead
   * @param eventSetup event conditions
   * @param trigPrims the ECAL trigger primitives used as input to the SR.
   * @param barrelDigis [in,out] the EB digi collection to filter
   * @param endcapDigis [in,out] the EE digi collection to filter
   */
  void run(const edm::EventSetup& eventSetup,
	   const EcalTrigPrimDigiCollection & trigPrims,
           EBDigiCollection & barrelDigis,
           EEDigiCollection & endcapDigis);

  /** Runs the selective readout (SR) algorithm.
   * @param eventSetup event conditions
   * @param trigPrims the ECAL trigger primitives used as input to the SR.
   * @param barrelDigis the input EB digi collection
   * @param endcapDigis the input EE digi collection
   * @param selectedBarrelDigis [out] the EB digi passing the SR. Pointer to
   *        the collection to fill. If null, no collection is filled.
   * @param selectedEndcapDigis [out] the EE digi passing the SR. Pointer to
   *        the collection to fill. If null, no collection is filled.
   * @param ebSrFlags [out] the computed SR flags for EB. Pointer to
   *        the collection to fill. If null, no collection is filled.
   * @param eeSrFlags [out] the computed SR flags for EE. Pointer to
   *        the collection to fill. If null, no collection is filled.
   */
  void run(const edm::EventSetup& eventSetup,
	   const EcalTrigPrimDigiCollection & trigPrims,
           const EBDigiCollection & barrelDigis,
           const EEDigiCollection & endcapDigis,
           EBDigiCollection* selectedBarrelDigis,
           EEDigiCollection* selectedEndcapDigis,
	   EBSrFlagCollection* ebSrFlags,
	   EESrFlagCollection* eeSrFlags);

  /** For debugging purposes.
   */
  EcalSelectiveReadout* getEcalSelectiveReadout(){
    return ecalSelectiveReadout.get();
  }

  /** Writes out TT flags. On of the 'run' method must be called beforehand.
   * Beware this method might be removed in future.
   * @param os stream to write to
   * @param iEvent event index. Ignored if <0.
   * @param withHeader. If true writes out a header with the legend.
   */
  void printTTFlags(std::ostream& os, int iEvent = -1,
                    bool withHeader=true) const;
  
 private:

  /** Returns true if a digi passes the zero suppression.
   * @param frame, data frame (aka digi). 
   * @param thr zero suppression threshold in thrUnit.
   * @return true if passed ZS filter, false if failed
   */
  
  bool accept(const edm::DataFrame& frame, int thr);
  
  /// helpers for constructors  
  /** Initializes ZS threshold and SR classificion to SR ("action") flags
   */
  void initCellThresholds(double barrelLowInterest, double endcapLowInterest,
			  double barrelHighInterest, double endcapHighInterest);
  /** Converts threshold in GeV to threshold in internal unit used by the
   * ZS FIR. 
   * @param thresholdInGeV the theshold in GeV
   * @param iSubDet 0 for barrel, 1 for endcap
   * @return threshold in thrUnit unit. INT_MAX means complete suppression,
   * INT_MIN means no zero suppression.
   */
  int internalThreshold(double thresholdInGeV, int iSubDet) const;

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

  /** Help function to set the srFlags field. Used in TrigPrimByPass mode
   * @param eventSetup the EDM event setup
   * @param ebDigi the ECAL barrel APD digis
   * @param eeDigi the ECAL endcap VPT digis
   */
  void setTtFlags(const edm::EventSetup& eventSetup,
		  const EBDigiCollection& ebDigis,
		  const EEDigiCollection& eeDigis);

  /** Help function to set the srFlags field.
   * @param trigPrim the trigger primitive digi collection
   */
  void setTtFlags(const EcalTrigPrimDigiCollection & trigPrims);

  template<class T>
  double frame2Energy(const T& frame, int timeOffset = 0) const;


//   /** Help function to get SR flag from ZS threshold using min/max convention
//    * for SUPPRESS and FULL_READOUT: see zsThreshold.
//    * @param thr ZS threshold in thrUnit
//    * @param flag for Zero suppression: EcalSrFlag::SRF_ZS1 or
//    * EcalSrFlag::SRF_ZS2
//    * @return the SR flag
//    */
//   int thr2Srf(int thr, int zsFlag) const;
  
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
  const static size_t nTriggerTowersInEta
  = 2*nEndcapTriggerTowersInEta+nBarrelTriggerTowersInEta;
  
  /** Number of phi divisions in trigger towers.
   */
  const static size_t nTriggerTowersInPhi = 72;


  /** Help class to comput selective readout flags. 
   */
  std::unique_ptr<EcalSelectiveReadout> ecalSelectiveReadout;

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

  /** Depth of DCC zero suppression FIR filter (number of taps),
   * in principal 6.
   */
  static const int nFIRTaps;

  /** DCC zero suppression FIR filter uncalibrated normalized weigths
   */
  std::vector<float> weights;

  /** Flag to use a symetric zero suppression (cut on absolute value)
   */
  bool symetricZS;
  
  /** Zero suppresion threshold for the ECAL expressed in ebThrUnit and
   * eeThrUnit. Set to numeric_limits<int>::min() for FULL READOUT and
   * to numeric_limits<int>::max() for SUPPRESS.
   * First index: 0 for barrel, 1 for endcap
   * 2nd index: channel interest (see EcalSelectiveReadout::towerInterest_t
   */
  int zsThreshold[2][8];

  /** Internal unit for Zero Suppression threshold (1/4th ADC count) used by
   * the FIR.
   * Index: 0 for barrel, 1 for endcap
   */
  double thrUnit[2];

  /** Switch for trigger primitive simulation module bypass debug mode.
   */
  bool trigPrimBypass_;

  /** Mode selection for "Trig bypass" mode
   * 0: TT thresholds applied on sum of crystal Et's
   * 1: TT thresholds applies on compressed Et from Trigger primitive
   * @see trigPrimByPass_ switch
   */
  int trigPrimBypassMode_;

  /** SR flag (low interest/single/neighbor/center) to action flag
   * (suppress, ZS1, ZS2, FRO) map.
   */
  std::vector<int> actions_;
  
  /** Switch to applies trigPrimBypassLTH_ and trigPrimBypassHTH_ thresholds
   * on TPG compressed ET instead of using flags from TPG: trig prim bypass mode
   * 1.
   */
  bool ttThresOnCompressedEt_;
  
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


  /** Maps RU interest flag (low interest, single neighbour, center) to
   * Selective readout action flag (type of readout).
   * 1st index: 0 for barrel, 1 for endcap
   * 2nd index: RU interest (low, single, neighbour, center,
   *                         forced low, forced single...)
   */
  int srFlags[2][8];

  /** Default TTF to substitute if absent from the trigger primitive collection
   */
  EcalSelectiveReadout::ttFlag_t defaultTtf_;

  /** Number of produced events
   */
  int ievt_;
};
#endif
