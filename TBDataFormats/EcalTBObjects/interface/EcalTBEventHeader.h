#ifndef RECECAL_ECALTBEVENTHEADER_H
#define RECECAL_ECALTBEVENTHEADER_H 1

#include <ostream>
#include <string>
#include <vector>
#include "DataFormats/EcalDetId/interface/EBDetId.h"

/** \class EcalTBEventHeader
 *  Container for event ancilllary informations defined in TB raw data formats  
 *
 *  $Id: EcalTBEventHeader.h,v 1.9 2007/01/08 15:30:34 meridian Exp $
 */


class EcalTBEventHeader {

 public:
  
  EcalTBEventHeader() {};
  
  ~EcalTBEventHeader() {};

  typedef struct magnetsMeasurement {
    int    magnet6IRead_ampere;
    int    magnet6ISet_ampere;

    int    magnet7IRead_ampere;
    int    magnet7ISet_ampere;

    int    magnet7VMeas_uvolt;
    int    magnet7IMeas_uampere;

    int    magnet6VMeas_uvolt;
    int    magnet6IMeas_uampere;
  } magnetsMeasurement_t;

  //! Returns the event number
  int eventNumber() const{
    return eventNumber_;
  }

  int runNumber() const{
    return runNumber_;
  }

  //! Returns the burst number
  short burstNumber() const{
    return burstNumber_;
  }

  //! Returns the burst number
  short smInBeam() const {
    return smInBeam_;
  }

  //! Returns the begin burst time (sec)
  int begBurstTimeSec() const{
    return begBurstTime_sec_;
  }

  //! Returns the begin burst time (msec)
  int begBurstTimeMsec() const{
    return begBurstTime_msec_;
  }

  //! Returns the end burst time (sec)
  int endBurstTimeSec() const{
    return endBurstTime_sec_;
  }

  //! Returns the end burst time (msec)
  int endBurstTimeMsec() const{
    return endBurstTime_msec_;
  }

  //! Returns the begin burst time (msec)
  int begBurstLV1A() const{
    return begBurstLV1A_;
  }

  //! Returns the end burst time (sec)
  int endBurstLV1A() const{
    return endBurstLV1A_;
  }
  

  // Return the event type: "beam", "laser", "pedestal". "error"
  // or a number orresponding to the orginal eventype stored in 
  // the RRF. 
  std::string eventType() const ;

  //! Returns the event type as in the H4ROOTDB
  int dbEventType() const ;

  //! Returns the trigger mask
  int triggerMask() const {
    return triggerMask_;
  }

  //! Returns the date in Unix time
  int date() const {
    return date_;
  }

  //! Returns the crystal which is being hit by the beam (in the internal SM numbering scheme)

  //Table supervisor information
  int crystalInBeam() const {
    return EBDetId(crystalInBeam_).ic();
  }
  int nominalCrystalInBeam() const {
    return EBDetId(nominalCrystalInBeam_).ic();
  }
  int nextCrystalInBeam() const {
    return EBDetId(nextCrystalInBeam_).ic();
  }
  //! Returns the theta table index
  unsigned int thetaTableIndex() const { return thetaTableIndex_; }
  //! Returns the phi table index
  unsigned int phiTableIndex() const { return phiTableIndex_; }
  //! Tell if the table is Moving
  bool tableIsMoving() const { return tableIsMoving_;} 
  //! Tell if the table is Moving At the begin of the spill (available in runs only after 1st August 06)
  bool tableIsMovingAtBegSpill() const { return tableIsMovingAtBegSpill_; }
  //! is there any sync error
  bool syncError() const { return syncError_; }

  ///SHOULD WE REMOVE ALL THIS???
  //! Unique codes for the 4 lasers
  /*! The Caltech laser system (resp. R. Zhu, contact at CERN A. Bornheim)
       is a two-laser (2 times "YLF and Ti:Sapphire lasers") which provides 
       25 ns (FWHM) pulses at 4 different wavelengths:
          -# Laser 1  Blue (violet)  440 nm 
                      Green (blue)   495 nm
          -# Laser 2  Red            709 nm
                      Infrared       801 nm
  */
  // FIXME: add the codes used by Jean Bourotte here.
  enum LaserType {
    LBlue     = 440, //! 440 nm
    LGreen    = 495, //! 495 nm
    LRed      = 709, //! 709 nm
    LInfrared = 800  //! 800 nm
  };
  //! return the laser intensity
  int lightIntensity() const {
    return lightIntensity_;
  }
  //! return the event laser type
  int laserType() const {
    return laserType_; // returns wavelength
  }

  LaserType laserTypeName() const {
    LaserType laser_type;
    switch(laserType_){
    case 440:  laser_type = LBlue;       break;
    case 495:  laser_type = LGreen;      break;
    case 709:  laser_type = LRed;        break;
    case 800:  laser_type = LInfrared;   break;
    default:   laser_type = LRed;        break;
    }
    return laser_type; // returns laserTypeName
  }

  //! returns S6 ADC value
  int S6ADC () const { 
    return S6ADC_ ; 
    }

  int  numberOfMagnetMeasurements() const { return numberOfMagnetMeasurements_; }
  
  const std::vector<magnetsMeasurement_t>&  magnetMeasurements() const { return magnetMeasurements_; }

  int  scaler(int scalerId) const {
    if (scalerId < ((int) scalers_.size() ) )
      return scalers_[scalerId];
    else
      return -1;
  }
  
  ///////////////
  //Set Methods

  void setEventNumber(const int& eventNumber) { eventNumber_=eventNumber; }

  void setRunNumber(const int& runNumber) { runNumber_=runNumber; }

  void setSmInBeam(const int& smInBeam) { smInBeam_ = smInBeam; }

  void setBurstNumber(const short& burstNumber ) { burstNumber_=burstNumber; }

  void setTriggerMask(const int& triggerMask ) { triggerMask_=triggerMask; }

  void setBegBurstTimeSec(const int&  begBurstTimeSec) { begBurstTime_sec_ = begBurstTimeSec; }

  void setBegBurstTimeMsec(const int&  begBurstTimeMsec) { begBurstTime_msec_ = begBurstTimeMsec; }

  void setEndBurstTimeSec(const int&  endBurstTimeSec) { endBurstTime_sec_ = endBurstTimeSec; }

  void setEndBurstTimeMsec(const int&  endBurstTimeMsec) { endBurstTime_msec_ = endBurstTimeMsec; }

  void setBegBurstLV1A(const int&  begBurstLV1A) { begBurstLV1A_ = begBurstLV1A; }

  void setEndBurstLV1A(const int&  endBurstLV1A) { endBurstLV1A_ = endBurstLV1A; }

  void setDate(const int& date ) { date_=date; }

  void setCrystalInBeam(const DetId& crystalInBeam ) { crystalInBeam_=crystalInBeam; }

  void setNominalCrystalInBeam(const DetId& crystalInBeam ) { nominalCrystalInBeam_=crystalInBeam; }

  void setNextCrystalInBeam(const DetId& crystalInBeam ) { nextCrystalInBeam_=crystalInBeam; }

  void setThetaTableIndex(const unsigned int& thetaTableIndex ) { thetaTableIndex_=thetaTableIndex; }

  void setPhiTableIndex(const unsigned int& phiTableIndex ) { phiTableIndex_=phiTableIndex; }

  void setTableIsMoving(const bool& tableIsMoving ) { tableIsMoving_=tableIsMoving; }

  void setTableIsMovingAtBegSpill(const bool& tableIsMoving ) { tableIsMovingAtBegSpill_=tableIsMoving; }

  void setSyncError(const bool& syncError ) { syncError_ = syncError; }

  void setLightIntensity(const int& lightIntensity) { lightIntensity_=lightIntensity; }

  void setLaserType(const int& laserType) { laserType_ = laserType; }

  void setS6ADC (const int& S6ADC) { S6ADC_ = S6ADC ; }

  void setNumberOfMagnetMeasurements(const int& numberOfMagnetMeasurements) { numberOfMagnetMeasurements_ = numberOfMagnetMeasurements;}
  void setMagnetMeasurements(const std::vector<magnetsMeasurement_t>& magnetMeasurements) { magnetMeasurements_ = magnetMeasurements;}

  void setScalers (const std::vector<int> & scalers) { scalers_ = scalers; }


 private:

  int      eventNumber_;      ///< The numner of the event 
  int      runNumber_;        ///< The number of the run
  short    burstNumber_;      ///< The number of the burst


  /// Information from the ecalSupervisor
  int      begBurstTime_sec_;
  int      begBurstTime_msec_;
  int      endBurstTime_sec_;
  int      endBurstTime_msec_;
  int      begBurstLV1A_;
  int      endBurstLV1A_;


  int      triggerMask_;      ///< The trigger mask 

  /// Magnet currents 
  int      numberOfMagnetMeasurements_;
  std::vector<magnetsMeasurement_t> magnetMeasurements_;
  
  /*   int      nominalMagnet6ReadAmpere_; */
  /*   int      nominalMagnet7ReadAmpere_; */
  /*   int      nominalMagnet6SetAmpere_; */
  /*   int      nominalMagnet7SetAmpere_; */
  /*   int      measuredMagnet6MicroAmpere_; */
  /*   int      measuredMagnet7MicroAmpere_; */
  /*   int      measuredMagnet6Volt_; */
  /*   int      measuredMagnet7Volt_; */


  int      date_;             ///< The date when the run was taken

  /// Information from the table Supervisor  
  DetId    crystalInBeam_;    ///< The current crystal hit by the beam
  DetId    nominalCrystalInBeam_;    ///< The nominal crystal which should be hit by the beam
  DetId    nextCrystalInBeam_;    ///< The nominal next crystal which should be hit by the beam
  unsigned int    thetaTableIndex_; ///< Theta table index (X)
  unsigned int    phiTableIndex_;   ///< Phi table index (Y)

  bool tableIsMoving_;

  bool tableIsMovingAtBegSpill_; //indipendent but from CAMAC one saying if table is moving at the begin of the spill

  //Sync error for Camac stuff
  bool syncError_;

  //FIXME for use in CMSSW(Probably unuseful when reading from new RawData Information will be stored in EcalDCCHeaderBlock)
  int      lightIntensity_;   ///< The light intensity
  int      laserType_;        ///< The laser type --see enum LaserType

  int      S6ADC_ ;           ///< ADC lecture of the S6 counter

  short smInBeam_;

  // array of 36 scalers
  std::vector<int> scalers_;

};

std::ostream& operator<<(std::ostream&, const EcalTBEventHeader&);
  
#endif
