#ifndef RECECAL_ECALTBEVENTHEADER_H
#define RECECAL_ECALTBEVENTHEADER_H 1

#include <ostream>
#include <string>
#include "DataFormats/EcalDetId/interface/EBDetId.h"

/** \class EcalTBEventHeader
 *  Simple container for TDC reconstructed informations 
 *
 *
 *  $Id: $
 */


class EcalTBEventHeader {

 public:
  
  EcalTBEventHeader() {};
  
  ~EcalTBEventHeader() {};

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
  

  //! Returns the event number
  int eventNumber() const{
    return eventNumber_;
  }

  //! Returns the burst number
  int burstNumber() const{
    return burstNumber_;
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
  int crystalInBeam() const {
    return EBDetId(crystalInBeam_).ic();
  }

  //! Returns the theta table index
  int thetaTableIndex() const { return thetaTableIndex_; }

  //! Returns the phi table index
  int phiTableIndex() const { return phiTableIndex_; }

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
  
  //Set Methods

  void setEventNumber(const int& eventNumber) { eventNumber_=eventNumber; }

  void setBurstNumber(const short& burstNumber ) { burstNumber_=burstNumber; }

  void setTriggerMask(const int& triggerMask ) { triggerMask_=triggerMask; }

  void setDate(const int& date ) { date_=date; }

  void setCrystalInBeam(const DetId& crystalInBeam ) { crystalInBeam_=crystalInBeam; }

  void setThetaTableIndex(const int& thetaTableIndex ) { thetaTableIndex_=thetaTableIndex; }

  void setPhiTableIndex(const int& phiTableIndex ) { phiTableIndex_=phiTableIndex; }

  void setLightIntensity(const int& lightIntensity) { lightIntensity_=lightIntensity; }

  void setLaserType(const int& laserType) { laserType_ = laserType; }

 private:

  int      eventNumber_;      ///< The number of the event
  short    burstNumber_;      ///< The number of the burst

  int      triggerMask_;      ///< The trigger mask 
  
  int      date_;             ///< The date when the run was taken
  
  DetId    crystalInBeam_;    ///< The crystal hit by the beam

  int      thetaTableIndex_; ///< Theta table index
  int      phiTableIndex_;   ///< Phi table index

  //FIXME for use in CMSSW(Probably unuseful when reading from new RawData Information will be stored in EcalDCCHeaderBlock)
  int      lightIntensity_;   ///< The light intensity
  int      laserType_;        ///< The laser type --see enum LaserType

};

std::ostream& operator<<(std::ostream&, const EcalTBEventHeader&);
  
#endif
