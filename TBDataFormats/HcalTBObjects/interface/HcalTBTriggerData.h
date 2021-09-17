#ifndef HCALTBTRIGGERDATA_H
#define HCALTBTRIGGERDATA_H 1

#include <string>
#include <iostream>
#include <cstdint>

/** \class HcalTBTriggerData

This class contains trigger information (mainly trigger type and time),
and run information such as run, event, spill, bunch and orbit numbers.
      
  $Date: 2005/10/06 22:21:33 $
  $Revision: 1.2 $
  \author P. Dudero - Minnesota
  */
class HcalTBTriggerData {
public:
  HcalTBTriggerData();

  // Getter methods

  const std::string& runNumberSequenceId() const { return runNumberSequenceId_; }

  /// Returns the current run number
  uint32_t runNumber() const { return runNumber_; }
  /// Returns the entire packed trigger word
  uint32_t triggerWord() const { return triggerWord_; }
  /// Returns the relative time of this trigger in microseconds
  uint32_t triggerTimeUsec() const { return triggerTimeUsec_; }
  /// Returns the base time of the run (in seconds, from a time() call)
  uint32_t triggerTimeBase() const { return triggerTimeBase_; }
  /// Returns the spill number of this trigger
  uint32_t spillNumber() const { return spillNumber_; }
  /// Returns the orbit number of this trigger
  uint32_t orbitNumber() const { return orbitNumber_; }
  /// Returns the bunch number of this trigger
  uint16_t bunchNumber() const { return bunchNumber_; }
  /// Returns the event number of this trigger
  uint16_t eventNumber() const { return eventNumber_; }
  uint32_t flagsDaqTtype() const { return flagsDaqTtype_; }
  uint32_t algoBits3() const { return algoBits3_; }
  uint32_t algoBits2() const { return algoBits2_; }
  uint32_t algoBits1() const { return algoBits1_; }
  uint32_t algoBits0() const { return algoBits0_; }
  uint32_t techBits() const { return techBits_; }
  uint32_t gps1234() const { return gps1234_; }
  uint32_t gps5678() const { return gps5678_; }

  // Setter methods
  void setStandardData(uint32_t orbitNumber,
                       uint32_t eventNumber,
                       uint16_t bunchNumber,
                       uint32_t flags_daq_ttype,
                       uint32_t algo_bits_3,
                       uint32_t algo_bits_2,
                       uint32_t algo_bits_1,
                       uint32_t algo_bits_0,
                       uint32_t tech_bits,
                       uint32_t gps_1234,
                       uint32_t gps_5678);

  void setExtendedData(uint32_t triggerWord,
                       uint32_t triggerTime_usec,
                       uint32_t triggerTime_base,
                       uint32_t spillNumber,
                       uint32_t runNumber,
                       const char* runNumberSequenceId);

  // Parse trigger word routines

  /// returns true if this trigger came from beam data
  inline bool wasBeamTrigger() const { return (triggerWord() & 0x0F) == bit_BeamTrigger; }

  /// returns true if this trigger was fake (from a non-H2 manager)
  inline bool wasFakeTrigger() const { return (triggerWord() & 0x0F) == bit_FakeTrigger; }

  /// returns true if this trigger was a calibration trigger
  inline bool wasSpillIgnorantPedestalTrigger() const {
    return (triggerWord() & 0x0F) == bit_spillIgnorantPedestalTrigger;
  }

  /// returns true if this was an out-of-spill pedestal trigger
  inline bool wasOutSpillPedestalTrigger() const { return (triggerWord() & 0x0F) == bit_OutSpillPedestalTrigger; }

  /// returns true if this was an in-spill pedestal trigger
  inline bool wasInSpillPedestalTrigger() const { return (triggerWord() & 0x0F) == bit_InSpillPedestalTrigger; }

  /// returns true if this was a laser trigger
  inline bool wasLaserTrigger() const { return (triggerWord() & 0x0F) == bit_LaserTrigger; }

  /// returns true if this was a LED trigger
  inline bool wasLEDTrigger() const { return (triggerWord() & 0x0F) == bit_LEDTrigger; }

  /// returns true if the "spill" bit was set
  inline bool wasInSpill() const { return (triggerWord() & bit_InSpill); }

  static const uint32_t bit_BeamTrigger;                   // = 1;
  static const uint32_t bit_InSpillPedestalTrigger;        // = 2;
  static const uint32_t bit_OutSpillPedestalTrigger;       // = 3;
  static const uint32_t bit_LaserTrigger;                  // = 4
  static const uint32_t bit_spillIgnorantPedestalTrigger;  // = 5;
  static const uint32_t bit_LEDTrigger;                    // = 6

  static const uint32_t bit_FakeTrigger;  // = 15

  static const uint32_t bit_InSpill;  // = 0x10;

private:
  std::string runNumberSequenceId_;
  uint32_t runNumber_;
  uint32_t triggerWord_;
  uint32_t triggerTimeUsec_;
  uint32_t triggerTimeBase_;
  uint32_t spillNumber_;
  uint32_t orbitNumber_;
  uint16_t bunchNumber_;
  uint32_t eventNumber_;
  uint32_t flagsDaqTtype_;  /// <extended type=31:28,extended size=27:24,zeros=23:7,daq#=6:4,type=3:0>
  uint32_t algoBits3_;
  uint32_t algoBits2_;
  uint32_t algoBits1_;
  uint32_t algoBits0_;
  uint32_t techBits_;
  uint32_t gps1234_;
  uint32_t gps5678_;
};

std::ostream& operator<<(std::ostream& s, const HcalTBTriggerData& htbtd);

#endif
