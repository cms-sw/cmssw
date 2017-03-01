#ifndef MU_END_STRIP_ELECTRONICS_SIM_H
#define MU_END_STRIP_ELECTRONICS_SIM_H

/** \class CSCStripElectronicsSim
 * Model the readout electronics chain for EMU CSC strips
 *
 * \author Rick Wilkinson
 *
 */

#include "SimMuon/CSCDigitizer/src/CSCBaseElectronicsSim.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "SimMuon/CSCDigitizer/src/CSCStripAmpResponse.h"

class CSCDetectorHit;
class CSCComparatorDigi;
class CSCCrosstalkGenerator;
class CSCStripConditions;
#include <vector>
#include <string>

namespace CLHEP {
  class HepRandomEngine;
}

class CSCStripElectronicsSim : public CSCBaseElectronicsSim
{
public:
  /// configurable parameters
  explicit CSCStripElectronicsSim(const edm::ParameterSet & p);

  virtual ~CSCStripElectronicsSim();

  void fillDigis(CSCStripDigiCollection & digis,
                 CSCComparatorDigiCollection & comparators,
                 CLHEP::HepRandomEngine*);

  void fillMissingLayer(const CSCLayer * layer, const CSCComparatorDigiCollection & comparators, 
                        CSCStripDigiCollection & digis, CLHEP::HepRandomEngine*);

  void setStripConditions(CSCStripConditions * cond) {theStripConditions = cond;}

  CSCAnalogSignal makeNoiseSignal(int element, CLHEP::HepRandomEngine*) override;

  void createDigi(int istrip, const CSCAnalogSignal & signal, std::vector<CSCStripDigi> & result, CLHEP::HepRandomEngine*);
 
private:
  /// initialization for each layer
  void initParameters() override;

  virtual int readoutElement(int strip) const override;

  float calculateAmpResponse(float t) const override;
  CSCStripAmpResponse theAmpResponse;

  void runComparator(std::vector<CSCComparatorDigi> & result, CLHEP::HepRandomEngine*);

  /// calculates the comparator reading, including saturation and offsets
  float comparatorReading(const CSCAnalogSignal & signal, float time, CLHEP::HepRandomEngine*) const;

  // tells which strips to read out around the input strip
  void getReadoutRange(int inputStrip, 
                       int & minStrip, int & maxStrip);

  /// finds the key strips from these comparators
  std::list<int>
  getKeyStrips(const std::vector<CSCComparatorDigi> & comparators) const;

  /// get ths strips that have detector hits
  std::list<int>
  getKeyStripsFromMC() const;
  /// finds what strips to read.  Will either take 5 strips around
  /// the keystrip, or the whole CFEB, based on doSuppression_
  std::list<int>
  channelsToRead(const std::list<int> & keyStrips, int window) const;

  void fillStripDigis(const std::list<int> & keyStrips,
                      CSCStripDigiCollection & digis,
                      CLHEP::HepRandomEngine*);

  void addCrosstalk(CLHEP::HepRandomEngine*);
  void addCrosstalk(const CSCAnalogSignal & signal,
                    int thisStrip, int otherStrip, CLHEP::HepRandomEngine*);


  void selfTest() const;

  // saturation of the 12-bit ADC.  Max reading is 4095
  void doSaturation(CSCStripDigi & digi);

  // useful constants
  float theComparatorThreshold;      // in fC
  float theComparatorNoise;
  float theComparatorRMSOffset;
  // note that we don't implement the effect of the x3.5 amplifier
  float theComparatorSaturation;
  // all of these times are in nanoseconds
  float theComparatorWait;
  float theComparatorDeadTime;
  float theDaqDeadTime;
  // save the calculation of time-of-flight+drift+shaping
  float theTimingOffset;

  int nScaBins_;
  bool doSuppression_;
  bool doCrosstalk_;
  CSCStripConditions * theStripConditions;
  CSCCrosstalkGenerator * theCrosstalkGenerator;

  int theComparatorClockJump;
  // the length of each SCA time bin, in ns.  50 by default
  float sca_time_bin_size;
  // the SCA bin which holds the peak signal.  4, by default.
  // that's really the 5th, since We start counting at 0
  int   sca_peak_bin;
  // which time bin the trigger crossing goes in
  double theComparatorTimeBinOffset;
  // to center comparator signals
  double theComparatorTimeOffset;
  double theComparatorSamplingTime;
  // tweaks the timing of the SCA
  std::vector<double> theSCATimingOffsets;
  // remeber toe TOF correction in comparators,
  // so we can undo it for SCA
  float theAverageTimeOfFlight;
};

#endif

