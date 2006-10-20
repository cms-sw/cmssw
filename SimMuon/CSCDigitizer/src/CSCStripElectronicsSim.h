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
class CSCDetectorHit;
class CSCComparatorDigi;
class CSCCrosstalkGenerator;
class CSCScaNoiseGenerator;
#include <vector>
#include <string>

class CSCStripElectronicsSim : public CSCBaseElectronicsSim
{
public:
  /// configurable parameters
  explicit CSCStripElectronicsSim(const edm::ParameterSet & p);

  virtual ~CSCStripElectronicsSim();

  // sets the RMS fluctuation of each SCA bin, in fC
  void setScaNoise(float noise) {sca_noise = noise;};
  void setComparatorThreshold(float threshold) 
     {theComparatorThreshold = threshold;};

  void fillDigis(CSCStripDigiCollection & digis,
                 CSCComparatorDigiCollection & comparators);

private:
  /// initialization for each layer
  void initParameters();

  virtual int readoutElement(int strip) const;

  float calculateAmpResponse(float t) const;

  std::vector<CSCComparatorDigi> runComparator();

  /// calculates the comparator reading, including saturation and offsets
  float comparatorReading(const CSCAnalogSignal & signal, float time) const;

  CSCAnalogSignal makeNoiseSignal(int element);

  virtual float signalDelay(int element, float pos) const;
  
  // tells which strips to read out around the input strip
  void getReadoutRange(int inputStrip, 
                       int & minStrip, int & maxStrip);
  void addCrosstalk();

  CSCStripDigi createDigi(int istrip,
                  float sca_start_time,
                  bool addScaNoise);

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

  int nScaBins_;
  bool doCrosstalk_;
  CSCCrosstalkGenerator * theCrosstalkGenerator;
  CSCScaNoiseGenerator  * theScaNoiseGenerator;

  int theComparatorClockJump;
  // the length of each SCA time bin, in ns.  50 by default
  float sca_time_bin_size;
  float sca_noise;
  // the following values are in ADC counts
  float theAnalogNoise;
  float thePedestal;
  float thePedestalWidth;
  // the SCA bin which holds the peak signal.  4, by default.
  // that's really the 5th, since We start counting at 0
  int   sca_peak_bin;
  // which time bin the trigger crossing goes in
  int theComparatorTimeBinOffset;

  // can be "simple" or "file"
  std::string scaNoiseMode_;
};

#endif

