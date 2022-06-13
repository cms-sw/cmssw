#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h>
#include <CondFormats/EcalObjects/interface/EcalTPGTPMode.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>

EcalFenixStripFormatEB::EcalFenixStripFormatEB() : shift_(0) {}

EcalFenixStripFormatEB::~EcalFenixStripFormatEB() {}

int EcalFenixStripFormatEB::setInput(
    int input_even, int inputEvenPeak, int input_odd, int inputOddPeak, int inputsFGVB) {
  inputsFGVB_ = inputsFGVB;
  inputEvenPeak_ = inputEvenPeak;
  input_even_ = input_even;
  inputOddPeak_ = inputOddPeak;
  input_odd_ = input_odd;
  return 0;
}

int EcalFenixStripFormatEB::process() {
  int even_output = 0;
  int odd_output = 0;

  // Applying sliding window on the strip output after the peak finder
  if (ecaltpgTPMode_->DisableEBEvenPeakFinder) {
    even_output = input_even_ >> shift_;
  } else {
    if (inputEvenPeak_ == 1)
      even_output = input_even_ >> shift_;
  }

  if (ecaltpgTPMode_->EnableEBOddPeakFinder) {
    if (inputOddPeak_ == 1)
      odd_output = input_odd_ >> shift_;
  } else {
    odd_output = input_odd_ >> shift_;
  }

  // Truncating the signals to 12 bit after peak finder sliding window
  if (odd_output > 0XFFF)
    odd_output = 0XFFF;
  if (even_output > 0XFFF)
    even_output = 0XFFF;

  // Prepare the amplitude output for the strip looking at the TPmode options
  int output = 0;
  bool is_odd_larger = false;
  if (ecaltpgTPMode_->EnableEBOddFilter && odd_output > even_output)
    is_odd_larger =
        true;  // If running with odd filter enabled, check if odd output is larger regardless of strip formatter output mode
  switch (ecaltpgTPMode_->FenixEBStripOutput) {
    case 0:  // even filter out
      output = even_output;
      break;
    case 1:  // odd filter out
      if (ecaltpgTPMode_->EnableEBOddFilter)
        output = odd_output;
      else
        output = even_output;
      break;
    case 2:  // larger between odd and even
      if (ecaltpgTPMode_->EnableEBOddFilter && odd_output > even_output) {
        output = odd_output;
      } else
        output = even_output;
      break;
    case 3:  // even + odd
      if (ecaltpgTPMode_->EnableEBOddFilter)
        output = even_output + odd_output;
      else
        output = even_output;
      break;
  }

  if (output > 0XFFF)
    output = 0XFFF;  // ok: barrel saturates at 12 bits

  // Info bits
  // bit12 is sFGVB, bit13 is for odd>even flagging
  output |= ((inputsFGVB_ & 0x1) << 12);

  // if the flagging mode is OFF the bit stays 0, since it is not used for other things
  if (ecaltpgTPMode_->EnableEBOddFilter && ecaltpgTPMode_->FenixEBStripInfobit2) {
    output |= ((is_odd_larger & 0x1) << 13);
  }

  return output;
}

void EcalFenixStripFormatEB::process(std::vector<int> &sFGVBout,
                                     std::vector<int> &peakout_even,
                                     std::vector<int> &filtout_even,
                                     std::vector<int> &peakout_odd,
                                     std::vector<int> &filtout_odd,
                                     std::vector<int> &output) {
  if (peakout_even.size() != filtout_even.size() || sFGVBout.size() != filtout_even.size() ||
      peakout_odd.size() != filtout_odd.size() || filtout_odd.size() != filtout_even.size()) {
    edm::LogWarning("EcalTPG") << " problem in EcalFenixStripFormatEB: sfgvb_out, peak_out and "
                                  "filt_out don't have the same size";
  }
  for (unsigned int i = 0; i < filtout_even.size(); i++) {
    setInput(filtout_even[i], peakout_even[i], filtout_odd[i], peakout_odd[i], sFGVBout[i]);
    output[i] = process();
  }
  return;
}

void EcalFenixStripFormatEB::setParameters(uint32_t &id,
                                           const EcalTPGSlidingWindow *&slWin,
                                           const EcalTPGTPMode *ecaltptTPMode) {
  // TP mode contains options for the formatter (odd/even filters config)
  ecaltpgTPMode_ = ecaltptTPMode;
  const EcalTPGSlidingWindowMap &slwinmap = slWin->getMap();
  EcalTPGSlidingWindowMapIterator it = slwinmap.find(id);
  if (it != slwinmap.end())
    shift_ = (*it).second;
  else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGSlidingWindowMap entry for " << id;
}
