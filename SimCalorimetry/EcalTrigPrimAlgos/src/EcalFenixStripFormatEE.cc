#include <CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h>
#include <CondFormats/EcalObjects/interface/EcalTPGStripStatus.h>
#include <CondFormats/EcalObjects/interface/EcalTPGTPMode.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEE.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

//-----------------------------------------------------------------------------------------
EcalFenixStripFormatEE::EcalFenixStripFormatEE() : shift_(0) {}
//------------------------------------------------------------------------------------------

EcalFenixStripFormatEE::~EcalFenixStripFormatEE() {}

//-----------------------------------------------------------------------------------------

int EcalFenixStripFormatEE::setInput(int input_even, int inputEvenPeak, int input_odd, int inputOddPeak, int fgvb) {
  inputEvenPeak_ = inputEvenPeak;
  input_even_ = input_even;
  inputOddPeak_ = inputOddPeak;
  input_odd_ = input_odd;
  fgvb_ = fgvb;
  return 0;
}
//-----------------------------------------------------------------------------------------

int EcalFenixStripFormatEE::process() {
  // Bad strip - zero everything
  if (stripStatus_ != 0)
    return 0;

  int even_output = 0;
  int odd_output = 0;

  // Applying sliding window on the strip output after the peak finder
  if (ecaltpgTPMode_->DisableEEEvenPeakFinder) {
    even_output = input_even_ >> shift_;
  } else {
    if (inputEvenPeak_ == 1)
      even_output = input_even_ >> shift_;
  }

  if (ecaltpgTPMode_->EnableEEOddPeakFinder) {
    if (inputOddPeak_ == 1)
      odd_output = input_odd_ >> shift_;
  } else {
    odd_output = input_odd_ >> shift_;
  }

  // Truncating the signals to  to 12 bit after peak finder sliding window
  if (odd_output > 0XFFF)
    odd_output = 0XFFF;
  if (even_output > 0XFFF)
    even_output = 0XFFF;

  // Prepare the amplitude output for the strip looking at the TPmode options
  int output = 0;
  bool is_odd_larger = false;

  if (ecaltpgTPMode_->EnableEEOddFilter && (odd_output > even_output))
    is_odd_larger =
        true;  // If running with odd filter enabled, check if odd output is larger regardless of strip formatter output mode
  switch (ecaltpgTPMode_->FenixEEStripOutput) {
    case 0:  // even filter out
      output = even_output;
      break;
    case 1:  // odd filter out
      if (ecaltpgTPMode_->EnableEEOddFilter)
        output = odd_output;
      else
        output = even_output;
      break;
    case 2:  // larger between odd and even
      if (ecaltpgTPMode_->EnableEEOddFilter && (odd_output > even_output)) {
        output = odd_output;
      } else
        output = even_output;
      break;
    case 3:  // even + odd
      if (ecaltpgTPMode_->EnableEEOddFilter)
        output = even_output + odd_output;
      else
        output = even_output;
      break;
  }

  // barrel saturates at 12 bits, endcap at 10!
  // Pascal: finally no,endcap has 12 bits as in EB (bug in FENIX!!!!)
  if (output > 0XFFF)
    output = 0XFFF;

  // Info bits
  // bit12 is sFGVB, bit13 is for odd>even flagging
  output |= ((fgvb_ & 0x1) << 12);

  if (ecaltpgTPMode_->EnableEEOddFilter && ecaltpgTPMode_->FenixEEStripInfobit2) {
    output |= ((is_odd_larger & 0x1) << 13);
  }

  return output;
}
//------------------------------------------------------------------------------------------

void EcalFenixStripFormatEE::process(std::vector<int> &fgvbout,
                                     std::vector<int> &peakout_even,
                                     std::vector<int> &filtout_even,
                                     std::vector<int> &peakout_odd,
                                     std::vector<int> &filtout_odd,
                                     std::vector<int> &output) {
  if (peakout_even.size() != filtout_even.size() || fgvbout.size() != filtout_even.size() ||
      peakout_odd.size() != filtout_odd.size() || filtout_odd.size() != filtout_even.size()) {
    edm::LogWarning("EcalTPG") << " problem in EcalFenixStripFormatEE: peak_out and filt_out don't "
                                  "have the same size";
    std::cout << " Size peak_out" << peakout_even.size() << ", size filt_out:" << filtout_even.size() << std::flush
              << std::endl;
  }

  for (unsigned int i = 0; i < filtout_even.size(); i++) {
    setInput(filtout_even[i], peakout_even[i], filtout_odd[i], peakout_odd[i], fgvbout[i]);
    output[i] = process();
  }
  return;
}
//-----------------------------------------------------------------------------------------

void EcalFenixStripFormatEE::setParameters(uint32_t id,
                                           const EcalTPGSlidingWindow *&slWin,
                                           const EcalTPGStripStatus *stripStatus,
                                           const EcalTPGTPMode *ecaltpgTPMode) {
  // TP mode contains options for the formatter (odd/even filters config)
  ecaltpgTPMode_ = ecaltpgTPMode;
  const EcalTPGSlidingWindowMap &slwinmap = slWin->getMap();
  EcalTPGSlidingWindowMapIterator it = slwinmap.find(id);
  if (it != slwinmap.end())
    shift_ = (*it).second;
  else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGSlidingWindowMap entry for " << id;

  const EcalTPGStripStatusMap &statusMap = stripStatus->getMap();
  EcalTPGStripStatusMapIterator sit = statusMap.find(id);
  if (sit != statusMap.end()) {
    stripStatus_ = (*sit).second;
  } else {
    stripStatus_ = 0;  // Assume strip OK
  }
}
//-----------------------------------------------------------------------------------------
