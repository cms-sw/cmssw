#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>

EcalFenixStripFormatEB::EcalFenixStripFormatEB() : shift_(0) {}

EcalFenixStripFormatEB::~EcalFenixStripFormatEB() {}

int EcalFenixStripFormatEB::setInput(int input, int inputPeak, int inputsFGVB) {
  inputsFGVB_ = inputsFGVB;
  inputPeak_ = inputPeak;
  input_ = input;
  return 0;
}

int EcalFenixStripFormatEB::process() {
  //    buffer_=input_>>shift_;  //FIXME: buffer why?

  if (inputPeak_ == 0)
    return ((inputsFGVB_ & 0x1) << 12);
  //    int output=buffer_;
  int output = input_ >> shift_;
  if (output > 0XFFF)
    output = 0XFFF;  // ok: barrel saturates at 12 bits
  // Add stripFGVB
  output |= ((inputsFGVB_ & 0x1) << 12);

  return output;
}

void EcalFenixStripFormatEB::process(std::vector<int> &sFGVBout,
                                     std::vector<int> &peakout,
                                     std::vector<int> &filtout,
                                     std::vector<int> &output) {
  if (peakout.size() != filtout.size() || sFGVBout.size() != filtout.size()) {
    edm::LogWarning("EcalTPG") << " problem in EcalFenixStripFormatEB: sfgvb_out, peak_out and "
                                  "filt_out don't have the same size";
  }
  for (unsigned int i = 0; i < filtout.size(); i++) {
    setInput(filtout[i], peakout[i], sFGVBout[i]);

    output[i] = process();
  }
  return;
}

void EcalFenixStripFormatEB::setParameters(uint32_t &id, const EcalTPGSlidingWindow *&slWin) {
  const EcalTPGSlidingWindowMap &slwinmap = slWin->getMap();
  EcalTPGSlidingWindowMapIterator it = slwinmap.find(id);
  if (it != slwinmap.end())
    shift_ = (*it).second;
  else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGSlidingWindowMap entry for " << id;
}
