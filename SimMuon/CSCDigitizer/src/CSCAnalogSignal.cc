#include "SimMuon/CSCDigitizer/src/CSCAnalogSignal.h"
#include <algorithm>
#include <iostream>

// =================================
float CSCAnalogSignal::peakTime() const {
  size_t imax = std::max_element(theBinValues.begin(), theBinValues.end()) - theBinValues.begin();
  return imax / invBinSize + theTimeOffset;
}

std::ostream &operator<<(std::ostream &stream, const CSCAnalogSignal &signal) {
  stream << "CSCAnalogSignal: Element " << signal.theElement << "   Total " << signal.theTotal << std::endl;
  for (int i = 0; i < int(signal.theBinValues.size()); ++i) {
    //@@ ptc 26-Feb-02 Don't both with very small amplitudes

    if (signal.theBinValues[i] > 1.E-10) {
      stream << i * signal.getBinSize() + signal.getTimeOffset() << "\t" << signal.theBinValues[i] << std::endl;
    }
  }
  return stream;
}
