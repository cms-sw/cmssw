#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixMaxof2.h>

// global type definitions for class implementation in source file defined by
// Tag entries in ArgoUML Result: typedef <typedef_global_source> <tag_value>;
EcalFenixMaxof2::EcalFenixMaxof2(int maxNrSamples, int nbMaxStrips) : nbMaxStrips_(nbMaxStrips) {
  std::vector<int> vec(maxNrSamples, 0);
  for (int i2strip = 0; i2strip < nbMaxStrips_ - 1; ++i2strip)
    sumby2_.push_back(vec);
}

EcalFenixMaxof2::~EcalFenixMaxof2() {}

void EcalFenixMaxof2::process(
    std::vector<std::vector<int>> &bypasslinout, int nstrip, int bitMask, int bitOddEven, std::vector<int> &output) {
  int mask = (1 << bitMask) - 1;
  bool strip_oddmask[nstrip][output.size()];

  for (int i2strip = 0; i2strip < nstrip - 1; ++i2strip)
    for (unsigned int i = 0; i < output.size(); i++)
      sumby2_[i2strip][i] = 0;
  for (unsigned int i = 0; i < output.size(); i++)
    output[i] = 0;

  // Prepare also the mask of strips to be avoided because of the odd>even energy flag
  for (int istrip = 0; istrip < nstrip; ++istrip) {
    for (unsigned int i = 0; i < output.size(); i++) {
      if ((bypasslinout[istrip][i] >> bitOddEven) & 1)
        strip_oddmask[istrip][i] = false;
      else
        strip_oddmask[istrip][i] = true;
    }
  }

  for (unsigned int i = 0; i < output.size(); i++) {
    if (nstrip - 1 == 0) {
      output[i] = strip_oddmask[0][i] * ((bypasslinout[0][i]) & mask);
    } else {
      for (int i2strip = 0; i2strip < nstrip - 1; ++i2strip) {
        sumby2_[i2strip][i] = strip_oddmask[i2strip][i] * ((bypasslinout[i2strip][i]) & mask) +
                              strip_oddmask[i2strip + 1][i] * ((bypasslinout[i2strip + 1][i]) & mask);
        if (sumby2_[i2strip][i] > output[i]) {
          output[i] = sumby2_[i2strip][i];
        }
      }
    }
  }
  return;
}
