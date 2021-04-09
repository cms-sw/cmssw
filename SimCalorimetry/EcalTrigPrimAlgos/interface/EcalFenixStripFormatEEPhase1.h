#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENISTRIPFORMATEEPHASE1_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENISTRIPFORMATEEPHASE1_H

#include <cstdint>
#include <vector>

class EcalTPGSlidingWindow;
class EcalTPGStripStatus;
class EcalTPGTPMode;

/**
  \class EcalFenixStripFormatEEPhase1
 \brief Formatting for Fenix strip
*  input: 18 bits + 3x 1bit (fgvb, gapflagbit, output from peakfinder)
 *  output:16 bits
 *  The output corresponds to 1 calodataframe per strip
 *  --- not really a calodataframe no?
 */

class EcalFenixStripFormatEEPhase1 {
private:
  int inputEvenPeak_;
  int inputOddPeak_;
  int input_even_;
  int input_odd_;
  uint32_t shift_;
  int fgvb_;
  uint16_t stripStatus_;
  const EcalTPGTPMode *ecaltpgTPMode_;

  int setInput(int input_even, int inputEvenPeak, int input_odd, int inputOddPeak, int fgvb);
  int process();

public:
  EcalFenixStripFormatEEPhase1();
  virtual ~EcalFenixStripFormatEEPhase1();

  virtual void process(std::vector<int> &fgvbout,
                       std::vector<int> &peakout_even,
                       std::vector<int> &filtout_even,
                       std::vector<int> &peakout_odd,
                       std::vector<int> &filtout_odd,
                       std::vector<int> &output);
  void setParameters(uint32_t id, const EcalTPGSlidingWindow *&, const EcalTPGStripStatus *, const EcalTPGTPMode *);
};

#endif
