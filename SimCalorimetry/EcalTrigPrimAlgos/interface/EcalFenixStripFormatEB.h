#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXSTRIPFORMATEB_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXSTRIPFORMATEB_H

#include <cstdint>
#include <vector>

class EcalTPGSlidingWindow;
class EcalTPGTPMode;

/**
  \class EcalFenixStripFormatEB
 \brief Formatting for Fenix strip
*  input: 18 bits + 3x 1bit (fgvb, gapflagbit, output from peakfinder)
 *  output:16 bits
 *  The output corresponds to 1 calodataframe per strip
 *  --- not really a calodataframe no?
 */

class EcalFenixStripFormatEB {
private:
  int inputsFGVB_;
  int inputEvenPeak_;
  int inputOddPeak_;
  int input_even_;
  int input_odd_;
  uint32_t shift_;
  const EcalTPGTPMode *ecaltpgTPMode_;
  //  int buffer_;

  int setInput(int input_even, int inputEvenPeak, int input_odd, int inputOddPeak, int inputsFGVB);
  int process();

public:
  EcalFenixStripFormatEB();
  virtual ~EcalFenixStripFormatEB();
  virtual void process(std::vector<int> &sFGVBout,
                       std::vector<int> &peakout_even,
                       std::vector<int> &filtout_even,
                       std::vector<int> &peakout_odd,
                       std::vector<int> &filtout_odd,
                       std::vector<int> &output);
  void setParameters(uint32_t &, const EcalTPGSlidingWindow *&, const EcalTPGTPMode *);
};
#endif
