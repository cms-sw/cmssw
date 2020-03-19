#ifndef ECAL_FENIX_STRIP_FORMAT_EE_H
#define ECAL_FENIX_STRIP_FORMAT_EE_H

#include <cstdint>
#include <vector>

class EcalTPGSlidingWindow;
class EcalTPGStripStatus;

/**
  \class EcalFenixStripFormatEE
 \brief Formatting for Fenix strip
*  input: 18 bits + 3x 1bit (fgvb, gapflagbit, output from peakfinder)
 *  output:16 bits
 *  The output corresponds to 1 calodataframe per strip
 *  --- not really a calodataframe no?
 */

class EcalFenixStripFormatEE {
private:
  int inputPeak_;
  int input_;
  uint32_t shift_;
  int fgvb_;
  uint16_t stripStatus_;

  int setInput(int input, int inputPeak, int fgvb);
  int process();

public:
  EcalFenixStripFormatEE();
  virtual ~EcalFenixStripFormatEE();

  virtual void process(std::vector<int> &, std::vector<int> &, std::vector<int> &, std::vector<int> &);
  void setParameters(uint32_t id, const EcalTPGSlidingWindow *&, const EcalTPGStripStatus *);
};

#endif
