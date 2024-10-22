#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBFenixStripFormatEB_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBFenixStripFormatEB_h

#include <vector>
#include <cstdint>

class EcalTPGSlidingWindow;

/** 
    \class EcalEBFenixStripFormatEB
   \brief Formatting for Fenix strip
  *  input: 18 bits + 3x 1bit (fgvb, gapflagbit, output from peakfinder)
   *  output:16 bits
   *  The output corresponds to 1 calodataframe per strip
   *  --- not really a calodataframe no?
   */

class EcalEBFenixStripFormatEB {
private:
  int inputsFGVB_;
  int inputPeak_;
  int input_;
  uint32_t shift_;
  //  int buffer_;

  int setInput(int input, int inputPeak, int inputsFGVB);
  int process();

public:
  EcalEBFenixStripFormatEB();
  virtual ~EcalEBFenixStripFormatEB();
  virtual void process(std::vector<int> &, std::vector<int> &, std::vector<int> &, std::vector<int> &);
  void setParameters(uint32_t &, const EcalTPGSlidingWindow *&);
};
#endif
