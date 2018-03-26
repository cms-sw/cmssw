#ifndef ECAL_FENIX_STRIP_FORMAT_EB_H
#define ECAL_FENIX_STRIP_FORMAT_EB_H

#include <vector>
#include <stdint.h>

class EcalTPGSlidingWindow;

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
  int inputPeak_;
  int input_;
  uint32_t shift_;
  //  int buffer_;

  int setInput(int input, int inputPeak, int inputsFGVB);
  int process();

 public:
  EcalFenixStripFormatEB();
  virtual ~EcalFenixStripFormatEB();
  virtual void  process(std::vector<int> &, std::vector<int> &, std::vector<int> &, std::vector<int> &) ;
  void setParameters(uint32_t&, const EcalTPGSlidingWindow*&);
};
#endif
