using namespace std;
#ifndef ECAL_FENIX_STRIP_FORMAT_H
#define ECAL_FENIX_STRIP_FORMAT_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <stdio.h>
#include <iostream>

class EcalVFormatter;
namespace tpg {

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
    \class EcalFenixStripFormat
   \brief Formatting for Fenix strip
  *  input: 18 bits + 3x 1bit (fgvb, gapflagbit, output from peakfinder)
   *  output:16 bits
   *  The output corresponds to 1 calodataframe per strip
   *  --- not really a calodataframe no?
   */
class EcalFenixStripFormat : public EcalVFormatter {
    
 private:
  int inputPeak_;
  int input_;
  int shift_;
  int buffer_[2];

  int setInput(int input, int inputPeak);
  int process();


 public:
  EcalFenixStripFormat();
  virtual ~EcalFenixStripFormat();
  virtual vector<int> process(vector<int> , vector<int>);

 
};


} /* End of namespace tpg */

#endif
