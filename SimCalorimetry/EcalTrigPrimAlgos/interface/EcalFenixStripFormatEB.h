#ifndef ECAL_FENIX_STRIP_FORMAT_EB_H
#define ECAL_FENIX_STRIP_FORMAT_EB_H

//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <vector>
#include <iostream>

//class EcalVFormatter;
class EcalTPParameters;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
    \class EcalFenixStripFormatEB
   \brief Formatting for Fenix strip
  *  input: 18 bits + 3x 1bit (fgvb, gapflagbit, output from peakfinder)
   *  output:16 bits
   *  The output corresponds to 1 calodataframe per strip
   *  --- not really a calodataframe no?
   */
   
//ENDCAP:MODIF   
//   class EcalFenixStripFormatEB : public EcalVFormatter {

 class EcalFenixStripFormatEB {
  
 private:
  const EcalTPParameters * ecaltpp_ ;
  int inputPeak_;
  int input_;
  int shift_;
  int buffer_;

  int setInput(int input, int inputPeak);
  int process();

 public:
  EcalFenixStripFormatEB(const EcalTPParameters * ecaltpp);
  virtual ~EcalFenixStripFormatEB();
  //  virtual std::vector<int> process(std::vector<int> &, std::vector<int> &) ;
  virtual void  process(std::vector<int> &, std::vector<int> &, std::vector<int> &) ;
  void setParameters(int SM, int towerInSM, int stripInTower);

 
};


#endif
