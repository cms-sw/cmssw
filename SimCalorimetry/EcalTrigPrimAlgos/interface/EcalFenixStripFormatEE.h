#ifndef ECAL_FENIX_STRIP_FORMAT_EE_H
#define ECAL_FENIX_STRIP_FORMAT_EE_H

//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFormatter.h>
#include <vector>
#include <iostream>

//class EcalVFormatter;
class EcalTPParameters;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
    \class EcalFenixStripFormatEE
   \brief Formatting for Fenix strip
  *  input: 18 bits + 3x 1bit (fgvb, gapflagbit, output from peakfinder)
   *  output:16 bits
   *  The output corresponds to 1 calodataframe per strip
   *  --- not really a calodataframe no?
   */
//ENDCAP:MODIF class EcalFenixStripFormatEB : public EcalVFormatter {
 class EcalFenixStripFormatEE {   
 private:
  const EcalTPParameters *ecaltpp_ ;
  int inputPeak_;
  int input_;
  int shift_;
  int fgvb_;
  int buffer_;

  int setInput(int input, int inputPeak, int fgvb);
  int process();


 public:
  EcalFenixStripFormatEE(const EcalTPParameters *ecaltpp);
  virtual ~EcalFenixStripFormatEE();
  //  virtual std::vector<int> process(std::vector<int>& ,std::vector<int>& , std::vector<int>&) ;
  virtual void  process(std::vector<int>& ,std::vector<int>& , std::vector<int>&,std::vector<int>&) ;
  void setParameters(int sector, int towerInSector, int stripInTower);

 
};


#endif
