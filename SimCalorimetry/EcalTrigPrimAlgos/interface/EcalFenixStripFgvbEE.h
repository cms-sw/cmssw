#ifndef ECAL_FENIXSTRIP_FGVB_EE_H
#define ECAL_FENIXSTRIP_FGVB_EE_H

#include <vector>

class EcalTPParameters;
class EEDataFrame ;


/** 
    \class EcalFenixStripFgvbEE
    \brief calculation of Fgvb for the endcap in Fenix Strip 
    *  calculates fgvb for the endcap in Fenix Strip
    *  
    *  
    *  input: 5X18 bits
    *  output: 1 bit 
    *  
    *  
    */
class EcalFenixStripFgvbEE  {

 private:
  const EcalTPParameters * ecaltpp_ ;
  std::vector<unsigned int> const * params_ ;

 public:
  EcalFenixStripFgvbEE(const EcalTPParameters * ecaltpp) ;
  virtual ~EcalFenixStripFgvbEE();
  void    setParameters(int sector, int towNum, int stripNr);
  void process( std::vector<std::vector<int> > &lin_out, std::vector<int> &output);
};


#endif
