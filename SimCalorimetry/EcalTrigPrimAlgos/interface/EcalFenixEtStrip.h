#ifndef ECAL_FENIX_ET_STRIP_H
#define ECAL_FENIX_ET_STRIP_H
#include <vector>

  /** 
     \class EcalFenixEtStrip

     class for calculation of Et for Fenix strip
   *  input: 5x18 bits
   *  output: 18 bits representing sum
   *  
   *  sum method gets vector of CaloTimeSamples
   *  as input (steph comment : Ursula, why CaloTimeSample ?)
   *  simple sum, test for max?
   *  max in h4ana is 0x3FFFF 
   *  
   *  ---> if overflow sum= (2^18-1)
   */

class EcalFenixEtStrip  {
 private:

 public:
  EcalFenixEtStrip();
  virtual ~EcalFenixEtStrip();
  void process(const std::vector<std::vector<int> > &linout, int nrXtals, std::vector<int> & output);
};

#endif
