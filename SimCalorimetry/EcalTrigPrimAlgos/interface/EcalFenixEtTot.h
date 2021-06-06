#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXETTOT_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXETTOT_H

#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include <vector>

/**
    \class EcalFenixEtTot

    \brief class for calculation of Et for Fenix tcp
    *  calculates the sum.
    * As in the firmware the Et sum is splitted in even and odd sum according to the OddEvenBit.
    * The bit (14th) is handled by strip.
    *
    *  inputs:
    *        -5x 12 bits (12 first bits of output of passlin or take 13 bits and
   select the first 12 ones in the class...
    *         for EBDataFrame
              5X 10 bits (10 first bits)
              according to the second parameter)
    *        -number of interesting bits according EE or EBDataFrame
             -mask to apply before checking for oddEven flaf
    *
    *  output :12 bits (EB) or 10(EE)
    *
    *  in case of overflow, result is set to (2**12)-1 or (2**10)-1
    */
class EcalFenixEtTot {
public:
  EcalFenixEtTot();
  virtual ~EcalFenixEtTot();
  virtual std::vector<int> process(const std::vector<EBDataFrame *> &);

  void process(std::vector<std::vector<int>> &,
               int nStr,
               int bitMask,
               int bitOddEven,
               std::vector<int> &out_even,
               std::vector<int> &out_odd);

private:
};

#endif
