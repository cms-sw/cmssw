#ifndef SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXMAXOF2_H
#define SIMCALORIMETRY_ECALTRIGPRIMALGOS_ECALFENIXMAXOF2_H

#include <vector>

/**
 *  finds max sum of two adjacent samples
 *
 *  input: 5x 12 bits (les 12 premiers bits sortant du bypasslin)
 *  output: 12 bits
 *
 *  computes 4 sums of 2 strips and gives the max
 *  max limited by 0xfff
 *
 * As in the firmware the computation is performed only on the even energy sum,
 * so the oddEvenFlag (14th bit) is used to exclude the energy of the odd strips from the computation. D.Valsecchi.
 *
 */

class EcalFenixMaxof2 {
public:
  EcalFenixMaxof2(int maxNrSamples, int nbMaxStrips);
  virtual ~EcalFenixMaxof2();
  void process(std::vector<std::vector<int>> &, int nStr, int bitMask, int bitOddEven, std::vector<int> &out);

private:
  int nbMaxStrips_;
  std::vector<std::vector<int>> sumby2_;
};

#endif
