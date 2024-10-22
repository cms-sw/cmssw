#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtTot.h>

//----------------------------------------------------------------------------------------
EcalFenixEtTot::EcalFenixEtTot() {}
//----------------------------------------------------------------------------------------
EcalFenixEtTot::~EcalFenixEtTot() {}
//----------------------------------------------------------------------------------------
std::vector<int> EcalFenixEtTot::process(const std::vector<EBDataFrame *> &calodatafr) {
  std::vector<int> out;
  return out;
}
//----------------------------------------------------------------------------------------
void EcalFenixEtTot::process(std::vector<std::vector<int>> &bypasslinout,
                             int nStr,
                             int bitMask,
                             int bitOddEven,
                             std::vector<int> &output_even,
                             std::vector<int> &output_odd) {
  for (unsigned int i = 0; i < output_even.size(); i++) {
    output_even[i] = 0;
    output_odd[i] = 0;
  }

  int mask = (1 << bitMask) - 1;
  for (int istrip = 0; istrip < nStr; istrip++) {
    for (unsigned int i = 0; i < bypasslinout[istrip].size(); i++) {
      int output = (bypasslinout[istrip][i] & mask);  // fix bug inn case of EE: MSB are set for FG, so
                                                      // need to apply mask in summation.
      if (output > mask)
        output = mask;
      // Check the oddeven flag to assign the amplitude to the correct sum
      // If the feature is off in the strip fenix the bit will be always 0 and only the even sum will be summed
      if ((bypasslinout[istrip][i] >> bitOddEven) & 1) {
        output_odd[i] += output;
      } else {
        output_even[i] += output;
      }
    }
  }
  return;
}
//----------------------------------------------------------------------------------------
