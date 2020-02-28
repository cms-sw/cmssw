#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>

EcalFenixEtStrip::EcalFenixEtStrip() {}
EcalFenixEtStrip::~EcalFenixEtStrip() {}

void EcalFenixEtStrip::process(const std::vector<std::vector<int>> &linout, int nrXtals, std::vector<int> &output) {
  for (unsigned int i = 0; i < output.size(); i++) {
    output[i] = 0;
  }
  for (int ixtal = 0; ixtal < nrXtals; ixtal++) {
    for (unsigned int i = 0; i < output.size(); i++) {
      output[i] += (linout[ixtal])[i];
    }
  }
  for (unsigned int i = 0; i < output.size(); i++) {
    if (output[i] > 0X3FFFF)
      output[i] = 0X3FFFF;
  }
  return;
}
