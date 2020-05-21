#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>

EcalFenixEtStrip::EcalFenixEtStrip() {}
EcalFenixEtStrip::~EcalFenixEtStrip() {}

void EcalFenixEtStrip::process(const std::vector<std::vector<int>> &linout, int nrXtals, std::vector<int> &output) {
  for (int &i : output) {
    i = 0;
  }
  for (int ixtal = 0; ixtal < nrXtals; ixtal++) {
    for (unsigned int i = 0; i < output.size(); i++) {
      output[i] += (linout[ixtal])[i];
    }
  }
  for (int &i : output) {
    if (i > 0X3FFFF)
      i = 0X3FFFF;
  }
  return;
}
