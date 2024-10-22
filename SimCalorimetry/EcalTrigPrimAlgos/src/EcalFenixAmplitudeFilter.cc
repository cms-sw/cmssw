#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include <iostream>

EcalFenixAmplitudeFilter::EcalFenixAmplitudeFilter(bool tpInfoPrintout)
    : inputsAlreadyIn_(0), stripid_{0}, shift_(6), tpInfoPrintout_(tpInfoPrintout) {}

EcalFenixAmplitudeFilter::~EcalFenixAmplitudeFilter() {}

int EcalFenixAmplitudeFilter::setInput(int input, int fgvb) {
  if (input > 0X3FFFF) {
    edm::LogError("EcalTPG") << "ERROR IN INPUT OF EVEN AMPLITUDE FILTER";
    return -1;
  }
  if (inputsAlreadyIn_ < 5) {
    buffer_[inputsAlreadyIn_] = input;
    fgvbBuffer_[inputsAlreadyIn_] = fgvb;
    inputsAlreadyIn_++;
  } else {
    for (int i = 0; i < 4; i++) {
      buffer_[i] = buffer_[i + 1];
      fgvbBuffer_[i] = fgvbBuffer_[i + 1];
    }
    buffer_[4] = input;
    fgvbBuffer_[4] = fgvb;
  }
  return 1;
}

void EcalFenixAmplitudeFilter::process(std::vector<int> &addout,
                                       std::vector<int> &output,
                                       std::vector<int> &fgvbIn,
                                       std::vector<int> &fgvbOut) {
  inputsAlreadyIn_ = 0;
  for (unsigned int i = 0; i < 5; i++) {
    buffer_[i] = 0;
    fgvbBuffer_[i] = 0;
  }

  for (unsigned int i = 0; i < addout.size(); i++) {
    setInput(addout[i], fgvbIn[i]);
    process();
    if (tpInfoPrintout_) {
      if (i >= 4) {
        edm::LogVerbatim("EcalTPG") << i << " " << stripid_ << " " << weights_[0] << " " << weights_[1] << " "
                                    << weights_[2] << " " << weights_[3] << " " << weights_[4] << " "
                                    << weights_[0] / 64.0 << " " << weights_[1] / 64.0 << " " << weights_[2] / 64.0
                                    << " " << weights_[3] / 64.0 << " " << weights_[4] / 64.0 << " [" << buffer_[0]
                                    << ", " << buffer_[1] << ", " << buffer_[2] << ", " << buffer_[3] << ", "
                                    << buffer_[4] << "]"
                                    << " --> output: " << processedOutput_ << " EVEN";
      }
    }
    output[i] = processedOutput_;
    fgvbOut[i] = processedFgvbOutput_;
  }
  // shift the result by 1!
  for (unsigned int i = 0; i < (output.size()); i++) {
    if (i != output.size() - 1) {
      output[i] = output[i + 1];
      fgvbOut[i] = fgvbOut[i + 1];
    } else {
      output[i] = 0;
      fgvbOut[i] = 0;
    }
  }
  return;
}

void EcalFenixAmplitudeFilter::process() {
  processedOutput_ = 0;
  processedFgvbOutput_ = 0;
  if (inputsAlreadyIn_ < 5)  // 5 digis required to produce first ET value
    return;
  int output = 0;
  int fgvbInt = 0;
  for (int i = 0; i < 5; i++) {
    output += (weights_[i] * buffer_[i]) >> shift_;
    if ((fgvbBuffer_[i] == 1 && i == 3) || fgvbInt == 1) {
      fgvbInt = 1;
    }
  }
  if (output < 0)
    output = 0;
  if (output > 0X3FFFF)
    output = 0X3FFFF;
  processedOutput_ = output;
  processedFgvbOutput_ = fgvbInt;
}

void EcalFenixAmplitudeFilter::setParameters(uint32_t raw,
                                             const EcalTPGWeightIdMap *ecaltpgWeightMap,
                                             const EcalTPGWeightGroup *ecaltpgWeightGroup) {
  stripid_ = raw;
  uint32_t params_[5];
  const EcalTPGGroups::EcalTPGGroupsMap &groupmap = ecaltpgWeightGroup->getMap();
  EcalTPGGroups::EcalTPGGroupsMapItr it = groupmap.find(raw);
  if (it != groupmap.end()) {
    uint32_t weightid = (*it).second;
    const EcalTPGWeightIdMap::EcalTPGWeightMap &weightmap = ecaltpgWeightMap->getMap();
    EcalTPGWeightIdMap::EcalTPGWeightMapItr itw = weightmap.find(weightid);
    (*itw).second.getValues(params_[0], params_[1], params_[2], params_[3], params_[4]);

    for (int i = 0; i < 5; ++i) {
      weights_[i] = (params_[i] & 0x40) ? (int)(params_[i] | 0xffffffc0) : (int)(params_[i]);
    }
  } else
    edm::LogWarning("EcalTPG") << " could not find EcalTPGGroupsMap entry for " << raw;
}
