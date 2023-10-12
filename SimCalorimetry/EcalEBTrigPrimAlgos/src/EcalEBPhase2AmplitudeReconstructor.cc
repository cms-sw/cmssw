#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2AmplitudeReconstructor.h>
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

const  int EcalEBPhase2AmplitudeReconstructor::maxSamplesUsed_=12;



EcalEBPhase2AmplitudeReconstructor::EcalEBPhase2AmplitudeReconstructor(bool debug)
    : debug_(debug), inputsAlreadyIn_(0), shift_(13) {}

EcalEBPhase2AmplitudeReconstructor::~EcalEBPhase2AmplitudeReconstructor() {}

int EcalEBPhase2AmplitudeReconstructor::setInput(int input) {
  if (input > 0X3FFF) {
    std::cout << "ERROR IN INPUT OF AMPLITUDE FILTER" << std::endl;
    return -1;
  }

  if (inputsAlreadyIn_ < maxSamplesUsed_ ) {
    if (debug_)
      std::cout << " EcalEBPhase2AmplitudeReconstructor::setInput inputsAlreadyIn_<5 input " << input << std::endl;
    buffer_[inputsAlreadyIn_] = input;
    inputsAlreadyIn_++;
  } else {

    for (int i = 0; i < (maxSamplesUsed_-1) ; i++) {
      buffer_[i] = buffer_[i + 1];
      if (debug_)
        std::cout << " EcalEBPhase2AmplitudeReconstructor::setInput inputsAlreadyIn buffer " << buffer_[i] << std::endl;
    }
    buffer_[maxSamplesUsed_-1] = input;
  }
  return 1;
}

void EcalEBPhase2AmplitudeReconstructor::process(std::vector<int> &linout, std::vector<int> &output) {
  inputsAlreadyIn_ = 0;
  for (unsigned int i = 0; i < maxSamplesUsed_; i++) {
    buffer_[i] = 0;
  }

  for (unsigned int i = 0; i < linout.size(); i++) {
    setInput(linout[i]);
    if (debug_) {
      for (unsigned int j = 0; j < maxSamplesUsed_; j++) {
        std::cout << " buffer_ " << buffer_[j];
      }
      std::cout << "  " << std::endl;
    }

    if (i == (maxSamplesUsed_-1)) {
      process();
      output[0] = processedOutput_;
    } else if (i == (ecalPh2::sampleSize-1)) {
      process();
      output[1] = processedOutput_;
    }
  }
  return;
}

void EcalEBPhase2AmplitudeReconstructor::process() {
  processedOutput_ = 0;
  if (inputsAlreadyIn_ < maxSamplesUsed_)
    return;
  int64_t tmpIntOutput = 0;
  for (int i = 0; i < maxSamplesUsed_; i++) {
    tmpIntOutput += (weights_[i] * buffer_[i]);
    if (debug_)
      std::cout << " AmplitudeFilter buffer " << buffer_[i] << " weight " << weights_[i] << std::endl;
  }
  if (tmpIntOutput < 0)
    tmpIntOutput = 0;
  tmpIntOutput = tmpIntOutput >> shift_;
  if (debug_)
    std::cout << " AmplitudeFilter tmpIntOutput " << tmpIntOutput << " shift_ " << shift_ << std::endl;
  if (tmpIntOutput > 0X1FFF)
    tmpIntOutput = 0X1FFF;
  uint output = tmpIntOutput;  // should be 13 bit uint at this point
  processedOutput_ = output;
  if (debug_)
    std::cout << " AmplitudeFilter  processedOutput_ " << processedOutput_ << std::endl;
}

void EcalEBPhase2AmplitudeReconstructor::setParameters(uint32_t raw,
                                                       const EcalEBPhase2TPGAmplWeightIdMap *ecaltpgWeightMap,
                                                       const EcalTPGWeightGroup *ecaltpgWeightGroup) {
  uint32_t params_[maxSamplesUsed_];
  const EcalTPGGroups::EcalTPGGroupsMap &groupmap = ecaltpgWeightGroup->getMap();
  if (debug_)
    std::cout << " EcalEBPhase2AmplitudeReconstructor::setParameters groupmap size " << groupmap.size()
              << "  channel ID " << raw << std::endl;
  EcalTPGGroups::EcalTPGGroupsMapItr it = groupmap.find(raw);
  if (it != groupmap.end()) {
    uint32_t weightid = (*it).second;

    const EcalEBPhase2TPGAmplWeightIdMap::EcalEBPhase2TPGAmplWeightMap &weightmap = ecaltpgWeightMap->getMap();
    EcalEBPhase2TPGAmplWeightIdMap::EcalEBPhase2TPGAmplWeightMapItr itw = weightmap.find(weightid);

    (*itw).second.getValues(params_[0],
                            params_[1],
                            params_[2],
                            params_[3],
                            params_[4],
                            params_[5],
                            params_[6],
                            params_[7],
                            params_[8],
                            params_[9],
                            params_[10],
                            params_[11]);

    if (debug_)
      std::cout << " EcalEBPhase2AmplitudeReconstructor::setParameters weights after the map  " << params_[0] << " "
                << params_[1] << " " << params_[2] << " " << params_[3] << " " << params_[4] << " " << params_[5] << " "
                << params_[6] << " " << params_[7] << " " << params_[8] << " " << params_[9] << " " << params_[10]
                << " " << params_[11] << std::endl;

    // we have to transform negative coded in 13 bits into negative coded in 32 bits
    // maybe this should go into the getValue method??

    for (int i = 0; i < maxSamplesUsed_; ++i) {
      weights_[i] = (params_[i] & 0x1000) ? (int)(params_[i] | 0xfffff000) : (int)(params_[i]);
    }

    if (debug_) {
      for (int i = 0; i < maxSamplesUsed_ ; ++i) {
        std::cout << " EcalEBPhase2AmplitudeReconstructor::setParameters weights after the cooking " << weights_[i]
                  << std::endl;
      }
      std::cout << std::endl;
    }

  } else
    edm::LogWarning("EcalTPG")
        << " EcalEBPhase2AmplitudeReconstructor::setParameters could not find EcalTPGGroupsMap entry for " << raw;
}
