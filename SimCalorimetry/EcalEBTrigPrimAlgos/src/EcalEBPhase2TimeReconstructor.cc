#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2TimeReconstructor.h>
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"

#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

EcalEBPhase2TimeReconstructor::EcalEBPhase2TimeReconstructor(bool debug)
    : debug_(debug), inputsAlreadyIn_(0), shift_(12) {}

EcalEBPhase2TimeReconstructor::~EcalEBPhase2TimeReconstructor() {}

int EcalEBPhase2TimeReconstructor::setInput(int input) {
  if (input > 0X7FFF) {
    std::cout << "ERROR IN INPUT OF TIME FILTER" << std::endl;
    return -1;
  }
  if (inputsAlreadyIn_ < 12) {
    if (debug_)
      std::cout << " EcalEBPhase2TimeReconstructor::setInput inputsAlreadyIn_<5 input " << input << std::endl;
    buffer_[inputsAlreadyIn_] = input;
    inputsAlreadyIn_++;
  } else {
    for (int i = 0; i < 11; i++) {
      buffer_[i] = buffer_[i + 1];
      if (debug_)
        std::cout << " EcalEBPhase2TimeReconstructor::setInput inputsAlreadyIn buffer " << buffer_[i] << std::endl;
    }
    buffer_[11] = input;
    inputsAlreadyIn_++;
  }
  return 1;
}

void EcalEBPhase2TimeReconstructor::process(std::vector<int> &addout,
                                            std::vector<int> &ampRecoOutput,
                                            std::vector<int64_t> &output) {
  inputsAlreadyIn_ = 0;
  for (unsigned int i = 0; i < 12; i++) {
    buffer_[i] = 0;
  }

  //Taking in the results of the amplitude reconstruction
  //Bit shifting them for use as index of invAmpAr_ lookup table
  // move input amplitude (13 bits) to 9 bits to use as array index

  ampIn_[0] = ampRecoOutput[0] >> 4;
  ampIn_[1] = ampRecoOutput[1] >> 4;

  for (unsigned int i = 0; i < addout.size(); i++) {
    setInput(addout[i]);

    if (debug_) {
      std::cout << "  EcalEBPhase2TimeReconstructor::process(std::vector<int> buffer_ " << std::endl;
      ;
      for (unsigned int j = 0; j < 12; j++) {
        std::cout << " buffer_ " << buffer_[j];
      }
      std::cout << "  " << std::endl;
    }

    if (i == 11) {
      if (debug_)
        std::cout << "  EcalEBPhase2TimeReconstructor::process(std::vector<int>)    i = 11 " << std::endl;
      process();
      if (debug_)
        std::cout << "  EcalEBPhase2TimeReconstructor::process(std::vector<int>)    after process() "
                  << processedOutput_ << std::endl;
      output[0] = processedOutput_;
      if (debug_)
        std::cout << "  EcalEBPhase2TimeReconstructor::process(std::vector<int>)    after setting the output "
                  << output[0] << std::endl;
    } else if (i == 15) {
      if (debug_)
        std::cout << "  EcalEBPhase2TimeReconstructor::process(std::vector<int>)    i = 15 " << std::endl;
      process();
      output[1] = processedOutput_;
    }
  }

  return;
}

void EcalEBPhase2TimeReconstructor::process() {
  //UB FIXME: 5
  processedOutput_ = 0;
  if (inputsAlreadyIn_ < 12)
    return;
  int64_t output = 0;
  for (int i = 0; i < 12; i++) {
    output += (weights_[i] * buffer_[i]);
    if (debug_)
      std::cout << " TimeFilter buffer " << buffer_[i] << " weight " << weights_[i] << " output " << output
                << std::endl;
  }
  output = output >> shift_;
  if (debug_)
    std::cout << " TimeFilter local  output " << output << std::endl;
  //Dividing output by the result of the amplitude reconstruction via an approximation using the invAmpAr lookup table
  int ampInd = 0;
  if (debug_)
    std::cout << " inputsAlreadyIn_ " << inputsAlreadyIn_ << std::endl;
  if (inputsAlreadyIn_ > 12) {
    ampInd = 1;
  }

  if (debug_)
    std::cout << " Begininning Final TimeFilter Calculation" << std::endl;

  int64_t tmpOutput = output * invAmpAr_[ampIn_[ampInd]];
  if (debug_)
    std::cout << " output*tmpInvAmpAr " << tmpOutput << std::endl;

  output = tmpOutput >> 20;
  if (debug_)
    std::cout << " output after bit shift " << output << std::endl;

  if (output < -1024)
    output = -1023;
  else if (output > 1024)
    output = 1023;
  if (debug_)
    std::cout << " output after if/else " << output << std::endl;
  processedOutput_ = output;

  if (debug_)
    std::cout << " TimeFilter final output " << processedOutput_ << std::endl;
}

void EcalEBPhase2TimeReconstructor::setParameters(uint32_t raw,
                                                  const EcalEBPhase2TPGTimeWeightIdMap *ecaltpgWeightMap,
                                                  const EcalTPGWeightGroup *ecaltpgWeightGroup) {
  uint32_t params_[12];
  const EcalTPGGroups::EcalTPGGroupsMap &groupmap = ecaltpgWeightGroup->getMap();
  if (debug_)
    std::cout << " EcalEBPhase2TimeReconstructor::setParameters groupmap size " << groupmap.size() << std::endl;
  EcalTPGGroups::EcalTPGGroupsMapItr it = groupmap.find(raw);
  if (it != groupmap.end()) {
    uint32_t weightid = (*it).second;
    const EcalEBPhase2TPGTimeWeightIdMap::EcalEBPhase2TPGTimeWeightMap &weightmap = ecaltpgWeightMap->getMap();
    EcalEBPhase2TPGTimeWeightIdMap::EcalEBPhase2TPGTimeWeightMapItr itw = weightmap.find(weightid);

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
      std::cout << " EcalEBPhase2TimeReconstructor::setParameters time weights after the map  " << params_[0] << " "
                << params_[1] << " " << params_[2] << " " << params_[3] << " " << params_[4] << " " << params_[5] << " "
                << params_[6] << " " << params_[7] << " " << params_[8] << " " << params_[9] << " " << params_[10]
                << " " << params_[11] << std::endl;

    // we have to transform negative coded in 16 bits into negative coded in 32 bits
    // maybe this should go into the getValue method??
    //std::cout << "peak flag settings" << std::endl;
    for (int i = 0; i < 12; ++i) {
      weights_[i] = (params_[i] & 0x8000) ? (int)(params_[i] | 0xffff8000) : (int)(params_[i]);
    }

  } else
    edm::LogWarning("EcalTPG")
        << " EcalEBPhase2TimeReconstructor::setParameters could not find EcalTPGGroupsMap entry for " << raw;
}
