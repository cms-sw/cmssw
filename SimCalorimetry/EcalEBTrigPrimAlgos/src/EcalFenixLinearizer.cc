#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixLinearizer.h>

#include <CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h>
#include <CondFormats/EcalObjects/interface/EcalTPGPedestals.h>
#include <CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalFenixLinearizer::EcalFenixLinearizer(bool famos)
    : famos_(famos), init_(false), linConsts_(nullptr), peds_(nullptr), badXStatus_(nullptr) {}

EcalFenixLinearizer::~EcalFenixLinearizer() {
  if (init_) {
    for (int i = 0; i < (int)vectorbadXStatus_.size(); i++) {
      delete vectorbadXStatus_[i];
    }
  }
}

void EcalFenixLinearizer::setParameters(uint32_t raw,
                                        const EcalTPGPedestals *ecaltpPed,
                                        const EcalTPGLinearizationConst *ecaltpLin,
                                        const EcalTPGCrystalStatus *ecaltpBadX) {
  const EcalTPGLinearizationConstMap &linMap = ecaltpLin->getMap();
  EcalTPGLinearizationConstMapIterator it = linMap.find(raw);

  if (it != linMap.end()) {
    linConsts_ = &(*it);
  } else
    std::cout << " could not find EcalTPGLinearizationConstMap entry for " << raw << std::endl;

  EcalTPGPedestalsMapIterator itped = ecaltpPed->find(raw);
  if (itped != ecaltpPed->end())
    peds_ = &(*itped);
  else
    std::cout << " could not find EcalTPGPedestalsMap entry for " << raw << std::endl;

  const EcalTPGCrystalStatusMap &badXMap = ecaltpBadX->getMap();
  EcalTPGCrystalStatusMapIterator itbadX = badXMap.find(raw);

  if (itbadX != badXMap.end()) {
    badXStatus_ = &(*itbadX);
  } else {
    edm::LogWarning("EcalTPG") << " could not find EcalTPGCrystalStatusMap entry for " << raw;
    badXStatus_ = new EcalTPGCrystalStatusCode();
    vectorbadXStatus_.push_back(&(*badXStatus_));
    init_ = true;
  }
}

int EcalFenixLinearizer::process() {
  int output = (uncorrectedSample_ - base_);  //Substract base
  //std::cout << " EcalFenixLinearizer::process() output non bit shifted " << output << std::endl;
  if (famos_ || output < 0)
    return 0;

  if (output < 0)
    return shift_ << 12;                      // FENIX bug(!)
  output = (output * mult_) >> (shift_ + 2);  //Apply multiplicative factor
  //std::cout << " EcalFenixLinearizer::process() output 2nd step " << output << std::endl;
  if (output > 0X3FFFF)
    output = 0X3FFFF;  //Saturation if too high
  //std::cout << " EcalFenixLinearizer::process() output 3rd step " << output << std::endl;
  return output;
}

int EcalFenixLinearizer::setInput(const EcalMGPASample &RawSam) {
  //std::cout << "  EcalFenixLinearizer::setInput RawSam.raw() " << RawSam.raw() << std::endl;
  if (RawSam.raw() > 0X3FFF) {
    LogDebug("EcalTPG") << "ERROR IN INPUT SAMPLE OF FENIX LINEARIZER";
    return -1;
  }

  uncorrectedSample_ = RawSam.adc();  //uncorrectedSample_ is coded in the 12 LSB
  gainID_ = RawSam.gainId();          //uncorrectedSample_ is coded in the 2 next bits!

  if (gainID_ == 0) {
    base_ = 0;
    shift_ = 0;
    mult_ = 0xFF;
    if ((linConsts_->mult_x12 == 0) && (linConsts_->mult_x6 == 0) && (linConsts_->mult_x1 == 0)) {
      mult_ = 0;  // Implemented in CCSSupervisor to
                  // reject overflow cases in rejected channels
    }
  } else if (gainID_ == 1) {
    base_ = peds_->mean_x12;
    shift_ = linConsts_->shift_x12;

    // take into account the badX
    // badXStatus_ == 0 if the crystal works
    // badXStatus_ !=0 some problem with the crystal
    if (badXStatus_->getStatusCode() != 0) {
      mult_ = 0;
    } else {
      mult_ = linConsts_->mult_x12;
    }
  } else if (gainID_ == 2) {
    base_ = peds_->mean_x6;
    shift_ = linConsts_->shift_x6;

    // take into account the badX
    // check if the badX has a status code=0 or 1
    if (badXStatus_->getStatusCode() != 0) {
      mult_ = 0;
    } else {
      mult_ = linConsts_->mult_x6;
    }
  } else if (gainID_ == 3) {
    base_ = peds_->mean_x1;
    shift_ = linConsts_->shift_x1;

    // take into account the badX
    // check if the badX has a status code=0 or 1
    if (badXStatus_->getStatusCode() != 0) {
      mult_ = 0;
    } else {
      mult_ = linConsts_->mult_x1;
    }
  }

  //std::cout << "  EcalFenixLinearizer::setInput   uncorrectedSample_ " << RawSam.adc() << " gainID " << gainID_ << " baseline " << base_ << std::endl;

  if (famos_)
    base_ = 200;  //FIXME by preparing a correct TPG.txt for Famos

  return 1;
}
