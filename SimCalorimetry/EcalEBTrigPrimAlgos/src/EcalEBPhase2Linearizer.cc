#include <SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2Linearizer.h>

//#include <CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

EcalEBPhase2Linearizer::EcalEBPhase2Linearizer(bool debug)
    : debug_(debug), init_(false), peds_(nullptr), badXStatus_(nullptr) {}

EcalEBPhase2Linearizer::~EcalEBPhase2Linearizer() {
  if (init_) {
    for (int i = 0; i < (int)vectorbadXStatus_.size(); i++) {
      delete vectorbadXStatus_[i];
    }
  }
}

void EcalEBPhase2Linearizer::setParameters(EBDetId detId,
                                           const EcalLiteDTUPedestalsMap *ecaltpPed,
                                           const EcalEBPhase2TPGLinearizationConstMap *ecaltpLin,
                                           const EcalTPGCrystalStatus *ecaltpBadX)

{
  EcalLiteDTUPedestalsMap::const_iterator itped = ecaltpPed->getMap().find(detId);
  if (itped != ecaltpPed->end())
    peds_ = &(*itped);
  else
    std::cout << " could not find EcalLiteDTUPedestal entry for " << detId << std::endl;

  const EcalEBPhase2TPGLinearizationConstMap &linMap = ecaltpLin->getMap();
  EcalEBPhase2TPGLinearizationConstMapIterator it = linMap.find(detId.rawId());
  if (it != linMap.end()) {
    linConsts_ = &(*it);
  } else
    std::cout << " could not find EcalEBPhase2TPGLinearizationConstMap entry for " << detId.rawId() << std::endl;

  const EcalTPGCrystalStatusMap &badXMap = ecaltpBadX->getMap();
  EcalTPGCrystalStatusMapIterator itbadX = badXMap.find(detId.rawId());
  if (itbadX != badXMap.end()) {
    badXStatus_ = &(*itbadX);
  } else {
    edm::LogWarning("EcalTPG") << " could not find EcalTPGCrystalStatusMap entry for " << detId.rawId();
    badXStatus_ = new EcalTPGCrystalStatusCode();
    vectorbadXStatus_.push_back(&(*badXStatus_));
    init_ = true;
  }
}

int EcalEBPhase2Linearizer::doOutput() {
  int tmpIntOut;
  if (uncorrectedSample_) {
    tmpIntOut = (uncorrectedSample_ - base_ + I2CSub_);  //Substract base. Add I2C
  } else {
    tmpIntOut = 0;
  }
  if (tmpIntOut < 0) {
    tmpIntOut = 0;
  }
  uint output = tmpIntOut;
  output = (output * mult_) >> shift_;
  // protect against saturation
  // ...........

  return output;
}

int EcalEBPhase2Linearizer::setInput(const EcalLiteDTUSample &RawSam)

{
  uncorrectedSample_ = RawSam.adc();  //uncorrectedSample_
  gainID_ = RawSam.gainId();

  base_ = peds_->mean(gainID_);

  if (gainID_ == 0) {
    mult_ = linConsts_->mult_x10;
    shift_ = linConsts_->shift_x10;
    I2CSub_ = linConsts_->i2cSub_x10;
  } else {
    mult_ = linConsts_->mult_x1;
    shift_ = linConsts_->shift_x1;
    I2CSub_ = linConsts_->i2cSub_x1;
  }

  return 1;
}

void EcalEBPhase2Linearizer::process(const EBDigiCollectionPh2::Digi &df, std::vector<int> &output_percry) {
  //We know a tower numbering is:                                                                                                                      // S1 S2 S3 S4 S5

  // 4  5  14 15 24
  // 3  6  13 16 23
  // 2  7  12 17 22
  // 1  8  11 18 21
  // 0  9  10 19 20

  for (int i = 0; i < df.size(); i++) {
    EcalLiteDTUSample thisSample = df[i];
    setInput(thisSample);
    output_percry[i] = doOutput();
  }

  if (debug_) {
    std::cout << " mult "
              << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      std::cout << mult_ << " ";
    }
    std::cout << " " << std::endl;

    std::cout << " gainID "
              << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      std::cout << gainID_ << " ";
    }
    std::cout << " " << std::endl;

    std::cout << " Ped "
              << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      std::cout << base_ << " ";
    }
    std::cout << " " << std::endl;

    std::cout << " i2c "
              << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      std::cout << I2CSub_ << " ";
    }
    std::cout << " " << std::endl;

    std::cout << " shift "
              << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      std::cout << shift_ << " ";
    }
    std::cout << " " << std::endl;

    std::cout << " lin out "
              << " ";
    for (int i = 0; i < df.size(); i++) {
      std::cout << output_percry[i] << " ";
    }

    std::cout << " " << std::endl;

    std::cout << " EcalEBPhase2Linearizer::process(const  .. Final output " << std::endl;
    std::cout << " output_percry "
              << " ";
    for (int i = 0; i < df.size(); i++) {
      std::cout << output_percry[i] << " ";
    }
    std::cout << " " << std::endl;
  }
  return;
}
