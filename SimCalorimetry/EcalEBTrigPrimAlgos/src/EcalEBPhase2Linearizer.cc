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
    edm::LogError("EcalEBPhase2Linearizer") << " could not find EcalLiteDTUPedestal entry for " << detId << std::endl;

  const EcalEBPhase2TPGLinearizationConstMap &linMap = ecaltpLin->getMap();
  EcalEBPhase2TPGLinearizationConstMapIterator it = linMap.find(detId.rawId());
  if (it != linMap.end()) {
    linConsts_ = &(*it);
  } else
    edm::LogError("EcalEBPhase2Linearizer")
        << " could not find EcalEBPhase2TPGLinearizationConstMap entry for " << detId.rawId() << std::endl;

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
    LogDebug("EcalEBPhase2Linearizer") << " mult "
                                       << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      LogDebug("") << mult_ << " ";
    }
    LogDebug("") << " " << std::endl;

    LogDebug("") << " gainID "
                 << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      LogDebug("") << gainID_ << " ";
    }
    LogDebug("") << " " << std::endl;

    LogDebug("") << " Ped "
                 << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      LogDebug("") << base_ << " ";
    }
    LogDebug("") << " " << std::endl;

    LogDebug("") << " i2c "
                 << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      LogDebug("") << I2CSub_ << " ";
    }
    LogDebug("") << " " << std::endl;

    LogDebug("") << " shift "
                 << " ";
    for (int i = 0; i < df.size(); i++) {
      EcalLiteDTUSample thisSample = df[i];
      setInput(thisSample);
      LogDebug("") << shift_ << " ";
    }
    LogDebug("") << " " << std::endl;

    LogDebug("") << " lin out "
                 << " ";
    for (int i = 0; i < df.size(); i++) {
      LogDebug("") << output_percry[i] << " ";
    }

    LogDebug("") << " " << std::endl;

    LogDebug("") << " EcalEBPhase2Linearizer::process(const  .. Final output " << std::endl;
    LogDebug("") << " output_percry "
                 << " ";
    for (int i = 0; i < df.size(); i++) {
      LogDebug("") << output_percry[i] << " ";
    }
    LogDebug("") << " " << std::endl;
  }
  return;
}
