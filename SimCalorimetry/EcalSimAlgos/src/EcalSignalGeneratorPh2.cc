#include "SimCalorimetry/EcalSimAlgos/interface/EcalSignalGeneratorPh2.h"

template <>
EcalSignalGeneratorPh2<EBDigitizerTraits_Ph2>::EcalSignalGeneratorPh2(edm::ConsumesCollector& cc,
                                                                      const edm::InputTag& inputTag,
                                                                      const double ebs25notCont,
                                                                      const double peToA,
                                                                      const bool timeDependent)
    : EcalBaseSignalGenerator(),
      gainRatiosToken_(cc.esConsumes()),
      interCalibConstantsMCToken_(cc.esConsumes()),
      adcToGeVConstantToken_(cc.esConsumes()),
      theEvent_(nullptr),
      theEventPrincipal_(nullptr),
      gainRatios_(nullptr),
      theInputTag_(inputTag),
      tok_(cc.consumes<COLLECTION>(inputTag)),
      ebs25notCont_(ebs25notCont),
      peToA_(peToA),
      ical_(nullptr),
      timeDependent_(timeDependent),
      lasercals_(nullptr),
      lasercals_prime_(nullptr) {
  if (timeDependent_) {
    laserDbToken_ = cc.esConsumes();
    laserDbMCToken_ = cc.esConsumes();
  }
}

template <>
void EcalSignalGeneratorPh2<EBDigitizerTraits_Ph2>::initializeEvent(const edm::Event* event,
                                                                    const edm::EventSetup* eventSetup) {
  theEvent_ = event;
  gainRatios_ = &eventSetup->getData(gainRatiosToken_);  // find the gains
  // Ecal Intercalibration Constants
  ical_ = &eventSetup->getData(interCalibConstantsMCToken_);
  // adc to GeV
  auto const& agc = eventSetup->getData(adcToGeVConstantToken_);
  maxEne_ = agc.getEBValue() * ecalPh2::gains[0] * ecalPh2::MAXADC * ebs25notCont_;

  if (timeDependent_) {
    //----
    //
    auto const eventTimeValue = theEvent_->getRun().runAuxiliary().beginTime().value();
    //
    //         The "time" will have to match in the generation of the tag
    //         for the MC from ECAL (apd/pn, alpha, whatever time dependent is needed)
    //
    iTime_ = eventTimeValue;

    // Ecal LaserCorrection Constants for laser correction ratio
    lasercals_ = &eventSetup->getData(laserDbToken_);

    //
    // the "prime" is exactly the same as the usual laser service, BUT
    // it has only 1 IOV, so that effectively you are dividing IOV_n / IOV_0
    // NB: in the creation of the tag make sure the "prime" (MC) tag is prepared properly!
    // NB again: if many IOVs also in "MC" tag, then fancy things could be perfomed ... left for the future
    //
    lasercals_prime_ = &eventSetup->getData(laserDbMCToken_);

    //clear the laser cache for each event time
    CalibCache().swap(valueLCCache_LC_);
    CalibCache().swap(valueLCCache_LC_prime_);  //--- also the "prime" ... yes
    //----
  }
}

template <>
void EcalSignalGeneratorPh2<EBDigitizerTraits_Ph2>::initializeEvent(const edm::EventPrincipal* eventPrincipal,
                                                                    const edm::EventSetup* eventSetup) {
  theEventPrincipal_ = eventPrincipal;
  gainRatios_ = &eventSetup->getData(gainRatiosToken_);  // find the gains
  // Ecal Intercalibration Constants
  ical_ = &eventSetup->getData(interCalibConstantsMCToken_);
  // adc to GeV
  auto const& agc = eventSetup->getData(adcToGeVConstantToken_);
  maxEne_ = agc.getEBValue() * ecalPh2::gains[0] * ecalPh2::MAXADC * ebs25notCont_;

  if (timeDependent_) {
    //----
    edm::TimeValue_t eventTimeValue = 0;
    if (theEventPrincipal_) {
      //
      eventTimeValue = theEventPrincipal_->runPrincipal().beginTime().value();
      //
      //         The "time" will have to match in the generation of the tag
      //         for the MC from ECAL (apd/pn, alpha, whatever time dependent is needed)
      //
    } else {
      edm::LogError("EcalSignalGeneratorPh2") << " theEventPrincipal not defined??? " << std::endl;
    }
    iTime_ = eventTimeValue;

    // Ecal LaserCorrection Constants for laser correction ratio
    lasercals_ = &eventSetup->getData(laserDbToken_);
    lasercals_prime_ = &eventSetup->getData(laserDbMCToken_);

    //clear the laser cache for each event time
    CalibCache().swap(valueLCCache_LC_);
    CalibCache().swap(valueLCCache_LC_prime_);  //--- also the "prime" ... yes
    //----
  }
}

template <>
CaloSamples EcalSignalGeneratorPh2<EBDigitizerTraits_Ph2>::samplesInPE(const DIGI& digi) {
  const auto detId = digi.id();

  double icalconst = 1.;  // find the correct value.
  auto const& icalMap = ical_->getMap();
  auto icalit = icalMap.find(detId);
  if (icalit != icalMap.end()) {
    icalconst = *icalit;
  }

  auto const gainRatio = (*gainRatios_)[detId];

  double LSB[ecalPh2::NGAINS];
  for (unsigned int igain(0); igain < ecalPh2::NGAINS; ++igain) {
    LSB[igain] = maxEne_ / (ecalPh2::MAXADC * gainRatio);
  }

  LogDebug("EcalSignalGeneratorPh2") << "Intercalibration: " << icalconst << ", LSBs: " << LSB[0] << " " << LSB[1]
                                     << ", gain ratio: " << gainRatio << ", max. energy: " << maxEne_;

  CaloSamples result(detId, digi.size());

  // correction factor for premixed sample: ratio of laser corrections
  // LC' /  LC  (see formula)
  double value_LC = 1.;
  double value_LC_prime = 1.;
  if (timeDependent_) {
    auto const& evtTimeStamp = edm::Timestamp(iTime_);

    // LC that depends with time
    auto cache = valueLCCache_LC_.find(detId);
    if (cache != valueLCCache_LC_.end()) {
      value_LC = cache->second;
    } else {
      value_LC = lasercals_->getLaserCorrection(detId, evtTimeStamp);
      valueLCCache_LC_.emplace(detId, value_LC);
    }

    // LC at the beginning of the time (first IOV of the GT == first time)
    // Using the different "tag", the one with "MC"
    cache = valueLCCache_LC_prime_.find(detId);
    if (cache != valueLCCache_LC_prime_.end()) {
      value_LC_prime = cache->second;
    } else {
      value_LC_prime = lasercals_prime_->getLaserCorrection(detId, evtTimeStamp);
      valueLCCache_LC_prime_.emplace(detId, value_LC_prime);
    }
  }
  auto const correction_factor_for_premixed_sample_transparency = value_LC_prime / value_LC;

  for (int isample = 0; isample < digi.size(); ++isample) {
    auto const gainId = digi[isample].gainId();
    result[isample] = float(digi[isample].adc()) * LSB[gainId] * icalconst / peToA_ *
                      correction_factor_for_premixed_sample_transparency;
  }

  LogDebug("EcalSignalGeneratorPh2").log([&](auto& li) {
    li << "Noise input: " << digi;
    li << "\nConverted noise sample:\n";
    for (int isample = 0; isample < digi.size(); ++isample) {
      li << " " << result[isample];
    }
  });

  return result;
}

template <>
void EcalSignalGeneratorPh2<EBDigitizerTraits_Ph2>::fill(edm::ModuleCallingContext const* mcc) {
  theNoiseSignals.clear();
  edm::Handle<COLLECTION> pDigis;
  const COLLECTION* digis = nullptr;
  // try accessing by whatever is set, Event or EventPrincipal
  if (theEvent_) {
    if (theEvent_->getByToken(tok_, pDigis)) {
      digis = pDigis.product();  // get a ptr to the product
    } else {
      throw cms::Exception("EcalSignalGeneratorPh2") << "Cannot find input data " << theInputTag_;
    }
  } else if (theEventPrincipal_) {
    auto const digisPTR = edm::getProductByTag<COLLECTION>(*theEventPrincipal_, theInputTag_, mcc);
    if (digisPTR) {
      digis = digisPTR->product();
    }
  } else {
    throw cms::Exception("EcalSignalGeneratorPh2") << "No Event or EventPrincipal was set";
  }

  if (digis) {
    // loop over digis, adding these to the existing maps
    for (typename COLLECTION::const_iterator it = digis->begin(); it != digis->end(); ++it) {
      // need to convert to something useful
      if (validDigi(*it)) {
        theNoiseSignals.push_back(samplesInPE(*it));
      }
    }
  }
}
