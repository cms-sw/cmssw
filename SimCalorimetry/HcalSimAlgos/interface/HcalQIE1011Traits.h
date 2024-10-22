#ifndef HcalSimAlgos_HcalQIE10Traits_h
#define HcalSimAlgos_HcalQIE10Traits_h

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"

class HcalQIE10DigitizerTraits {
public:
  typedef QIE10DigiCollection DigiCollection;
  typedef QIE10DataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
  static constexpr double PreMixFactor = 10.0;
  static const unsigned PreMixBits = 254;
};

class HcalQIE11DigitizerTraits {
public:
  typedef QIE11DigiCollection DigiCollection;
  typedef QIE11DataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
  static constexpr double PreMixFactor = 10.0;
  static const unsigned PreMixBits = 254;
};

template <class Traits>
class CaloTDigitizerQIE1011Run {
public:
  typedef typename Traits::ElectronicsSim ElectronicsSim;
  typedef typename Traits::Digi Digi;
  typedef typename Traits::DigiCollection DigiCollection;

  void operator()(DigiCollection& output,
                  CLHEP::HepRandomEngine* engine,
                  CaloSamples* analogSignal,
                  std::vector<DetId>::const_iterator idItr,
                  ElectronicsSim* theElectronicsSim) {
    output.push_back(idItr->rawId());
    Digi digi(output.back());  //QIEDataFrame gets ptr to edm::DataFrame data
    theElectronicsSim->analogToDigital(engine, *analogSignal, digi, Traits::PreMixFactor, Traits::PreMixBits);
  }
};

#endif
