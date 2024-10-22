#ifndef HcalSimAlgos_HcalDigitizerTraits_h
#define HcalSimAlgos_HcalDigitizerTraits_h
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalElectronicsSim.h"

class HBHEDigitizerTraits {
public:
  typedef HBHEDigiCollection DigiCollection;
  typedef HBHEDataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
  static constexpr double PreMixFactor = 10.0;
  static const unsigned PreMixBits = 126;
};

class HODigitizerTraits {
public:
  typedef HODigiCollection DigiCollection;
  typedef HODataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
  static constexpr double PreMixFactor = 10.0;
  static const unsigned PreMixBits = 126;
};

class HFDigitizerTraits {
public:
  typedef HFDigiCollection DigiCollection;
  typedef HFDataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
  static constexpr double PreMixFactor = 10.0;
  static const unsigned PreMixBits = 126;
};

class ZDCDigitizerTraits {
public:
  typedef ZDCDigiCollection DigiCollection;
  typedef ZDCDataFrame Digi;
  typedef HcalElectronicsSim ElectronicsSim;
  static constexpr double PreMixFactor = 10.0;
  static const unsigned PreMixBits = 126;
};

template <class Traits>
class CaloTDigitizerQIE8Run {
public:
  typedef typename Traits::ElectronicsSim ElectronicsSim;
  typedef typename Traits::Digi Digi;
  typedef typename Traits::DigiCollection DigiCollection;

  void operator()(DigiCollection& output,
                  CLHEP::HepRandomEngine* engine,
                  CaloSamples* analogSignal,
                  std::vector<DetId>::const_iterator idItr,
                  ElectronicsSim* theElectronicsSim) {
    Digi digi(*idItr);
    theElectronicsSim->analogToDigital(engine, *analogSignal, digi, Traits::PreMixFactor, Traits::PreMixBits);
    output.push_back(std::move(digi));
  }
};

#endif
