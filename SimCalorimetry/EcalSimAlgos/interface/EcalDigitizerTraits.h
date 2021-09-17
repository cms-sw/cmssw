#ifndef EcalSimAlgos_EcalDigitizerTraits_h
#define EcalSimAlgos_EcalDigitizerTraits_h

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalLiteDTUCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDataFrame_Ph2.h"

class EcalHitResponse;

class EBDigitizerTraits {
public:
  /// the digis collection
  typedef EBDigiCollection DigiCollection;
  /// the dataframes
  typedef EBDataFrame Digi;
  /// the electronics simulation
  typedef CaloTSamples<float, 10> EcalSamples;
  typedef EcalElectronicsSim<EcalCoder, EcalSamples, EcalDataFrame> ElectronicsSim;

  static void fix(Digi& digi, edm::DataFrame df){};
};

class EEDigitizerTraits {
public:
  /// the digis collection
  typedef EEDigiCollection DigiCollection;
  /// the dataframes
  typedef EEDataFrame Digi;
  /// the electronics simulation
  typedef CaloTSamples<float, 10> EcalSamples;
  typedef EcalElectronicsSim<EcalCoder, EcalSamples, EcalDataFrame> ElectronicsSim;

  static void fix(Digi& digi, edm::DataFrame df) {}
};

class ESDigitizerTraits {
public:
  /// the digis collection
  typedef ESDigiCollection DigiCollection;
  /// the dataframes
  typedef ESDataFrame Digi;
  /// the electronics simulation
  typedef ESElectronicsSimFast ElectronicsSim;

  typedef CaloTSamples<float, 3> EcalSamples;

  static void fix(Digi& digi, edm::DataFrame df) {
    for (unsigned int i(0); i != 3; ++i) {
      static const int offset(65536);  // for int16 to uint16
      const int16_t dshort(digi[i].raw());
      const int dint((int)dshort +  // add offset for uint16 conversion
                     ((int16_t)0 > dshort ? offset : (int)0));
      df[i] = dint;
    }
  }
};

class ESOldDigitizerTraits {
public:
  /// the digis collection
  typedef ESDigiCollection DigiCollection;
  /// the dataframes
  typedef ESDataFrame Digi;
  /// the electronics simulation
  typedef ESElectronicsSim ElectronicsSim;

  //      typedef CaloTSamples<float,3> EcalSamples ;
};

class EBDigitizerTraits_Ph2 {
public:
  /// the digis collection
  typedef EBDigiCollectionPh2 DigiCollection;
  /// the dataframes
  typedef EcalDataFrame_Ph2 Digi;
  /// the electronics simulation
  typedef CaloTSamples<float, ecalPh2::sampleSize> EcalSamples;
  typedef EcalElectronicsSim<EcalLiteDTUCoder, EcalSamples, Digi> ElectronicsSim;

  static void fix(Digi& digi, edm::DataFrame df){};
};

#endif
