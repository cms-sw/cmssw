#ifndef EcalSimAlgos_EcalDigitizerTraits_h
#define EcalSimAlgos_EcalDigitizerTraits_h

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"

/* \class EBDigitizerTraits
 * \brief typenames for the ECAL barrel digitization
 *
 */
class EBDigitizerTraits {
  /// the digis collection
  typedef EBDigiCollection DigiCollection;
  /// the dataframes
  typedef EBDataFrame Digi;
  /// the electronics simulation
  typedef EcalElectronicsSim ElectronicsSim;
};


/* \class EEDigitizerTraits
 * \brief typenames for the ECAL endcap digitization
 *
 */
class EEDigitizerTraits {
  /// the digis collection
  typedef EEDigiCollection DigiCollection;
  /// the dataframes
  typedef EEDataFrame Digi;
  /// the electronics simulation
  typedef EcalElectronicsSim ElectronicsSim;
};


/* \class ESDigitizerTraits
 * \brief typenames for the preshower digitization
 *
 */
class ESDigitizerTraits {
  /// the digis collection
  typedef ESDigiCollection DigiCollection;
  /// the dataframes
  typedef ESDataFrame Digi;
  /// the electronics simulation
  typedef ESElectronicsSim ElectronicsSim;
};


#endif

