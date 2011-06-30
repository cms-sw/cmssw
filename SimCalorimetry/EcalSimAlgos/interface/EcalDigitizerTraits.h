#ifndef EcalSimAlgos_EcalDigitizerTraits_h
#define EcalSimAlgos_EcalDigitizerTraits_h

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"

class EcalHitResponse ;

class EBDigitizerTraits 
{
   public:
      /// the digis collection
      typedef EBDigiCollection DigiCollection;
      /// the dataframes
      typedef EBDataFrame Digi;
      /// the electronics simulation
      typedef EcalElectronicsSim ElectronicsSim;

      void addNoiseHits( EcalHitResponse* hr ) const { } ;      
};

class EEDigitizerTraits 
{
   public:
      /// the digis collection
      typedef EEDigiCollection DigiCollection;
      /// the dataframes
      typedef EEDataFrame Digi;
      /// the electronics simulation
      typedef EcalElectronicsSim ElectronicsSim;
      
      void addNoiseHits( EcalHitResponse* hr ) const { } ;
};

class ESDigitizerTraits 
{
   public:
      /// the digis collection
      typedef ESDigiCollection DigiCollection;
      /// the dataframes
      typedef ESDataFrame Digi;
      /// the electronics simulation
      typedef ESElectronicsSim ElectronicsSim;

      ESDigitizerTraits() ;

      void addNoiseHits( EcalHitResponse* hr ) ;
};


#endif

