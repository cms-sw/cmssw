#ifndef EcalSimAlgos_EcalDigitizerTraits_h
#define EcalSimAlgos_EcalDigitizerTraits_h

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"

#include "DataFormats/EcalDigi/interface/EcalDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EKDataFrame.h"

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

      typedef CaloTSamples<float,10> EcalSamples ;

      static void fix( Digi& digi, edm::DataFrame df ) {};
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

      typedef CaloTSamples<float,10> EcalSamples ;

      static void fix( Digi& digi, edm::DataFrame df ) {}
};

class EKDigitizerTraits 
{
   public:
      /// the digis collection
      typedef EKDigiCollection DigiCollection;
      /// the dataframes
      typedef EKDataFrame Digi;
      /// the electronics simulation
      typedef EcalElectronicsSim ElectronicsSim;

      typedef CaloTSamples<float,10> EcalSamples ;

      static void fix( Digi& digi, edm::DataFrame df ) {
	for(int i=0; i < digi.size(); i++){
	  df[i]=digi[i];
	}
      }
};

class ESDigitizerTraits 
{
   public:
      /// the digis collection
      typedef ESDigiCollection DigiCollection ;
      /// the dataframes
      typedef ESDataFrame Digi ;
      /// the electronics simulation
      typedef ESElectronicsSimFast ElectronicsSim ;

      typedef CaloTSamples<float,3> EcalSamples ;

      static void fix( Digi& digi, edm::DataFrame df ) {
	 for( unsigned int i ( 0 ) ; i != 3; ++i )
	 {
	    static const int offset ( 65536 ) ; // for int16 to uint16
	    const int16_t dshort ( digi[i].raw() ) ;
	    const int     dint   ( (int) dshort + // add offset for uint16 conversion
				   ( (int16_t) 0 > dshort ? 
				     offset : (int) 0 ) ) ;
	    df[i] = dint ;
	 }
      }
};

class ESOldDigitizerTraits 
{
   public:
      /// the digis collection
      typedef ESDigiCollection DigiCollection ;
      /// the dataframes
      typedef ESDataFrame Digi ;
      /// the electronics simulation
      typedef ESElectronicsSim ElectronicsSim ;

//      typedef CaloTSamples<float,3> EcalSamples ;
};


#endif

