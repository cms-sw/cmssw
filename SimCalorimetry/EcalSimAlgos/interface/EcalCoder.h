
#ifndef EcalSimAlgos_EcalCoder_h
#define EcalSimAlgos_EcalCoder_h 1

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCorrelatedNoiseMatrix.h"

template<typename M> class CorrelatedNoisifier ;
class EcalMGPASample;
class EcalDataFrame;
class DetId;

#include<vector>

/* \class EEDigitizerTraits
 * \brief Converts CaloDataFrame in CaloTimeSample and vice versa.
 *
 */
class EcalCoder
{
   public:

      typedef CaloTSamples<float,10> EcalSamples ;
      
      typedef CorrelatedNoisifier<EcalCorrMatrix> Noisifier ;

      enum { NBITS         =   12 , // number of available bits
	     MAXADC        = 4095 , // 2^12 -1,  adc max range
	     ADCGAINSWITCH = 4079 , // adc gain switch
	     NGAINS        =    3   // number of electronic gains
      };

      /// ctor
      EcalCoder( bool        addNoise        , 
		 bool        PreMix1         ,
		 Noisifier* ebCorrNoise0     ,
		 Noisifier* eeCorrNoise0 = 0 ,
		 Noisifier* ebCorrNoise1 = 0 ,
		 Noisifier* eeCorrNoise1 = 0 ,
		 Noisifier* ebCorrNoise2 = 0 ,
		 Noisifier* eeCorrNoise2 = 0   ) ; // make EE version optional for tb compatibility
      /// dtor
      virtual ~EcalCoder() ;

      /// can be fetched every event from the EventSetup
      void setPedestals( const EcalPedestals* pedestals ) ;

      void setGainRatios( const EcalGainRatios* gainRatios ) ;

      void setFullScaleEnergy( double EBscale ,
			       double EEscale   ) ;

      void setIntercalibConstants( const EcalIntercalibConstantsMC* ical ) ; 
 

      /// from EcalSamples to EcalDataFrame
      virtual void analogToDigital( const EcalSamples& clf , 
				    EcalDataFrame&     df    ) const;
 
   private:

      /// limit on the energy scale due to the electronics range
      double fullScaleEnergy( const DetId & did ) const ;

      /// produce the pulse-shape
      void encode( const EcalSamples& ecalSamples , 
		   EcalDataFrame&     df            ) const ;

//      double decode( const EcalMGPASample& sample , 
//		     const DetId&          detId    ) const ;

      /// not yet implemented
      //      void noisify( const EcalIntercalibConstantsMC* values ,
      //		    int                              size     ) const ;

      void findPedestal( const DetId& detId    , 
			 int          gainId   , 
			 double&      pedestal ,
			 double&      width      ) const ;
    
      void findGains( const DetId& detId, 
		      double       theGains[] ) const ;

      void findIntercalibConstant( const DetId& detId ,
				   double&      icalconst ) const ;
   
      const EcalPedestals* m_peds ;
      
      const EcalGainRatios* m_gainRatios ; // the electronics gains

      const EcalIntercalibConstantsMC* m_intercals ; //record specific for simulation of gain variation in MC

      double m_maxEneEB ; // max attainable energy in the ecal barrel
      double m_maxEneEE ; // max attainable energy in the ecal endcap
      
      bool m_addNoise ;   // whether add noise to the pedestals and the gains
      bool m_PreMix1 ;   // Follow necessary steps for PreMixing input

      const Noisifier* m_ebCorrNoise[3] ;
      const Noisifier* m_eeCorrNoise[3] ;
};

#endif
