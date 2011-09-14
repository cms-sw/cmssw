#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESFastTDigitizer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"


EcalDigiProducer::EcalDigiProducer( const edm::ParameterSet& params ) :
   m_APDDigitizer    ( 0 ) ,
   m_BarrelDigitizer ( 0 ) ,
   m_EndcapDigitizer ( 0 ) ,
   m_ESDigitizer     ( 0 ) ,
   m_ESDigitizerFast ( 0 ) ,
   m_ParameterMap    ( 0 ) ,
   m_APDShape        ( params.getParameter<double>( "apdShapeTstart" ) ,
		       params.getParameter<double>( "apdShapeTau"    )   )  ,
   m_EBShape         (   ) ,
   m_EEShape         (   ) ,
   m_ESShape         ( new ESShape() ) ,
   m_APDResponse     ( 0 ) ,
   m_EBResponse      ( 0 ) ,
   m_EEResponse      ( 0 ) ,
   m_ESResponse      ( 0 ) ,
   m_ElectronicsSim    ( 0 ) ,
   m_ESElectronicsSim  ( 0 ) ,
   m_ESElectronicsSimFast ( 0 ) ,
   m_Coder                ( 0 ) ,
   m_APDElectronicsSim    ( 0 ) ,
   m_APDCoder             ( 0 ) ,
   m_Geometry             ( 0 ) ,
   m_apdParameters        ( 0 )
{
   /// output collections names

   m_EBdigiCollection                        = params.getParameter<std::string>("EBdigiCollection");
   m_EEdigiCollection                        = params.getParameter<std::string>("EEdigiCollection");
   m_ESdigiCollection                        = params.getParameter<std::string>("ESdigiCollection");
   const bool addNoise                       = params.getParameter<bool>       ("doNoise"); 
   const double simHitToPhotoelectronsBarrel = params.getParameter<double>     ("simHitToPhotoelectronsBarrel");
   const double simHitToPhotoelectronsEndcap = params.getParameter<double>     ("simHitToPhotoelectronsEndcap");
   const double photoelectronsToAnalogBarrel = params.getParameter<double>     ("photoelectronsToAnalogBarrel");
   const double photoelectronsToAnalogEndcap = params.getParameter<double>     ("photoelectronsToAnalogEndcap");
   const double samplingFactor               = params.getParameter<double>     ("samplingFactor");
   const double timePhase                    = params.getParameter<double>     ("timePhase");
   const unsigned int readoutFrameSize       = params.getParameter<int>        ("readoutFrameSize");
   const int binOfMaximum                    = params.getParameter<int>        ("binOfMaximum");
   const bool doPhotostatistics              = params.getParameter<bool>       ("doPhotostatistics");
   const bool syncPhase                      = params.getParameter<bool>       ("syncPhase");
   const bool cosmicsPhase                   = params.getParameter<bool>       ("cosmicsPhase");
   const double cosmicsShift                 = params.getParameter<double>     ("cosmicsShift");
   const std::vector<double> ebCorMatG12     = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG12");
   const std::vector<double> eeCorMatG12     = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG12");
   const std::vector<double> ebCorMatG06     = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG06");
   const std::vector<double> eeCorMatG06     = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG06");
   const std::vector<double> ebCorMatG01     = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG01");
   const std::vector<double> eeCorMatG01     = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG01");
   const bool applyConstantTerm              = params.getParameter<bool>       ("applyConstantTerm");
   const double rmsConstantTerm              = params.getParameter<double>     ("ConstantTerm");
   const bool addESNoise                     = params.getParameter<bool>       ("doESNoise");
   const int numESdetId                      = params.getParameter<int>        ("numESdetId");
   m_doFast                                  = params.getParameter<bool>       ("doFast");
   m_EBs25notCont                            = params.getParameter<double>     ("EBs25notContainment");
   m_EEs25notCont                            = params.getParameter<double>     ("EEs25notContainment");
   m_hitsProducerTag                         = params.getParameter<std::string>("hitsProducer");

   const bool   apdAddToBarrel  ( params.getParameter<bool>   ("apdAddToBarrel") ) ;
   const bool   apdSeparateDigi ( params.getParameter<bool>   ("apdSeparateDigi") ) ;
   const double apdSimToPELow   ( params.getParameter<double> ("apdSimToPELow" ) ) ;
   const double apdSimToPEHigh  ( params.getParameter<double> ("apdSimToPEHigh") ) ;
   const double apdTimeOffset   ( params.getParameter<double> ("apdTimeOffset" ) ) ;
   const double apdTimeOffWidth ( params.getParameter<double> ("apdTimeOffWidth" ) ) ;
   const bool   apdDoPEStats    ( params.getParameter<bool>   ("apdDoPEStats"  ) ) ;
   const std::string apdDigiTag ( params.getParameter<std::string>("apdDigiTag") ) ;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   if( apdSeparateDigi ) produces<EBDigiCollection>( apdDigiTag ) ;
   produces<EBDigiCollection>( m_EBdigiCollection ) ;
   produces<EEDigiCollection>( m_EEdigiCollection ) ;
   produces<ESDigiCollection>( m_ESdigiCollection ) ;

   // initialize the default valuer for hardcoded parameters and the EB/EE shape

   m_ParameterMap = new EcalSimParameterMap( simHitToPhotoelectronsBarrel ,
					     simHitToPhotoelectronsEndcap , 
					     photoelectronsToAnalogBarrel ,
					     photoelectronsToAnalogEndcap , 
					     samplingFactor               ,
					     timePhase                    ,
					     readoutFrameSize             , 
					     binOfMaximum                 ,
					     doPhotostatistics            ,
					     syncPhase                     ) ;

   m_apdParameters = new APDSimParameters( apdAddToBarrel  ,
					   apdSeparateDigi ,
					   apdSimToPELow   ,
					   apdSimToPEHigh  ,
					   apdTimeOffset   ,
					   apdTimeOffWidth ,
					   apdDoPEStats    ,
					   apdDigiTag        ) ;
  


   if( apdSeparateDigi ) 
      m_APDResponse = new EBHitResponse( m_ParameterMap  ,
					 &m_EBShape      ,
					 true            ,
					 m_apdParameters ,
					 &m_APDShape       ) ;
   
   m_EBResponse = new EBHitResponse( m_ParameterMap  ,
				     &m_EBShape      ,
				     false           , // barrel
				     m_apdParameters ,
				     &m_APDShape       ) ;

   m_EEResponse = new CaloHitRespoNew( m_ParameterMap, &m_EEShape ) ;
   m_ESResponse = new CaloHitResponse( m_ParameterMap, m_ESShape )  ;

   // further phase for cosmics studies
   if( cosmicsPhase ) 
   {
      m_EBResponse->setPhaseShift( 1. + cosmicsShift ) ;
      m_EEResponse->setPhaseShift( 1. + cosmicsShift ) ;
   }

   EcalCorrMatrix ebMatrix[ 3 ] ;
   EcalCorrMatrix eeMatrix[ 3 ] ;

   assert( ebCorMatG12.size() == readoutFrameSize ) ;
   assert( eeCorMatG12.size() == readoutFrameSize ) ;
   assert( ebCorMatG06.size() == readoutFrameSize ) ;
   assert( eeCorMatG06.size() == readoutFrameSize ) ;
   assert( ebCorMatG01.size() == readoutFrameSize ) ;
   assert( eeCorMatG01.size() == readoutFrameSize ) ;

   assert( 1.e-7 > fabs( ebCorMatG12[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( ebCorMatG06[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( ebCorMatG01[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( eeCorMatG12[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( eeCorMatG06[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( eeCorMatG01[0] - 1.0 ) ) ;

   for ( unsigned int row ( 0 ) ; row != readoutFrameSize ; ++row )
   {
      assert( 0 == row || 1. >= ebCorMatG12[row] ) ;
      assert( 0 == row || 1. >= ebCorMatG06[row] ) ;
      assert( 0 == row || 1. >= ebCorMatG01[row] ) ;
      assert( 0 == row || 1. >= eeCorMatG12[row] ) ;
      assert( 0 == row || 1. >= eeCorMatG06[row] ) ;
      assert( 0 == row || 1. >= eeCorMatG01[row] ) ;
      for ( unsigned int column ( 0 ) ; column <= row ; ++column )
      {
	 const unsigned int index ( row - column ) ;
	 ebMatrix[0]( row, column ) = ebCorMatG12[ index ] ;
	 eeMatrix[0]( row, column ) = eeCorMatG12[ index ] ;
	 ebMatrix[1]( row, column ) = ebCorMatG06[ index ] ;
	 eeMatrix[1]( row, column ) = eeCorMatG06[ index ] ;
	 ebMatrix[2]( row, column ) = ebCorMatG01[ index ] ;
	 eeMatrix[2]( row, column ) = eeCorMatG01[ index ] ;
      }
   }
			  
   m_EBCorrNoise[0] = new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[0] ) ;
   m_EECorrNoise[0] = new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[0] ) ;
   m_EBCorrNoise[1] = new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[1] ) ;
   m_EECorrNoise[1] = new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[1] ) ;
   m_EBCorrNoise[2] = new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[2] ) ;
   m_EECorrNoise[2] = new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[2] ) ;

   m_Coder = new EcalCoder( addNoise         , 
			    m_EBCorrNoise[0] ,
			    m_EECorrNoise[0] ,
			    m_EBCorrNoise[1] ,
			    m_EECorrNoise[1] ,
			    m_EBCorrNoise[2] ,
			    m_EECorrNoise[2]   ) ;

   m_ElectronicsSim = new EcalElectronicsSim( m_ParameterMap    ,
					      m_Coder           ,
					      applyConstantTerm ,
					      rmsConstantTerm     ) ;

				  
   if( apdSeparateDigi )
   {
      m_APDCoder = new EcalCoder( false            , 
				  m_EBCorrNoise[0] ,
				  m_EECorrNoise[0] ,
				  m_EBCorrNoise[1] ,
				  m_EECorrNoise[1] ,
				  m_EBCorrNoise[2] ,
				  m_EECorrNoise[2]   ) ;

      m_APDElectronicsSim = new EcalElectronicsSim( m_ParameterMap    ,
						    m_APDCoder        ,
						    applyConstantTerm ,
						    rmsConstantTerm     ) ;

      m_APDDigitizer = new EBDigitizer( m_APDResponse       , 
					m_APDElectronicsSim ,
					false                 ) ;
   }

   m_BarrelDigitizer = new EBDigitizer( m_EBResponse     , 
					m_ElectronicsSim ,
					addNoise            ) ;

   m_EndcapDigitizer = new EEDigitizer( m_EEResponse     ,
					m_ElectronicsSim , 
					addNoise            ) ;

   if( !m_doFast ) 
   {
      m_ESElectronicsSim = new ESElectronicsSim( addESNoise ) ;
      
      m_ESDigitizer = new ESDigitizer( m_ESResponse       , 
				       m_ESElectronicsSim ,
				       addESNoise           );
   }
   else
   {
      m_ESElectronicsSimFast = new ESElectronicsSimFast( addESNoise ) ; 

      m_ESDigitizerFast = new ESFastTDigitizer( m_ESResponse           ,
						m_ESElectronicsSimFast ,
						addESNoise             ,
						numESdetId               ) ;
   }
}

EcalDigiProducer::~EcalDigiProducer() 
{
   delete m_APDDigitizer         ;
   delete m_BarrelDigitizer      ;
   delete m_EndcapDigitizer      ;
   delete m_ESDigitizer          ;
   delete m_ESDigitizerFast      ;
   delete m_ParameterMap         ;
   delete m_ESShape              ;
   delete m_APDResponse          ; 
   delete m_EBResponse           ; 
   delete m_EEResponse           ; 
   delete m_ESResponse           ; 
   delete m_EBCorrNoise[0]       ; 
   delete m_EECorrNoise[0]       ; 
   delete m_EBCorrNoise[1]       ; 
   delete m_EECorrNoise[1]       ; 
   delete m_EBCorrNoise[2]       ; 
   delete m_EECorrNoise[2]       ; 
   delete m_ElectronicsSim       ;
   delete m_ESElectronicsSim     ;
   delete m_ESElectronicsSimFast ;
   delete m_Coder                ;
   delete m_APDElectronicsSim    ;
   delete m_APDCoder             ;
   delete m_apdParameters        ;
}

void 
EcalDigiProducer::produce( edm::Event&            event      ,
			   const edm::EventSetup& eventSetup   ) 
{
   // Step A: Get Inputs

   checkGeometry( eventSetup );
   checkCalibrations( eventSetup );

   // Get input
   edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;

   // test access to SimHits
   const std::string barrelHitsName    ( m_hitsProducerTag + "EcalHitsEB" ) ;
   const std::string endcapHitsName    ( m_hitsProducerTag + "EcalHitsEE" ) ;
   const std::string preshowerHitsName ( m_hitsProducerTag + "EcalHitsES" ) ;

   event.getByLabel( "mix",
		     barrelHitsName ,
		     crossingFrame    ) ;

   MixCollection<PCaloHit>* EBHits (
      !crossingFrame.isValid() ? 0 :
      new MixCollection<PCaloHit>( crossingFrame.product() ) ) ;

   const bool isEB ( crossingFrame.isValid() &&
		     0 != EBHits             &&
		     EBHits->inRegistry()       ) ;

   if( !crossingFrame.isValid() )
      edm::LogError("EcalDigiProducer") << "Error! can't get the product " 
					<< barrelHitsName.c_str() ;

   event.getByLabel( "mix",
		     endcapHitsName ,
		     crossingFrame    ) ;

   MixCollection<PCaloHit>* EEHits (
      !crossingFrame.isValid() ? 0 :
      new MixCollection<PCaloHit>( crossingFrame.product() ) ) ;

   const bool isEE ( crossingFrame.isValid() &&
		     0 != EEHits             &&
		     EEHits->inRegistry()       ) ;

   if( !crossingFrame.isValid() ) 
      edm::LogError("EcalDigiProducer") << "Error! can't get the product " 
					<< endcapHitsName.c_str() ;

   event.getByLabel( "mix",
		     preshowerHitsName ,
		     crossingFrame       ) ;

   MixCollection<PCaloHit>* ESHits (
      !crossingFrame.isValid() ? 0 :
      new MixCollection<PCaloHit>( crossingFrame.product() ) ) ;

   const bool isES ( crossingFrame.isValid()   &&
		     0 != ESHits               &&
		     ESHits->inRegistry()          ) ;

   if( !crossingFrame.isValid() ) 
      edm::LogError("EcalDigiProducer") << "Error! can't get the product " 
					<< preshowerHitsName.c_str() ;

   const bool apdSeparateDigi ( 0 != m_APDResponse ) ;

   // Step B: Create empty output
   std::auto_ptr<EBDigiCollection> apdResult      ( !apdSeparateDigi ? 0 :
						    new EBDigiCollection() ) ;
   std::auto_ptr<EBDigiCollection> barrelResult   ( new EBDigiCollection() ) ;
   std::auto_ptr<EEDigiCollection> endcapResult   ( new EEDigiCollection() ) ;
   std::auto_ptr<ESDigiCollection> preshowerResult( new ESDigiCollection() ) ;
   
   // run the algorithm

   if( isEB )
   {
      std::auto_ptr<MixCollection<PCaloHit> >  barrelHits( EBHits ) ;
      m_BarrelDigitizer->run( *barrelHits   , 
			      *barrelResult   ) ;

      edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size() ;

      if( apdSeparateDigi )
      {
	 m_APDDigitizer->run( *barrelHits , 
			      *apdResult    ) ;

	 edm::LogInfo("DigiInfo") << "APD Digis: " << apdResult->size() ;
      }
   }

   if( isEE )
   {
      std::auto_ptr<MixCollection<PCaloHit> >  endcapHits( EEHits ) ;
      m_EndcapDigitizer->run( *endcapHits   ,
			      *endcapResult   ) ;
      edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size() ;
   }

   if( isES ) 
   {
      std::auto_ptr<MixCollection<PCaloHit> >  preshowerHits( ESHits ) ;
      if (!m_doFast) 
      {
	 m_ESDigitizer->run( *preshowerHits   , 
			     *preshowerResult   ) ; 
      }
      else
      {
	 m_ESDigitizerFast->run(*preshowerHits, *preshowerResult); 
      }
      edm::LogInfo("EcalDigi") << "ES Digis: " << preshowerResult->size();
   }

   const std::string& apdDigiTag ( m_apdParameters->digiTag() ) ;

   // Step D: Put outputs into event
   if( apdSeparateDigi )
      event.put( apdResult,    apdDigiTag         ) ;
   event.put( barrelResult,    m_EBdigiCollection ) ;
   event.put( endcapResult,    m_EEdigiCollection ) ;
   event.put( preshowerResult, m_ESdigiCollection ) ;

}

void  
EcalDigiProducer::checkCalibrations( const edm::EventSetup& eventSetup ) 
{
   // Pedestals from event setup

   edm::ESHandle<EcalPedestals>            dbPed   ;
   eventSetup.get<EcalPedestalsRcd>().get( dbPed ) ;
   const EcalPedestals* pedestals        ( dbPed.product() ) ;
  
   m_Coder->setPedestals( pedestals ) ;
   if( 0 != m_APDCoder ) m_APDCoder->setPedestals( pedestals ) ;

   // Ecal Intercalibration Constants
   edm::ESHandle<EcalIntercalibConstantsMC>            pIcal   ;
   eventSetup.get<EcalIntercalibConstantsMCRcd>().get( pIcal ) ;
   const EcalIntercalibConstantsMC* ical             ( pIcal.product() ) ;
  
   m_Coder->setIntercalibConstants( ical ) ;
   if( 0 != m_APDCoder) m_APDCoder->setIntercalibConstants( ical ) ;

   m_EBResponse->setIntercal( ical ) ;
   if( 0 != m_APDResponse ) m_APDResponse->setIntercal( ical ) ;

   // ADC -> GeV Scale
   edm::ESHandle<EcalADCToGeVConstant> pAgc;
   eventSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
   const EcalADCToGeVConstant* agc = pAgc.product();
  
   // Gain Ratios
   edm::ESHandle<EcalGainRatios> pRatio;
   eventSetup.get<EcalGainRatiosRcd>().get(pRatio);
   const EcalGainRatios* gr = pRatio.product();

   m_Coder->setGainRatios( gr );
   if( 0 != m_APDCoder) m_APDCoder->setGainRatios( gr );

   EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();

   double theGains[m_Coder->NGAINS+1];
   theGains[0] = 0.;
   theGains[3] = 1.;
   theGains[2] = defaultRatios->gain6Over1() ;
   theGains[1] = theGains[2]*(defaultRatios->gain12Over6()) ;

   LogDebug("EcalDigi") << " Gains: " << "\n" << " g1 = " << theGains[1] 
			<< "\n" << " g2 = " << theGains[2] 
			<< "\n" << " g3 = " << theGains[3] ;

   delete defaultRatios;

   const double EBscale (
      ( agc->getEBValue())*theGains[1]*(m_Coder->MAXADC)*m_EBs25notCont ) ;

   LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() 
			<< "\n" << " notCont = " << m_EBs25notCont 
			<< "\n" << " saturation for EB = " << EBscale 
			<< ", " << m_EBs25notCont ;

   const double EEscale (
      (agc->getEEValue())*theGains[1]*(m_Coder->MAXADC)*m_EEs25notCont ) ;

   LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEEValue() 
			<< "\n" << " notCont = " << m_EEs25notCont 
			<< "\n" << " saturation for EB = " << EEscale 
			<< ", " << m_EEs25notCont ;

   m_Coder->setFullScaleEnergy( EBscale , 
				EEscale   ) ;
   if( 0 != m_APDCoder ) m_APDCoder->setFullScaleEnergy( EBscale ,
							 EEscale   ) ;


   // ES condition objects
   edm::ESHandle<ESGain>                hesgain      ;
   edm::ESHandle<ESMIPToGeVConstant>    hesMIPToGeV  ;
   edm::ESHandle<ESPedestals>           hesPedestals ;
   edm::ESHandle<ESIntercalibConstants> hesMIPs      ;

   eventSetup.get<ESGainRcd>().               get( hesgain      ) ;
   eventSetup.get<ESMIPToGeVConstantRcd>().   get( hesMIPToGeV  ) ;
   eventSetup.get<ESPedestalsRcd>().          get( hesPedestals ) ;
   eventSetup.get<ESIntercalibConstantsRcd>().get( hesMIPs      ) ;

   const ESGain*                esgain     ( hesgain.product()      ) ;
   const ESPedestals*           espeds     ( hesPedestals.product() ) ;
   const ESIntercalibConstants* esmips     ( hesMIPs.product()      ) ;
   const ESMIPToGeVConstant*    esMipToGeV ( hesMIPToGeV.product()  ) ;
   const int ESGain ( 1.1 > esgain->getESGain() ? 1 : 2 ) ;  
   const double ESMIPToGeV ( ( 1 == ESGain ) ?
			     esMipToGeV->getESValueLow()  :
			     esMipToGeV->getESValueHigh()   ) ; 
   
   m_ESShape->setGain( ESGain );

   if( !m_doFast )
   {
      m_ESElectronicsSim->setGain(      ESGain     ) ;
      m_ESElectronicsSim->setPedestals( espeds     ) ;
      m_ESElectronicsSim->setMIPs(      esmips     ) ;
      m_ESElectronicsSim->setMIPToGeV(  ESMIPToGeV ) ;
   }
   else
   {
      m_ESDigitizerFast->setGain(           ESGain     ) ;
      m_ESElectronicsSimFast->setGain(      ESGain     ) ;
      m_ESElectronicsSimFast->setPedestals( espeds     ) ;
      m_ESElectronicsSimFast->setMIPs(      esmips     ) ;
      m_ESElectronicsSimFast->setMIPToGeV(  ESMIPToGeV ) ;
   }
}

void 
EcalDigiProducer::checkGeometry( const edm::EventSetup & eventSetup ) 
{
   // TODO find a way to avoid doing this every event
   edm::ESHandle<CaloGeometry>               hGeometry   ;
   eventSetup.get<CaloGeometryRecord>().get( hGeometry ) ;

   const CaloGeometry* pGeometry = &*hGeometry;

   if( pGeometry != m_Geometry )
   {
      m_Geometry = pGeometry;
      updateGeometry();
   }
}

void
EcalDigiProducer::updateGeometry() 
{
   if( 0 != m_APDResponse ) m_APDResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
   m_EBResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
   m_EEResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalEndcap    ) ) ;
   m_ESResponse->setGeometry( m_Geometry ) ;
//      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalPreshower ) ) ;


   const std::vector<DetId>& theESDets ( 
      m_Geometry->getSubdetectorGeometry(
	 DetId::Ecal, EcalPreshower)->getValidDetIds() ) ;

   if( !m_doFast ) 
   {
      m_ESDigitizer->setDetIds( theESDets ) ;
   }
   else
   {
      m_ESDigitizerFast->setDetIds( theESDets ) ; 
   }
}
