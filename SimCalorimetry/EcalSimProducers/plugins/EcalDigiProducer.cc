
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
   m_ESShape         ( 0 ) ,
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
   const int ESGain                          = params.getParameter<int>        ("ESGain");
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
   const double ESNoiseSigma                 = params.getParameter<double>     ("ESNoiseSigma");
   const int ESBaseline                      = params.getParameter<int>        ("ESBaseline");
   const double ESMIPADC                     = params.getParameter<double>     ("ESMIPADC");
   const double ESMIPkeV                     = params.getParameter<double>     ("ESMIPkeV");
   const int numESdetId                      = params.getParameter<int>        ("numESdetId");
   const double zsThreshold                  = params.getParameter<double>     ("zsThreshold");
   const std::string refFile                 = params.getParameter<std::string>("refHistosFile");
   m_doFast                                  = params.getParameter<bool>       ("doFast");
   m_EBs25notCont                            = params.getParameter<double>     ("EBs25notContainment");
   m_EEs25notCont                            = params.getParameter<double>     ("EEs25notContainment");
   m_hitsProducerTag                         = params.getParameter<std::string>("hitsProducer");

   const bool   apdAddToBarrel  ( params.getParameter<bool>   ("apdAddToBarrel") ) ;
   const bool   apdSeparateDigi ( params.getParameter<bool>   ("apdSeparateDigi") ) ;
   const double apdSimToPELow   ( params.getParameter<double> ("apdSimToPELow" ) ) ;
   const double apdSimToPEHigh  ( params.getParameter<double> ("apdSimToPEHigh") ) ;
   const double apdTimeOffset   ( params.getParameter<double> ("apdTimeOffset" ) ) ;
   const bool   apdDoPEStats    ( params.getParameter<bool>   ("apdDoPEStats"  ) ) ;
   const std::string apdDigiTag ( params.getParameter<std::string>("apdDigiTag") ) ;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   if( apdSeparateDigi ) produces<EBDigiCollection>( apdDigiTag ) ;
   produces<EBDigiCollection>( m_EBdigiCollection ) ;
   produces<EEDigiCollection>( m_EEdigiCollection ) ;
   produces<ESDigiCollection>( m_ESdigiCollection ) ;

   // initialize the default valuer for hardcoded parameters and the EB/EE shape

   m_ParameterMap = new EcalSimParameterMap( simHitToPhotoelectronsBarrel,
					     simHitToPhotoelectronsEndcap, 
					     photoelectronsToAnalogBarrel,
					     photoelectronsToAnalogEndcap, 
					     samplingFactor,
					     timePhase,
					     readoutFrameSize, 
					     binOfMaximum,
					     doPhotostatistics,
					     syncPhase);

   m_apdParameters = new APDSimParameters( apdAddToBarrel  ,
					   apdSeparateDigi ,
					   apdSimToPELow   ,
					   apdSimToPEHigh  ,
					   apdTimeOffset   ,
					   apdDoPEStats    ,
					   apdDigiTag        ) ;
  
   m_ESShape   = new ESShape(ESGain);
   
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

   m_EEResponse = new CaloHitResponse( m_ParameterMap, &m_EEShape );
   m_ESResponse = new CaloHitResponse( m_ParameterMap, m_ESShape );

   // further phase for cosmics studies
   if( cosmicsPhase ) 
   {
      m_EBResponse->setPhaseShift( 1. + cosmicsShift ) ;
      m_EEResponse->setPhaseShift( 1. + cosmicsShift ) ;
   }


   EcalCorrMatrix ebMatrix[3] ;
   EcalCorrMatrix eeMatrix[3] ;

   const unsigned int size2 ( readoutFrameSize*readoutFrameSize ) ;
   assert( ebCorMatG12.size() == size2 ) ;
   assert( eeCorMatG12.size() == size2 ) ;
   assert( ebCorMatG06.size() == size2 ) ;
   assert( eeCorMatG06.size() == size2 ) ;
   assert( ebCorMatG01.size() == size2 ) ;
   assert( eeCorMatG01.size() == size2 ) ;

   for ( unsigned int row ( 0 ) ; row != readoutFrameSize; ++row )
   {
      for ( unsigned int column ( 0 ) ; column != readoutFrameSize; ++column )
      {
	 const unsigned int index = column + readoutFrameSize*row;
	 ebMatrix[0]( row, column ) = ebCorMatG12[index];
	 eeMatrix[0]( row, column ) = eeCorMatG12[index];
	 ebMatrix[1]( row, column ) = ebCorMatG06[index];
	 eeMatrix[1]( row, column ) = eeCorMatG06[index];
	 ebMatrix[2]( row, column ) = ebCorMatG01[index];
	 eeMatrix[2]( row, column ) = eeCorMatG01[index];
      }
   }
			  
   m_EBCorrNoise[0] = new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[0] ) ;
   m_EECorrNoise[0] = new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[0] ) ;
   m_EBCorrNoise[1] = new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[1] ) ;
   m_EECorrNoise[1] = new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[1] ) ;
   m_EBCorrNoise[2] = new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[2] ) ;
   m_EECorrNoise[2] = new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[2] ) ;

   m_Coder = new EcalCoder( addNoise, 
			    m_EBCorrNoise[0] ,
			    m_EECorrNoise[0] ,
			    m_EBCorrNoise[1] ,
			    m_EECorrNoise[1] ,
			    m_EBCorrNoise[2] ,
			    m_EECorrNoise[2]   ) ;

   m_ElectronicsSim = new EcalElectronicsSim( m_ParameterMap,
					      m_Coder,
					      applyConstantTerm,
					      rmsConstantTerm   );

				  
   if( apdSeparateDigi )
   {
      m_APDCoder = new EcalCoder( false, 
				  m_EBCorrNoise[0] ,
				  m_EECorrNoise[0] ,
				  m_EBCorrNoise[1] ,
				  m_EECorrNoise[1] ,
				  m_EBCorrNoise[2] ,
				  m_EECorrNoise[2]   ) ;

      m_APDElectronicsSim = new EcalElectronicsSim( m_ParameterMap,
						    m_APDCoder,
						    applyConstantTerm,
						    rmsConstantTerm   );

      m_APDDigitizer = new EBDigitizer( m_APDResponse       , 
					m_APDElectronicsSim ,
					false                 );
   }

   m_BarrelDigitizer = new EBDigitizer( m_EBResponse     , 
					m_ElectronicsSim ,
					addNoise            );

   m_EndcapDigitizer = new EEDigitizer( m_EEResponse     ,
					m_ElectronicsSim , 
					addNoise            );


   if (!m_doFast) 
   {
      m_ESElectronicsSim     =
	 new ESElectronicsSim( addESNoise,
			       ESNoiseSigma,
			       ESGain, 
			       ESBaseline,
			       ESMIPADC,
			       ESMIPkeV);
      
      m_ESDigitizer = new ESDigitizer( m_ESResponse, 
				       m_ESElectronicsSim,
				       addESNoise           );
   }
   else
   {
      m_ESElectronicsSimFast = 
	 new ESElectronicsSimFast( addESNoise,
				   ESNoiseSigma,
				   ESGain, 
				   ESBaseline,
				   ESMIPADC,
				   ESMIPkeV); 

      m_ESDigitizerFast = new ESFastTDigitizer( m_ESResponse,
						m_ESElectronicsSimFast,
						addESNoise,
						numESdetId, 
						zsThreshold, 
						refFile);
   }
}


EcalDigiProducer::~EcalDigiProducer() 
{
   delete m_APDDigitizer;
   delete m_BarrelDigitizer;
   delete m_EndcapDigitizer;
   delete m_ESDigitizer;
   delete m_ESDigitizerFast;
   delete m_ParameterMap;
   delete m_ESShape;
   delete m_APDResponse; 
   delete m_EBResponse; 
   delete m_EEResponse; 
   delete m_ESResponse; 
   delete m_EBCorrNoise[0]; 
   delete m_EECorrNoise[0]; 
   delete m_EBCorrNoise[1]; 
   delete m_EECorrNoise[1]; 
   delete m_EBCorrNoise[2]; 
   delete m_EECorrNoise[2]; 
   delete m_ElectronicsSim;
   delete m_ESElectronicsSim;
   delete m_ESElectronicsSimFast;
   delete m_Coder;
   delete m_APDElectronicsSim;
   delete m_APDCoder;
   delete m_apdParameters;
}

void 
EcalDigiProducer::produce( edm::Event&            event,
			   const edm::EventSetup& eventSetup ) 
{
   // Step A: Get Inputs

   checkGeometry( eventSetup );
   checkCalibrations( eventSetup );

   // Get input
   edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;

   const std::vector<DetId>& barrelDets    =  m_Geometry->getValidDetIds(DetId::Ecal, EcalBarrel    ) ;
   const std::vector<DetId>& endcapDets    =  m_Geometry->getValidDetIds(DetId::Ecal, EcalEndcap    ) ;
   const std::vector<DetId>& preshowerDets =  m_Geometry->getValidDetIds(DetId::Ecal, EcalPreshower ) ;

   // test access to SimHits
   const std::string barrelHitsName    ( m_hitsProducerTag + "EcalHitsEB" ) ;
   const std::string endcapHitsName    ( m_hitsProducerTag + "EcalHitsEE" ) ;
   const std::string preshowerHitsName ( m_hitsProducerTag + "EcalHitsES" ) ;

   event.getByLabel( "mix",
		     barrelHitsName ,
		     crossingFrame    );

   MixCollection<PCaloHit>* EBHits ( !crossingFrame.isValid() ? 0 :
				     new MixCollection<PCaloHit>( crossingFrame.product() ) ) ;

   const bool isEB ( 0 != barrelDets.size()  &&
		     crossingFrame.isValid() &&
		     0 != EBHits             &&
		     EBHits->inRegistry()       ) ;

   if( !crossingFrame.isValid() ) edm::LogError("EcalDigiProducer") << "Error! can't get the product " 
								    << barrelHitsName.c_str() ;



   event.getByLabel( "mix",
		     endcapHitsName ,
		     crossingFrame    );

   MixCollection<PCaloHit>* EEHits ( !crossingFrame.isValid() ? 0 :
				     new MixCollection<PCaloHit>( crossingFrame.product() ) ) ;

   const bool isEE ( 0 != endcapDets.size()  &&
		     crossingFrame.isValid() &&
		     0 != EEHits             &&
		     EEHits->inRegistry()       ) ;

   if( !crossingFrame.isValid() ) edm::LogError("EcalDigiProducer") << "Error! can't get the product " 
								    << endcapHitsName.c_str() ;



   event.getByLabel( "mix",
		     preshowerHitsName ,
		     crossingFrame    );

   MixCollection<PCaloHit>* ESHits ( !crossingFrame.isValid() ? 0 :
				     new MixCollection<PCaloHit>( crossingFrame.product() ) ) ;

   const bool isES ( 0 != preshowerDets.size() &&
		     crossingFrame.isValid()   &&
		     0 != ESHits               &&
		     ESHits->inRegistry()          ) ;

   if( !crossingFrame.isValid() ) edm::LogError("EcalDigiProducer") << "Error! can't get the product " 
								    << preshowerHitsName.c_str() ;

   const bool apdSeparateDigi ( 0 != m_APDResponse ) ;


   // Step B: Create empty output
   std::auto_ptr<EBDigiCollection> apdResult   ( !apdSeparateDigi ? 0 :
						 new EBDigiCollection() ) ;
   std::auto_ptr<EBDigiCollection> barrelResult(    new EBDigiCollection() ) ;
   std::auto_ptr<EEDigiCollection> endcapResult(    new EEDigiCollection() ) ;
   std::auto_ptr<ESDigiCollection> preshowerResult( new ESDigiCollection() ) ;
   
   // run the algorithm

   if( isEB )
   {
      std::auto_ptr<MixCollection<PCaloHit> >  barrelHits( EBHits );
      m_BarrelDigitizer->run( *barrelHits, 
			      *barrelResult ) ;

      edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();

      if( apdSeparateDigi )
      {
	 m_APDDigitizer->run( *barrelHits , 
			      *apdResult    ) ;

	 edm::LogInfo("DigiInfo") << "APD Digis: " << apdResult->size();
      }
   }

   if( isEE )
   {
      std::auto_ptr<MixCollection<PCaloHit> >  endcapHits( EEHits );
      m_EndcapDigitizer->run(*endcapHits, *endcapResult);
      edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size();
   }

   if( isES ) 
   {
      std::auto_ptr<MixCollection<PCaloHit> >  preshowerHits( ESHits );
      if (!m_doFast) 
      {
	 m_ESDigitizer->run(*preshowerHits, *preshowerResult); 
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

   edm::ESHandle<EcalPedestals> dbPed;
   eventSetup.get<EcalPedestalsRcd>().get( dbPed );
   const EcalPedestals* thePedestals=dbPed.product();
  
   m_Coder->setPedestals( thePedestals ) ;
   if( 0 != m_APDCoder ) m_APDCoder->setPedestals( thePedestals ) ;

   // Ecal Intercalibration Constants
   edm::ESHandle<EcalIntercalibConstantsMC> pIcal;
   eventSetup.get<EcalIntercalibConstantsMCRcd>().get(pIcal);
   const EcalIntercalibConstantsMC *ical = pIcal.product();
  
   m_Coder->setIntercalibConstants( ical );
   if( 0 != m_APDCoder) m_APDCoder->setIntercalibConstants( ical );

   m_EBResponse->setIntercal( ical );
   if( 0 != m_APDResponse ) m_APDResponse->setIntercal( ical );

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

   const double EBscale = (agc->getEBValue())*theGains[1]*(m_Coder->MAXADC)*m_EBs25notCont;

   LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() 
			<< "\n" << " notCont = " << m_EBs25notCont 
			<< "\n" << " saturation for EB = " << EBscale 
			<< ", " << m_EBs25notCont ;

   const double EEscale = (agc->getEEValue())*theGains[1]*(m_Coder->MAXADC)*m_EEs25notCont;

   LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEEValue() 
			<< "\n" << " notCont = " << m_EEs25notCont 
			<< "\n" << " saturation for EB = " << EEscale 
			<< ", " << m_EEs25notCont ;

   m_Coder->setFullScaleEnergy( EBscale , EEscale ) ;
   if( 0 != m_APDCoder ) m_APDCoder->setFullScaleEnergy( EBscale , EEscale ) ;
}


void 
EcalDigiProducer::checkGeometry( const edm::EventSetup & eventSetup ) 
{
   // TODO find a way to avoid doing this every event
   edm::ESHandle<CaloGeometry>               hGeometry   ;
   eventSetup.get<CaloGeometryRecord>().get( hGeometry ) ;

   const CaloGeometry* pGeometry = &*hGeometry;
  
   // see if we need to update
   if( pGeometry != m_Geometry )
   {
      m_Geometry = pGeometry;
      updateGeometry();
   }
}


void
EcalDigiProducer::updateGeometry() 
{
   m_APDResponse->setGeometry( m_Geometry ) ;
   m_EBResponse->setGeometry( m_Geometry ) ;
   m_EEResponse->setGeometry( m_Geometry ) ;
   m_ESResponse->setGeometry( m_Geometry ) ;
   
   const std::vector<DetId>& theBarrelDets 
      ( m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel   )->getValidDetIds() ) ;
   const std::vector<DetId>& theEndcapDets 
      ( m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalEndcap   )->getValidDetIds() ) ;
   const std::vector<DetId>& theESDets     
      ( m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalPreshower)->getValidDetIds() ) ;

   edm::LogInfo("EcalDigi") << "deb geometry: " << "\n" 
			    << "\t barrel: " << theBarrelDets.size () << "\n"
			    << "\t endcap: " << theEndcapDets.size () << "\n"
			    << "\t preshower: " << theESDets.size();
  
   if( 0 != m_APDDigitizer ) m_APDDigitizer->setDetIds( theBarrelDets ) ;
   m_BarrelDigitizer->setDetIds( theBarrelDets ) ;
   m_EndcapDigitizer->setDetIds( theEndcapDets ) ;
   if( !m_doFast ) 
   {
      m_ESDigitizer->setDetIds( theESDets ) ;
   }
   else
   {
      m_ESDigitizerFast->setDetIds( theESDets ) ; 
   }
}

