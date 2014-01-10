#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalPhaseIIDigiProducer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"

EcalPhaseIIDigiProducer::EcalPhaseIIDigiProducer( const edm::ParameterSet& params, edm::EDProducer& mixMod ) :
   DigiAccumulatorMixMod(),
   m_APDShape         ( params.getParameter<double>( "apdShapeTstart" ) ,
			params.getParameter<double>( "apdShapeTau"    )   )  ,
   m_EBShape          (   ) ,
   m_EEShape          (   ) ,
   m_EBdigiCollection ( params.getParameter<std::string>("EBdigiCollection") ) ,
   m_EEdigiCollection ( params.getParameter<std::string>("EEdigiCollection") ) ,
   m_ESdigiCollection ( params.getParameter<std::string>("ESdigiCollection") ) ,
   m_hitsProducerTag  ( params.getParameter<std::string>("hitsProducer"    ) ) ,
   m_useLCcorrection  ( params.getUntrackedParameter<bool>("UseLCcorrection") ) ,
   m_apdSeparateDigi  ( params.getParameter<bool>       ("apdSeparateDigi") ) ,

   m_EBs25notCont     ( params.getParameter<double>     ("EBs25notContainment") ) ,
   m_EEs25notCont     ( params.getParameter<double>     ("EEs25notContainment") ) ,

   m_readoutFrameSize ( params.getParameter<int>       ("readoutFrameSize") ) ,
   m_ParameterMap     ( new EcalSimParameterMap(
			   params.getParameter<double> ("simHitToPhotoelectronsBarrel") ,
			   params.getParameter<double> ("simHitToPhotoelectronsEndcap") , 
			   params.getParameter<double> ("photoelectronsToAnalogBarrel") ,
			   params.getParameter<double> ("photoelectronsToAnalogEndcap") , 
			   params.getParameter<double> ("samplingFactor") ,
			   params.getParameter<double> ("timePhase") ,
			   m_readoutFrameSize ,
			   params.getParameter<int>    ("binOfMaximum") , 
			   params.getParameter<bool>   ("doPhotostatistics") ,
			   params.getParameter<bool>   ("syncPhase") ) ) ,
   
   m_apdDigiTag    ( params.getParameter<std::string> ("apdDigiTag"  )      ) ,
   m_apdParameters ( new APDSimParameters( 
			params.getParameter<bool>        ("apdAddToBarrel"  ) ,
			m_apdSeparateDigi ,
			params.getParameter<double>      ("apdSimToPELow"   ) ,
			params.getParameter<double>      ("apdSimToPEHigh"  ) ,
			params.getParameter<double>      ("apdTimeOffset"   ) ,
			params.getParameter<double>      ("apdTimeOffWidth" ) ,
			params.getParameter<bool>        ("apdDoPEStats"    ) ,
			m_apdDigiTag ,
			params.getParameter<std::vector<double> > ( "apdNonlParms" ) ) ) ,

   m_APDResponse ( !m_apdSeparateDigi ? 0 :
		   new EBHitResponse( m_ParameterMap  ,
				      &m_EBShape      ,
				      true            ,
				      m_apdParameters ,
				      &m_APDShape       ) ) ,
   
   m_EBResponse ( new EBHitResponse( m_ParameterMap  ,
				     &m_EBShape      ,
				     false           , // barrel
				     m_apdParameters ,
				     &m_APDShape       ) ) ,

   m_EEResponse ( new EEHitResponse( m_ParameterMap,
				     &m_EEShape       ) ) ,
   m_addESNoise           ( params.getParameter<bool> ("doESNoise") ) ,
   m_doFastES             ( params.getParameter<bool> ("doFast"   ) ) ,
   m_APDDigitizer      ( 0 ) ,
   m_BarrelDigitizer   ( 0 ) ,
   m_EndcapDigitizer   ( 0 ) ,
   m_ElectronicsSim    ( 0 ) ,
   m_Coder             ( 0 ) ,
   m_APDElectronicsSim ( 0 ) ,
   m_APDCoder          ( 0 ) ,
   m_Geometry          ( 0 ) ,
   m_EBCorrNoise       (   ) ,
   m_EECorrNoise       (   ) 
{
   if(m_apdSeparateDigi) mixMod.produces<EBDigiCollection>(m_apdDigiTag);
   mixMod.produces<EBDigiCollection>(m_EBdigiCollection);
   mixMod.produces<EEDigiCollection>(m_EEdigiCollection);
   mixMod.produces<ESDigiCollection>(m_ESdigiCollection);

   const std::vector<double> ebCorMatG12 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG12");
   const std::vector<double> eeCorMatG12 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG12");
   const std::vector<double> ebCorMatG06 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG06");
   const std::vector<double> eeCorMatG06 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG06");
   const std::vector<double> ebCorMatG01 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG01");
   const std::vector<double> eeCorMatG01 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG01");

   const bool applyConstantTerm          = params.getParameter<bool>       ("applyConstantTerm");
   const double rmsConstantTerm          = params.getParameter<double>     ("ConstantTerm");

   const bool addNoise                   = params.getParameter<bool>       ("doNoise"); 
   const bool cosmicsPhase               = params.getParameter<bool>       ("cosmicsPhase");
   const double cosmicsShift             = params.getParameter<double>     ("cosmicsShift");

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   // further phase for cosmics studies
   if( cosmicsPhase ) 
   {
      m_EBResponse->setPhaseShift( 1. + cosmicsShift ) ;
      m_EEResponse->setPhaseShift( 1. + cosmicsShift ) ;
   }

   EcalCorrMatrix ebMatrix[ 3 ] ;
   EcalCorrMatrix eeMatrix[ 3 ] ;

   assert( ebCorMatG12.size() == m_readoutFrameSize ) ;
   assert( eeCorMatG12.size() == m_readoutFrameSize ) ;
   assert( ebCorMatG06.size() == m_readoutFrameSize ) ;
   assert( eeCorMatG06.size() == m_readoutFrameSize ) ;
   assert( ebCorMatG01.size() == m_readoutFrameSize ) ;
   assert( eeCorMatG01.size() == m_readoutFrameSize ) ;

   assert( 1.e-7 > fabs( ebCorMatG12[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( ebCorMatG06[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( ebCorMatG01[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( eeCorMatG12[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( eeCorMatG06[0] - 1.0 ) ) ;
   assert( 1.e-7 > fabs( eeCorMatG01[0] - 1.0 ) ) ;

   for ( unsigned int row ( 0 ) ; row != m_readoutFrameSize ; ++row )
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
				  
   if( m_apdSeparateDigi )
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
}

EcalPhaseIIDigiProducer::~EcalPhaseIIDigiProducer() 
{
   delete m_EndcapDigitizer      ;
   delete m_BarrelDigitizer      ;
   delete m_APDDigitizer         ;
   delete m_APDElectronicsSim    ;
   delete m_APDCoder             ;
   delete m_ElectronicsSim       ;
   delete m_Coder                ;
   delete m_EBCorrNoise[0]       ; 
   delete m_EECorrNoise[0]       ; 
   delete m_EBCorrNoise[1]       ; 
   delete m_EECorrNoise[1]       ; 
   delete m_EBCorrNoise[2]       ; 
   delete m_EECorrNoise[2]       ; 

   delete m_EEResponse           ; 
   delete m_EBResponse           ; 
   delete m_APDResponse          ; 

   delete m_apdParameters        ;
   delete m_ParameterMap         ;
}

void
EcalPhaseIIDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& eventSetup) {
   checkGeometry( eventSetup );
   checkCalibrations( event, eventSetup );
   m_BarrelDigitizer->initializeHits();
   if(m_apdSeparateDigi) {
      m_APDDigitizer->initializeHits();
   }
   m_EndcapDigitizer->initializeHits();
}

void
EcalPhaseIIDigiProducer::accumulateCaloHits(HitsHandle const& ebHandle, HitsHandle const& eeHandle, HitsHandle const& esHandle, int bunchCrossing) {
  if(ebHandle.isValid()) {
    m_BarrelDigitizer->add(*ebHandle.product(), bunchCrossing);

    if(m_apdSeparateDigi) {
      m_APDDigitizer->add(*ebHandle.product(), bunchCrossing);
    }
  }

  if(eeHandle.isValid()) {
    m_EndcapDigitizer->add(*eeHandle.product(), bunchCrossing);
  }
}

void
EcalPhaseIIDigiProducer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup) {
  // Step A: Get Inputs
  edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  e.getByLabel(ebTag, ebHandle);

  edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
  edm::Handle<std::vector<PCaloHit> > eeHandle;
  e.getByLabel(eeTag, eeHandle);

  edm::InputTag esTag(m_hitsProducerTag, "EcalHitsES");
  edm::Handle<std::vector<PCaloHit> > esHandle;
  e.getByLabel(esTag, esHandle);

  accumulateCaloHits(ebHandle, eeHandle, esHandle, 0);
}

void
EcalPhaseIIDigiProducer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup) {
  // Step A: Get Inputs
  edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  e.getByLabel(ebTag, ebHandle);

  edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
  edm::Handle<std::vector<PCaloHit> > eeHandle;
  e.getByLabel(eeTag, eeHandle);

  edm::InputTag esTag(m_hitsProducerTag, "EcalHitsES");
  edm::Handle<std::vector<PCaloHit> > esHandle;
  e.getByLabel(esTag, esHandle);

  accumulateCaloHits(ebHandle, eeHandle, esHandle, e.bunchCrossing());
}

void 
EcalPhaseIIDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& eventSetup) {
   // Step B: Create empty output
   std::auto_ptr<EBDigiCollection> apdResult      ( !m_apdSeparateDigi ? 0 :
						    new EBDigiCollection() ) ;
   std::auto_ptr<EBDigiCollection> barrelResult   ( new EBDigiCollection() ) ;
   std::auto_ptr<EEDigiCollection> endcapResult   ( new EEDigiCollection() ) ;
   std::auto_ptr<ESDigiCollection> preshowerResult( new ESDigiCollection() ) ;
   
   // run the algorithm

   m_BarrelDigitizer->run( *barrelResult ) ;
   cacheEBDigis( &*barrelResult ) ;

   edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size() ;

   if( m_apdSeparateDigi ) {
      m_APDDigitizer->run( *apdResult ) ;
      edm::LogInfo("DigiInfo") << "APD Digis: " << apdResult->size() ;
   }

   m_EndcapDigitizer->run( *endcapResult ) ;
   edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size() ;
   cacheEEDigis( &*endcapResult ) ;

   // Step D: Put outputs into event
   if( m_apdSeparateDigi ) {
      event.put( apdResult,    m_apdDigiTag         ) ;
   }

   event.put( barrelResult,    m_EBdigiCollection ) ;
   event.put( endcapResult,    m_EEdigiCollection ) ;
   event.put( preshowerResult, m_ESdigiCollection ) ;
}

void  
EcalPhaseIIDigiProducer::checkCalibrations(const edm::Event& event, const edm::EventSetup& eventSetup ) 
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

   // Ecal LaserCorrection Constants                                
   edm::ESHandle<EcalLaserDbService> laser;
   eventSetup.get<EcalLaserDbRecord>().get(laser);
   const edm::TimeValue_t eventTimeValue = event.time().value();

   m_EBResponse->setEventTime(eventTimeValue);
   m_EBResponse->setLaserConstants(laser.product(), m_useLCcorrection);

   m_EEResponse->setEventTime(eventTimeValue);
   m_EEResponse->setLaserConstants(laser.product(), m_useLCcorrection);

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
}

void 
EcalPhaseIIDigiProducer::checkGeometry( const edm::EventSetup & eventSetup ) 
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
EcalPhaseIIDigiProducer::updateGeometry() 
{
   if( 0 != m_APDResponse ) m_APDResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
   m_EBResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
   m_EEResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalEndcap    ) ) ;
}
