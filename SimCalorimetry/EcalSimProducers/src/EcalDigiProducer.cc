#include "FWCore/Framework/interface/Event.h"
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESHitResponse.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/APDSimParameters.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
//#include "SimCalorimetry/EcalSimAlgos/interface/ESFastTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESDigitizer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"
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

#include "CondFormats/ESObjects/interface/ESIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/ESObjects/interface/ESMIPToGeVConstant.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/ESObjects/interface/ESGain.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "CondFormats/ESObjects/interface/ESPedestals.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"


EcalDigiProducer::EcalDigiProducer( const edm::ParameterSet& params, edm::ProducerBase& mixMod, edm::ConsumesCollector& iC) :
  EcalDigiProducer(params, iC)
{
   if(m_apdSeparateDigi) mixMod.produces<EBDigiCollection>(m_apdDigiTag);
     
   mixMod.produces<EBDigiCollection>(m_EBdigiCollection);
   mixMod.produces<EEDigiCollection>(m_EEdigiCollection);
   mixMod.produces<ESDigiCollection>(m_ESdigiCollection);
}

// version for Pre-Mixing, for use outside of MixingModule
EcalDigiProducer::EcalDigiProducer( const edm::ParameterSet& params,  edm::ConsumesCollector& iC) :
   DigiAccumulatorMixMod(),
   m_APDShape         ( true ) ,
   m_EBShape          ( true ) ,
   m_EEShape          ( true ) ,
   m_ESShape          (   ) ,
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

   m_APDResponse ( !m_apdSeparateDigi ? nullptr :
		   new EBHitResponse( m_ParameterMap.get()  ,
				      &m_EBShape      ,
				      true            ,
				      m_apdParameters.get() ,
				      &m_APDShape       ) ) ,
   
   m_EBResponse ( new EBHitResponse( m_ParameterMap.get()  ,
				     &m_EBShape      ,
				     false           , // barrel
				     m_apdParameters.get() ,
				     &m_APDShape       ) ) ,

   m_EEResponse ( new EEHitResponse( m_ParameterMap.get(),
				     &m_EEShape       ) ) ,
   m_ESResponse ( new ESHitResponse( m_ParameterMap.get(), &m_ESShape ) ) ,
   m_ESOldResponse ( new CaloHitResponse( m_ParameterMap.get(), &m_ESShape ) ) ,

   m_addESNoise           ( params.getParameter<bool> ("doESNoise") ) ,
   m_PreMix1              ( params.getParameter<bool> ("EcalPreMixStage1") ) ,
   m_PreMix2              ( params.getParameter<bool> ("EcalPreMixStage2") ) ,

   m_doFastES             ( params.getParameter<bool> ("doFast"   ) ) ,

   m_doEB                 ( params.getParameter<bool> ("doEB"     ) ) ,
   m_doEE                 ( params.getParameter<bool> ("doEE"     ) ) ,
   m_doES                 ( params.getParameter<bool> ("doES"     ) ) ,

   m_ESElectronicsSim     ( m_doFastES ? nullptr :
			    new ESElectronicsSim( m_addESNoise ) ) ,
	 
   m_ESOldDigitizer       ( m_doFastES ? nullptr :
			    new ESOldDigitizer( m_ESOldResponse.get()    , 
						m_ESElectronicsSim.get() ,
						m_addESNoise         ) ) ,
   
   m_ESElectronicsSimFast ( !m_doFastES ? nullptr :
			    new ESElectronicsSimFast( m_addESNoise, 
						      m_PreMix1      ) ) ,

   m_ESDigitizer          ( !m_doFastES ? nullptr :
			    new ESDigitizer( m_ESResponse.get()           ,
					     m_ESElectronicsSimFast.get() ,
					     m_addESNoise            ) ) ,

   m_APDDigitizer      ( nullptr ) ,
   m_BarrelDigitizer   ( nullptr ) ,
   m_EndcapDigitizer   ( nullptr ) ,
   m_ElectronicsSim    ( nullptr ) ,
   m_Coder             ( nullptr ) ,
   m_APDElectronicsSim ( nullptr ) ,
   m_APDCoder          ( nullptr ) ,
   m_Geometry          ( nullptr ) ,
   m_EBCorrNoise       ( { {nullptr, nullptr, nullptr} } ) ,
   m_EECorrNoise       ( { {nullptr, nullptr, nullptr} } ) 
{
  // "produces" statements taken care of elsewhere.
  //   if(m_apdSeparateDigi) mixMod.produces<EBDigiCollection>(m_apdDigiTag);
  // mixMod.produces<EBDigiCollection>(m_EBdigiCollection);
  // mixMod.produces<EEDigiCollection>(m_EEdigiCollection);
  // mixMod.produces<ESDigiCollection>(m_ESdigiCollection);
   if ( m_doEB ) iC.consumes<std::vector<PCaloHit> >(edm::InputTag(m_hitsProducerTag, "EcalHitsEB"));   
   if ( m_doEE ) iC.consumes<std::vector<PCaloHit> >(edm::InputTag(m_hitsProducerTag, "EcalHitsEE"));
   if ( m_doES ) iC.consumes<std::vector<PCaloHit> >(edm::InputTag(m_hitsProducerTag, "EcalHitsES"));

   const std::vector<double> ebCorMatG12 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG12");
   const std::vector<double> eeCorMatG12 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG12");
   const std::vector<double> ebCorMatG06 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG06");
   const std::vector<double> eeCorMatG06 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG06");
   const std::vector<double> ebCorMatG01 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG01");
   const std::vector<double> eeCorMatG01 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG01");

   const bool applyConstantTerm          = params.getParameter<bool>       ("applyConstantTerm");
   const double rmsConstantTerm          = params.getParameter<double>     ("ConstantTerm");

   const bool addNoise                   = params.getParameter<bool>       ("doENoise"); 
   const bool cosmicsPhase               = params.getParameter<bool>       ("cosmicsPhase");
   const double cosmicsShift             = params.getParameter<double>     ("cosmicsShift");

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

   // further phase for cosmics studies
   if( cosmicsPhase ) 
   {
     if( m_doEB ) m_EBResponse->setPhaseShift( 1. + cosmicsShift ) ;
     if( m_doEE ) m_EEResponse->setPhaseShift( 1. + cosmicsShift ) ;
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
			  
   m_EBCorrNoise[0].reset( new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[0] ) );
   m_EECorrNoise[0].reset( new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[0] ) );
   m_EBCorrNoise[1].reset( new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[1] ) );
   m_EECorrNoise[1].reset( new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[1] ) );
   m_EBCorrNoise[2].reset( new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix[2] ) );
   m_EECorrNoise[2].reset( new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix[2] ) );

   m_Coder.reset( new EcalCoder( addNoise         , 
                                 m_PreMix1        ,
                                 m_EBCorrNoise[0].get() ,
                                 m_EECorrNoise[0].get() ,
                                 m_EBCorrNoise[1].get() ,
                                 m_EECorrNoise[1].get() ,
                                 m_EBCorrNoise[2].get() ,
                                 m_EECorrNoise[2].get()   ) );

   m_ElectronicsSim.reset( new EcalElectronicsSim( m_ParameterMap.get()    ,
                                                   m_Coder.get()           ,
                                                   applyConstantTerm ,
                                                   rmsConstantTerm     ) );
				  
   if( m_apdSeparateDigi )
   {
     m_APDCoder.reset( new EcalCoder( false            , 
                                      m_PreMix1        ,
                                      m_EBCorrNoise[0].get() ,
                                      m_EECorrNoise[0].get() ,
                                      m_EBCorrNoise[1].get() ,
                                      m_EECorrNoise[1].get() ,
                                      m_EBCorrNoise[2].get() ,
                                      m_EECorrNoise[2].get()   ) );
     
     m_APDElectronicsSim.reset( new EcalElectronicsSim( m_ParameterMap.get()    ,
                                                        m_APDCoder.get()        ,
                                                        applyConstantTerm ,
                                                        rmsConstantTerm     ) );
     
     m_APDDigitizer.reset( new EBDigitizer( m_APDResponse.get()       , 
                                            m_APDElectronicsSim.get() ,
                                            false                 ) );
   }

   if( m_doEB ) {
     m_BarrelDigitizer.reset( new EBDigitizer( m_EBResponse.get()     , 
                                               m_ElectronicsSim.get() ,
                                               addNoise            ) );
   }

   if( m_doEE ) {
     m_EndcapDigitizer.reset( new EEDigitizer( m_EEResponse.get()     ,
                                               m_ElectronicsSim.get() , 
                                               addNoise            ) );
   }
}


EcalDigiProducer::~EcalDigiProducer() 
{}

void
EcalDigiProducer::initializeEvent(edm::Event const& event, edm::EventSetup const& eventSetup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  randomEngine_ = &rng->getEngine(event.streamID());

   checkGeometry( eventSetup );
   checkCalibrations( event, eventSetup );
   if( m_doEB ) {
     m_BarrelDigitizer->initializeHits();
     if(m_apdSeparateDigi) {
       m_APDDigitizer->initializeHits();
     }
   }
   if( m_doEE ) {
     m_EndcapDigitizer->initializeHits();
   }
   if( m_doES ) {
     if(m_doFastES) {
       m_ESDigitizer->initializeHits();
     } else {
       m_ESOldDigitizer->initializeHits();
     }
   }
}

void
EcalDigiProducer::accumulateCaloHits(HitsHandle const& ebHandle, HitsHandle const& eeHandle, HitsHandle const& esHandle, int bunchCrossing) {
  if(m_doEB && ebHandle.isValid()) {
    m_BarrelDigitizer->add(*ebHandle.product(), bunchCrossing, randomEngine_);

    if(m_apdSeparateDigi) {
      m_APDDigitizer->add(*ebHandle.product(), bunchCrossing, randomEngine_);
    }
  }

  if(m_doEE && eeHandle.isValid()) {
    m_EndcapDigitizer->add(*eeHandle.product(), bunchCrossing, randomEngine_);
  }

  if(m_doES && esHandle.isValid()) {
    if(m_doFastES) {
      m_ESDigitizer->add(*esHandle.product(), bunchCrossing, randomEngine_);
    } else {
      m_ESOldDigitizer->add(*esHandle.product(), bunchCrossing, randomEngine_);
    }
  }
}

void
EcalDigiProducer::accumulate(edm::Event const& e, edm::EventSetup const& eventSetup) {
  // Step A: Get Inputs
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  if(m_doEB) {
    m_EBShape.setEventSetup(eventSetup);  // need to set the eventSetup here, otherwise pre-mixing module will not wrk
    m_APDShape.setEventSetup(eventSetup); //
    edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
    e.getByLabel(ebTag, ebHandle);
  }

  edm::Handle<std::vector<PCaloHit> > eeHandle;
  if(m_doEE) {
    m_EEShape.setEventSetup(eventSetup); // need to set the eventSetup here, otherwise pre-mixing module will not work
    edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
    e.getByLabel(eeTag, eeHandle);
  }

  edm::Handle<std::vector<PCaloHit> > esHandle;
  if(m_doES) {
    edm::InputTag esTag(m_hitsProducerTag, "EcalHitsES");
    e.getByLabel(esTag, esHandle);
  }

  accumulateCaloHits(ebHandle, eeHandle, esHandle, 0);
}

void
EcalDigiProducer::accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& eventSetup, edm::StreamID const& streamID) {
  // Step A: Get Inputs
  edm::Handle<std::vector<PCaloHit> > ebHandle;
  if(m_doEB) {
    edm::InputTag ebTag(m_hitsProducerTag, "EcalHitsEB");
    e.getByLabel(ebTag, ebHandle);
  }

  edm::Handle<std::vector<PCaloHit> > eeHandle;
  if(m_doEE) {
    edm::InputTag eeTag(m_hitsProducerTag, "EcalHitsEE");
    e.getByLabel(eeTag, eeHandle);
  }

  edm::Handle<std::vector<PCaloHit> > esHandle;
  if(m_doES) {
    edm::InputTag esTag(m_hitsProducerTag, "EcalHitsES");
    e.getByLabel(esTag, esHandle);
  }

  accumulateCaloHits(ebHandle, eeHandle, esHandle, e.bunchCrossing());
}

void 
EcalDigiProducer::finalizeEvent(edm::Event& event, edm::EventSetup const& eventSetup) {
   // Step B: Create empty output
   std::unique_ptr<EBDigiCollection> apdResult      ( !m_apdSeparateDigi || !m_doEB ? nullptr :
						    new EBDigiCollection() ) ;
   std::unique_ptr<EBDigiCollection> barrelResult   ( new EBDigiCollection() ) ;
   std::unique_ptr<EEDigiCollection> endcapResult   ( new EEDigiCollection() ) ;
   std::unique_ptr<ESDigiCollection> preshowerResult( new ESDigiCollection() ) ;
   
   // run the algorithm

   if( m_doEB ) {
     m_BarrelDigitizer->run( *barrelResult, randomEngine_ ) ;
     cacheEBDigis( &*barrelResult ) ;
     
     edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size() ;
     
     if( m_apdSeparateDigi ) {
       m_APDDigitizer->run( *apdResult, randomEngine_ ) ;
       edm::LogInfo("DigiInfo") << "APD Digis: " << apdResult->size() ;
     }
   }

   if( m_doEE ) {
     m_EndcapDigitizer->run( *endcapResult, randomEngine_ ) ;
     edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size() ;
     cacheEEDigis( &*endcapResult ) ;
   }
   if( m_doES ) {
     if(m_doFastES) {
       m_ESDigitizer->run( *preshowerResult, randomEngine_ ) ;
     } else {
       m_ESOldDigitizer->run( *preshowerResult, randomEngine_ ) ;
     }
     edm::LogInfo("EcalDigi") << "ES Digis: " << preshowerResult->size();
   }


   // Step D: Put outputs into event
   if( m_apdSeparateDigi ) {
     //event.put(std::move(apdResult),    m_apdDigiTag         ) ;
   }

   event.put(std::move(barrelResult),    m_EBdigiCollection ) ;
   event.put(std::move(endcapResult),    m_EEdigiCollection ) ;
   event.put(std::move(preshowerResult), m_ESdigiCollection ) ;

   randomEngine_ = nullptr; // to prevent access outside event
}

void
EcalDigiProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup)
{
   edm::Service<edm::RandomNumberGenerator> rng;
   if ( ! rng.isAvailable() ) {
      throw cms::Exception("Configuration") <<
         "RandomNumberGenerator service is not available.\n"
         "You must add the service in the configuration file\n"
         "or remove the module that requires it.";
   }
   CLHEP::HepRandomEngine* engine = &rng->getEngine(lumi.index());

   if( m_doEB ) {
     if( nullptr != m_APDResponse ) m_APDResponse->initialize(engine);
     m_EBResponse->initialize(engine);
   }
}

void  
EcalDigiProducer::checkCalibrations(const edm::Event& event, const edm::EventSetup& eventSetup ) 
{
   // Pedestals from event setup

   edm::ESHandle<EcalPedestals>            dbPed   ;
   eventSetup.get<EcalPedestalsRcd>().get( dbPed ) ;
   const EcalPedestals* pedestals        ( dbPed.product() ) ;
  
   m_Coder->setPedestals( pedestals ) ;
   if( nullptr != m_APDCoder ) m_APDCoder->setPedestals( pedestals ) ;

   // Ecal Intercalibration Constants
   edm::ESHandle<EcalIntercalibConstantsMC>            pIcal   ;
   eventSetup.get<EcalIntercalibConstantsMCRcd>().get( pIcal ) ;
   const EcalIntercalibConstantsMC* ical             ( pIcal.product() ) ;
  
   m_Coder->setIntercalibConstants( ical ) ;
   if( nullptr != m_APDCoder) m_APDCoder->setIntercalibConstants( ical ) ;

   m_EBResponse->setIntercal( ical ) ;
   if( nullptr != m_APDResponse ) m_APDResponse->setIntercal( ical ) ;

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
   if( nullptr != m_APDCoder) m_APDCoder->setGainRatios( gr );

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
   if( nullptr != m_APDCoder ) m_APDCoder->setFullScaleEnergy( EBscale ,
							 EEscale   ) ;

   if( nullptr != m_ESOldDigitizer ||
       nullptr != m_ESDigitizer       )
   {
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
   
      if( m_doES ) {
        m_ESShape.setGain( ESGain );      
        if( !m_doFastES )
          {
            m_ESElectronicsSim->setGain(      ESGain     ) ;
            m_ESElectronicsSim->setPedestals( espeds     ) ;
            m_ESElectronicsSim->setMIPs(      esmips     ) ;
            m_ESElectronicsSim->setMIPToGeV(  ESMIPToGeV ) ;
          }
        else
          {
            m_ESDigitizer->setGain(               ESGain     ) ;
            m_ESElectronicsSimFast->setPedestals( espeds     ) ;
            m_ESElectronicsSimFast->setMIPs(      esmips     ) ;
            m_ESElectronicsSimFast->setMIPToGeV(  ESMIPToGeV ) ;
          }
      }
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
  if( m_doEB ) {
    if( nullptr != m_APDResponse ) m_APDResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
    m_EBResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
  }
  if( m_doEE ) {
    m_EEResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalEndcap    ) ) ;
  }
  if( m_doES ) {
    m_ESResponse->setGeometry(
      m_Geometry->getSubdetectorGeometry( DetId::Ecal, EcalPreshower    ) ) ;
    m_ESOldResponse->setGeometry( m_Geometry ) ;
   
   const std::vector<DetId>* theESDets ( 
      nullptr != m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower) ?
      &m_Geometry->getSubdetectorGeometry(DetId::Ecal, EcalPreshower)->getValidDetIds() : nullptr ) ;

   if( !m_doFastES ) 
   {
      if( nullptr != m_ESOldDigitizer &&
	  nullptr != theESDets             )
	 m_ESOldDigitizer->setDetIds( *theESDets ) ;
   }
   else
   {
      if( nullptr != m_ESDigitizer &&
	  nullptr != theESDets         )
	 m_ESDigitizer->setDetIds( *theESDets ) ; 
   }
  }
}

void EcalDigiProducer::setEBNoiseSignalGenerator(EcalBaseSignalGenerator * noiseGenerator) {
  //noiseGenerator->setParameterMap(theParameterMap);
  if(nullptr != m_BarrelDigitizer) m_BarrelDigitizer->setNoiseSignalGenerator(noiseGenerator);
}

void EcalDigiProducer::setEENoiseSignalGenerator(EcalBaseSignalGenerator * noiseGenerator) {
  //noiseGenerator->setParameterMap(theParameterMap);
  if(nullptr != m_EndcapDigitizer) m_EndcapDigitizer->setNoiseSignalGenerator(noiseGenerator);
}

void EcalDigiProducer::setESNoiseSignalGenerator(EcalBaseSignalGenerator * noiseGenerator) {
  //noiseGenerator->setParameterMap(theParameterMap);
  if(nullptr != m_ESDigitizer) m_ESDigitizer->setNoiseSignalGenerator(noiseGenerator);  
}


void EcalDigiProducer::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
   m_EBShape.setEventSetup(setup); 
   m_EEShape.setEventSetup(setup);
   m_APDShape.setEventSetup(setup);
}

