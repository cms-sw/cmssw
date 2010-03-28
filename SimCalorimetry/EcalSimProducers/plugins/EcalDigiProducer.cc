#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalCoder.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSim.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESFastTDigitizer.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
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
#include <iostream>
EcalDigiProducer::EcalDigiProducer( const edm::ParameterSet& params ) :
   m_BarrelDigitizer ( 0 ) ,
   m_EndcapDigitizer ( 0 ) ,
   m_ESDigitizer     ( 0 ) ,
   m_ESDigitizerFast ( 0 ) ,
   m_ParameterMap    ( 0 ) ,
   m_EBShape         (   ) ,
   m_EEShape         (   ) ,
   m_ESShape         (   ) ,
   m_EBResponse      ( 0 ) ,
   m_EEResponse      ( 0 ) ,
   m_ESResponse      ( 0 ) ,
   m_EBCorrNoise       ( 0 ) ,
   m_EECorrNoise       ( 0 ) ,
   m_ElectronicsSim    ( 0 ) ,
   m_ESElectronicsSim  ( 0 ) ,
   m_ESElectronicsSimFast ( 0 ) ,
   m_Coder                ( 0 ) ,
   m_Geometry             ( 0 )
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
   const std::vector<double> ebCorMatVec     = params.getParameter< std::vector<double> >("EBCorrelatedNoiseMatrix");
   const std::vector<double> eeCorMatVec     = params.getParameter< std::vector<double> >("EECorrelatedNoiseMatrix");
   const bool applyConstantTerm              = params.getParameter<bool>       ("applyConstantTerm");
   const double rmsConstantTerm              = params.getParameter<double>     ("ConstantTerm");
   const bool addESNoise                     = params.getParameter<bool>       ("doESNoise");
   const int numESdetId                      = params.getParameter<int>        ("numESdetId");
   m_doFast                                  = params.getParameter<bool>       ("doFast");
   m_EBs25notCont                            = params.getParameter<double>     ("EBs25notContainment");
   m_EEs25notCont                            = params.getParameter<double>     ("EEs25notContainment");
   m_hitsProducerTag                         = params.getParameter<std::string>("hitsProducer");

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
   
   m_ESShape   = new ESShape();

   m_EBResponse = new CaloHitResponse( m_ParameterMap, &m_EBShape );
   m_EEResponse = new CaloHitResponse( m_ParameterMap, &m_EEShape );
   m_ESResponse = new CaloHitResponse( m_ParameterMap, m_ESShape );

   // further phase for cosmics studies
   if( cosmicsPhase ) 
   {
      m_EBResponse->setPhaseShift( 1. + cosmicsShift ) ;
      m_EEResponse->setPhaseShift( 1. + cosmicsShift ) ;
   }


   EcalCorrMatrix ebMatrix ;
   EcalCorrMatrix eeMatrix ;

   assert( ebCorMatVec.size() == (unsigned int)(readoutFrameSize*readoutFrameSize) ) ;
   assert( eeCorMatVec.size() == (unsigned int)(readoutFrameSize*readoutFrameSize) ) ;
   for ( unsigned int row ( 0 ) ; row != readoutFrameSize; ++row )
   {
      for ( unsigned int column ( 0 ) ; column != readoutFrameSize; ++column )
      {
	 const unsigned int index = column + readoutFrameSize*row;
	 ebMatrix( row, column ) = ebCorMatVec[index];
	 eeMatrix( row, column ) = eeCorMatVec[index];
      }
   }

   m_EBCorrNoise = new CorrelatedNoisifier<EcalCorrMatrix>( ebMatrix ) ;
   m_EECorrNoise = new CorrelatedNoisifier<EcalCorrMatrix>( eeMatrix ) ;

   m_Coder = new EcalCoder( addNoise, m_EBCorrNoise, m_EECorrNoise ) ;

   m_ElectronicsSim = new EcalElectronicsSim( m_ParameterMap,
					      m_Coder,
					      applyConstantTerm,
					      rmsConstantTerm   );



   m_BarrelDigitizer = new EBDigitizer( m_EBResponse, 
					m_ElectronicsSim,
					addNoise           );

   m_EndcapDigitizer = new EEDigitizer( m_EEResponse,
					m_ElectronicsSim, 
					addNoise           );


   if (!m_doFast) 
   {
     m_ESElectronicsSim     =
       new ESElectronicsSim( addESNoise );
     
     m_ESDigitizer = new ESDigitizer( m_ESResponse, 
				      m_ESElectronicsSim,
				      addESNoise           );
   }
   else
     {
       m_ESElectronicsSimFast = 
	 new ESElectronicsSimFast( addESNoise );

      m_ESDigitizerFast = new ESFastTDigitizer( m_ESResponse,
						m_ESElectronicsSimFast,
						addESNoise,
						numESdetId);
   }
}


EcalDigiProducer::~EcalDigiProducer() 
{
   delete m_BarrelDigitizer;
   delete m_EndcapDigitizer;
   delete m_ESDigitizer;
   delete m_ESDigitizerFast;
   delete m_ParameterMap;
   delete m_ESShape;
   delete m_EBResponse; 
   delete m_EEResponse; 
   delete m_ESResponse; 
   delete m_EBCorrNoise; 
   delete m_EECorrNoise; 
   delete m_ElectronicsSim;
   delete m_ESElectronicsSim;
   delete m_ESElectronicsSimFast;
   delete m_Coder;
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


   // Step B: Create empty output
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

   // Step D: Put outputs into event
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
  
   m_Coder->setPedestals( thePedestals );

   // Ecal Intercalibration Constants
   edm::ESHandle<EcalIntercalibConstantsMC> pIcal;
   eventSetup.get<EcalIntercalibConstantsMCRcd>().get(pIcal);
   const EcalIntercalibConstantsMC *ical = pIcal.product();
  
   m_Coder->setIntercalibConstants( ical );

   // ADC -> GeV Scale
   edm::ESHandle<EcalADCToGeVConstant> pAgc;
   eventSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
   const EcalADCToGeVConstant* agc = pAgc.product();
  
   // Gain Ratios
   edm::ESHandle<EcalGainRatios> pRatio;
   eventSetup.get<EcalGainRatiosRcd>().get(pRatio);
   const EcalGainRatios* gr = pRatio.product();

   m_Coder->setGainRatios( gr );

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

   // ES condition objects
   edm::ESHandle<ESGain> esgain_;
   edm::ESHandle<ESMIPToGeVConstant> esMIPToGeV_;
   edm::ESHandle<ESPedestals> esPedestals_;
   edm::ESHandle<ESIntercalibConstants> esMIPs_;

   eventSetup.get<ESGainRcd>().get(esgain_);
   eventSetup.get<ESMIPToGeVConstantRcd>().get(esMIPToGeV_);
   eventSetup.get<ESPedestalsRcd>().get(esPedestals_);
   eventSetup.get<ESIntercalibConstantsRcd>().get(esMIPs_);

   const ESGain *esgain = esgain_.product();
   const ESPedestals *espeds = esPedestals_.product();
   const ESIntercalibConstants *esmips = esMIPs_.product();
   const ESMIPToGeVConstant *esMipToGeV = esMIPToGeV_.product();
   int ESGain = (int) esgain->getESGain();  
   double ESMIPToGeV = (ESGain == 1) ? esMipToGeV->getESValueLow() : esMipToGeV->getESValueHigh(); 

   m_ESShape->setGain(ESGain);
   if (!m_doFast) {
     m_ESElectronicsSim->setGain(ESGain);
     m_ESElectronicsSim->setPedestals(espeds);
     m_ESElectronicsSim->setMIPs(esmips);
     m_ESElectronicsSim->setMIPToGeV(ESMIPToGeV);
   } else {
     m_ESDigitizerFast->setGain(ESGain);
     m_ESElectronicsSimFast->setGain(ESGain);
     m_ESElectronicsSimFast->setPedestals(espeds);
     m_ESElectronicsSimFast->setMIPs(esmips);
     m_ESElectronicsSimFast->setMIPToGeV(ESMIPToGeV);
   }
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

