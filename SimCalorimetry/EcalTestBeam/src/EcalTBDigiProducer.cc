
#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalSimParameterMap.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBHitResponse.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEHitResponse.h"

EcalTBDigiProducer::EcalTBDigiProducer( const edm::ParameterSet& params ) :
   EcalDigiProducer( params )
{
   m_EBdigiFinalTag = params.getParameter<std::string>( "EBdigiFinalCollection" ) ;
   m_EBdigiTempTag  = params.getParameter<std::string>( "EBdigiCollection");

   produces<EBDigiCollection>( m_EBdigiFinalTag ) ; // after selective readout
   produces<EcalTBTDCRawInfo>() ;

   const bool syncPhase ( params.getParameter<bool>("syncPhase") ) ;

   // possible phase shift for asynchronous trigger (e.g. test-beam)

   m_doPhaseShift = !syncPhase ;
   m_thisPhaseShift = 1. ;

   typedef std::vector< edm::ParameterSet > Parameters;
   Parameters ranges=params.getParameter<Parameters>( "tdcRanges" ) ;
   for( Parameters::iterator itRanges = ranges.begin(); 
	itRanges != ranges.end(); ++itRanges )
   {
      EcalTBTDCRecInfoAlgo::EcalTBTDCRanges aRange;
      aRange.runRanges.first = itRanges->getParameter<int>("startRun");
      aRange.runRanges.second = itRanges->getParameter<int>("endRun");
      aRange.tdcMin = itRanges->getParameter< std::vector<double> >("tdcMin");
      aRange.tdcMax = itRanges->getParameter< std::vector<double> >("tdcMax");
      m_tdcRanges.push_back(aRange);
   }

   m_use2004OffsetConvention = 
      params.getUntrackedParameter< bool >("use2004OffsetConvention",
					   false ) ;

   m_ecalTBInfoLabel =
      params.getUntrackedParameter<std::string>( "EcalTBInfoLabel" ,
						 "SimEcalTBG4Object" ) ;

   m_doReadout = params.getParameter<bool>( "doReadout" ) ;

   m_theTBReadout = new EcalTBReadout( m_ecalTBInfoLabel ) ;

   m_tunePhaseShift =  params.getParameter<double>( "tunePhaseShift" ) ;
}

EcalTBDigiProducer::~EcalTBDigiProducer()
{
}

void EcalTBDigiProducer::produce( edm::Event&            event      ,
				  const edm::EventSetup& eventSetup   ) 
{
  std::cout<<"====****Entering EcalTBDigiProducer produce()"<<std::endl ;
   edm::ESHandle<CaloGeometry>               hGeometry ;
   eventSetup.get<CaloGeometryRecord>().get( hGeometry ) ;
   const std::vector<DetId>& theBarrelDets (
      hGeometry->getValidDetIds(DetId::Ecal, EcalBarrel) ) ;

   m_theTBReadout->setDetIds( theBarrelDets ) ;

   std::auto_ptr<EcalTBTDCRawInfo> TDCproduct( new EcalTBTDCRawInfo(1) ) ;
   if( m_doPhaseShift ) 
   {
      edm::Handle<PEcalTBInfo>             theEcalTBInfo   ;
      event.getByLabel( m_ecalTBInfoLabel, theEcalTBInfo ) ;
      m_thisPhaseShift = theEcalTBInfo->phaseShift() ;

      DetId detId( DetId::Ecal, 1 ) ;
      setPhaseShift( detId ) ;

      fillTBTDCRawInfo( *TDCproduct ) ; // fill the TDC info in the event    
   }

   m_ebDigis = std::auto_ptr<EBDigiCollection> ( new EBDigiCollection ) ;

   EcalDigiProducer::produce( event, eventSetup ) ;

   const EBDigiCollection* barrelResult ( &*m_ebDigis ) ;

   std::auto_ptr<EBDigiCollection> barrelReadout( new EBDigiCollection() ) ;
   if( m_doReadout ) 
   {
      m_theTBReadout->performReadout( event,
				      m_theTTmap,
				      *barrelResult,
				      *barrelReadout ) ;
   }
   else 
   {
      *barrelReadout = *barrelResult ;
   }

   std::cout<< "===**** EcalTBDigiProducer: number of barrel digis = "
	    << barrelReadout->size()<<std::endl ;

   event.put( barrelReadout, m_EBdigiFinalTag ) ;
   event.put( TDCproduct ) ;

   m_ebDigis.reset() ; // release memory
   m_eeDigis.reset() ; // release memory
}

void 
EcalTBDigiProducer::setPhaseShift( const DetId& detId ) 
{  
   const CaloSimParameters& parameters ( 
      EcalDigiProducer::m_ParameterMap->simParameters( detId ) ) ;

   if ( !parameters.syncPhase() ) 
   {
      const int myDet ( detId.subdetId() ) ;

      LogDebug("EcalDigi") << "Setting the phase shift " 
			   << m_thisPhaseShift 
			   << " and the offset " 
			   << m_tunePhaseShift 
			   << " for the subdetector " 
			   << myDet;

      if( myDet == 1 ) 
      {
	 double passPhaseShift ( m_thisPhaseShift + m_tunePhaseShift ) ;
	 if( m_use2004OffsetConvention ) passPhaseShift = 1. - passPhaseShift ;
	 EcalDigiProducer::m_EBResponse->setPhaseShift( passPhaseShift ) ;
	 EcalDigiProducer::m_EEResponse->setPhaseShift( passPhaseShift ) ;
      }
   }
}

void 
EcalTBDigiProducer::fillTBTDCRawInfo( EcalTBTDCRawInfo& theTBTDCRawInfo ) 
{
   const unsigned int thisChannel ( 1 ) ;
  
   const unsigned int thisCount ( 
      (unsigned int)( m_thisPhaseShift*( m_tdcRanges[0].tdcMax[0]
					 - m_tdcRanges[0].tdcMin[0] ) 
		      + m_tdcRanges[0].tdcMin[0] ) ) ;

   EcalTBTDCSample theTBTDCSample ( thisChannel, thisCount ) ;

   const unsigned int sampleIndex ( 0 ) ;
   theTBTDCRawInfo.setSample( sampleIndex, theTBTDCSample ) ;

   LogDebug("EcalDigi") << theTBTDCSample << "\n" << theTBTDCRawInfo ;
}

void 
EcalTBDigiProducer::cacheEBDigis( const EBDigiCollection* ebDigiPtr ) const
{
   m_ebDigis.reset( new EBDigiCollection ) ;
   *m_ebDigis = *ebDigiPtr ;
}

void 
EcalTBDigiProducer::cacheEEDigis( const EEDigiCollection* eeDigiPtr ) const
{
   std::cout<< "===**** EcalTBDigiProducer: number of endcap digis = "
	    << eeDigiPtr->size()<<std::endl ;
}
