
#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsMCRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"


EcalTBDigiProducer::EcalTBDigiProducer(const edm::ParameterSet& params)
{

  /// output collections names

  EBdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
  EEdigiCollection_ = params.getParameter<std::string>("EEdigiCollection");

  const std::string hitsProducer (
     params.getParameter<std::string>("hitsProducer") ) ;
  
  m_barrelHitsName = hitsProducer + "EcalHitsEB" ;
  m_endcapHitsName = hitsProducer + "EcalHitsEE" ;

  produces<EBDigiCollection>( EBdigiCollection_ );
  produces<EEDigiCollection>( EEdigiCollection_ );

  produces<EcalTBTDCRawInfo>(); //For TB

  // initialize the default valuer for hardcoded parameters and the EB/EE shape

  bool addNoise = params.getParameter<bool>("doNoise"); 
  double simHitToPhotoelectronsBarrel = params.getParameter<double>("simHitToPhotoelectronsBarrel");
  double simHitToPhotoelectronsEndcap = params.getParameter<double>("simHitToPhotoelectronsEndcap");
  double photoelectronsToAnalogBarrel = params.getParameter<double>("photoelectronsToAnalogBarrel");
  double photoelectronsToAnalogEndcap = params.getParameter<double>("photoelectronsToAnalogEndcap");
  double samplingFactor = params.getParameter<double>("samplingFactor");
  double timePhase = params.getParameter<double>("timePhase");
  int readoutFrameSize = params.getParameter<int>("readoutFrameSize");
  int binOfMaximum = params.getParameter<int>("binOfMaximum");
  bool doPhotostatistics = params.getParameter<bool>("doPhotostatistics");
  bool syncPhase = params.getParameter<bool>("syncPhase");

  // possible phase shift for asynchronous trigger (e.g. test-beam)

  doPhaseShift = !syncPhase; //For TB
  thisPhaseShift = 1.; //For TB

  theParameterMap = new EcalSimParameterMap(simHitToPhotoelectronsBarrel, simHitToPhotoelectronsEndcap, 
                                            photoelectronsToAnalogBarrel, photoelectronsToAnalogEndcap, 
                                            samplingFactor, timePhase, readoutFrameSize, binOfMaximum,
                                            doPhotostatistics, syncPhase);

  
  theEBResponse = new CaloHitRespoNew(theParameterMap, &theEBShape);
  theEEResponse = new CaloHitRespoNew(theParameterMap, &theEEShape);

  EcalCorrMatrix thisMatrix;

  std::vector<double> corrNoiseMatrix = params.getParameter< std::vector<double> >("CorrelatedNoiseMatrix");
  if ( corrNoiseMatrix.size() == (unsigned int)(readoutFrameSize*readoutFrameSize) ) {
    for ( int row = 0 ; row < readoutFrameSize; ++row ) {
      for ( int column = 0 ; column < readoutFrameSize; ++column ) {
        int index = column + readoutFrameSize*row;
        thisMatrix(row,column) = corrNoiseMatrix[index];
      }
    }
  }
  theNoiseMatrix = new EcalCorrMatrix(thisMatrix);

  theCorrNoise = new CorrelatedNoisifier<EcalCorrMatrix>(thisMatrix);

  theCoder = new EcalCoder(addNoise, theCorrNoise);
  bool applyConstantTerm = params.getParameter<bool>("applyConstantTerm");
  double rmsConstantTerm = params.getParameter<double> ("ConstantTerm");
  theElectronicsSim = new EcalElectronicsSim(theParameterMap, theCoder, applyConstantTerm, rmsConstantTerm);


  theBarrelDigitizer = new EBDigitizer( theEBResponse, 
					theElectronicsSim, 
					addNoise           );

  theEndcapDigitizer = new EEDigitizer( theEEResponse, 
					theElectronicsSim, 
					addNoise           );


  // not containment corrections
  EBs25notCont = params.getParameter<double>("EBs25notContainment");
  EEs25notCont = params.getParameter<double>("EEs25notContainment");

//For TB --------------------------------------

  /// Test Beam specific part

  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters ranges=params.getParameter<Parameters>("tdcRanges");
  for(Parameters::iterator itRanges = ranges.begin(); itRanges != ranges.end(); ++itRanges) 
    {
      EcalTBTDCRecInfoAlgo::EcalTBTDCRanges aRange;
      aRange.runRanges.first = itRanges->getParameter<int>("startRun");
      aRange.runRanges.second = itRanges->getParameter<int>("endRun");
      aRange.tdcMin = itRanges->getParameter< std::vector<double> >("tdcMin");
      aRange.tdcMax = itRanges->getParameter< std::vector<double> >("tdcMax");
      tdcRanges.push_back(aRange);
    }

  use2004OffsetConvention_ = params.getUntrackedParameter< bool >("use2004OffsetConvention",false);

  ecalTBInfoLabel = params.getUntrackedParameter<std::string>("EcalTBInfoLabel","SimEcalTBG4Object");
  doReadout = params.getParameter<bool>("doReadout");

  theTBReadout = new EcalTBReadout(ecalTBInfoLabel);

  tunePhaseShift =  params.getParameter<double>("tunePhaseShift");

//For TB --------------------------------------
}


EcalTBDigiProducer::~EcalTBDigiProducer() 
{
  if (theParameterMap)  { delete theParameterMap; }
  if (theEBResponse)  { delete theEBResponse; }
  if (theEEResponse)  { delete theEEResponse; }
  if (theCorrNoise)     { delete theCorrNoise; }
  if (theNoiseMatrix)   { delete theNoiseMatrix; }
  if (theCoder)         { delete theCoder; }
  if (theElectronicsSim){ delete theElectronicsSim; }
  if (theBarrelDigitizer){ delete theBarrelDigitizer; }
  if (theEndcapDigitizer){ delete theEndcapDigitizer; }
}

void EcalTBDigiProducer::produce( edm::Event&            event      ,
				  const edm::EventSetup& eventSetup   ) 
{
   // Step A: Get Inputs

//For TB ----------------
   edm::ESHandle<CaloGeometry>               hGeometry ;
   eventSetup.get<CaloGeometryRecord>().get( hGeometry ) ;
   theEBResponse->setGeometry( 
      hGeometry->getSubdetectorGeometry( DetId::Ecal, EcalBarrel    ) ) ;
   theEEResponse->setGeometry( 
      hGeometry->getSubdetectorGeometry( DetId::Ecal, EcalEndcap    ) ) ;

   // takes no time because gives back const ref
   const std::vector<DetId>& theBarrelDets (
      hGeometry->getValidDetIds(DetId::Ecal, EcalBarrel) ) ;
//   const std::vector<DetId>& theEndcapDets (
//      hGeometry->getValidDetIds(DetId::Ecal, EcalEndcap) ) ;

//   theBarrelDigitizer->setDetIds( theBarrelDets ) ;
//   theEndcapDigitizer->setDetIds( theEndcapDets ) ;
   theTBReadout      ->setDetIds( theBarrelDets ) ;

//For TB ----------------

   checkCalibrations(eventSetup);

   // Get input
   edm::Handle<CrossingFrame<PCaloHit> >      crossingFrame;
   event.getByLabel( "mix", m_barrelHitsName, crossingFrame ) ;

   MixCollection<PCaloHit>* EBHits ( crossingFrame.isValid() ?
				     new MixCollection<PCaloHit>( crossingFrame.product() ) : 0 ) ;

   const bool isEB ( 0 != EBHits &&
		     0 != EBHits->size() ) ;

   event.getByLabel( "mix", m_endcapHitsName, crossingFrame ) ;
   MixCollection<PCaloHit>* EEHits ( crossingFrame.isValid() ?
				     new MixCollection<PCaloHit>( crossingFrame.product() ) : 0 ) ;

   const bool isEE ( 0 != EEHits &&
		     0 != EEHits->size() ) ;

//For TB ----------------------------------------  
   std::auto_ptr<EcalTBTDCRawInfo> TDCproduct(new EcalTBTDCRawInfo(1));
   if( doPhaseShift ) 
   {
      edm::Handle<PEcalTBInfo>           theEcalTBInfo   ;
      event.getByLabel( ecalTBInfoLabel, theEcalTBInfo ) ;
      thisPhaseShift = theEcalTBInfo->phaseShift();
    
      DetId detId( DetId::Ecal, 1 ) ;
      setPhaseShift( detId );

      fillTBTDCRawInfo( *TDCproduct ) ; // fill the TDC info in the event
    
  }
//For TB ----------------------------------------  

  // Step B: Create empty output and then fill it
   std::auto_ptr<EBDigiCollection> barrelResult( new EBDigiCollection() ) ;
   std::auto_ptr<EEDigiCollection> endcapResult( new EEDigiCollection() ) ;

   if ( isEB ) 
   {
      std::auto_ptr<MixCollection<PCaloHit> >  barrelHits( EBHits );
      theBarrelDigitizer->run( *barrelHits, *barrelResult ) ;
      edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();
      std::cout << "EB Digis: " << barrelResult->size()<<std::endl;

/*
	CaloDigiCollectionSorter sorter(5) ;

	std::vector<EBDataFrame> sortedDigisEB = sorter.sortedVector(*barrelResult);
	LogDebug("EcalDigi") << "Top 10 EB digis";
	std::cout<< "Top 10 EB digis\n";
	for(int i = 0; i < std::min(10,(int) sortedDigisEB.size()); ++i) 
	{
	   LogDebug("EcalDigi") << sortedDigisEB[i];
	   std::cout << sortedDigisEB[i]<<"\n";
	}
	std::cout<< std::endl ;
*/      
   }

   if( isEE ) 
   {
      std::auto_ptr<MixCollection<PCaloHit> >  endcapHits( EEHits );
      theEndcapDigitizer->run( *endcapHits, *endcapResult ) ;
      edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size();
      std::cout << "EE Digis: " << endcapResult->size()<<std::endl;
   }

   //For TB -------------------------------------------
   // perform the TB readout if required, 
   // otherwise simply fill the proper object

   std::auto_ptr<EBDigiCollection> barrelReadout( new EBDigiCollection() ) ;
   if ( doReadout ) 
   {
      theTBReadout->performReadout( event,
				    *theTTmap,
				    *barrelResult,
				    *barrelReadout);
   }
   else 
   {
      barrelResult->swap(*barrelReadout);
   }

  // Step D: Put outputs into event
//TB  event.put(barrelResult, EBdigiCollection_);
   event.put( barrelReadout, EBdigiCollection_ ) ;
   event.put( endcapResult,  EEdigiCollection_ ) ;
   event.put( TDCproduct ) ;

//For TB -------------------------------------------
}


void  EcalTBDigiProducer::checkCalibrations(const edm::EventSetup & eventSetup) 
{

  // Pedestals from event setup

  edm::ESHandle<EcalPedestals> dbPed;
  eventSetup.get<EcalPedestalsRcd>().get( dbPed );
  const EcalPedestals* thePedestals=dbPed.product();
  
  theCoder->setPedestals( thePedestals );

  // Ecal Intercalibration Constants
  edm::ESHandle<EcalIntercalibConstantsMC> pIcal;
  eventSetup.get<EcalIntercalibConstantsMCRcd>().get(pIcal);
  const EcalIntercalibConstantsMC *ical = pIcal.product();
  
  theCoder->setIntercalibConstants( ical );

  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  eventSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  
  // Gain Ratios
  edm::ESHandle<EcalGainRatios> pRatio;
  eventSetup.get<EcalGainRatiosRcd>().get(pRatio);
  const EcalGainRatios* gr = pRatio.product();

  theCoder->setGainRatios( gr );

  EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();

  double theGains[theCoder->NGAINS+1];
  theGains[0] = 0.;
  theGains[3] = 1.;
  theGains[2] = defaultRatios->gain6Over1() ;
  theGains[1] = theGains[2]*(defaultRatios->gain12Over6()) ;

  LogDebug("EcalDigi") << " Gains: " << "\n" << " g1 = " << theGains[1] << "\n" << " g2 = " << theGains[2] << "\n" << " g3 = " << theGains[3];

  delete defaultRatios;

  const double EBscale = (agc->getEBValue())*theGains[1]*(theCoder->MAXADC)*EBs25notCont;
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() << "\n" << " notCont = " << EBs25notCont << "\n" << " saturation for EB = " << EBscale << ", " << EBs25notCont;
  const double EEscale = (agc->getEEValue())*theGains[1]*(theCoder->MAXADC)*EEs25notCont;
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEEValue() << "\n" << " notCont = " << EEs25notCont << "\n" << " saturation for EB = " << EEscale << ", " << EEs25notCont;
  theCoder->setFullScaleEnergy( EBscale , EEscale );

}

/* //TB
void EcalTBDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) 
{
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<CaloGeometryRecord>().get(hGeometry);

  const CaloGeometry * pGeometry = &*hGeometry;
  
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    updateGeometry();
  }
}

void EcalTBDigiProducer::updateGeometry() {
  theEcalResponse->setGeometry(theGeometry);
//TB  theESResponse->setGeometry(theGeometry);

  const std::vector<DetId>& theBarrelDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  const std::vector<DetId>& theEndcapDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
//TB  const std::vector<DetId>& theESDets     =  theGeometry->getValidDetIds(DetId::Ecal, EcalPreshower);

  edm::LogInfo("EcalDigi") << "deb geometry: " << "\n" 
			   << "\t barrel: " << theBarrelDets.size () << "\n"
			   << "\t endcap: " << theEndcapDets.size () << "\n";
//TB                      << "\t preshower: " << theESDets.size();
  
  theBarrelDigitizer->setDetIds(theBarrelDets);
  theEndcapDigitizer->setDetIds(theEndcapDets);
//TB  if (!doFast) { theESDigitizer->setDetIds(theESDets); }
//TB  if ( doFast) { theESDigitizerFast->setDetIds(theESDets); }

  theTBReadout->setDetIds(theBarrelDets);
}
*/
//For TB --------------------------------------------------------

void EcalTBDigiProducer::setPhaseShift(const DetId & detId) {
  
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  if ( !parameters.syncPhase() ) {

    int myDet = detId.subdetId();

    LogDebug("EcalDigi") << "Setting the phase shift " << thisPhaseShift << " and the offset " << tunePhaseShift << " for the subdetector " << myDet;

    if ( myDet == 1) {
      double passPhaseShift = thisPhaseShift+tunePhaseShift;
      if ( use2004OffsetConvention_ ) passPhaseShift = 1.-passPhaseShift;
      theEBResponse->setPhaseShift(passPhaseShift);
      theEEResponse->setPhaseShift(passPhaseShift);
    }
    
  }
  
}

void EcalTBDigiProducer::fillTBTDCRawInfo(EcalTBTDCRawInfo & theTBTDCRawInfo) {

  unsigned int thisChannel = 1;
  
  unsigned int thisCount = (unsigned int)(thisPhaseShift*(tdcRanges[0].tdcMax[0]-tdcRanges[0].tdcMin[0]) + tdcRanges[0].tdcMin[0]);

  EcalTBTDCSample theTBTDCSample(thisChannel, thisCount);

  unsigned int sampleIndex = 0;
  theTBTDCRawInfo.setSample(sampleIndex, theTBTDCSample);

  LogDebug("EcalDigi") << theTBTDCSample << "\n" << theTBTDCRawInfo;

}
//For TB --------------------------------------------------------
