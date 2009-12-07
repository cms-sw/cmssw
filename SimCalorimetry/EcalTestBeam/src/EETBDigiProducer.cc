
#include "SimCalorimetry/EcalTestBeam/interface/EETBDigiProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CLHEP/Random/RandFlat.h"

EETBDigiProducer::EETBDigiProducer(const edm::ParameterSet& params) 
: theGeometry(0)
{
  /// output collections names

  digiCollectionLabel_ = params.getParameter<std::string>("EEdigiCollection");
  hitsProducerLabel_   = params.getParameter<std::string>("hitsProducer");

  produces<EEDigiCollection>();
  produces<EcalTBTDCRawInfo>();
  
  // initialize the default valuer for hardcoded parameters and the EB/EE shape

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

  doPhaseShift = !syncPhase;
  thisPhaseShift = 1.;

  theParameterMap = new EcalSimParameterMap(simHitToPhotoelectronsBarrel, simHitToPhotoelectronsEndcap, 
                                            photoelectronsToAnalogBarrel, photoelectronsToAnalogEndcap, 
                                            samplingFactor, timePhase, readoutFrameSize, binOfMaximum,
                                            doPhotostatistics, syncPhase);
  theEcalShape = new EcalShape(timePhase);

  theEcalResponse = new CaloHitResponse(theParameterMap, theEcalShape);

  bool addNoise = params.getParameter<bool>("doNoise"); 

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
  theNoiseMatrix = new EcalCorrelatedNoiseMatrix(thisMatrix);

  theCorrNoise = new CorrelatedNoisifier<EcalCorrMatrix>(thisMatrix);

  theCoder = new EcalCoder(addNoise, theCorrNoise);
  bool applyConstantTerm = params.getParameter<bool>("applyConstantTerm");
  double rmsConstantTerm = params.getParameter<double> ("ConstantTerm");
  theElectronicsSim = new EcalElectronicsSim(theParameterMap, theCoder, applyConstantTerm, rmsConstantTerm);

  theDigitizer = new EEDigitizer( theEcalResponse,
				  theElectronicsSim,
				  addNoise           ) ;

  // not containment corrections
  EEs25notCont = params.getParameter<double>("EEs25notContainment");
  EBs25notCont = params.getParameter<double>("EBs25notContainment");

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

}


EETBDigiProducer::~EETBDigiProducer() 
{
  delete theParameterMap;
  delete theEcalShape;
  delete theEcalResponse;
  delete theCorrNoise;
  delete theNoiseMatrix;
  delete theCoder;
  delete theElectronicsSim;
  delete theDigitizer;
  delete theTBReadout;
}


void EETBDigiProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{

   std::cout<<"****** in tbdigi produce at start *****"<<std::endl;

  // Step A: Get Inputs

  checkGeometry(eventSetup);

   std::cout<<"****** in tbdigi produce at 1 *****"<<std::endl;
  checkCalibrations(eventSetup);
   std::cout<<"****** in tbdigi produce at 2 *****"<<std::endl;

  const std::string hitsLabel ( hitsProducerLabel_ +
				"EcalHitsEE"           ) ;

  // Get input
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame ;
  event.getByLabel( "mix", hitsLabel, crossingFrame ) ;
  std::cout<<"****** in tbdigi produce at 3 *****"<<std::endl;

  std::cout<<"cFrame is "<<crossingFrame->getSubDet()
	   <<", "<<crossingFrame->getSignal().size()<<std::endl;

  std::auto_ptr<MixCollection<PCaloHit> > 
     hits( new MixCollection<PCaloHit>( crossingFrame.product() ) ) ;
  std::cout<<"****** in tbdigi produce at 4 *****"<<std::endl;

  std::cout<<"**Number of endcap hits is "<<hits->size()<<std::endl ;

  std::auto_ptr<EcalTBTDCRawInfo> TDCproduct(new EcalTBTDCRawInfo(1));
  std::cout<<"****** in tbdigi produce at 5 *****"<<std::endl;

  // Step B: Create empty output
  std::auto_ptr<EEDigiCollection> finalDigis ( new EEDigiCollection() ) ;
  std::cout<<"****** in tbdigi produce at 6 *****"<<std::endl;

  // run the algorithm

  CaloDigiCollectionSorter sorter(5);
  std::cout<<"****** in tbdigi produce at 7 *****"<<std::endl;
  
  if (doPhaseShift) 
  {
     std::cout<<"****** in tbdigi produce at 8 *****"<<std::endl;
    
     edm::Handle<PEcalTBInfo> theEcalTBInfo;
     event.getByLabel(ecalTBInfoLabel,theEcalTBInfo);
     std::cout<<"****** in tbdigi produce at 9 *****"<<std::endl;
     thisPhaseShift = theEcalTBInfo->phaseShift();
     std::cout<<"****** in tbdigi produce at A *****"<<std::endl;
    
     DetId detId(DetId::Ecal, 1);
     setPhaseShift(detId);
     std::cout<<"****** in tbdigi produce at B *****"<<std::endl;
    
    // fill the TDC information in the event
     
     fillTBTDCRawInfo(*TDCproduct);
     std::cout<<"****** in tbdigi produce at C *****"<<std::endl;
  }
  
  std::cout<<"****** in tbdigi produce at D *****"<<std::endl;

  std::auto_ptr<EEDigiCollection> digiCollec(new EEDigiCollection());
  if( 0 < hits->size() )
  {
     theDigitizer->run(*hits, *finalDigis);
     std::cout<<"****** in tbdigi produce at E *****"<<std::endl;
     edm::LogInfo("DigiInfo") << "EE Digis: " << finalDigis->size();

     // perform the TB readout if required, 
     // otherwise simply fill the proper object
      
      if ( doReadout ) 
      {
	 theTBReadout->performReadout( event,
				       *theTTmap,
				       *finalDigis,
				       *digiCollec ) ;
      }
      else 
      {
	 finalDigis->swap(*digiCollec);
      }
   }
   std::cout<<"****** in tbdigi produce at F *****"<<std::endl;
  
/*
   std::vector<EEDataFrame> sortedDigis = sorter.sortedVector(*digiCollec);
   edm::LogInfo("EcalDigi") << "Top 10 digis";
   for(int i = 0; i < std::min(10,(int) sortedDigis.size()); ++i) 
   {
      edm::LogInfo("EcalDigi") << sortedDigis[i];
   }
*/
  // Step D: Put outputs into event
  event.put(digiCollec);
   std::cout<<"****** in tbdigi produce at G *****"<<std::endl;
  event.put(TDCproduct);
   std::cout<<"****** in tbdigi produce at H *****"<<std::endl;
  
}



void  EETBDigiProducer::checkCalibrations(const edm::EventSetup & eventSetup) 
{

  // Pedestals from event setup

  edm::ESHandle<EcalPedestals> dbPed;
  eventSetup.get<EcalPedestalsRcd>().get( dbPed );
  const EcalPedestals* thePedestals=dbPed.product();
  
  theCoder->setPedestals( thePedestals );

  // Ecal Intercalibration Constants
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  eventSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants *ical = pIcal.product();
  
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
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() << "\n" << " saturation for EB = " << EBscale;
  const double EEscale = (agc->getEEValue())*theGains[1]*(theCoder->MAXADC)*EEs25notCont;
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEEValue() << "\n" << " saturation for EE = " << EEscale;
  theCoder->setFullScaleEnergy( EBscale , EEscale );
}


void EETBDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) 
{
   std::cout<<"**in checkGeometry 1**"<<std::endl;

  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<CaloGeometryRecord>().get(hGeometry);

  const CaloGeometry * pGeometry = &*hGeometry;

  std::cout<<"**in checkGeometry 2**"<<std::endl;
  
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    updateGeometry();
  }
}


void EETBDigiProducer::updateGeometry() {
  theEcalResponse->setGeometry(theGeometry);
   std::cout<<"**in update Geometry 1**"<<std::endl;

  const std::vector<DetId>& theBarrelDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);

  edm::LogInfo("EcalDigi") << "deb geometry: " << "\n" 
                       << "\t endcap: " << theBarrelDets.size ();

   std::cout<<"**in updateGeometry  2**"<<std::endl;
  theDigitizer->setDetIds(theBarrelDets);
   std::cout<<"**in updateGeometry 3**"<<std::endl;

  theTBReadout->setDetIds(theBarrelDets);
   std::cout<<"**in updateGeometry 4**"<<std::endl;
}


void EETBDigiProducer::setPhaseShift(const DetId & detId) {
  
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  if ( !parameters.syncPhase() ) {

    int myDet = detId.subdetId();

    LogDebug("EcalDigi") << "Setting the phase shift " << thisPhaseShift << " and the offset " << tunePhaseShift << " for the subdetector " << myDet;

    if ( myDet == 1) {
      double passPhaseShift = thisPhaseShift+tunePhaseShift;
      if ( use2004OffsetConvention_ ) passPhaseShift = 1.-passPhaseShift;
      theEcalResponse->setPhaseShift(passPhaseShift);
    }
    
  }
  
}


void EETBDigiProducer::fillTBTDCRawInfo(EcalTBTDCRawInfo & theTBTDCRawInfo) {

  unsigned int thisChannel = 1;
  
  unsigned int thisCount = (unsigned int)(thisPhaseShift*(tdcRanges[0].tdcMax[0]-tdcRanges[0].tdcMin[0]) + tdcRanges[0].tdcMin[0]);

  EcalTBTDCSample theTBTDCSample(thisChannel, thisCount);

  unsigned int sampleIndex = 0;
  theTBTDCRawInfo.setSample(sampleIndex, theTBTDCSample);

  LogDebug("EcalDigi") << theTBTDCSample << "\n" << theTBTDCRawInfo;

}


