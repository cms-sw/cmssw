
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
  :  theGeometry(0)
{

  /// output collections names

  EBdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
  EEdigiCollection_ = params.getParameter<std::string>("EEdigiCollection");
//TB  ESdigiCollection_ = params.getParameter<std::string>("ESdigiCollection");

  hitsProducer_ = params.getParameter<std::string>("hitsProducer");
  
  produces<EBDigiCollection>(EBdigiCollection_);
  produces<EEDigiCollection>(EEdigiCollection_);
//TB  produces<ESDigiCollection>(ESdigiCollection_);

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

  theEcalShape = new EcalShape(timePhase);
  
//TB  int ESGain = params.getParameter<int>("ESGain");
//TB  theESShape = new ESShape(ESGain);

  theEcalResponse = new CaloHitResponse(theParameterMap, theEcalShape);
//TB  theESResponse = new CaloHitResponse(theParameterMap, theESShape);


/* //TB
  // further phase for cosmics studies
  cosmicsPhase = params.getParameter<bool>("cosmicsPhase");
  cosmicsShift = params.getParameter<double>("cosmicsShift");
  if (cosmicsPhase) {
    theEcalResponse->setPhaseShift(1.+cosmicsShift);
  }
*/

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

//TB  doFast = params.getParameter<bool>("doFast");
//TB  bool addESNoise = params.getParameter<bool>("doESNoise");
//TB  double ESNoiseSigma = params.getParameter<double> ("ESNoiseSigma");
//TB  int ESBaseline  = params.getParameter<int>("ESBaseline");
//TB  double ESMIPADC = params.getParameter<double>("ESMIPADC");
//TB  double ESMIPkeV = params.getParameter<double>("ESMIPkeV");
//TB  int numESdetId  = params.getParameter<int>("numESdetId");
//TB  double zsThreshold = params.getParameter<double>("zsThreshold");
//TB  std::string refFile = params.getParameter<std::string>("refHistosFile");

//TB  theESElectronicsSim     = 0;
//TB  theESElectronicsSimFast = 0;
//TB  if (!doFast) { theESElectronicsSim     = new ESElectronicsSim(addESNoise, ESNoiseSigma, ESGain, ESBaseline, ESMIPADC, ESMIPkeV); }
//TB  if ( doFast) { theESElectronicsSimFast = new ESElectronicsSimFast(addESNoise, ESNoiseSigma, ESGain, ESBaseline, ESMIPADC, ESMIPkeV);  }

  theBarrelDigitizer = new EBDigitizer(theEcalResponse, theElectronicsSim, addNoise);
  theEndcapDigitizer = new EEDigitizer(theEcalResponse, theElectronicsSim, addNoise);

//TB  theESDigitizer = 0;
//TB  theESDigitizerFast = 0;
//TB  if (!doFast){ theESDigitizer = new ESDigitizer(theESResponse, theESElectronicsSim, addESNoise); }
//TB  if ( doFast){ theESDigitizerFast = new ESFastTDigitizer(theESResponse, theESElectronicsSimFast, addESNoise, numESdetId, zsThreshold, refFile);}

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
  if (theEcalShape)     { delete theEcalShape; }
//TB  if (theESShape)       { delete theESShape; }
  if (theEcalResponse)  { delete theEcalResponse; }
//TB  if (theESResponse)    { delete theESResponse; }
  if (theCorrNoise)     { delete theCorrNoise; }
  if (theNoiseMatrix)   { delete theNoiseMatrix; }
  if (theCoder)         { delete theCoder; }
  if (theElectronicsSim){ delete theElectronicsSim; }
//TB  if (theESElectronicsSim)    { delete theESElectronicsSim; }
//TB  if (theESElectronicsSimFast){ delete theESElectronicsSimFast; }
  if (theBarrelDigitizer){ delete theBarrelDigitizer; }
  if (theEndcapDigitizer){ delete theEndcapDigitizer; }
//TB  if (theESDigitizer)    { delete theESDigitizer; }
//TB  if (theESDigitizerFast){ delete theESDigitizerFast; }
}


void EcalTBDigiProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{

  // Step A: Get Inputs

  checkGeometry(eventSetup);
  checkCalibrations(eventSetup);

  // Get input
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;

  const std::vector<DetId>& theBarrelDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  const std::vector<DetId>& theEndcapDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
//TB  const std::vector<DetId>& theESDets     =  theGeometry->getValidDetIds(DetId::Ecal, EcalPreshower);

  // test access to SimHits
  const std::string barrelHitsName(hitsProducer_+"EcalHitsEB");
  const std::string endcapHitsName(hitsProducer_+"EcalHitsEE");
//TB  const std::string preshowerHitsName(hitsProducer_+"EcalHitsES");

  bool isEB = true;
  event.getByLabel("mix",barrelHitsName,crossingFrame);
  MixCollection<PCaloHit> * EBHits = 0 ;
  if (crossingFrame.isValid()) { 
    EBHits = new MixCollection<PCaloHit>(crossingFrame.product());
  }
  else { 
    edm::LogError("EcalTBDigiProducer") << "Error! can't get the product " << barrelHitsName.c_str() ;
    isEB = false;
  }
  if ( ! EBHits->inRegistry() || theBarrelDets.size() == 0 ) isEB = false;

  bool isEE = true;
  event.getByLabel("mix",endcapHitsName,crossingFrame);
  MixCollection<PCaloHit> * EEHits = 0 ;
  if (crossingFrame.isValid()) {
    EEHits = new MixCollection<PCaloHit>(crossingFrame.product());
  }
  else {
    edm::LogError("EcalTBDigiProducer") << "Error! can't get the product " << endcapHitsName.c_str() ;
    isEE = false;
  }
  if ( ! EEHits->inRegistry() || theEndcapDets.size() == 0 ) isEE = false;

/*  //TB
  bool isES = true;
  event.getByLabel("mix",preshowerHitsName,crossingFrame);
  MixCollection<PCaloHit> * ESHits = 0 ;
  if (crossingFrame.isValid()) {
    ESHits = new MixCollection<PCaloHit>(crossingFrame.product());
  }
  else { 
    edm::LogError("EcalTBDigiProducer") << "Error! can't get the product " << preshowerHitsName.c_str() ;
    isES = false; 
  }    
  if ( ! ESHits->inRegistry() || theESDets.size() == 0 ) isES = false;
*/

  // Step B: Create empty output
  std::auto_ptr<EBDigiCollection> barrelResult(new EBDigiCollection());
  std::auto_ptr<EEDigiCollection> endcapResult(new EEDigiCollection());
//TB  std::auto_ptr<ESDigiCollection> preshowerResult(new ESDigiCollection());

  // run the algorithm

  CaloDigiCollectionSorter sorter(5);

//For TB ----------------------------------------  
  std::auto_ptr<EcalTBTDCRawInfo> TDCproduct(new EcalTBTDCRawInfo(1));
  if (doPhaseShift) {
    
    edm::Handle<PEcalTBInfo> theEcalTBInfo;
    event.getByLabel(ecalTBInfoLabel,theEcalTBInfo);
    thisPhaseShift = theEcalTBInfo->phaseShift();
    
    DetId detId(DetId::Ecal, 1);
    setPhaseShift(detId);
    
    // fill the TDC information in the event
    
    fillTBTDCRawInfo(*TDCproduct);
    
  }
//For TB ----------------------------------------  

  if ( isEB ) {

    std::auto_ptr<MixCollection<PCaloHit> >  barrelHits( EBHits );
    theBarrelDigitizer->run(*barrelHits, *barrelResult);
    edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();

    /*
    std::vector<EBDataFrame> sortedDigisEB = sorter.sortedVector(*barrelResult);
    LogDebug("EcalDigi") << "Top 10 EB digis";
    for(int i = 0; i < std::min(10,(int) sortedDigisEB.size()); ++i) 
      {
        LogDebug("EcalDigi") << sortedDigisEB[i];
      }
    */
  }

  if ( isEE ) {

    std::auto_ptr<MixCollection<PCaloHit> >  endcapHits( EEHits );
    theEndcapDigitizer->run(*endcapHits, *endcapResult);
    edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size();

    /*
    std::vector<EEDataFrame> sortedDigisEE = sorter.sortedVector(*endcapResult);
    LogDebug("EcalDigi")  << "Top 10 EE digis";
    for(int i = 0; i < std::min(10,(int) sortedDigisEE.size()); ++i) 
      {
        LogDebug("EcalDigi") << sortedDigisEE[i];
      }
    */
  }

/*   //TB
  if ( isES ) {

    std::auto_ptr<MixCollection<PCaloHit> >  preshowerHits( ESHits );
    if (!doFast) { theESDigitizer->run(*preshowerHits, *preshowerResult); }
    if ( doFast) { theESDigitizerFast->run(*preshowerHits, *preshowerResult); }
    edm::LogInfo("EcalDigi") << "ES Digis: " << preshowerResult->size();
    
//   CaloDigiCollectionSorter sorter_es(7);
//   std::vector<ESDataFrame> sortedDigis_es = sorter_es.sortedVector(*preshowerResult);
//   LogDebug("DigiDump") << "List all ES digis";
//   for(int i = 0; i < sortedDigis_es.size(); ++i) 
//     {
//       LogDebug("DigiDump") << sortedDigis_es[i];
//     }
  }
*/
//TB  event.put(preshowerResult, ESdigiCollection_);

//For TB -------------------------------------------
  // perform the TB readout if required, 
  // otherwise simply fill the proper object

  std::auto_ptr<EBDigiCollection> barrelReadout(new EBDigiCollection());
  if ( doReadout ) {

     theTBReadout->performReadout(event, *theTTmap, *barrelResult, *barrelReadout);
  }
  else {
    barrelResult->swap(*barrelReadout);
  }
  event.put(TDCproduct);

  // Step D: Put outputs into event
//TB  event.put(barrelResult, EBdigiCollection_);
  event.put(barrelReadout, EBdigiCollection_);
  event.put(endcapResult, EEdigiCollection_);

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

//For TB --------------------------------------------------------

void EcalTBDigiProducer::setPhaseShift(const DetId & detId) {
  
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

void EcalTBDigiProducer::fillTBTDCRawInfo(EcalTBTDCRawInfo & theTBTDCRawInfo) {

  unsigned int thisChannel = 1;
  
  unsigned int thisCount = (unsigned int)(thisPhaseShift*(tdcRanges[0].tdcMax[0]-tdcRanges[0].tdcMin[0]) + tdcRanges[0].tdcMin[0]);

  EcalTBTDCSample theTBTDCSample(thisChannel, thisCount);

  unsigned int sampleIndex = 0;
  theTBTDCRawInfo.setSample(sampleIndex, theTBTDCSample);

  LogDebug("EcalDigi") << theTBTDCSample << "\n" << theTBTDCRawInfo;

}
//For TB --------------------------------------------------------
