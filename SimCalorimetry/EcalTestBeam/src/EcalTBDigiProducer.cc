
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CLHEP/Random/RandFlat.h"

EcalTBDigiProducer::EcalTBDigiProducer(const edm::ParameterSet& params) 
: theGeometry(0)
{
  /// output collections names

  EBdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");

  produces<EBDigiCollection>();
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

  //theNoiseMatrix = new EcalCorrelatedNoiseMatrix(readoutFrameSize);
  HepSymMatrix thisMatrix(readoutFrameSize,1);
  //theNoiseMatrix->getMatrix(thisMatrix);

  std::vector<double> corrNoiseMatrix = params.getParameter< std::vector<double> >("CorrelatedNoiseMatrix");
  if ( corrNoiseMatrix.size() == (unsigned int)(readoutFrameSize*readoutFrameSize) ) {
    for ( int row = 0 ; row < readoutFrameSize; ++row ) {
      for ( int column = 0 ; column < readoutFrameSize; ++column ) {
        int index = column + readoutFrameSize*row;
        thisMatrix[row][column] = corrNoiseMatrix[index];
      }
    }
  }
  theNoiseMatrix = new EcalCorrelatedNoiseMatrix(thisMatrix);

  theCorrNoise = new CorrelatedNoisifier(thisMatrix);

  theCoder = new EcalCoder(addNoise, theCorrNoise);
  bool applyConstantTerm = params.getParameter<bool>("applyConstantTerm");
  double rmsConstantTerm = params.getParameter<double> ("ConstantTerm");
  theElectronicsSim = new EcalElectronicsSim(theParameterMap, theCoder, applyConstantTerm, rmsConstantTerm);

  theBarrelDigitizer = new EBDigitizer(theEcalResponse, theElectronicsSim, addNoise);

  // not containment corrections
  EBs25notCont = params.getParameter<double>("EBs25notContainment");
  EEs25notCont = params.getParameter<double>("EEs25notContainment");

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


EcalTBDigiProducer::~EcalTBDigiProducer() 
{
  delete theParameterMap;
  delete theEcalShape;
  delete theEcalResponse;
  delete theCorrNoise;
  delete theNoiseMatrix;
  delete theCoder;
  delete theElectronicsSim;
  delete theBarrelDigitizer;
  delete theTBReadout;
}


void EcalTBDigiProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{

  // Step A: Get Inputs

  checkGeometry(eventSetup);
  checkCalibrations(eventSetup);

  const std::string barrelHitsName("EcalHitsEB");

  // Get input
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;
  event.getByLabel("mix",barrelHitsName,crossingFrame);
  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits( new MixCollection<PCaloHit>(crossingFrame.product()));

  std::auto_ptr<EcalTBTDCRawInfo> TDCproduct(new EcalTBTDCRawInfo(1));

  // Step B: Create empty output
  std::auto_ptr<EBDigiCollection> barrelResult(new EBDigiCollection());

  // run the algorithm

  CaloDigiCollectionSorter sorter(5);
  
  if (doPhaseShift) {
    
    edm::Handle<PEcalTBInfo> theEcalTBInfo;
    event.getByLabel(ecalTBInfoLabel,theEcalTBInfo);
    thisPhaseShift = theEcalTBInfo->phaseShift();
    
    DetId detId(DetId::Ecal, 1);
    setPhaseShift(detId);
    
    // fill the TDC information in the event
    
    fillTBTDCRawInfo(*TDCproduct);
    
  }
  
  theBarrelDigitizer->run(*barrelHits, *barrelResult);
  edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();

  // perform the TB readout if required, 
  // otherwise simply fill the proper object

  std::auto_ptr<EBDigiCollection> barrelReadout(new EBDigiCollection());
  if ( doReadout ) {
    theTBReadout->performReadout(event, *theTTmap, *barrelResult, *barrelReadout);
  }
  else {
    barrelResult->swap(*barrelReadout);
  }
  
  /*
  std::vector<EBDataFrame> sortedDigisEB = sorter.sortedVector(*barrelReadout);
  LogDebug("EcalDigi") << "Top 10 EB digis";
  for(int i = 0; i < std::min(10,(int) sortedDigisEB.size()); ++i) 
    {
      LogDebug("EcalDigi") << sortedDigisEB[i];
    }
  */

  // Step D: Put outputs into event
  event.put(barrelReadout);
  event.put(TDCproduct);
  
}



void  EcalTBDigiProducer::checkCalibrations(const edm::EventSetup & eventSetup) 
{

  // Pedestals from event setup

  edm::ESHandle<EcalPedestals> dbPed;
  eventSetup.get<EcalPedestalsRcd>().get( dbPed );
  const EcalPedestals* thePedestals=dbPed.product();
  
  theCoder->setPedestals( thePedestals );

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


void EcalTBDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) 
{
  // TODO find a way to avoid doing this every event
  edm::ESHandle<CaloGeometry> hGeometry;
  eventSetup.get<IdealGeometryRecord>().get(hGeometry);

  const CaloGeometry * pGeometry = &*hGeometry;
  
  // see if we need to update
  if(pGeometry != theGeometry) {
    theGeometry = pGeometry;
    updateGeometry();
  }
}


void EcalTBDigiProducer::updateGeometry() {
  theEcalResponse->setGeometry(theGeometry);

  theBarrelDets.clear();

  theBarrelDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);

  edm::LogInfo("EcalDigi") << "deb geometry: " << "\n" 
                       << "\t barrel: " << theBarrelDets.size ();

  theBarrelDigitizer->setDetIds(theBarrelDets);

  theTBReadout->setDetIds(theBarrelDets);
}


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


