
#include "SimCalorimetry/EcalTestBeam/interface/EcalTBDigiProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CLHEP/Random/RandFlat.h"

EcalTBDigiProducer::EcalTBDigiProducer(const edm::ParameterSet& params) 
: theGeometry(0)
{
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
  theCoder = new EcalCoder(addNoise);
  bool applyConstantTerm = params.getParameter<bool>("applyConstantTerm");
  double rmsConstantTerm = params.getParameter<double> ("ConstantTerm");
  theElectronicsSim = new EcalElectronicsSim(theParameterMap, theCoder, applyConstantTerm, rmsConstantTerm);

  theBarrelDigitizer = new EBDigitizer(theEcalResponse, theElectronicsSim, addNoise);

}


EcalTBDigiProducer::~EcalTBDigiProducer() 
{
  delete theParameterMap;
  delete theEcalShape;
  delete theEcalResponse;
  delete theCoder;
  delete theElectronicsSim;
  delete theBarrelDigitizer;
}


void EcalTBDigiProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{

  // Step A: Get Inputs

  checkGeometry(eventSetup);
  checkCalibrations(eventSetup);

  // Get input
  edm::Handle<CrossingFrame> crossingFrame;
  event.getByType(crossingFrame);

  // test access to SimHits
  const std::string barrelHitsName("EcalHitsEB");

  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits( new MixCollection<PCaloHit>(crossingFrame.product(), barrelHitsName) );

  std::auto_ptr<EcalTBTDCRawInfo> TDCproduct(new EcalTBTDCRawInfo(1));

  // Step B: Create empty output
  std::auto_ptr<EBDigiCollection> barrelResult(new EBDigiCollection());

  // run the algorithm

  CaloDigiCollectionSorter sorter(5);

  if (doPhaseShift) thisPhaseShift = RandFlat::shoot();
  
   if (doPhaseShift) {
     DetId detId(DetId::Ecal, 1);
     setPhaseShift(detId);

     // fill the TDC information in the event

     fillTBTDCRawInfo(*TDCproduct);

   }
  
  theBarrelDigitizer->run(*barrelHits, *barrelResult);
  edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();
  
  std::vector<EBDataFrame> sortedDigisEB = sorter.sortedVector(*barrelResult);
  LogDebug("EcalDigi") << "Top 10 EB digis";
  for(int i = 0; i < std::min(10,(int) sortedDigisEB.size()); ++i) 
    {
      LogDebug("EcalDigi") << sortedDigisEB[i];
    }
  
  
  // Step D: Put outputs into event
  event.put(barrelResult);
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

  double theGains[theCoder->NGAINS];
  theGains[0] = 1.;
  theGains[1] = defaultRatios->gain6Over1() ;
  theGains[2] = theGains[1]*(defaultRatios->gain12Over6()) ;

  LogDebug("EcalDigi") << " Gains: " << "\n" << " g0 = " << theGains[0] << "\n" << " g1 = " << theGains[1] << "\n" << " g2 = " << theGains[2];

  delete defaultRatios;

  const double EBscale = (agc->getEBValue())*theGains[2]*(theCoder->MAXADC);
  LogDebug("SetupInfo") << " GeV/ADC = " << agc->getEBValue() << "\n" << " saturation for EB = " << EBscale;
  const double EEscale = (agc->getEEValue())*theGains[2]*(theCoder->MAXADC);
  LogDebug("SetupInfo") << " GeV/ADC = " << agc->getEEValue() << "\n" << " saturation for EE = " << EEscale;
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

  LogInfo("EcalDigi") << "deb geometry: " << "\n" 
                       << "\t barrel: " << theBarrelDets.size ();

  theBarrelDigitizer->setDetIds(theBarrelDets);
}


void EcalTBDigiProducer::setPhaseShift(const DetId & detId) {
  
  const CaloSimParameters & parameters = theParameterMap->simParameters(detId);
  if ( !parameters.syncPhase() ) {

    int myDet = detId.subdetId();

    LogDebug("EcalDigi") << "Setting the phase shift " << thisPhaseShift << " for the subdetector " << myDet;

    if ( myDet == 1) {
      theEcalResponse->setPhaseShift(thisPhaseShift);
    }
    
  }
  
}


void EcalTBDigiProducer::fillTBTDCRawInfo(EcalTBTDCRawInfo & theTBTDCRawInfo) {

  unsigned int thisChannel = 1;
  
  // to be taken from ParameterSet, use hardcoded values 
  // for the time being as in 
  // RecTBCalo/EcalTBTDCReconstructor/data/EcalTBTDCReconstructor.cfi

  unsigned int tdcMin = 430;
  unsigned int tdcMax = 958;

  unsigned int thisCount = (unsigned int)(thisPhaseShift*(tdcMax-tdcMin)) + tdcMin;

  EcalTBTDCSample theTBTDCSample(thisChannel, thisCount);

  unsigned int sampleIndex = 0;
  theTBTDCRawInfo.setSample(sampleIndex, theTBTDCSample);

  LogDebug("EcalDigi") << theTBTDCSample << "\n" << theTBTDCRawInfo;

}
