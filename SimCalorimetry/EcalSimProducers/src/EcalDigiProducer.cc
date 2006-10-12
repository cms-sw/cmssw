
#include "SimCalorimetry/EcalSimProducers/interface/EcalDigiProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Provenance.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CLHEP/Matrix/SymMatrix.h"

EcalDigiProducer::EcalDigiProducer(const edm::ParameterSet& params) 
:  theGeometry(0)
{

  /// output collections names

  EBdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
  EEdigiCollection_ = params.getParameter<std::string>("EEdigiCollection");
  ESdigiCollection_ = params.getParameter<std::string>("ESdigiCollection");

  produces<EBDigiCollection>(EBdigiCollection_);
  produces<EEDigiCollection>(EEdigiCollection_);
  produces<ESDigiCollection>(ESdigiCollection_);


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

  theParameterMap = new EcalSimParameterMap(simHitToPhotoelectronsBarrel, simHitToPhotoelectronsEndcap, 
                                            photoelectronsToAnalogBarrel, photoelectronsToAnalogEndcap, 
                                            samplingFactor, timePhase, readoutFrameSize, binOfMaximum,
                                            doPhotostatistics, syncPhase);
  theEcalShape = new EcalShape(timePhase);

  int ESGain = params.getParameter<int>("ESGain");
  theESShape = new ESShape(ESGain);

  theEcalResponse = new CaloHitResponse(theParameterMap, theEcalShape);
  theESResponse = new CaloHitResponse(theParameterMap, theESShape);
  
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

  theCorrNoise = new CaloCorrelatedNoisifier(thisMatrix);

  theCoder = new EcalCoder(addNoise, theCorrNoise);
  bool applyConstantTerm = params.getParameter<bool>("applyConstantTerm");
  double rmsConstantTerm = params.getParameter<double> ("ConstantTerm");
  theElectronicsSim = new EcalElectronicsSim(theParameterMap, theCoder, applyConstantTerm, rmsConstantTerm);

  bool addESNoise = params.getParameter<bool>("doESNoise");
  double ESNoiseSigma = params.getParameter<double> ("ESNoiseSigma");
  int ESBaseline = params.getParameter<int>("ESBaseline");
  double ESMIPADC = params.getParameter<double>("ESMIPADC");
  double ESMIPkeV = params.getParameter<double>("ESMIPkeV");
  theESElectronicsSim = new ESElectronicsSim(addESNoise, ESNoiseSigma, ESGain, ESBaseline, ESMIPADC, ESMIPkeV);

  theBarrelDigitizer = new EBDigitizer(theEcalResponse, theElectronicsSim, addNoise);
  theEndcapDigitizer = new EEDigitizer(theEcalResponse, theElectronicsSim, addNoise);
  theESDigitizer = new ESDigitizer(theESResponse, theESElectronicsSim, addESNoise);

}


EcalDigiProducer::~EcalDigiProducer() 
{
  delete theParameterMap;
  delete theEcalShape;
  delete theESShape;
  delete theEcalResponse;
  delete theESResponse;
  delete theCorrNoise;
  delete theNoiseMatrix;
  delete theCoder;
  delete theElectronicsSim;
  delete theESElectronicsSim;
  delete theBarrelDigitizer;
  delete theEndcapDigitizer;
  delete theESDigitizer;
}


void EcalDigiProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{

  // Step A: Get Inputs

  checkGeometry(eventSetup);
  checkCalibrations(eventSetup);

  // Get input
  edm::Handle<CrossingFrame> crossingFrame;
  event.getByType(crossingFrame);

  // test access to SimHits
  const std::string barrelHitsName("EcalHitsEB");
  const std::string endcapHitsName("EcalHitsEE");
  const std::string preshowerHitsName("EcalHitsES");

  bool isEB = true;
  MixCollection<PCaloHit> * EBHits = 0 ;
  try {
    EBHits = new MixCollection<PCaloHit>(crossingFrame.product(), barrelHitsName);
  } catch ( cms::Exception &e ) { isEB = false; }
  if ( theBarrelDets.size() == 0 ) isEB = false;
  
  //  std::auto_ptr<MixCollection<PCaloHit> > 
  //    barrelHits( new MixCollection<PCaloHit>(crossingFrame.product(), barrelHitsName) );

  bool isEE = true;
  MixCollection<PCaloHit> * EEHits = 0 ;
  try {
    EEHits = new MixCollection<PCaloHit>(crossingFrame.product(), endcapHitsName);
  } catch ( cms::Exception &e ) { isEE = false; }
  if ( theEndcapDets.size() == 0 ) isEE = false;

  //  std::auto_ptr<MixCollection<PCaloHit> > 
  //    endcapHits( new MixCollection<PCaloHit>(crossingFrame.product(),endcapHitsName) );

  bool isES = true;
  MixCollection<PCaloHit> * ESHits = 0 ;
  try {
    ESHits = new MixCollection<PCaloHit>(crossingFrame.product(), preshowerHitsName);
  } catch ( cms::Exception &e ) { isES = false; }
  if ( theESDets.size() == 0 ) isES = false;

  //    std::auto_ptr<MixCollection<PCaloHit> >
  //      preshowerHits( new MixCollection<PCaloHit>(crossingFrame.product(), preshowerHitsName) ); 
  
  // Step B: Create empty output
  std::auto_ptr<EBDigiCollection> barrelResult(new EBDigiCollection());
  std::auto_ptr<EEDigiCollection> endcapResult(new EEDigiCollection());
  std::auto_ptr<ESDigiCollection> preshowerResult(new ESDigiCollection());

  // run the algorithm

  CaloDigiCollectionSorter sorter(5);

  if ( isEB ) {

    std::auto_ptr<MixCollection<PCaloHit> >  barrelHits( EBHits );
    theBarrelDigitizer->run(*barrelHits, *barrelResult);
    edm::LogInfo("DigiInfo") << "EB Digis: " << barrelResult->size();

    std::vector<EBDataFrame> sortedDigisEB = sorter.sortedVector(*barrelResult);
    LogDebug("EcalDigi") << "Top 10 EB digis";
    for(int i = 0; i < std::min(10,(int) sortedDigisEB.size()); ++i) 
      {
        LogDebug("EcalDigi") << sortedDigisEB[i];
      }
  }

  if ( isEE ) {

    std::auto_ptr<MixCollection<PCaloHit> >  endcapHits( EEHits );
    theEndcapDigitizer->run(*endcapHits, *endcapResult);
    edm::LogInfo("EcalDigi") << "EE Digis: " << endcapResult->size();

    std::vector<EEDataFrame> sortedDigisEE = sorter.sortedVector(*endcapResult);
    LogDebug("EcalDigi")  << "Top 10 EE digis";
    for(int i = 0; i < std::min(10,(int) sortedDigisEE.size()); ++i) 
      {
        LogDebug("EcalDigi") << sortedDigisEE[i];
      }
  }

  if ( isES ) {

    std::auto_ptr<MixCollection<PCaloHit> >  preshowerHits( ESHits );
    theESDigitizer->run(*preshowerHits, *preshowerResult);
    edm::LogInfo("EcalDigi") << "ES Digis: " << preshowerResult->size();
    
//   CaloDigiCollectionSorter sorter_es(7);
//   std::vector<ESDataFrame> sortedDigis_es = sorter_es.sortedVector(*preshowerResult);
//   LogDebug("DigiDump") << "List all ES digis";
//   for(int i = 0; i < sortedDigis_es.size(); ++i) 
//     {
//       LogDebug("DigiDump") << sortedDigis_es[i];
//     }
  }

  // Step D: Put outputs into event
  event.put(barrelResult, EBdigiCollection_);
  event.put(endcapResult, EEdigiCollection_);
  event.put(preshowerResult, ESdigiCollection_);

}


void  EcalDigiProducer::checkCalibrations(const edm::EventSetup & eventSetup) 
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

  const double EBscale = (agc->getEBValue())*theGains[1]*(theCoder->MAXADC);
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() << "\n" << " saturation for EB = " << EBscale;
  const double EEscale = (agc->getEEValue())*theGains[1]*(theCoder->MAXADC);
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEEValue() << "\n" << " saturation for EE = " << EEscale;
  theCoder->setFullScaleEnergy( EBscale , EEscale );

}


void EcalDigiProducer::checkGeometry(const edm::EventSetup & eventSetup) 
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


void EcalDigiProducer::updateGeometry() {
  theEcalResponse->setGeometry(theGeometry);
  theESResponse->setGeometry(theGeometry);

  theBarrelDets.clear();
  theEndcapDets.clear();
  theESDets.clear();

  theBarrelDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  theEndcapDets =  theGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  theESDets     =  theGeometry->getValidDetIds(DetId::Ecal, EcalPreshower);

  LogInfo("EcalDigi") << "deb geometry: " << "\n" 
                      << "\t barrel: " << theBarrelDets.size () << "\n"
                      << "\t endcap: " << theEndcapDets.size () << "\n"
                      << "\t preshower: " << theESDets.size();
  
  theBarrelDigitizer->setDetIds(theBarrelDets);
  theEndcapDigitizer->setDetIds(theEndcapDets);
  theESDigitizer->setDetIds(theESDets);
}

