
#include "EcalDigiProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloDigiCollectionSorter.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

EcalDigiProducer::EcalDigiProducer(const edm::ParameterSet& params) 
:  theGeometry(0)
{

  /// output collections names

  EBdigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
  EEdigiCollection_ = params.getParameter<std::string>("EEdigiCollection");
  ESdigiCollection_ = params.getParameter<std::string>("ESdigiCollection");

  hitsProducer_ = params.getParameter<std::string>("hitsProducer");
  
  produces<EBDigiCollection>(EBdigiCollection_);
  produces<EEDigiCollection>(EEdigiCollection_);
  produces<ESDigiCollection>(ESdigiCollection_);


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

  theParameterMap = new EcalSimParameterMap(simHitToPhotoelectronsBarrel, simHitToPhotoelectronsEndcap, 
                                            photoelectronsToAnalogBarrel, photoelectronsToAnalogEndcap, 
                                            samplingFactor, timePhase, readoutFrameSize, binOfMaximum,
                                            doPhotostatistics, syncPhase);

  theEcalShape = new EcalShape(timePhase);
  
  int ESGain = params.getParameter<int>("ESGain");
  theESShape = new ESShape(ESGain);

  theEcalResponse = new CaloHitResponse(theParameterMap, theEcalShape);
  theESResponse = new CaloHitResponse(theParameterMap, theESShape);

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

  doFast = params.getParameter<bool>("doFast");
  bool addESNoise = params.getParameter<bool>("doESNoise");
  double ESNoiseSigma = params.getParameter<double> ("ESNoiseSigma");
  int ESBaseline  = params.getParameter<int>("ESBaseline");
  double ESMIPADC = params.getParameter<double>("ESMIPADC");
  double ESMIPkeV = params.getParameter<double>("ESMIPkeV");
  int numESdetId  = params.getParameter<int>("numESdetId");
  double zsThreshold = params.getParameter<double>("zsThreshold");
  std::string refFile = params.getParameter<std::string>("refHistosFile");

  theESElectronicsSim     = 0;
  theESElectronicsSimFast = 0;
  if (!doFast) { theESElectronicsSim     = new ESElectronicsSim(addESNoise, ESNoiseSigma, ESGain, ESBaseline, ESMIPADC, ESMIPkeV); }
  if ( doFast) { theESElectronicsSimFast = new ESElectronicsSimFast(addESNoise, ESNoiseSigma, ESGain, ESBaseline, ESMIPADC, ESMIPkeV);  }

  theBarrelDigitizer = new EBDigitizer(theEcalResponse, theElectronicsSim, addNoise);
  theEndcapDigitizer = new EEDigitizer(theEcalResponse, theElectronicsSim, addNoise);

  theESDigitizer = 0;
  theESDigitizerFast = 0;
  if (!doFast){ theESDigitizer = new ESDigitizer(theESResponse, theESElectronicsSim, addESNoise); }
  if ( doFast){ theESDigitizerFast = new ESFastTDigitizer(theESResponse, theESElectronicsSimFast, addESNoise, numESdetId, zsThreshold, refFile);}

  // not containment corrections
  EBs25notCont = params.getParameter<double>("EBs25notContainment");
  EEs25notCont = params.getParameter<double>("EEs25notContainment");
}


EcalDigiProducer::~EcalDigiProducer() 
{
  if (theParameterMap)  { delete theParameterMap; }
  if (theEcalShape)     { delete theEcalShape; }
  if (theESShape)       { delete theESShape; }
  if (theEcalResponse)  { delete theEcalResponse; }
  if (theESResponse)    { delete theESResponse; }
  if (theCorrNoise)     { delete theCorrNoise; }
  if (theNoiseMatrix)   { delete theNoiseMatrix; }
  if (theCoder)         { delete theCoder; }
  if (theElectronicsSim){ delete theElectronicsSim; }
  if (theESElectronicsSim)    { delete theESElectronicsSim; }
  if (theESElectronicsSimFast){ delete theESElectronicsSimFast; }
  if (theBarrelDigitizer){ delete theBarrelDigitizer; }
  if (theEndcapDigitizer){ delete theEndcapDigitizer; }
  if (theESDigitizer)    { delete theESDigitizer; }
  if (theESDigitizerFast){ delete theESDigitizerFast; }
}


void EcalDigiProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) 
{

  // Step A: Get Inputs

  checkGeometry(eventSetup);
  checkCalibrations(eventSetup);

  // Get input
  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;

  // test access to SimHits
  const std::string barrelHitsName(hitsProducer_+"EcalHitsEB");
  const std::string endcapHitsName(hitsProducer_+"EcalHitsEE");
  const std::string preshowerHitsName(hitsProducer_+"EcalHitsES");

  bool isEB = true;
  event.getByLabel("mix",barrelHitsName,crossingFrame);
  MixCollection<PCaloHit> * EBHits = 0 ;
  if (crossingFrame.isValid()) { 
    EBHits = new MixCollection<PCaloHit>(crossingFrame.product());
  }
  else { 
    edm::LogError("EcalDigiProducer") << "Error! can't get the product " << barrelHitsName.c_str() ;
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
    edm::LogError("EcalDigiProducer") << "Error! can't get the product " << endcapHitsName.c_str() ;
    isEE = false;
  }
  if ( ! EEHits->inRegistry() || theEndcapDets.size() == 0 ) isEE = false;

  bool isES = true;
  event.getByLabel("mix",preshowerHitsName,crossingFrame);
  MixCollection<PCaloHit> * ESHits = 0 ;
  if (crossingFrame.isValid()) {
    ESHits = new MixCollection<PCaloHit>(crossingFrame.product());
  }
  else { 
    edm::LogError("EcalDigiProducer") << "Error! can't get the product " << preshowerHitsName.c_str() ;
    isES = false; 
  }    
  if ( ! ESHits->inRegistry() || theESDets.size() == 0 ) isES = false;

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

  const double EBscale = (agc->getEBValue())*theGains[1]*(theCoder->MAXADC)*EBs25notCont;
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEBValue() << "\n" << " notCont = " << EBs25notCont << "\n" << " saturation for EB = " << EBscale << ", " << EBs25notCont;
  const double EEscale = (agc->getEEValue())*theGains[1]*(theCoder->MAXADC)*EEs25notCont;
  LogDebug("EcalDigi") << " GeV/ADC = " << agc->getEEValue() << "\n" << " notCont = " << EEs25notCont << "\n" << " saturation for EB = " << EEscale << ", " << EEs25notCont;
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

  edm::LogInfo("EcalDigi") << "deb geometry: " << "\n" 
                      << "\t barrel: " << theBarrelDets.size () << "\n"
                      << "\t endcap: " << theEndcapDets.size () << "\n"
                      << "\t preshower: " << theESDets.size();
  
  theBarrelDigitizer->setDetIds(theBarrelDets);
  theEndcapDigitizer->setDetIds(theEndcapDets);
  if (!doFast) { theESDigitizer->setDetIds(theESDets); }
  if ( doFast) { theESDigitizerFast->setDetIds(theESDets); }
}

