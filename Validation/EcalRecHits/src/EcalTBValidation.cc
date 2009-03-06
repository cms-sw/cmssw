// Ecal H4 tesbeam analysis 
#include "Validation/EcalRecHits/interface/EcalTBValidation.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

#include <iostream>
#include <string>


EcalTBValidation::EcalTBValidation( const edm::ParameterSet& config ) {  

  data_                  = config.getUntrackedParameter<int>("data",-1000);
  xtalInBeam_            = config.getUntrackedParameter<int>("xtalInBeam",-1000);

  digiCollection_        = config.getParameter<std::string>("digiCollection");
  digiProducer_          = config.getParameter<std::string>("digiProducer");
  hitCollection_         = config.getParameter<std::string>("hitCollection");
  hitProducer_           = config.getParameter<std::string>("hitProducer");
  hodoRecInfoCollection_ = config.getParameter<std::string>("hodoRecInfoCollection");
  hodoRecInfoProducer_   = config.getParameter<std::string>("hodoRecInfoProducer");
  tdcRecInfoCollection_  = config.getParameter<std::string>("tdcRecInfoCollection");
  tdcRecInfoProducer_    = config.getParameter<std::string>("tdcRecInfoProducer");
  eventHeaderCollection_ = config.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_   = config.getParameter<std::string>("eventHeaderProducer");

  rootfile_              = config.getUntrackedParameter<std::string>("rootfile","EcalTBValidation.root");
}


EcalTBValidation::~EcalTBValidation(){}

void EcalTBValidation::beginJob(edm::EventSetup const&) {

  h_xib          = new TH2F("h_xib",    "xtal in beam position", 85, 0.,85.,20,0.,20.);
  h_ampltdc      = new TH2F("h_ampltdc","Max Amplitude vs TDC offset", 100,0.,1.,1000, 0., 4000.);
  h_hodoX        = new TH1F("h_hodoX",  "Beam Profile X",100,-20.,20.);
  h_hodoY        = new TH1F("h_hodoY",  "Beam Profile Y",100,-20.,20.);
  h_e1x1         = new TH1F("h_e1x1",        "E1x1 energy", 1000, 0., 4000.);
  h_e3x3         = new TH1F("h_e3x3",        "E3x3 energy", 1000, 0., 4000.);
  h_e5x5         = new TH1F("h_e5x5",        "E5x5 energy", 1000, 0., 4000.);
  h_e1x1_center  = new TH1F("h_e1x1_center", "E1x1 energy", 1000, 0., 4000.);
  h_e3x3_center  = new TH1F("h_e3x3_center", "E3x3 energy", 1000, 0., 4000.);
  h_e5x5_center  = new TH1F("h_e5x5_center", "E5x5 energy", 1000, 0., 4000.);
  h_e1e9         = new TH1F("h_e1e9",        "E1/E9 ratio",  600, 0., 1.2);
  h_e1e25        = new TH1F("h_e1e25",       "E1/E25 ratio", 600, 0., 1.2);
  h_e9e25        = new TH1F("h_e9e25",       "E9/E25 ratio", 600, 0., 1.2);
  h_e1vsX        = new TH2F("h_e1vsX",   "E1 vs X",    80,-20,20,1000,0.,4000.);
  h_e1vsY        = new TH2F("h_e1vsY",   "E1 vs Y",    80,-20,20,1000,0.,4000.);  
  h_e1e9vsX      = new TH2F("h_e1e9vsX", "E1/E9 vs X", 80,-20,20,600,0.,1.2);
  h_e1e9vsY      = new TH2F("h_e1e9vsY", "E1/E9 vs Y", 80,-20,20,600,0.,1.2);
  h_e1e25vsX     = new TH2F("h_e1e25vsX","E1/E25 vs X",80,-20,20,600,0.,1.2);
  h_e1e25vsY     = new TH2F("h_e1e25vsY","E1/E25 vs Y",80,-20,20,600,0.,1.2);
  h_e9e25vsX     = new TH2F("h_e9e25vsX","E9/E25 vs X",80,-20,20,600,0.,1.2);
  h_e9e25vsY     = new TH2F("h_e9e25vsY","E9/E25 vs Y",80,-20,20,600,0.,1.2);
  h_Shape        = new TH2F("h_Shape","Xtal in Beam Shape",250,0,10,350,0,3500);
}

void EcalTBValidation::endJob() {

  TFile f(rootfile_.c_str(),"RECREATE");
  h_xib          -> Write();
  h_ampltdc      -> Write();
  h_hodoX        -> Write();
  h_hodoY        -> Write();
  h_e1x1         -> Write();
  h_e3x3         -> Write();
  h_e5x5         -> Write();
  h_e1x1_center  -> Write();
  h_e3x3_center  -> Write();
  h_e5x5_center  -> Write();
  h_e1e9         -> Write();
  h_e1e25        -> Write();
  h_e9e25        -> Write();
  h_e1vsX        -> Write();
  h_e1vsY        -> Write();
  h_e1e9vsX      -> Write();
  h_e1e9vsY      -> Write();
  h_e1e25vsX     -> Write();
  h_e1e25vsY     -> Write();
  h_e9e25vsX     -> Write();
  h_e9e25vsY     -> Write();
  h_Shape        -> Write();
  f.Close();
}

void EcalTBValidation::analyze( const edm::Event& event, const edm::EventSetup& setup ) {

  using namespace edm;
  using namespace cms;

  // digis
  const EBDigiCollection* theDigis=0;
  Handle<EBDigiCollection> pdigis;
  event.getByLabel(digiProducer_, digiCollection_, pdigis);
  if(pdigis.isValid()){
    theDigis = pdigis.product(); 
  } 
  else {    
    std::cerr << "Error! can't get the product " << digiCollection_.c_str() << std::endl;
    return;
  }

  // rechits
  const EBUncalibratedRecHitCollection* theHits=0;  
  Handle<EBUncalibratedRecHitCollection> phits;
  event.getByLabel(hitProducer_, hitCollection_, phits);
  if(phits.isValid()){
    theHits = phits.product(); 
  } 
  else {
    std::cerr << "Error! can't get the product " << hitCollection_.c_str() << std::endl;
    return;
  }

  // hodoscopes
  const EcalTBHodoscopeRecInfo* theHodo=0;  
  Handle<EcalTBHodoscopeRecInfo> pHodo;
  event.getByLabel(hodoRecInfoProducer_, hodoRecInfoCollection_, pHodo);
  if(pHodo.isValid()){ 
    theHodo = pHodo.product(); 
  }
  else{ 
    std::cerr << "Error! can't get the product " << hodoRecInfoCollection_.c_str() << std::endl;
    return;
  }
  
  // tdc
  const EcalTBTDCRecInfo* theTDC=0;
  Handle<EcalTBTDCRecInfo> pTDC;
  event.getByLabel(tdcRecInfoProducer_, tdcRecInfoCollection_, pTDC);
  if(pTDC.isValid()){
    theTDC = pTDC.product(); 
  }
  else{ 
    std::cerr << "Error! can't get the product " << tdcRecInfoCollection_.c_str() << std::endl;
    return;
  }

  // event header
  const EcalTBEventHeader* evtHeader=0;
  Handle<EcalTBEventHeader> pEventHeader;
  event.getByLabel(eventHeaderProducer_ , pEventHeader);
  if(pEventHeader.isValid()){
    evtHeader = pEventHeader.product(); 
  }
  else{ 
    std::cerr << "Error! can't get the product " << eventHeaderProducer_.c_str() << std::endl;
    return;
  }
  
  
  // -----------------------------------------------------------------------
  // xtal-in-beam
  EBDetId xtalInBeamId(1,xtalInBeam_, EBDetId::SMCRYSTALMODE);
  if (xtalInBeamId==EBDetId(0)){ return; }
  int xibEta = xtalInBeamId.ieta();
  int xibPhi = xtalInBeamId.iphi();

  // skipping events with moving table (in data)
  if (data_ && (evtHeader->tableIsMoving())) return;
  
  // amplitudes
  EBDetId Xtals5x5[25];
  for (unsigned int icry=0;icry<25;icry++){
    unsigned int row    = icry/5;
    unsigned int column = icry%5;
    int ieta = xtalInBeamId.ieta()+column-2;
    int iphi = xtalInBeamId.iphi()+row-2;
    if(EBDetId::validDetId(ieta, iphi)){ 
      EBDetId tempId(ieta, iphi,EBDetId::ETAPHIMODE);
      if (tempId.ism()==1) 
	Xtals5x5[icry] = tempId;
      else
	Xtals5x5[icry] = EBDetId(0);
    } else {
      Xtals5x5[icry] = EBDetId(0);   
    }
  } 
 
  // matrices
  double ampl1x1 = 0.;
  double ampl3x3 = 0.;
  double ampl5x5 = 0.;
  for (unsigned int icry=0;icry<25;icry++) {
    if (!Xtals5x5[icry].null()){
      double theAmpl = (theHits->find(Xtals5x5[icry]))->amplitude();
      ampl5x5 += theAmpl;
      if (icry==12){ampl1x1 = theAmpl;}
      if (icry==6 || icry==7 || icry==8 || icry==11 || icry==12 || icry==13 || icry==16 || icry==17 || icry==18){ampl3x3 += theAmpl;}
    }}


  // pulse shape
  double sampleSave[10];
  for(int ii=0; ii < 10; ++ii){ sampleSave[ii] = 0.0; }
  EBDigiCollection::const_iterator thisDigi = theDigis->find(xtalInBeamId);
  int sMax = -1;
  double eMax = 0.;
  if (thisDigi != theDigis->end()){
    EBDataFrame myDigi = (*thisDigi);
    for (int sample=0; sample < myDigi.size(); ++sample){
      double analogSample = myDigi.sample(sample).adc();
      sampleSave[sample]  = analogSample;
      if ( eMax < analogSample ) {
	eMax = analogSample;
	sMax = sample;
      }
    }
  }
  
  // beam profile
  double xBeam = theHodo->posX();
  double yBeam = theHodo->posY();


  // filling histos
  h_xib      -> Fill(xibEta, xibPhi);
  h_hodoX    -> Fill(xBeam);
  h_hodoY    -> Fill(yBeam);
  h_ampltdc  -> Fill(theTDC->offset(),ampl1x1);
  h_e1x1     -> Fill(ampl1x1);
  h_e3x3     -> Fill(ampl3x3);
  h_e5x5     -> Fill(ampl5x5);
  h_e1e9     -> Fill(ampl1x1/ampl3x3);
  h_e1e25    -> Fill(ampl1x1/ampl5x5);
  h_e9e25    -> Fill(ampl3x3/ampl5x5);
  h_e1vsX    -> Fill(xBeam,ampl1x1);
  h_e1vsY    -> Fill(yBeam,ampl1x1);
  h_e1e9vsX  -> Fill(xBeam,ampl1x1/ampl3x3);
  h_e1e9vsY  -> Fill(yBeam,ampl1x1/ampl3x3);
  h_e1e25vsX -> Fill(xBeam,ampl1x1/ampl5x5);
  h_e1e25vsY -> Fill(yBeam,ampl1x1/ampl5x5);
  h_e9e25vsX -> Fill(xBeam,ampl3x3/ampl5x5);
  h_e9e25vsY -> Fill(yBeam,ampl3x3/ampl5x5);

  for(int ii=0; ii < 10; ++ii){ h_Shape->Fill(double(ii)+theTDC->offset(),sampleSave[ii]); }

  if ( (fabs(xBeam)<2.5) && (fabs(yBeam)<2.5) ){ 
    h_e1x1_center  -> Fill(ampl1x1);
    h_e3x3_center  -> Fill(ampl3x3);
    h_e5x5_center  -> Fill(ampl5x5);
  }
}


