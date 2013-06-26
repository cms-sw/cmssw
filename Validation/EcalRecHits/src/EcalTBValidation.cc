// Ecal H4 tesbeam analysis 
#include "Validation/EcalRecHits/interface/EcalTBValidation.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


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

  //rootfile_              = config.getUntrackedParameter<std::string>("rootfile","EcalTBValidation.root");

  // verbosity...
  verbose_ = config.getUntrackedParameter<bool>("verbose", false);

  dbe_ = edm::Service<DQMStore>().operator->();
  if( dbe_ ) {
    if( verbose_ ) {
      dbe_->setVerbose(1);
      dbe_->showDirStructure();
    }
    else {
      dbe_->setVerbose(0);
    }
  }
    
  meETBxib_ = 0;
  meETBampltdc_ = 0;
  meETBShape_ = 0;
  meETBhodoX_ = 0;
  meETBhodoY_ = 0;
  meETBe1x1_ = 0;
  meETBe3x3_ = 0;
  meETBe5x5_ = 0;
  meETBe1e9_ = 0;
  meETBe1e25_ = 0;
  meETBe9e25_ = 0;
  meETBe1x1_center_ = 0;
  meETBe3x3_center_ = 0;
  meETBe5x5_center_ = 0;
  meETBe1vsX_ = 0;
  meETBe1vsY_ = 0;
  meETBe1e9vsX_ = 0;
  meETBe1e9vsY_ = 0;
  meETBe1e25vsX_ = 0;
  meETBe1e25vsY_ = 0;
  meETBe9e25vsX_ = 0;
  meETBe9e25vsY_ = 0;

  if( dbe_ ) {

    std::string hname;   
    dbe_->setCurrentFolder( "EcalRecHitsV/EcalTBValidationTask" );

    hname = "xtal in beam position";
    meETBxib_          = dbe_->book2D( hname, hname, 85, 0., 85., 20,0., 20. );
    hname = "Max Amplitude vs TDC offset";
    meETBampltdc_      = dbe_->book2D( hname, hname, 100, 0., 1., 1000, 0., 4000. );
    hname = "Beam Profile X";
    meETBhodoX_        = dbe_->book1D( hname, hname, 100, -20., 20. );
    hname = "Beam Profile Y";
    meETBhodoY_        = dbe_->book1D( hname, hname, 100, -20., 20. );
    hname = "E1x1 energy";
    meETBe1x1_         = dbe_->book1D( hname, hname, 1000, 0., 4000. );
    hname = "E3x3 energy";
    meETBe3x3_         = dbe_->book1D( hname, hname, 1000, 0., 4000. );
    hname = "E5x5 energy";
    meETBe5x5_         = dbe_->book1D( hname, hname, 1000, 0., 4000. );
    hname = "E1x1 energy center";
    meETBe1x1_center_  = dbe_->book1D( hname, hname, 1000, 0., 4000. );
    hname = "E3x3 energy center";
    meETBe3x3_center_  = dbe_->book1D( hname, hname, 1000, 0., 4000. );
    hname = "E5x5 energy center";
    meETBe5x5_center_  = dbe_->book1D( hname, hname, 1000, 0., 4000. );
    hname = "E1 over E9 ratio";
    meETBe1e9_         = dbe_->book1D( hname, hname, 600, 0., 1.2 );
    hname = "E1 over E25 ratio";
    meETBe1e25_        = dbe_->book1D( hname, hname, 600, 0., 1.2 );
    hname = "E9 over E25 ratio";
    meETBe9e25_        = dbe_->book1D( hname, hname, 600, 0., 1.2 );
    hname = "E1 vs X";
    meETBe1vsX_        = dbe_->book2D( hname, hname, 80, -20, 20, 1000, 0., 4000. );
    hname = "E1 vs Y";
    meETBe1vsY_        = dbe_->book2D( hname, hname, 80, -20, 20, 1000, 0., 4000. );  
    hname = "E1 over E9 vs X";
    meETBe1e9vsX_      = dbe_->book2D( hname, hname, 80, -20, 20, 600, 0., 1.2 );
    hname = "E1 over E9 vs Y";
    meETBe1e9vsY_      = dbe_->book2D( hname, hname, 80, -20, 20, 600, 0., 1.2 );
    hname = "E1 over E25 vs X";
    meETBe1e25vsX_     = dbe_->book2D( hname, hname, 80, -20, 20, 600, 0., 1.2 );
    hname = "E1 over E25 vs Y";
    meETBe1e25vsY_     = dbe_->book2D( hname, hname, 80, -20, 20, 600, 0., 1.2 );
    hname = "E9 over E25 vs X";
    meETBe9e25vsX_     = dbe_->book2D( hname, hname, 80, -20, 20, 600, 0., 1.2 );
    hname = "E9 over E25 vs Y";
    meETBe9e25vsY_     = dbe_->book2D( hname, hname, 80, -20, 20, 600, 0., 1.2 );
    hname = "Xtal in Beam Shape";
    meETBShape_        = dbe_->book2D( hname, hname, 250, 0, 10, 350, 0, 3500 );
  }

}


EcalTBValidation::~EcalTBValidation(){}

void EcalTBValidation::beginJob() {}

void EcalTBValidation::endJob() {}

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
  // int sMax = -1; // UNUSED
  double eMax = 0.;
  if (thisDigi != theDigis->end()){
    EBDataFrame myDigi = (*thisDigi);
    for (int sample=0; sample < myDigi.size(); ++sample){
      double analogSample = myDigi.sample(sample).adc();
      sampleSave[sample]  = analogSample;
      if ( eMax < analogSample ) {
	eMax = analogSample;
	// sMax = sample; // UNUSED
      }
    }
  }
  
  // beam profile
  double xBeam = theHodo->posX();
  double yBeam = theHodo->posY();

  // filling histos
  meETBxib_      -> Fill(xibEta, xibPhi);
  meETBhodoX_    -> Fill(xBeam);
  meETBhodoY_    -> Fill(yBeam);
  meETBampltdc_  -> Fill(theTDC->offset(),ampl1x1);
  meETBe1x1_     -> Fill(ampl1x1);
  meETBe3x3_     -> Fill(ampl3x3);
  meETBe5x5_     -> Fill(ampl5x5);
  meETBe1e9_     -> Fill(ampl1x1/ampl3x3);
  meETBe1e25_    -> Fill(ampl1x1/ampl5x5);
  meETBe9e25_    -> Fill(ampl3x3/ampl5x5);
  meETBe1vsX_    -> Fill(xBeam,ampl1x1);
  meETBe1vsY_    -> Fill(yBeam,ampl1x1);
  meETBe1e9vsX_  -> Fill(xBeam,ampl1x1/ampl3x3);
  meETBe1e9vsY_  -> Fill(yBeam,ampl1x1/ampl3x3);
  meETBe1e25vsX_ -> Fill(xBeam,ampl1x1/ampl5x5);
  meETBe1e25vsY_ -> Fill(yBeam,ampl1x1/ampl5x5);
  meETBe9e25vsX_ -> Fill(xBeam,ampl3x3/ampl5x5);
  meETBe9e25vsY_ -> Fill(yBeam,ampl3x3/ampl5x5);

  for(int ii=0; ii < 10; ++ii){ meETBShape_->Fill(double(ii)+theTDC->offset(),sampleSave[ii]); }

  if ( (fabs(xBeam)<2.5) && (fabs(yBeam)<2.5) ){ 
    meETBe1x1_center_  -> Fill(ampl1x1);
    meETBe3x3_center_  -> Fill(ampl3x3);
    meETBe5x5_center_  -> Fill(ampl5x5);
  }
}


