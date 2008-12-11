/*
 * \file EcalRecHitsValidation.cc
 *
 * $Date: 2008/12/10 10:46:58 $
 * \author C. Rovelli
 *
*/

#include <Validation/EcalRecHits/interface/EcalRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <string>

using namespace cms;
using namespace edm;
using namespace std;

EcalRecHitsValidation::EcalRecHitsValidation(const ParameterSet& ps){

  // ---------------------- 
  HepMCLabel                 = ps.getParameter<std::string>("moduleLabelMC"); 
  hitsProducer_              = ps.getParameter<std::string>("hitsProducer");
  EBrechitCollection_        = ps.getParameter<edm::InputTag>("EBrechitCollection");
  EErechitCollection_        = ps.getParameter<edm::InputTag>("EErechitCollection");
  ESrechitCollection_        = ps.getParameter<edm::InputTag>("ESrechitCollection");
  EBuncalibrechitCollection_ = ps.getParameter<edm::InputTag>("EBuncalibrechitCollection");
  EEuncalibrechitCollection_ = ps.getParameter<edm::InputTag>("EEuncalibrechitCollection");
  
  // ---------------------- 
  // DQM ROOT output 
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "");

  if ( outputFile_.size() != 0 ) {
    LogInfo("OutputInfo") << " Ecal RecHits Task histograms will be saved to '" << outputFile_.c_str() << "'";
  } else {
    LogInfo("OutputInfo") << " Ecal RecHits Task histograms will NOT be saved";
  }
 
  // ---------------------- 
  // verbosity switch 
  verbose_ = ps.getUntrackedParameter<bool>("verbose", false);
  
  // ----------------------                 
  // get hold of back-end interface 
  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();                   
  if ( dbe_ ) {
    if ( verbose_ ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }                                                                  
  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }


  // ----------------------   
  meGunEnergy_               = 0;
  meGunEta_                  = 0;   
  meGunPhi_                  = 0;   
  meEBRecHitSimHitRatio_     = 0;
  meEERecHitSimHitRatio_     = 0;
  meESRecHitSimHitRatio_     = 0;
  meEBRecHitSimHitRatioGt35_ = 0;
  meEERecHitSimHitRatioGt35_ = 0;
  meEBUnRecHitSimHitRatio_     = 0;
  meEEUnRecHitSimHitRatio_     = 0;
  meEBUnRecHitSimHitRatioGt35_ = 0;
  meEEUnRecHitSimHitRatioGt35_ = 0;
  meEBe5x5_                    = 0;
  meEBe5x5OverSimHits_         = 0;
  meEBe5x5OverGun_             = 0;
  meEEe5x5_                    = 0;
  meEEe5x5OverSimHits_         = 0;
  meEEe5x5OverGun_             = 0;
  
  meEBRecHitLog10Energy_         = 0;
  meEERecHitLog10Energy_         = 0;
  meESRecHitLog10Energy_         = 0;
  meEBRecHitLog10EnergyContr_    = 0;
  meEERecHitLog10EnergyContr_    = 0;
  meESRecHitLog10EnergyContr_    = 0;
  meEBRecHitLog10Energy5x5Contr_ = 0;
  meEERecHitLog10Energy5x5Contr_ = 0;
  
  // ---------------------- 
  std::string histo;
   
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalRecHitsV/EcalRecHitsTask");
    
    histo = "EcalRecHitsTask Gun Momentum";
    meGunEnergy_ = dbe_->book1D(histo.c_str(), histo.c_str(), 100, 0., 1000.);
  
    histo = "EcalRecHitsTask Gun Eta";      
    meGunEta_ = dbe_->book1D(histo.c_str(), histo.c_str(), 700, -3.5, 3.5);
     
    histo = "EcalRecHitsTask Gun Phi";  
    meGunPhi_ = dbe_->book1D(histo.c_str(), histo.c_str(), 360, 0., 360.);    
    
    histo = "EcalRecHitsTask Barrel RecSimHit Ratio";  
    meEBRecHitSimHitRatio_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0., 2.);   

    histo = "EcalRecHitsTask Endcap RecSimHit Ratio"; 
    meEERecHitSimHitRatio_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0., 2.);

    histo = "EcalRecHitsTask Preshower RecSimHit Ratio"; 
    meESRecHitSimHitRatio_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0., 2.);

    histo = "EcalRecHitsTask Barrel RecSimHit Ratio gt 3p5 GeV";  
    meEBRecHitSimHitRatioGt35_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);   

    histo = "EcalRecHitsTask Endcap RecSimHit Ratio gt 3p5 GeV"; 
    meEERecHitSimHitRatioGt35_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);

    histo = "EcalRecHitsTask Barrel Unc RecSimHit Ratio";  
    meEBUnRecHitSimHitRatio_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0., 2.);   

    histo = "EcalRecHitsTask Endcap Unc RecSimHit Ratio"; 
    meEEUnRecHitSimHitRatio_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0., 2.);

    histo = "EcalRecHitsTask Barrel Unc RecSimHit Ratio gt 3p5 GeV";  
    meEBUnRecHitSimHitRatioGt35_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);   

    histo = "EcalRecHitsTask Endcap Unc RecSimHit Ratio gt 3p5 GeV"; 
    meEEUnRecHitSimHitRatioGt35_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);

    histo = "EcalRecHitsTask Barrel Rec E5x5";
    meEBe5x5_ = dbe_->book1D(histo.c_str(), histo.c_str(), 4000, 0., 400.);

    histo = "EcalRecHitsTask Barrel Rec E5x5 over Sim E5x5";
    meEBe5x5OverSimHits_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);

    histo = "EcalRecHitsTask Barrel Rec E5x5 over gun energy";
    meEBe5x5OverGun_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);

    histo = "EcalRecHitsTask Endcap Rec E5x5";
    meEEe5x5_ = dbe_->book1D(histo.c_str(), histo.c_str(), 4000, 0., 400.);

    histo = "EcalRecHitsTask Endcap Rec E5x5 over Sim E5x5";
    meEEe5x5OverSimHits_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);

    histo = "EcalRecHitsTask Endcap Rec E5x5 over gun energy";
    meEEe5x5OverGun_ = dbe_->book1D(histo.c_str(), histo.c_str(), 80, 0.9, 1.1);

    meEBRecHitLog10Energy_ = dbe_->book1D( "EcalRecHitsTask Barrel Log10 Energy", "EcalRecHitsTask Barrel Log10 Energy", 90, -5., 4. ); 
    meEERecHitLog10Energy_ = dbe_->book1D( "EcalRecHitsTask Endcap Log10 Energy", "EcalRecHitsTask Endcap Log10 Energy", 90, -5., 4. ); 
    meESRecHitLog10Energy_ = dbe_->book1D( "EcalRecHitsTask Preshower Log10 Energy", "EcalRecHitsTask Preshower Log10 Energy", 90, -5., 4. ); 
    meEBRecHitLog10EnergyContr_ = dbe_->bookProfile( "EcalRecHits Barrel Log10En vs Hit Contribution", "EcalRecHits Barrel Log10En vs Hit Contribution", 90, -5., 4., 100, 0., 1. ); 
    meEERecHitLog10EnergyContr_ = dbe_->bookProfile( "EcalRecHits Endcap Log10En vs Hit Contribution", "EcalRecHits Endcap Log10En vs Hit Contribution", 90, -5., 4., 100, 0., 1. ); 
    meESRecHitLog10EnergyContr_ = dbe_->bookProfile( "EcalRecHits Preshower Log10En vs Hit Contribution", "EcalRecHits Preshower Log10En vs Hit Contribution", 90, -5., 4., 100, 0., 1. ); 
    meEBRecHitLog10Energy5x5Contr_ = dbe_->bookProfile( "EcalRecHits Barrel Log10En5x5 vs Hit Contribution", "EcalRecHits Barrel Log10En5x5 vs Hit Contribution", 90, -5., 4., 100, 0., 1. ); 
    meEERecHitLog10Energy5x5Contr_ = dbe_->bookProfile( "EcalRecHits Endcap Log10En5x5 vs Hit Contribution", "EcalRecHits Endcap Log10En5x5 vs Hit Contribution", 90, -5., 4., 100, 0., 1. ); 

  }
}

EcalRecHitsValidation::~EcalRecHitsValidation(){   
  
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);  
}

void EcalRecHitsValidation::beginJob(const EventSetup& c){  

}

void EcalRecHitsValidation::endJob(){

}

void EcalRecHitsValidation::analyze(const Event& e, const EventSetup& c){

  //Temporary stuff


  
  LogInfo("EcalRecHitsTask, EventInfo: ") << " Run = " << e.id().run() << " Event = " << e.id().event();

  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  c.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  const double barrelADCtoGeV_  = agc->getEBValue();
  const double endcapADCtoGeV_ = agc->getEEValue();
  
  Handle<HepMCProduct> MCEvt;                   
  bool skipMC = false;
  e.getByLabel(HepMCLabel, MCEvt);  
  if (!MCEvt.isValid()) { skipMC = true; }

  edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;

  bool skipBarrel = false;
  const EBUncalibratedRecHitCollection *EBUncalibRecHit =0;
  Handle< EBUncalibratedRecHitCollection > EcalUncalibRecHitEB;
  e.getByLabel( EBuncalibrechitCollection_, EcalUncalibRecHitEB);
  if (EcalUncalibRecHitEB.isValid()) {
    EBUncalibRecHit = EcalUncalibRecHitEB.product() ;    
  } else {
    skipBarrel = true;
  }

  bool skipEndcap = false;
  const EEUncalibratedRecHitCollection *EEUncalibRecHit = 0;
  Handle< EEUncalibratedRecHitCollection > EcalUncalibRecHitEE;
  e.getByLabel( EEuncalibrechitCollection_, EcalUncalibRecHitEE);
  if (EcalUncalibRecHitEE.isValid()){ 
    EEUncalibRecHit = EcalUncalibRecHitEE.product () ;
  } else {
    skipEndcap = true;
  }

  const EBRecHitCollection *EBRecHit = 0;
  Handle<EBRecHitCollection> EcalRecHitEB;
  e.getByLabel( EBrechitCollection_, EcalRecHitEB);
  if (EcalRecHitEB.isValid()){ 
    EBRecHit = EcalRecHitEB.product();
  } else {
    skipBarrel = true;
  }

  const EERecHitCollection *EERecHit = 0;
  Handle<EERecHitCollection> EcalRecHitEE;
  e.getByLabel( EErechitCollection_, EcalRecHitEE);
  if (EcalRecHitEE.isValid()){
    EERecHit = EcalRecHitEE.product ();
  } else {
    skipEndcap = true;
  }

  bool skipPreshower = false;
  const ESRecHitCollection *ESRecHit = 0;
  Handle<ESRecHitCollection> EcalRecHitES;
  e.getByLabel( ESrechitCollection_, EcalRecHitES);
  if (EcalRecHitES.isValid()) {
    ESRecHit = EcalRecHitES.product ();      
  } else {
    skipPreshower = true;
  }


  // ---------------------- 
  // gun
  double eGun = 0.;
  if ( ! skipMC ) {
    for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin(); p != MCEvt->GetEvent()->particles_end(); ++p )  {      
      double htheta = (*p)->momentum().theta();
      double heta = -log(tan(htheta * 0.5));
      double hphi = (*p)->momentum().phi();
      hphi = (hphi>=0) ? hphi : hphi+2*M_PI;
      hphi = hphi / M_PI * 180.;
      
      LogDebug("EventInfo") << "EcalRecHitsTask: Particle gun type form MC = " << abs((*p)->pdg_id()) 
			    << "\n" << "Energy = "<< (*p)->momentum().e() 
			    << "\n" << "Eta = "   << heta 
			    << "\n" << "Phi = "   << hphi;
      
      if ( (*p)->momentum().e() > eGun ) eGun = (*p)->momentum().e();

      if (meGunEnergy_) meGunEnergy_->Fill((*p)->momentum().e());
      if (meGunEta_)    meGunEta_   ->Fill(heta);
      if (meGunPhi_)    meGunPhi_   ->Fill(hphi); 
    }
  }

  // -------------------------------------------------------------------
  // BARREL

  if ( ! skipBarrel) {

    // 1) loop over simHits  
    const std::string barrelHitsName(hitsProducer_+"EcalHitsEB");
    e.getByLabel("mix",barrelHitsName,crossingFrame);
    std::auto_ptr<MixCollection<PCaloHit> > 
      barrelHits (new MixCollection<PCaloHit>(crossingFrame.product ()));
    
    MapType ebSimMap;
    MapType ebRecMap;
    const int ebcSize = 90;
    double ebcontr[ebcSize];
    double ebcontr25[ebcSize];
    double ebtotal = 0.;

    for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin (); hitItr != barrelHits->end (); ++hitItr)  {   
      EBDetId ebid = EBDetId(hitItr->id());
      
      LogDebug("SimHitInfo, barrel") 
	<< "CaloHit "   << hitItr->getName() << " DetID = " << hitItr->id()   << "\n"	
	<< "Energy = "  << hitItr->energy()  << " Time = "  << hitItr->time() << "\n"
	<< "EBDetId = " << ebid.ieta()       << " "         << ebid.iphi();
      
      uint32_t crystid = ebid.rawId();
      ebSimMap[crystid] += hitItr->energy();
    }
    
    
    
    // 2) loop over RecHits 
    for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin(); uncalibRecHit != EBUncalibRecHit->end() ; ++uncalibRecHit) {
      EBDetId EBid = EBDetId(uncalibRecHit->id());
      
      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EBRecHit->find(EBid);
      ebRecMap[EBid.rawId()] += myRecHit->energy();
      
      // Fill log10(Energy) stuff...   
      ebtotal += myRecHit->energy();
      if( myRecHit->energy() > 0 ) meEBRecHitLog10Energy_->Fill( log10( myRecHit->energy() ) );
      int log10i = int( ( log10( myRecHit->energy() ) + 5. ) * 10. );
      if( log10i >=0 and log10i < ebcSize ) ebcontr[ log10i ] += myRecHit->energy();
    
      // comparison Rec/Sim hit
      if ( ebSimMap[EBid.rawId()] != 0. ) {
	double uncEnergy = uncalibRecHit->amplitude()*barrelADCtoGeV_;
	if (meEBUnRecHitSimHitRatio_)                                {meEBUnRecHitSimHitRatio_    ->Fill(uncEnergy/ebSimMap[EBid.rawId()]);}
	if (meEBUnRecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEBUnRecHitSimHitRatioGt35_->Fill(uncEnergy/ebSimMap[EBid.rawId()]);}
      }
      
      if (myRecHit != EBRecHit->end()) {
	if ( ebSimMap[EBid.rawId()] != 0. ) {
	  if (meEBRecHitSimHitRatio_)                                {meEBRecHitSimHitRatio_    ->Fill(myRecHit->energy()/ebSimMap[EBid.rawId()]);}
	  if (meEBRecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEBRecHitSimHitRatioGt35_->Fill(myRecHit->energy()/ebSimMap[EBid.rawId()]);}
	}
      }
      else
	continue;
    }  // loop over the UncalibratedRecHitCollection
    
    // RecHits matrix
    uint32_t  ebcenterid = getUnitWithMaxEnergy(ebRecMap);
    EBDetId myEBid(ebcenterid);
    int bx = myEBid.ietaAbs();
    int by = myEBid.iphi();
    int bz = myEBid.zside();
    findBarrelMatrix(5,5,bx,by,bz,ebRecMap);
    double e5x5rec = 0.;
    double e5x5sim = 0.;
    for ( unsigned int i = 0; i < crystalMatrix.size(); i++ ) {
      e5x5rec += ebRecMap[crystalMatrix[i]];
      e5x5sim += ebSimMap[crystalMatrix[i]];
      int log10i25 = int( ( log10( ebRecMap[crystalMatrix[i]] ) + 5. ) * 10. );
      if( log10i25 >=0 && log10i25 < ebcSize ) ebcontr25[ log10i25 ] += ebRecMap[crystalMatrix[i]];
    }
    
    meEBe5x5_->Fill(e5x5rec);
    if ( e5x5sim > 0. ) meEBe5x5OverSimHits_->Fill(e5x5rec/e5x5sim);
    if ( eGun > 0. ) meEBe5x5OverGun_->Fill(e5x5rec/eGun);
    

    if( meEBRecHitLog10EnergyContr_  && ebtotal != 0 ) {
      for( int i=0; i<ebcSize; i++ ) {
	meEBRecHitLog10EnergyContr_->Fill( -5.+(float(i)+0.5)/10., ebcontr[i]/ebtotal );
      }
    }
    
    if( meEBRecHitLog10Energy5x5Contr_  && e5x5rec != 0 ) {
      for( int i=0; i<ebcSize; i++ ) {
	meEBRecHitLog10Energy5x5Contr_->Fill( -5.+(float(i)+0.5)/10., ebcontr25[i]/e5x5rec );
      }
    }
  }

  // -------------------------------------------------------------------
  // ENDCAP

  if ( ! skipEndcap ) {

    // 1) loop over simHits
    const std::string endcapHitsName(hitsProducer_+"EcalHitsEE");
    e.getByLabel("mix",endcapHitsName,crossingFrame);
    std::auto_ptr<MixCollection<PCaloHit> > 
      endcapHits (new MixCollection<PCaloHit>(crossingFrame.product ()));
  
    MapType eeSimMap;
    MapType eeRecMap;
    const int eecSize = 90;
    double eecontr[eecSize];
    double eecontr25[eecSize];
    double eetotal = 0.;
 
    for (MixCollection<PCaloHit>::MixItr hitItr = endcapHits->begin(); hitItr != endcapHits->end(); ++hitItr) {   
      EEDetId eeid = EEDetId(hitItr->id()) ;
      
      LogDebug("Endcap, HitInfo")
	<<" CaloHit "      << hitItr->getName() << " DetID = "        << hitItr->id()   << "\n"
	<< "Energy = "     << hitItr->energy()  << " Time = "         << hitItr->time() << "\n"
	<< "EEDetId side " << eeid.zside()      << " = " << eeid.ix() << " " << eeid.iy();
      
      uint32_t crystid = eeid.rawId();
      eeSimMap[crystid] += hitItr->energy();
    }



    // 2) loop over RecHits
    for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EEUncalibRecHit->begin(); uncalibRecHit != EEUncalibRecHit->end(); ++uncalibRecHit) {
      EEDetId EEid = EEDetId(uncalibRecHit->id());
      
      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EERecHit->find(EEid);
      eeRecMap[EEid.rawId()] += myRecHit->energy();

      // Fill log10(Energy) stuff...
      eetotal += myRecHit->energy();   
      if( myRecHit->energy() > 0 ) meEERecHitLog10Energy_->Fill( log10( myRecHit->energy() ) );

      int log10i = int( ( log10( myRecHit->energy() ) + 5. ) * 10. );
      if( log10i >=0 and log10i < eecSize ) eecontr[ log10i ] += myRecHit->energy();

      // comparison Rec/Sim hit
      if ( eeSimMap[EEid.rawId()] != 0. ) {
	double uncEnergy = uncalibRecHit->amplitude()*endcapADCtoGeV_;
	if (meEEUnRecHitSimHitRatio_)                                {meEEUnRecHitSimHitRatio_    ->Fill(uncEnergy/eeSimMap[EEid.rawId()]);}
	if (meEEUnRecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEEUnRecHitSimHitRatioGt35_->Fill(uncEnergy/eeSimMap[EEid.rawId()]);}
      }

      if (myRecHit != EERecHit->end()) {
	if ( eeSimMap[EEid.rawId()] != 0. ) {
	  if (meEERecHitSimHitRatio_)                                {meEERecHitSimHitRatio_    ->Fill(myRecHit->energy()/eeSimMap[EEid.rawId()]); }
	  if (meEERecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEERecHitSimHitRatioGt35_->Fill(myRecHit->energy()/eeSimMap[EEid.rawId()]); }
	}
      }
      else
	continue;
    }  // loop over the UncalibratedechitCollection
  
    // RecHits matrix
    uint32_t  eecenterid = getUnitWithMaxEnergy(eeRecMap);
    EEDetId myEEid(eecenterid);
    int bx = myEEid.ix();
    int by = myEEid.iy();
    int bz = myEEid.zside();
    findEndcapMatrix(5,5,bx,by,bz,eeRecMap);
    double e5x5rec = 0.;
    double e5x5sim = 0.;
    for ( unsigned int i = 0; i < crystalMatrix.size(); i++ ) {
      e5x5rec += eeRecMap[crystalMatrix[i]];
      e5x5sim += eeSimMap[crystalMatrix[i]];
      int log10i25 = int( ( log10( eeRecMap[crystalMatrix[i]] ) + 5. ) * 10. );
      if( log10i25 >=0 && log10i25 < eecSize ) eecontr25[ log10i25 ] += eeRecMap[crystalMatrix[i]];
    }
    
    meEEe5x5_->Fill(e5x5rec);
    if ( e5x5sim > 0. ) meEEe5x5OverSimHits_->Fill(e5x5rec/e5x5sim);
    if ( eGun > 0. ) meEEe5x5OverGun_->Fill(e5x5rec/eGun);
    

    if( meEERecHitLog10EnergyContr_  && eetotal != 0 ) {
      for( int i=0; i<eecSize; i++ ) {
	meEERecHitLog10EnergyContr_->Fill( -5.+(float(i)+0.5)/10., eecontr[i]/eetotal );
      }
    }
    
    if( meEERecHitLog10Energy5x5Contr_  && e5x5rec != 0 ) {
      for( int i=0; i<eecSize; i++ ) {
	meEERecHitLog10Energy5x5Contr_->Fill( -5.+(float(i)+0.5)/10., eecontr25[i]/e5x5rec );
      }
    }

  }

  // -------------------------------------------------------------------
  // PRESHOWER

  if ( ! skipPreshower ) {

    // 1) loop over simHits
    const std::string preshowerHitsName(hitsProducer_+"EcalHitsES");
    e.getByLabel("mix",preshowerHitsName,crossingFrame);
    std::auto_ptr<MixCollection<PCaloHit> > 
      preshowerHits (new MixCollection<PCaloHit>(crossingFrame.product ()));

    MapType esSimMap;
    const int escSize = 90;
    double escontr[escSize];
    double estotal = 0.;

  
    for (MixCollection<PCaloHit>::MixItr hitItr = preshowerHits->begin(); hitItr != preshowerHits->end(); ++hitItr) {   
      ESDetId esid = ESDetId(hitItr->id()) ;

      LogDebug("Preshower, HitInfo")
	<<" CaloHit "       << hitItr->getName() << " DetID = "         << hitItr->id()   << "\n"
	<< "Energy = "      << hitItr->energy()  << " Time = "          << hitItr->time() << "\n"
	<< "ESDetId strip " << esid.strip()      << " = " << esid.six() << " " << esid.siy();
      
      uint32_t crystid = esid.rawId();
      esSimMap[crystid] += hitItr->energy();
    }


    // 2) loop over RecHits
    for (EcalRecHitCollection::const_iterator recHit = ESRecHit->begin(); recHit != ESRecHit->end(); ++recHit) {
      ESDetId ESid = ESDetId(recHit->id());
      if ( esSimMap[ESid.rawId()] != 0. ) { 
	
	// Fill log10(Energy) stuff...
	estotal += recHit->energy();   
	if( recHit->energy() > 0 ) meESRecHitLog10Energy_->Fill( log10( recHit->energy() ) );

	int log10i = int( ( log10( recHit->energy() ) + 5. ) * 10. );
	if( log10i >=0 and log10i < escSize ) escontr[ log10i ] += recHit->energy();

	if (meESRecHitSimHitRatio_) { 
	  meESRecHitSimHitRatio_ ->Fill(recHit->energy()/esSimMap[ESid.rawId()]); 
	}
      }
      else
	continue;
    }  // loop over the RechitCollection

    if( meESRecHitLog10EnergyContr_  && estotal != 0 ) {
      for( int i=0; i<escSize; i++ ) {
	meESRecHitLog10EnergyContr_->Fill( -5.+(float(i)+0.5)/10., escontr[i]/estotal );
      }
    }

    
  }
  
}

  
uint32_t EcalRecHitsValidation::getUnitWithMaxEnergy(MapType& themap) {
  
  //look for max
  uint32_t unitWithMaxEnergy = 0;
  float    maxEnergy = 0.;
  
  MapType::iterator iter;
  for (iter = themap.begin(); iter != themap.end(); iter++) {
    
    if (maxEnergy < (*iter).second) {
      maxEnergy = (*iter).second;       
      unitWithMaxEnergy = (*iter).first;
    }                           
  }
  
  return unitWithMaxEnergy;
}

void EcalRecHitsValidation::findBarrelMatrix(int nCellInEta, int nCellInPhi,
                                             int CentralEta, int CentralPhi,int CentralZ,
                                             MapType& themap) {
  
  int goBackInEta = nCellInEta/2;
  int goBackInPhi = nCellInPhi/2;
  int matrixSize = nCellInEta*nCellInPhi; 
  crystalMatrix.clear();
  crystalMatrix.resize(matrixSize);

  int startEta  =  CentralZ*CentralEta - goBackInEta;
  int startPhi  =  CentralPhi - goBackInPhi;
  
  int i = 0 ;
  for ( int ieta = startEta; ieta < startEta+nCellInEta; ieta ++ ) {
    for( int iphi = startPhi; iphi < startPhi + nCellInPhi; iphi++ ) {
      uint32_t  index;
      if (abs(ieta) > 85 || abs(ieta)<1 ) { continue; }
      if (iphi< 1)      { index = EBDetId(ieta,iphi+360).rawId(); }
      else if(iphi>360) { index = EBDetId(ieta,iphi-360).rawId(); }
      else              { index = EBDetId(ieta,iphi).rawId();     }
      crystalMatrix[i++] = index;
    }
  }
  
}
 
void EcalRecHitsValidation::findEndcapMatrix(int nCellInX, int nCellInY,
                                             int CentralX, int CentralY,int CentralZ,
                                             MapType&  themap) {
  int goBackInX = nCellInX/2;
  int goBackInY = nCellInY/2;
  crystalMatrix.clear();

   int startX  =  CentralX - goBackInX;
   int startY  =  CentralY - goBackInY;

   for ( int ix = startX; ix < startX+nCellInX; ix ++ ) {

      for( int iy = startY; iy < startY + nCellInY; iy++ ) {

        uint32_t index ;

	if(EEDetId::validDetId(ix,iy,CentralZ)) {
          index = EEDetId(ix,iy,CentralZ).rawId();
	}
	else { continue; }
        crystalMatrix.push_back(index);
      }
   }
}
