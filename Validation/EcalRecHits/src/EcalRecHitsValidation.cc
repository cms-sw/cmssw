/*
 * \file EcalRecHitsValidation.cc
 *
 * $Date: 2006/05/22 $
 * \author C. Rovelli
 *
*/

#include <Validation/EcalRecHits/interface/EcalRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>

EcalRecHitsValidation::EcalRecHitsValidation(const ParameterSet& ps){

  // ---------------------- 
  HepMCLabel  = ps.getUntrackedParameter("moduleLabelMC", string("PythiaSource")); 
  recHitProducer_            = ps.getParameter<std::string>("recHitProducer");
  ESrecHitProducer_          = ps.getParameter<std::string>("ESrecHitProducer");
  EBrechitCollection_        = ps.getParameter<std::string>("EBrechitCollection");
  EErechitCollection_        = ps.getParameter<std::string>("EErechitCollection");
  ESrechitCollection_        = ps.getParameter<std::string>("ESrechitCollection");
  uncalibrecHitProducer_     = ps.getParameter<std::string>("uncalibrecHitProducer");
  EBuncalibrechitCollection_ = ps.getParameter<std::string>("EBuncalibrechitCollection");
  EEuncalibrechitCollection_ = ps.getParameter<std::string>("EEuncalibrechitCollection");
  
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
  
  if ( verbose_ ) {
    cout << " verbose switch is ON" << endl;
  } else {
    cout << " verbose switch is OFF" << endl;
  }
  
  // ----------------------                 
  // get hold of back-end interface 
  dbe_ = 0;
  dbe_ = Service<DaqMonitorBEInterface>().operator->();                   
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
  meGunEnergy_           = 0;
  meGunEta_              = 0;   
  meGunPhi_              = 0;   
  meEBRecHitSimHitRatio_ = 0;
  meEERecHitSimHitRatio_ = 0;
  meESRecHitSimHitRatio_ = 0;
  


  // ---------------------- 
  Char_t histo[20];
   
  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalRecHitsTask");
    
    sprintf (histo, "EcalRecHitsTask, Gun Momentum" );    
    meGunEnergy_ = dbe_->book1D(histo, histo, 100, 0., 1000.);
  
    sprintf (histo, "EcalRecHitsTask, Gun Eta" );      
    meGunEta_ = dbe_->book1D(histo, histo, 700, -3.5, 3.5);
     
    sprintf (histo, "EcalRecHitsTask, Gun Phi" );  
    meGunPhi_ = dbe_->book1D(histo, histo, 360, 0., 360.);    
    
    sprintf (histo, "EcalRecHitsTask, Barrel RecSimHit Ratio");  
    meEBRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 100, 0., 2.);

    sprintf (histo, "EcalRecHitsTask, Endcap RecSimHit Ratio"); 
    meEERecHitSimHitRatio_ = dbe_->book1D(histo, histo, 100, 0., 2.);

    sprintf (histo, "EcalRecHitsTask, Preshower RecSimHit Ratio"); 
    meESRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 100, 0., 2.);
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
  
  LogInfo("EcalRecHitsTask, EventInfo: ") << " Run = " << e.id().run() << " Event = " << e.id().event();
  
  Handle<HepMCProduct> MCEvt;                   
  e.getByLabel(HepMCLabel, MCEvt);  

  Handle<CrossingFrame> crossingFrame;    
  e.getByType(crossingFrame);

  Handle< EBUncalibratedRecHitCollection > EcalUncalibRecHitEB;
  Handle< EEUncalibratedRecHitCollection > EcalUncalibRecHitEE;
  try {
    e.getByLabel( uncalibrecHitProducer_, EBuncalibrechitCollection_, EcalUncalibRecHitEB);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EBuncalibrechitCollection_.c_str() << std::endl;
  }
  try {
    e.getByLabel( uncalibrecHitProducer_, EEuncalibrechitCollection_, EcalUncalibRecHitEE);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitdTaskError") << "Error! can't get the product " << EEuncalibrechitCollection_.c_str() << std::endl;
  }

  Handle<EBRecHitCollection> EcalRecHitEB;
  Handle<EERecHitCollection> EcalRecHitEE;
  Handle<ESRecHitCollection> EcalRecHitES;
  try {
    e.getByLabel( recHitProducer_, EBrechitCollection_, EcalRecHitEB);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EBrechitCollection_.c_str() << std::endl;
  }
  try {
    e.getByLabel( recHitProducer_, EErechitCollection_, EcalRecHitEE);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EErechitCollection_.c_str() << std::endl;
  }
  try {
    e.getByLabel( ESrecHitProducer_, ESrechitCollection_, EcalRecHitES);
   } catch ( std::exception& ex ) {
     edm::LogError("EcalLocalRecoTaskError") << "Error! can't get the product " << ESrechitCollection_.c_str() << std::endl;
   }

  // ---------------------- 
  // gun 
  for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin(); p != MCEvt->GetEvent()->particles_end(); ++p ) 
    {      
      Hep3Vector hmom = Hep3Vector((*p)->momentum().vect());
      double htheta = hmom.theta();
      double heta = -log(tan(htheta * 0.5));
      double hphi = hmom.phi();
      hphi = (hphi>=0) ? hphi : hphi+2*M_PI;
      hphi = hphi / M_PI * 180.;

      LogDebug("EventInfo") << "EcalRecHitsTask: Particle gun type form MC = " << abs((*p)->pdg_id()) 
			    << "\n" << "Energy = "<< (*p)->momentum().e() 
			    << "\n" << "Eta = "   << heta 
			    << "\n" << "Phi = "   << hphi;

      if (meGunEnergy_) meGunEnergy_->Fill((*p)->momentum().e());
      if (meGunEta_)    meGunEta_   ->Fill(heta);
      if (meGunPhi_)    meGunPhi_   ->Fill(hphi); 
    }
  

  // -------------------------------------------------------------------
  // BARREL

  // 1) loop over simHits  
  const std::string barrelHitsName ("EcalHitsEB");
  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits (new MixCollection<PCaloHit>(crossingFrame.product (), barrelHitsName)) ;
  
  MapType ebSimMap;
  
  for (MixCollection<PCaloHit>::MixItr hitItr = barrelHits->begin (); hitItr != barrelHits->end (); ++hitItr) 
    {   
      EBDetId ebid = EBDetId(hitItr->id());
      
      LogDebug("SimHitInfo, barrel") 
	<< "CaloHit "   << hitItr->getName() << " DetID = " << hitItr->id()   << "\n"	
	<< "Energy = "  << hitItr->energy()  << " Time = "  << hitItr->time() << "\n"
	<< "EBDetId = " << ebid.ieta()       << " "         << ebid.iphi();
      
      uint32_t crystid = ebid.rawId();
      ebSimMap[crystid] += hitItr->energy();
    }

  

  // 2) loop over RecHits 
  const EBUncalibratedRecHitCollection *EBUncalibRecHit = EcalUncalibRecHitEB.product() ;
  const EBRecHitCollection *EBRecHit = EcalRecHitEB.product();
  
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin(); uncalibRecHit != EBUncalibRecHit->end() ; ++uncalibRecHit)
    {
      EBDetId EBid = EBDetId(uncalibRecHit->id());
      
      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EBRecHit->find(EBid);
      
      // comparison Rec/Sim hit
      if (myRecHit != EBRecHit->end())
	{
	  if ( ebSimMap[EBid.rawId()] != 0. )
	    {
	      if (meEBRecHitSimHitRatio_) { meEBRecHitSimHitRatio_->Fill(myRecHit->energy()/ebSimMap[EBid.rawId()]); }
	    }
	}
      else
	continue;
    }  // loop over the UncalibratedRecHitCollection




  // -------------------------------------------------------------------
  // ENDCAP

  // 1) loop over simHits
  const std::string endcapHitsName ("EcalHitsEE") ;
  std::auto_ptr<MixCollection<PCaloHit> > 
    endcapHits (new MixCollection<PCaloHit>(crossingFrame.product (), endcapHitsName)) ;
  
  MapType eeSimMap;
  
  for (MixCollection<PCaloHit>::MixItr hitItr = endcapHits->begin(); hitItr != endcapHits->end(); ++hitItr) 
    {   
      EEDetId eeid = EEDetId(hitItr->id()) ;

      LogDebug("Endcap, HitInfo")
	<<" CaloHit "      << hitItr->getName() << " DetID = "        << hitItr->id()   << "\n"
	<< "Energy = "     << hitItr->energy()  << " Time = "         << hitItr->time() << "\n"
	<< "EEDetId side " << eeid.zside()      << " = " << eeid.ix() << " " << eeid.iy();
      
      uint32_t crystid = eeid.rawId();
      eeSimMap[crystid] += hitItr->energy();
    }



  // 2) loop over RecHits
  const EEUncalibratedRecHitCollection *EEUncalibRecHit = EcalUncalibRecHitEE.product () ;
  const EERecHitCollection *EERecHit = EcalRecHitEE.product ();
  
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EEUncalibRecHit->begin(); uncalibRecHit != EEUncalibRecHit->end(); ++uncalibRecHit)
    {
      EEDetId EEid = EEDetId(uncalibRecHit->id());
      
      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EERecHit->find(EEid);

      // comparison Rec/Sim hit
      if (myRecHit != EERecHit->end())
	{
	  if ( eeSimMap[EEid.rawId()] != 0. )
	    {
	      if (meEERecHitSimHitRatio_) { meEERecHitSimHitRatio_->Fill(myRecHit->energy()/eeSimMap[EEid.rawId()]); }
	    }
	}
      else
	continue;
    }  // loop over the UncalibratedechitCollection



  // -------------------------------------------------------------------
  // PRESHOWER

  // 1) loop over simHits
  const std::string preshowerHitsName ("EcalHitsES") ;
  std::auto_ptr<MixCollection<PCaloHit> > 
    preshowerHits (new MixCollection<PCaloHit>(crossingFrame.product (), preshowerHitsName)) ;
  
  MapType esSimMap;
  
  for (MixCollection<PCaloHit>::MixItr hitItr = preshowerHits->begin(); hitItr != preshowerHits->end(); ++hitItr) 
    {   
      ESDetId esid = ESDetId(hitItr->id()) ;

      LogDebug("Preshower, HitInfo")
	<<" CaloHit "       << hitItr->getName() << " DetID = "         << hitItr->id()   << "\n"
	<< "Energy = "      << hitItr->energy()  << " Time = "          << hitItr->time() << "\n"
	<< "ESDetId strip " << esid.strip()      << " = " << esid.six() << " " << esid.siy();
      
      uint32_t crystid = esid.rawId();
      esSimMap[crystid] += hitItr->energy();
    }


  // 2) loop over RecHits
  const ESRecHitCollection *ESRecHit = EcalRecHitES.product ();  
  for (EcalRecHitCollection::const_iterator recHit = ESRecHit->begin(); recHit != ESRecHit->end(); ++recHit)
    {
      ESDetId ESid = ESDetId(recHit->id());
      
      if ( esSimMap[ESid.rawId()] != 0. )
	{
	  if (meESRecHitSimHitRatio_) { meESRecHitSimHitRatio_->Fill(recHit->energy()/esSimMap[ESid.rawId()]); }
	}
      else
	continue;
    }  // loop over the RechitCollection

}
