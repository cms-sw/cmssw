/*
 * \file EcalRecHitsValidation.cc
 *
 * $Date: 2007/12/18 18:43:48 $
 * \author C. Rovelli
 *
*/

#include <Validation/EcalRecHits/interface/EcalRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>
#include "CalibCalorimetry/EcalTrivialCondModules/interface/EcalTrivialConditionRetriever.h"

using namespace cms;
using namespace edm;
using namespace std;

EcalRecHitsValidation::EcalRecHitsValidation(const ParameterSet& ps){

  // ---------------------- 
  HepMCLabel                 = ps.getParameter<std::string>("moduleLabelMC"); 
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
    meEBRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 80, 0., 2.);   

    sprintf (histo, "EcalRecHitsTask, Endcap RecSimHit Ratio"); 
    meEERecHitSimHitRatio_ = dbe_->book1D(histo, histo, 80, 0., 2.);

    sprintf (histo, "EcalRecHitsTask, Preshower RecSimHit Ratio"); 
    meESRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 80, 0., 2.);

    sprintf (histo, "EcalRecHitsTask, Barrel RecSimHit Ratio gt 3.5 GeV");  
    meEBRecHitSimHitRatioGt35_ = dbe_->book1D(histo, histo, 80, 0.9, 1.1);   

    sprintf (histo, "EcalRecHitsTask, Endcap RecSimHit Ratio gt 3.5 GeV"); 
    meEERecHitSimHitRatioGt35_ = dbe_->book1D(histo, histo, 80, 0.9, 1.1);

    sprintf (histo, "EcalRecHitsTask, Barrel Unc RecSimHit Ratio");  
    meEBUnRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 80, 0., 2.);   

    sprintf (histo, "EcalRecHitsTask, Endcap Unc RecSimHit Ratio"); 
    meEEUnRecHitSimHitRatio_ = dbe_->book1D(histo, histo, 80, 0., 2.);

    sprintf (histo, "EcalRecHitsTask, Barrel Unc RecSimHit Ratio gt 3.5 GeV");  
    meEBUnRecHitSimHitRatioGt35_ = dbe_->book1D(histo, histo, 80, 0.9, 1.1);   

    sprintf (histo, "EcalRecHitsTask, Endcap Unc RecSimHit Ratio gt 3.5 GeV"); 
    meEEUnRecHitSimHitRatioGt35_ = dbe_->book1D(histo, histo, 80, 0.9, 1.1);
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
  if (EcalUncalibRecHitEB.isValid()){
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
  if ( ! skipMC ) {
    for ( HepMC::GenEvent::particle_const_iterator p = MCEvt->GetEvent()->particles_begin(); p != MCEvt->GetEvent()->particles_end(); ++p ) 
      {      
        double htheta = (*p)->momentum().theta();
        double heta = -log(tan(htheta * 0.5));
        double hphi = (*p)->momentum().phi();
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
  }

  // -------------------------------------------------------------------
  // BARREL

  if ( ! skipBarrel) {

  // 1) loop over simHits  
  const std::string barrelHitsName ("EcalHitsEB");
  e.getByLabel("mix",barrelHitsName,crossingFrame);
  std::auto_ptr<MixCollection<PCaloHit> > 
    barrelHits (new MixCollection<PCaloHit>(crossingFrame.product ()));
  
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
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EBUncalibRecHit->begin(); uncalibRecHit != EBUncalibRecHit->end() ; ++uncalibRecHit)
    {
      EBDetId EBid = EBDetId(uncalibRecHit->id());
      
      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EBRecHit->find(EBid);
      
      // comparison Rec/Sim hit
	  if ( ebSimMap[EBid.rawId()] != 0. )
	    {
          double uncEnergy = uncalibRecHit->amplitude()*barrelADCtoGeV_;
	      if (meEBUnRecHitSimHitRatio_)                                {meEBUnRecHitSimHitRatio_    ->Fill(uncEnergy/ebSimMap[EBid.rawId()]);}
	      if (meEBUnRecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEBUnRecHitSimHitRatioGt35_->Fill(uncEnergy/ebSimMap[EBid.rawId()]);}
	    }

      if (myRecHit != EBRecHit->end())
	{
	  if ( ebSimMap[EBid.rawId()] != 0. )
	    {
	      if (meEBRecHitSimHitRatio_)                                {meEBRecHitSimHitRatio_    ->Fill(myRecHit->energy()/ebSimMap[EBid.rawId()]);}
	      if (meEBRecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEBRecHitSimHitRatioGt35_->Fill(myRecHit->energy()/ebSimMap[EBid.rawId()]);}
	    }
	}
      else
	continue;
    }  // loop over the UncalibratedRecHitCollection

  }


  // -------------------------------------------------------------------
  // ENDCAP

  if ( ! skipEndcap ) {

  // 1) loop over simHits
  const std::string endcapHitsName ("EcalHitsEE") ;
  e.getByLabel("mix",endcapHitsName,crossingFrame);
  std::auto_ptr<MixCollection<PCaloHit> > 
    endcapHits (new MixCollection<PCaloHit>(crossingFrame.product ()));
  
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
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EEUncalibRecHit->begin(); uncalibRecHit != EEUncalibRecHit->end(); ++uncalibRecHit)
    {
      EEDetId EEid = EEDetId(uncalibRecHit->id());
      
      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EERecHit->find(EEid);

      // comparison Rec/Sim hit
	  if ( eeSimMap[EEid.rawId()] != 0. )
	    {
          double uncEnergy = uncalibRecHit->amplitude()*endcapADCtoGeV_;
	      if (meEEUnRecHitSimHitRatio_)                                {meEEUnRecHitSimHitRatio_    ->Fill(uncEnergy/eeSimMap[EEid.rawId()]);}
	      if (meEEUnRecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEEUnRecHitSimHitRatioGt35_->Fill(uncEnergy/eeSimMap[EEid.rawId()]);}
	    }

      if (myRecHit != EERecHit->end())
	{
	  if ( eeSimMap[EEid.rawId()] != 0. )
	    {
	      if (meEERecHitSimHitRatio_)                                {meEERecHitSimHitRatio_    ->Fill(myRecHit->energy()/eeSimMap[EEid.rawId()]); }
	      if (meEERecHitSimHitRatioGt35_ && (myRecHit->energy()>3.5)){meEERecHitSimHitRatioGt35_->Fill(myRecHit->energy()/eeSimMap[EEid.rawId()]); }
	    }
	}
      else
	continue;
    }  // loop over the UncalibratedechitCollection

  }

  // -------------------------------------------------------------------
  // PRESHOWER

  if ( ! skipPreshower ) {

  // 1) loop over simHits
  const std::string preshowerHitsName ("EcalHitsES") ;
  e.getByLabel("mix",preshowerHitsName,crossingFrame);
  std::auto_ptr<MixCollection<PCaloHit> > 
    preshowerHits (new MixCollection<PCaloHit>(crossingFrame.product ()));

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
  for (EcalRecHitCollection::const_iterator recHit = ESRecHit->begin(); recHit != ESRecHit->end(); ++recHit)
    {
      ESDetId ESid = ESDetId(recHit->id());
      if ( esSimMap[ESid.rawId()] != 0. ){ if (meESRecHitSimHitRatio_){ meESRecHitSimHitRatio_ ->Fill(recHit->energy()/esSimMap[ESid.rawId()]); }}
      else
	continue;
    }  // loop over the RechitCollection

  }

}
