/*
 * \file EcalPreshowerRecHitsValidation.cc
 *
 * $Date: 2006/10/17 09:56:12 $
 * \author C. Rovelli
 *
 */

#include <Validation/EcalRecHits/interface/EcalPreshowerRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

using namespace cms;
using namespace edm;
using namespace std;


EcalPreshowerRecHitsValidation::EcalPreshowerRecHitsValidation(const ParameterSet& ps){


  // ----------------------
  EEuncalibrechitCollection_ = ps.getParameter<edm::InputTag>("EEuncalibrechitCollection");
  EErechitCollection_        = ps.getParameter<edm::InputTag>("EErechitCollection");
  ESrechitCollection_        = ps.getParameter<edm::InputTag>("ESrechitCollection");


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
  meESRecHitsEnergy_           = 0;                // total energy
  meESRecHitsEnergy_zp1st_     = 0;    
  meESRecHitsEnergy_zp2nd_     = 0;    
  meESRecHitsEnergy_zm1st_     = 0;    
  meESRecHitsEnergy_zm2nd_     = 0;    
  meESRecHitsMultip_           = 0;                // total multiplicity
  meESRecHitsMultip_zp1st_     = 0;    
  meESRecHitsMultip_zp2nd_     = 0;    
  meESRecHitsMultip_zm1st_     = 0;    
  meESRecHitsMultip_zm2nd_     = 0;    
  meESEERecHitsEnergy_zp_      = 0;                // versus EE energy 
  meESEERecHitsEnergy_zm_      = 0;      

  for (int kk=0; kk<32; kk++)
    { 
      meESRecHitsStripOccupancy_zp1st_[kk] = 0;   
      meESRecHitsStripOccupancy_zm1st_[kk] = 0;   
      meESRecHitsStripOccupancy_zp2nd_[kk] = 0;   
      meESRecHitsStripOccupancy_zm2nd_[kk] = 0;   
    }


  // ---------------------- 
  Char_t histo[200];
  if ( dbe_ ) 
    {
      dbe_->setCurrentFolder("EcalPreshowerRecHitsTask");

      sprintf (histo, "ES Energy" );
      meESRecHitsEnergy_ = dbe_->book1D(histo, histo, 110, -0.0005, 0.005);
      
      sprintf (histo, "ES Energy Plane1 Side+" );
      meESRecHitsEnergy_zp1st_ = dbe_->book1D(histo, histo, 110, -0.0005, 0.005);

      sprintf (histo, "ES Energy Plane2 Side+");
      meESRecHitsEnergy_zp2nd_ = dbe_->book1D(histo, histo, 110, -0.0005, 0.005);
     
      sprintf (histo, "ES Energy Plane1 Side-");
      meESRecHitsEnergy_zm1st_ = dbe_->book1D(histo, histo, 110, -0.0005, 0.005);

      sprintf (histo, "ES Energy Plane2 Side-");
      meESRecHitsEnergy_zm2nd_ = dbe_->book1D(histo, histo, 110, -0.0005, 0.005);

      sprintf (histo, "ES Multiplicity" );
      meESRecHitsMultip_ = dbe_->book1D(histo, histo, 100, 0., 700.);

      sprintf (histo, "ES Multiplicity Plane1 Side+");
      meESRecHitsMultip_zp1st_ = dbe_->book1D(histo, histo, 100, 0., 700.);

      sprintf (histo, "ES Multiplicity Plane2 Side+");
      meESRecHitsMultip_zp2nd_ = dbe_->book1D(histo, histo, 100, 0., 700.);

      sprintf (histo, "ES Multiplicity Plane1 Side-");
      meESRecHitsMultip_zm1st_ = dbe_->book1D(histo, histo, 100, 0., 700.);

      sprintf (histo, "ES Multiplicity Plane2 Side-");
      meESRecHitsMultip_zm2nd_ = dbe_->book1D(histo, histo, 100, 0., 700.);

      sprintf (histo, "Preshower EE vs ES energy Side+");
      meESEERecHitsEnergy_zp_ = dbe_->book2D(histo, histo, 100, 0., 0.1, 100, 0., 40.);

      sprintf (histo, "Preshower EE vs ES energy Side-");
      meESEERecHitsEnergy_zm_ = dbe_->book2D(histo, histo, 100, 0., 0.1, 100, 0., 40.);

      for (int kk=0; kk<32; kk++)
	{ 
	  sprintf(histo, "ES Occupancy Plane1 Side+ Strip%02d", kk+1);    
	  meESRecHitsStripOccupancy_zp1st_[kk] = dbe_->book2D(histo, histo, 40, 0., 40., 40, 0., 40.);

	  sprintf(histo, "ES Occupancy Plane2 Side+ Strip%02d", kk+1);    
	  meESRecHitsStripOccupancy_zp2nd_[kk] = dbe_->book2D(histo, histo, 40, 0., 40., 40, 0., 40.);

	  sprintf(histo, "ES Occupancy Plane1 Side- Strip%02d", kk+1);    
	  meESRecHitsStripOccupancy_zm1st_[kk] = dbe_->book2D(histo, histo, 40, 0., 40., 40, 0., 40.);

	  sprintf(histo, "ES Occupancy Plane2 Side- Strip%02d", kk+1);    
	  meESRecHitsStripOccupancy_zm2nd_[kk] = dbe_->book2D(histo, histo, 40, 0., 40., 40, 0., 40.);
	}
    }
}

EcalPreshowerRecHitsValidation::~EcalPreshowerRecHitsValidation(){   

}

void EcalPreshowerRecHitsValidation::beginJob(const EventSetup& c){  

}

void EcalPreshowerRecHitsValidation::endJob(){

}

void EcalPreshowerRecHitsValidation::analyze(const Event& e, const EventSetup& c){
  
  Handle<ESRecHitCollection> EcalRecHitES;
  try {
    e.getByLabel( ESrechitCollection_, EcalRecHitES);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EcalPreshowerRecHitTaskError") << "Error! can't get the product " << ESrechitCollection_.label() << ":" << ESrechitCollection_.instance();
  }

  Handle<EERecHitCollection> EcalRecHitEE;
  try {
    e.getByLabel( EErechitCollection_, EcalRecHitEE);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EErechitCollection_.label() << ":" << EErechitCollection_.instance();
  }
  
  Handle< EEUncalibratedRecHitCollection > EcalUncalibRecHitEE;
  try {
    e.getByLabel( EEuncalibrechitCollection_, EcalUncalibRecHitEE);
  } catch ( cms::Exception& ex ) {
    edm::LogError("EcalRecHitsTaskError") << "Error! can't get the product " << EEuncalibrechitCollection_.label() << ":" << EEuncalibrechitCollection_.instance();
  }



  // ---------------------- 
  // loop over RecHits
  const ESRecHitCollection *ESRecHit = EcalRecHitES.product ();
  const EERecHitCollection *EERecHit = EcalRecHitEE.product ();
  const EEUncalibratedRecHitCollection *EEUncalibRecHit = EcalUncalibRecHitEE.product() ;

  // multiplicities
  int mult_tot       = 0;
  int mult_zp1st     = 0;
  int mult_zp2nd     = 0;
  int mult_zm1st     = 0;
  int mult_zm2nd     = 0;

  // energies
  float ene_zp1st      = 0.;
  float ene_zp2nd      = 0.;
  float ene_zm1st      = 0.;
  float ene_zm2nd      = 0.;


  // ES
  for (ESRecHitCollection::const_iterator recHit = ESRecHit->begin(); recHit != ESRecHit->end() ; ++recHit)
    {
      ESDetId ESid = ESDetId(recHit->id());

      int zside = ESid.zside();
      int plane = ESid.plane();
      int six   = ESid.six();
      int siy   = ESid.siy();
      int strip = ESid.strip();
      
      // global
      mult_tot++;
      if (meESRecHitsEnergy_) meESRecHitsEnergy_ ->Fill(recHit->energy());      
      
      // side +, plane 1
      if ( (zside == +1) && (plane == 1) )
	{ 
	  mult_zp1st++;
	  ene_zp1st += recHit->energy();
	  if ( meESRecHitsEnergy_zp1st_ )                 { meESRecHitsEnergy_zp1st_                 -> Fill(recHit->energy()); }
	  if ( meESRecHitsStripOccupancy_zp1st_[strip-1] ){ meESRecHitsStripOccupancy_zp1st_[strip-1]-> Fill( six, siy ); }
	}
      

      // side +, plane 2
      if ( (zside == +1) && (plane == 2) )
	{ 
	  mult_zp2nd++; 
	  ene_zp2nd += recHit->energy();
	  if ( meESRecHitsEnergy_zp2nd_ )                 { meESRecHitsEnergy_zp2nd_                 -> Fill(recHit->energy()); }
	  if ( meESRecHitsStripOccupancy_zp2nd_[strip-1] ){ meESRecHitsStripOccupancy_zp2nd_[strip-1]-> Fill( six, siy ); }
	}


      // side -, plane 1
      if ( (zside == -1) && (plane == 1) )
	{ 
	  mult_zm1st++;
	  ene_zm1st += recHit->energy(); 
	  if ( meESRecHitsEnergy_zm1st_ )                 { meESRecHitsEnergy_zm1st_                 -> Fill(recHit->energy()); }
	  if ( meESRecHitsStripOccupancy_zm1st_[strip-1] ){ meESRecHitsStripOccupancy_zm1st_[strip-1]-> Fill( six, siy ); }
	}


      // side +, plane 2
      if ( (zside == -1) && (plane == 2) )
	{ 
	  mult_zm2nd ++; 
	  ene_zm2nd += recHit->energy();
	  if ( meESRecHitsEnergy_zm2nd_ )                 { meESRecHitsEnergy_zm2nd_                 -> Fill(recHit->energy()); }
	  if ( meESRecHitsStripOccupancy_zm2nd_[strip-1] ){ meESRecHitsStripOccupancy_zm2nd_[strip-1]-> Fill( six, siy ); }
	}

    }  // loop over the ES RecHitCollection



  // EE
  double zpEE = 0.;
  double zmEE = 0.;
  for (EcalUncalibratedRecHitCollection::const_iterator uncalibRecHit = EEUncalibRecHit->begin(); uncalibRecHit != EEUncalibRecHit->end() ; ++uncalibRecHit)
    {
      EEDetId EEid = EEDetId(uncalibRecHit->id());
      int mySide = EEid.zside();
      
      // Find corresponding recHit
      EcalRecHitCollection::const_iterator myRecHit = EERecHit->find(EEid);
      
      if (myRecHit != EERecHit->end() )
	{
	  if (mySide > 0) { zpEE = zpEE + myRecHit->energy(); }
	  if (mySide < 0) { zmEE = zmEE + myRecHit->energy(); }
	}
    } 
  
  

  // filling histos
  if (meESRecHitsMultip_)            { meESRecHitsMultip_           -> Fill(mult_tot);   }
  if (meESRecHitsMultip_zp1st_ )     { meESRecHitsMultip_zp1st_     -> Fill(mult_zp1st); }
  if (meESRecHitsMultip_zp2nd_ )     { meESRecHitsMultip_zp2nd_     -> Fill(mult_zp2nd); }
  if (meESRecHitsMultip_zm1st_ )     { meESRecHitsMultip_zm1st_     -> Fill(mult_zm1st); }
  if (meESRecHitsMultip_zm2nd_ )     { meESRecHitsMultip_zm2nd_     -> Fill(mult_zm2nd); }
  if (meESEERecHitsEnergy_zp_)       { meESEERecHitsEnergy_zp_      -> Fill( (ene_zp1st + 0.7*ene_zp2nd)/0.09, zpEE ); }
  if (meESEERecHitsEnergy_zm_)       { meESEERecHitsEnergy_zm_      -> Fill( (ene_zm1st + 0.7*ene_zm2nd)/0.09, zmEE ); }
}
