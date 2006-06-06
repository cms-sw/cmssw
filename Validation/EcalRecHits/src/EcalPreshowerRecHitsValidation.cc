/*
 * \file EcalPreshowerRecHitsValidation.cc
 *
 * $Date: 2006/05/22 $
 * \author C. Rovelli
 *
 */

#include <Validation/EcalRecHits/interface/EcalPreshowerRecHitsValidation.h>
#include <DataFormats/EcalDetId/interface/ESDetId.h>

EcalPreshowerRecHitsValidation::EcalPreshowerRecHitsValidation(const ParameterSet& ps){

  // ----------------------
  ESrecHitProducer_   = ps.getParameter<std::string>("ESrecHitProducer");
  ESrechitCollection_ = ps.getParameter<std::string>("ESrechitCollection");
  
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
  meESRecHitsEnergy_ = 0;
  for (int ii=0; ii<2; ii++){  // plane
    for (int jj=0; jj<2; jj++){  // side
      for (int kk=0; kk<32; kk++){ meESRecHitsStripOccupancy_[ii][jj][kk] = 0; }  // strip
    }}


  // ---------------------- 
  Char_t histo[200];
  if ( dbe_ ) 
    {
      dbe_->setCurrentFolder("EcalPreshowerRecHitsTask");
            
      sprintf (histo, "ES Energy" );
      meESRecHitsEnergy_ = dbe_->book1D(histo, histo, 100, 0., 1000.);

      for (int ii=0; ii<2; ii++) {
	for (int jj=0; jj<2; jj++) {

	  int pp;
	  if ( jj == 0 ){ pp = -1; } else { pp = 1; } 

	  for (int kk=0; kk<32; kk++)
	    { 
	      sprintf(histo, "ES Occupancy Plane%01d Side%01d Strip%02d", ii+1, pp, kk+1);    
	      meESRecHitsStripOccupancy_[ii][jj][kk] = dbe_->book2D(histo, histo, 40, 0., 40., 40, 0., 40.);
	    }
	}}
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
    e.getByLabel( ESrecHitProducer_, ESrechitCollection_, EcalRecHitES);
  } catch ( std::exception& ex ) {
    edm::LogError("EcalPreshowerRecHitTaskError") << "Error! can't get the product " << ESrechitCollection_.c_str() << std::endl;
  }


  // ---------------------- 
  // loop over RecHits
  const ESRecHitCollection *ESRecHit = EcalRecHitES.product ();
  
  for (ESRecHitCollection::const_iterator recHit = ESRecHit->begin(); recHit != ESRecHit->end() ; ++recHit)
    {
      ESDetId ESid = ESDetId(recHit->id());

      if (meESRecHitsEnergy_) meESRecHitsEnergy_ ->Fill(recHit->energy());

      int zside = ESid.zside();
      int plane = ESid.plane();
      int six   = ESid.six();
      int siy   = ESid.siy();
      int strip = ESid.strip();
      int pp =0;
      if (zside == -1){ pp = 0; }
      if (zside == +1){ pp = 1; }
      
      meESRecHitsStripOccupancy_[plane-1][pp][strip-1] ->Fill( six, siy );
      
    }  // loop over theRecHitCollection

}
