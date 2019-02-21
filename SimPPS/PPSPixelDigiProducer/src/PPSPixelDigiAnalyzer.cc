
#include <FWCore/Framework/interface/Event.h>

#include "SimPPS/PPSPixelDigiProducer/interface/PPSPixelDigiAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"



#include <iostream>
#include <string>

#include "TFile.h"


#define SELECTED_PIXEL_ROW 89
#define SELECTED_PIXEL_COLUMN 23
#define SELECTED_UNITID 2014314496
#define TG184 0.332655724

#define USE_MIDDLE_OF_PIXEL_2
#define CENTERX 1.05
#define CENTERY -8.475 

using namespace edm;
using namespace std;

PPSPixelDigiAnalyzer:: PPSPixelDigiAnalyzer(const ParameterSet& pset) : theRPixDetTopology_()
{

  label = pset.getUntrackedParameter<string>("label");  
  verbosity_ = pset.getParameter<int> ("Verbosity");
#ifdef USE_MIDDLE_OF_PIXEL
  file = new TFile("CTPPSPixelDigiPlots2.root","RECREATE");

  hOneHitperEvent = new TH2D("OneHitperEvent","One Hit per Event",30,-8.511,-8.361,20,1,1.1);
  hOneHitperEvent2 = new TH2D("OneHitperEvent2","One Hit per Event 2",30,-8.511,-8.361,20,1,1.1);
#else
  file = new TFile("CTPPSPixelDigiPlots.root","RECREATE");
  hOneHitperEvent = new TH2D("OneHitperEvent","One Hit per Event",30,-8.55,-8.4,20,1,1.1);
  hOneHitperEvent2 = new TH2D("OneHitperEvent2","One Hit per Event 2",30,-8.55,-8.4,20,1,1.1);
  hOneHitperEventCenter = new TH2D("OneHitperEventCenter","One Hit per Event Center",30,-0.075,0.075,20,-0.05,0.05);
  hOneHitperEvent2Center = new TH2D("OneHitperEvent2Center","Cluster Size 2",30,-0.075,0.075,20,-0.05,0.05);
#endif
  file->cd();
  hAllHits = new TH2D("AllHits","All Hits",10,1,1.1,10,-8.55,-8.4);

  if(file->IsOpen()) edm::LogInfo("PPSPixelDigiAnalyzer")<<"file open!";
  else edm::LogInfo("PPSPixelDigiAnalyzer")<<"*** Error in opening file ***";

  psim_token = consumes<PSimHitContainer>( edm::InputTag("g4SimHits","CTPPSPixelHits") );
  pixel_token = consumes<edm::DetSetVector<CTPPSPixelDigi>>( edm::InputTag(label,"") ); //label=RPixDetDigitizer???
}

PPSPixelDigiAnalyzer::~PPSPixelDigiAnalyzer(){
}

void PPSPixelDigiAnalyzer::beginJob(){
  found_corresponding_digi_count=0;
  for(int a=0; a<3; a++)  cumulative_cluster_size[a]=0;
}
void PPSPixelDigiAnalyzer::endJob(){
  file->cd();
  hAllHits->Write();
  hOneHitperEvent->Write();
  hOneHitperEvent2->Write();
  hOneHitperEventCenter->Write();
  hOneHitperEvent2Center->Write();
  file->Close();

  edm::LogInfo("PPSPixelDigiAnalyzer") << "found_corresponding_digi_count: " << found_corresponding_digi_count ;
  edm::LogInfo("PPSPixelDigiAnalyzer") << "Cumulative cluster size (1,2,>2) = " << cumulative_cluster_size[0] << ", " << cumulative_cluster_size[1] << ", " << cumulative_cluster_size[2] ;

  delete file;
  delete hAllHits;
  delete hOneHitperEvent;
  delete hOneHitperEvent2;
  delete hOneHitperEventCenter;
  delete hOneHitperEvent2Center;

}

void  PPSPixelDigiAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){
  if(verbosity_>0)edm::LogInfo("PPSPixelDigiAnalyzer") << "--- Run: " << event.id().run()
		       << " Event: " << event.id().event() ;
  
  edm::LogInfo("PPSPixelDigiAnalyzer") << "                                                            I do love Pixels     " ;  
  Handle<PSimHitContainer> simHits; 
  event.getByToken(psim_token,simHits);    

  edm::Handle< edm::DetSetVector<CTPPSPixelDigi> > CTPPSPixelDigis;
  event.getByToken(pixel_token, CTPPSPixelDigis);
       
  if(verbosity_>0)
    edm::LogInfo("PPSPixelDigiAnalyzer") << "\n=================== RPDA Starting SimHit access" << "  ===================" ;

  if(verbosity_>1)
    edm::LogInfo("PPSPixelDigiAnalyzer") << simHits->size() ;

  double selected_pixel_lower_x;
  double selected_pixel_lower_y;
  double selected_pixel_upper_x;
  double selected_pixel_upper_y;
  double myX=0;
  double myY=0;

  theRPixDetTopology_.pixelRange(SELECTED_PIXEL_ROW,SELECTED_PIXEL_COLUMN,selected_pixel_lower_x,selected_pixel_upper_x,selected_pixel_lower_y,selected_pixel_upper_y);

  double hit_inside_selected_pixel[2];
  bool found_hit_inside_selected_pixel = false;

  for(vector<PSimHit>::const_iterator hit = simHits->begin();
      hit != simHits->end(); hit++){    

    LocalPoint entryP = hit->entryPoint();
    LocalPoint exitP = hit->exitPoint();
    LocalPoint midP ((entryP.x()+exitP.x())/2.,(entryP.y()+exitP.y())/2.);
    
#ifdef USE_MIDDLE_OF_PIXEL
    if(entryP.x() > selected_pixel_lower_x && entryP.x() < selected_pixel_upper_x && entryP.y() > (selected_pixel_lower_y+0.115*TG184) && entryP.y() < (selected_pixel_upper_y+0.115*TG184) 
#else
#ifdef USE_MIDDLE_OF_PIXEL_2
       if(midP.x() > selected_pixel_lower_x && midP.x() < selected_pixel_upper_x && midP.y() > selected_pixel_lower_y && midP.y() < selected_pixel_upper_y 
#else
	  if(entryP.x() > selected_pixel_lower_x && entryP.x() < selected_pixel_upper_x && entryP.y() > selected_pixel_lower_y && entryP.y() < selected_pixel_upper_y 
#endif
#endif
	     && hit->detUnitId() ==  SELECTED_UNITID){
	    hit_inside_selected_pixel[0]=entryP.x();
	    hit_inside_selected_pixel[1]=entryP.y();
	    found_hit_inside_selected_pixel = true;
#ifdef USE_MIDDLE_OF_PIXEL_2
	    hAllHits->Fill(midP.x(),midP.y());
	    myX=midP.x();
	    myY=midP.y();
#else
	    hAllHits->Fill(entryP.x(),entryP.y());
	    myX=entryP.x();
	    myY=entryP.y();
#endif
	    if(verbosity_>2)   
	      edm::LogInfo("PPSPixelDigiAnalyzer") << hit_inside_selected_pixel[0] << " " << hit_inside_selected_pixel[1] ;
	  }

//--------------

	  if(verbosity_>1)
	    if( hit->timeOfFlight() > 0){
	      edm::LogInfo("PPSPixelDigiAnalyzer") << "DetId: " << hit->detUnitId()
		   <<"PID: "<<hit->particleType()
		   <<" TOF: "<<hit->timeOfFlight()
		   <<" Proc Type: "<<hit->processType() 
		   <<" p: " << hit->pabs()
		   <<" x = " << entryP.x() << "   y = " <<entryP.y() <<  "  z = " << entryP.z() ;
	    }
	  }

       if(verbosity_>0)
	 edm::LogInfo("PPSPixelDigiAnalyzer") << "\n=================== RPDA Starting Digi access" << "  ===================" ;
       int numberOfDetUnits = 0;
    
// Iterate on detector units
       edm::DetSetVector<CTPPSPixelDigi>::const_iterator DSViter = CTPPSPixelDigis->begin();

       for( ; DSViter != CTPPSPixelDigis->end(); DSViter++) {
	 ++numberOfDetUnits;
 
	 DetId detIdObject(DSViter->detId());
	 if(verbosity_>1)       edm::LogInfo("PPSPixelDigiAnalyzer") << "DetId: " << DSViter->detId();

	 bool found_corresponding_digi = false;
	 unsigned int corresponding_digi_cluster_size = 0;

// looping over digis in a unit id
	 edm::DetSet<CTPPSPixelDigi>::const_iterator begin = (*DSViter).begin();
	 edm::DetSet<CTPPSPixelDigi>::const_iterator end = (*DSViter).end();

	 if(verbosity_>2){
	   edm::LogInfo("PPSPixelDigiAnalyzer") << "FF  "<< DSViter->detId() ;
           for( edm::DetSet<CTPPSPixelDigi>::const_iterator di = begin; di != end; di++){
	     edm::LogInfo("PPSPixelDigiAnalyzer") << "           Digi row  " << di->row() << ", col "<<di->column() ;

	   // reconvert the digi to local coordinates
	     double lx;
	     double ly;
	     double ux;
	     double uy;
	     unsigned int rr = di->row();
	     unsigned int cc = di->column();
	     theRPixDetTopology_.pixelRange(rr,cc,lx,ux,ly,uy);

	     edm::LogInfo("PPSPixelDigiAnalyzer") << " pixel boundaries x low up, y low up " << lx << " "<< ux << " "<< ly << " "<< uy ;

	   }
	 }
	 if(DSViter->detId() == SELECTED_UNITID && found_hit_inside_selected_pixel
	    ){
	   for( edm::DetSet<CTPPSPixelDigi>::const_iterator di = begin; di != end; di++){

	     if(verbosity_>1)  	
	       edm::LogInfo("PPSPixelDigiAnalyzer") << "           Digi row  " << di->row() << ", col "<<di->column() ;
	
	     if( di->row() == SELECTED_PIXEL_ROW && di->column() == SELECTED_PIXEL_COLUMN ){
	       found_corresponding_digi_count++;
	       found_corresponding_digi = true;
	       corresponding_digi_cluster_size = 1;
	     }
	   }
//if coresponding digi found, re-loop to look for adjacent pixels
	   if( found_corresponding_digi){
	     for( edm::DetSet<CTPPSPixelDigi>::const_iterator di = begin; di != end; di++){
	  
	       if(verbosity_>1)  	
		 edm::LogInfo("PPSPixelDigiAnalyzer") << "           Digi row  " << di->row() << ", col "<<di->column() ;
	  
	       if( 
		  (di->row() == SELECTED_PIXEL_ROW+1 && di->column() == SELECTED_PIXEL_COLUMN)
		  || 	     (di->row() == SELECTED_PIXEL_ROW-1 && di->column() == SELECTED_PIXEL_COLUMN)
		  || 	     (di->row() == SELECTED_PIXEL_ROW && di->column() == SELECTED_PIXEL_COLUMN+1)
		  || 	     (di->row() == SELECTED_PIXEL_ROW && di->column() == SELECTED_PIXEL_COLUMN-1)
		   ){
		 corresponding_digi_cluster_size++;	  
		 edm::LogInfo("PPSPixelDigiAnalyzer") << "           Digi row  " << di->row() << ", col "<<di->column() ;
	       }
	     }
	   }
	 }
	 if( corresponding_digi_cluster_size >0){
	   edm::LogInfo("PPSPixelDigiAnalyzer") << "corresponding_digi_cluster_size in the event: " << corresponding_digi_cluster_size ;
	   hOneHitperEvent->Fill(myY,myX);
	   hOneHitperEventCenter->Fill(myY-CENTERY,myX-CENTERX);
	   if(corresponding_digi_cluster_size<3){
	     cumulative_cluster_size[corresponding_digi_cluster_size-1]++;
	     if(corresponding_digi_cluster_size>1){
	       hOneHitperEvent2->Fill(myY,myX);
	       hOneHitperEvent2Center->Fill(myY-CENTERY,myX-CENTERX);
	     }
	   }
	   else{
	     cumulative_cluster_size[2]++;
	     hOneHitperEvent2->Fill(myY,myX);
	     hOneHitperEvent2Center->Fill(myY-CENTERY,myX-CENTERX);
	   }
	 }
       }
     
       if(verbosity_>1)       edm::LogInfo("PPSPixelDigiAnalyzer") << "numberOfDetUnits in the event: " << numberOfDetUnits ;

       }

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

    DEFINE_FWK_MODULE(PPSPixelDigiAnalyzer);
