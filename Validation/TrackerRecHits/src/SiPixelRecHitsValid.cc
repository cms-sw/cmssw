// SiPixelRecHitsValid.cc
// Description: see SiPixelRecHitsValid.h
// Author: Jason Shaev, JHU
// Created 6/7/06
//
// G. Giurgiu, JHU (ggiurgiu@pha.jhu.edu)
//             added pull distributions (12/27/06)
//--------------------------------

#include "Validation/TrackerRecHits/interface/SiPixelRecHitsValid.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <math.h>

SiPixelRecHitsValid::SiPixelRecHitsValid(const edm::ParameterSet& ps)
  : conf_(ps)
  , siPixelRecHitCollectionToken_( consumes<SiPixelRecHitCollection>( ps.getParameter<edm::InputTag>( "src" ) ) ) {

}

SiPixelRecHitsValid::~SiPixelRecHitsValid() {
}

void SiPixelRecHitsValid::beginJob() {
}

void SiPixelRecHitsValid::bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es){
  ibooker.setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Pixel/clustBPIX");
  
  Char_t histo[200];
  
  // ---------------------------------------------------------------
  // All histograms that depend on plaquette number have 7 indexes.
  // The first 4 (0-3) correspond to Panel 1 plaquettes 1-4.
  // The last 3 (4-6) correspond to Panel 2 plaquettes 1-3.
  // ---------------------------------------------------------------
  
  //Cluster y-size by module number for barrel
  for (int i=0; i<8; i++) {
    sprintf(histo, "Clust_y_size_Module%d", i+1);
    clustYSizeModule[i] = ibooker.book1D(histo,"Cluster y-size by Module", 20, 0.5, 20.5); 
  } // end for
  
  //Cluster x-size by layer for barrel
  for (int i=0; i<3; i++) {
    sprintf(histo, "Clust_x_size_Layer%d", i+1);
    clustXSizeLayer[i] = ibooker.book1D(histo,"Cluster x-size by Layer", 20, 0.5, 20.5);
  } // end for
  
  //Cluster charge by module for 3 layers of barrel
  for (int i=0; i<8; i++) {
    //Cluster charge by module for Layer1
    sprintf(histo, "Clust_charge_Layer1_Module%d", i+1);
    clustChargeLayer1Modules[i] = ibooker.book1D(histo, "Cluster charge Layer 1 by Module", 50, 0., 200000.);
    
    //Cluster charge by module for Layer2
    sprintf(histo, "Clust_charge_Layer2_Module%d", i+1);
    clustChargeLayer2Modules[i] = ibooker.book1D(histo, "Cluster charge Layer 2 by Module", 50, 0., 200000.);
    
    //Cluster charge by module for Layer3
    sprintf(histo, "Clust_charge_Layer3_Module%d", i+1);
    clustChargeLayer3Modules[i] = ibooker.book1D(histo, "Cluster charge Layer 3 by Module",50, 0., 200000.);	
  } // end for
  
  ibooker.setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Pixel/clustFPIX");
  //Cluster x-size, y-size, and charge by plaquette for Disks in Forward
  for (int i=0; i<7; i++) {
    //Cluster x-size for Disk1 by Plaquette
    sprintf(histo, "Clust_x_size_Disk1_Plaquette%d", i+1);
    clustXSizeDisk1Plaquettes[i] = ibooker.book1D(histo, "Cluster X-size for Disk1 by Plaquette", 20, 0.5, 20.5);
    
    //Cluster x-size for Disk2 by Plaquette
    sprintf(histo, "Clust_x_size_Disk2_Plaquette%d", i+1);
    clustXSizeDisk2Plaquettes[i] = ibooker.book1D(histo, "Cluster X-size for Disk2 by Plaquette", 20, 0.5, 20.5);
    
    //Cluster y-size for Disk1 by Plaquette
    sprintf(histo, "Clust_y_size_Disk1_Plaquette%d", i+1);
    clustYSizeDisk1Plaquettes[i] = ibooker.book1D(histo, "Cluster Y-size for Disk1 by Plaquette", 20, 0.5, 20.5);
    
    //Cluster y-size for Disk2 by Plaquette
    sprintf(histo, "Clust_y_size_Disk2_Plaquette%d", i+1);
    clustYSizeDisk2Plaquettes[i] = ibooker.book1D(histo, "Cluster Y-size for Disk2 by Plaquette", 20, 0.5, 20.5);
    
    //Cluster charge for Disk1 by Plaquette
    sprintf(histo, "Clust_charge_Disk1_Plaquette%d", i+1);
    clustChargeDisk1Plaquettes[i] = ibooker.book1D(histo, "Cluster charge for Disk1 by Plaquette", 50, 0., 200000.);
    
    //Cluster charge for Disk2 by Plaquette
    sprintf(histo, "Clust_charge_Disk2_Plaquette%d", i+1);
    clustChargeDisk2Plaquettes[i] = ibooker.book1D(histo, "Cluster charge for Disk2 by Plaquette", 50, 0., 200000.);
  } // end for
  


  ibooker.setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Pixel/recHitBPIX");
  //RecHit Bunch crossing all barrel hits
  recHitBunchB = ibooker.book1D("RecHit_Bunch_Barrel", "RecHit Bunch Crossing, Barrel", 20, -10., 10.);
  
  //RecHit Event, in-time bunch, all barrel hits
  recHitEventB = ibooker.book1D("RecHit_Event_Barrel", "RecHit Event (in-time bunch), Barrel", 100, 0., 100.);
  
  //RecHit X Resolution all barrel hits
  recHitXResAllB = ibooker.book1D("RecHit_xres_b_All","RecHit X Res All Modules in Barrel", 100, -200., 200.);
  
  //RecHit Y Resolution all barrel hits
  recHitYResAllB = ibooker.book1D("RecHit_yres_b_All","RecHit Y Res All Modules in Barrel", 100, -200., 200.);
  
  //RecHit X distribution for full modules for barrel
  recHitXFullModules = ibooker.book1D("RecHit_x_FullModules", "RecHit X distribution for full modules", 100,-2., 2.);
  
  //RecHit X distribution for half modules for barrel
  recHitXHalfModules = ibooker.book1D("RecHit_x_HalfModules", "RecHit X distribution for half modules", 100, -1., 1.);
  
  //RecHit Y distribution all modules for barrel
  recHitYAllModules = ibooker.book1D("RecHit_y_AllModules", "RecHit Y distribution for all modules", 100, -4., 4.);
  
  //RecHit X resolution for flipped and unflipped ladders by layer for barrel
  for (int i=0; i<3; i++) {
    //RecHit no. of matched simHits all ladders by layer
    sprintf(histo, "RecHit_NsimHit_Layer%d", i+1);
    recHitNsimHitLayer[i] = ibooker.book1D(histo, "RecHit Number of simHits by Layer", 30, 0., 30.);

    //RecHit X resolution for flipped ladders by layer
    sprintf(histo, "RecHit_XRes_FlippedLadder_Layer%d", i+1);
    recHitXResFlippedLadderLayers[i] = ibooker.book1D(histo, "RecHit XRes Flipped Ladders by Layer", 100, -200., 200.);
    
    //RecHit X resolution for unflipped ladders by layer
    sprintf(histo, "RecHit_XRes_UnFlippedLadder_Layer%d", i+1);
    recHitXResNonFlippedLadderLayers[i] = ibooker.book1D(histo, "RecHit XRes NonFlipped Ladders by Layer", 100, -200., 200.);
  } // end for
  
  //RecHit Y resolutions for layers by module for barrel
  for (int i=0; i<8; i++) {
    //Rec Hit Y resolution by module for Layer1
    sprintf(histo, "RecHit_YRes_Layer1_Module%d", i+1);
    recHitYResLayer1Modules[i] = ibooker.book1D(histo, "RecHit YRes Layer1 by module", 100, -200., 200.);
    
    //RecHit Y resolution by module for Layer2
    sprintf(histo, "RecHit_YRes_Layer2_Module%d", i+1);
    recHitYResLayer2Modules[i] = ibooker.book1D(histo, "RecHit YRes Layer2 by module", 100, -200., 200.);
    
    //RecHit Y resolution by module for Layer3
    sprintf(histo, "RecHit_YRes_Layer3_Module%d", i+1);
    recHitYResLayer3Modules[i] = ibooker.book1D(histo, "RecHit YRes Layer3 by module", 100, -200., 200.); 
  } // end for
  
  ibooker.setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Pixel/recHitFPIX");
  //RecHit Bunch crossing all plaquettes
  recHitBunchF = ibooker.book1D("RecHit_Bunch_Forward", "RecHit Bunch Crossing, Forward", 20, -10., 10.);
  
  //RecHit Event, in-time bunch, all plaquettes
  recHitEventF = ibooker.book1D("RecHit_Event_Forward", "RecHit Event (in-time bunch), Forward", 100, 0., 100.);

  //RecHit No. of simHits, by disk
  recHitNsimHitDisk1 = ibooker.book1D("RecHit_NsimHit_Disk1", "RecHit Number of simHits, Disk1", 30, 0., 30.);
  recHitNsimHitDisk2 = ibooker.book1D("RecHit_NsimHit_Disk2", "RecHit Number of simHits, Disk2", 30, 0., 30.);
  
  //RecHit X resolution all plaquettes
  recHitXResAllF = ibooker.book1D("RecHit_xres_f_All", "RecHit X Res All in Forward", 100, -200., 200.);
  
  //RecHit Y resolution all plaquettes
  recHitYResAllF = ibooker.book1D("RecHit_yres_f_All", "RecHit Y Res All in Forward", 100, -200., 200.);
  
  //RecHit X distribution for plaquette with x-size 1 in forward
  recHitXPlaquetteSize1 = ibooker.book1D("RecHit_x_Plaquette_xsize1", "RecHit X Distribution for plaquette x-size1", 100, -2., 2.);
  
  //RecHit X distribution for plaquette with x-size 2 in forward
  recHitXPlaquetteSize2 = ibooker.book1D("RecHit_x_Plaquette_xsize2", "RecHit X Distribution for plaquette x-size2", 100, -2., 2.);
  
  //RecHit Y distribution for plaquette with y-size 2 in forward
  recHitYPlaquetteSize2 = ibooker.book1D("RecHit_y_Plaquette_ysize2", "RecHit Y Distribution for plaquette y-size2", 100, -4., 4.);
  
  //RecHit Y distribution for plaquette with y-size 3 in forward
  recHitYPlaquetteSize3 = ibooker.book1D("RecHit_y_Plaquette_ysize3", "RecHit Y Distribution for plaquette y-size3", 100, -4., 4.);
  
  //RecHit Y distribution for plaquette with y-size 4 in forward
  recHitYPlaquetteSize4 = ibooker.book1D("RecHit_y_Plaquette_ysize4", "RecHit Y Distribution for plaquette y-size4", 100, -4., 4.);
  
  //RecHit Y distribution for plaquette with y-size 5 in forward
  recHitYPlaquetteSize5 = ibooker.book1D("RecHit_y_Plaquette_ysize5", "RecHit Y Distribution for plaquette y-size5", 100, -4., 4.);
  
  //X and Y resolutions for both disks by plaquette in forward
  for (int i=0; i<7; i++) {
    //X resolution for Disk1 by plaquette
    sprintf(histo, "RecHit_XRes_Disk1_Plaquette%d", i+1);
    recHitXResDisk1Plaquettes[i] = ibooker.book1D(histo, "RecHit XRes Disk1 by plaquette", 100, -200., 200.); 
    //X resolution for Disk2 by plaquette
    sprintf(histo, "RecHit_XRes_Disk2_Plaquette%d", i+1);
    recHitXResDisk2Plaquettes[i] = ibooker.book1D(histo, "RecHit XRes Disk2 by plaquette", 100, -200., 200.);  
    
    //Y resolution for Disk1 by plaquette
    sprintf(histo, "RecHit_YRes_Disk1_Plaquette%d", i+1);
    recHitYResDisk1Plaquettes[i] = ibooker.book1D(histo, "RecHit YRes Disk1 by plaquette", 100, -200., 200.);
    //Y resolution for Disk2 by plaquette
    sprintf(histo, "RecHit_YRes_Disk2_Plaquette%d", i+1);
    recHitYResDisk2Plaquettes[i] = ibooker.book1D(histo, "RecHit YRes Disk2 by plaquette", 100, -200., 200.);
    
  }


  ibooker.setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Pixel/recHitPullsBPIX");
  recHitXPullAllB        = ibooker.book1D("RecHit_xres_b_All"       , "RecHit X Pull All Modules in Barrel"        , 100, -10.0, 10.0);
  recHitYPullAllB        = ibooker.book1D("RecHit_yres_b_All"       , "RecHit Y Pull All Modules in Barrel"        , 100, -10.0, 10.0);

  for (int i=0; i<3; i++) 
    {
      sprintf(histo, "RecHit_XPull_FlippedLadder_Layer%d", i+1);
      recHitXPullFlippedLadderLayers[i] = ibooker.book1D(histo, "RecHit XPull Flipped Ladders by Layer", 100, -10.0, 10.0);
      
      sprintf(histo, "RecHit_XPull_UnFlippedLadder_Layer%d", i+1);
      recHitXPullNonFlippedLadderLayers[i] = ibooker.book1D(histo, "RecHit XPull NonFlipped Ladders by Layer", 100, -10.0, 10.0);
    }
  
  for (int i=0; i<8; i++) 
    {
      sprintf(histo, "RecHit_YPull_Layer1_Module%d", i+1);
      recHitYPullLayer1Modules[i] = ibooker.book1D(histo, "RecHit YPull Layer1 by module", 100, -10.0, 10.0);
      
      sprintf(histo, "RecHit_YPull_Layer2_Module%d", i+1);
      recHitYPullLayer2Modules[i] = ibooker.book1D(histo, "RecHit YPull Layer2 by module", 100, -10.0, 10.0);
      
      sprintf(histo, "RecHit_YPull_Layer3_Module%d", i+1);
      recHitYPullLayer3Modules[i] = ibooker.book1D(histo, "RecHit YPull Layer3 by module", 100, -10.0, 10.0); 
    }
  
  ibooker.setCurrentFolder("TrackerRecHitsV/TrackerRecHits/Pixel/recHitPullsFPIX");
  recHitXPullAllF = ibooker.book1D("RecHit_XPull_f_All", "RecHit X Pull All in Forward", 100, -10.0, 10.0);
  
  recHitYPullAllF = ibooker.book1D("RecHit_YPull_f_All", "RecHit Y Pull All in Forward", 100, -10.0, 10.0);
  
  for (int i=0; i<7; i++) 
    {
      sprintf(histo, "RecHit_XPull_Disk1_Plaquette%d", i+1);
      recHitXPullDisk1Plaquettes[i] = ibooker.book1D(histo, "RecHit XPull Disk1 by plaquette", 100, -10.0, 10.0); 
      sprintf(histo, "RecHit_XPull_Disk2_Plaquette%d", i+1);
      recHitXPullDisk2Plaquettes[i] = ibooker.book1D(histo, "RecHit XPull Disk2 by plaquette", 100, -10.0, 10.0);  
      
      sprintf(histo, "RecHit_YPull_Disk1_Plaquette%d", i+1);
      recHitYPullDisk1Plaquettes[i] = ibooker.book1D(histo, "RecHit YPull Disk1 by plaquette", 100, -10.0, 10.0);
      
      sprintf(histo, "RecHit_YPull_Disk2_Plaquette%d", i+1);
      recHitYPullDisk2Plaquettes[i] = ibooker.book1D(histo, "RecHit YPull Disk2 by plaquette", 100, -10.0, 10.0);
    }
}

void SiPixelRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es) 
{

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  edm::LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
  if (e.id().event() % 1000 == 0)
    std::cout << " Run = " << e.id().run() << " Event = " << e.id().event() << std::endl;
  
  //Get RecHits
  edm::Handle<SiPixelRecHitCollection> recHitColl;
  e.getByToken( siPixelRecHitCollectionToken_, recHitColl );
  
  //Get event setup
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get(geom); 
  const TrackerGeometry& theTracker(*geom);
  
  TrackerHitAssociator associate( e, conf_ ); 
  
  //iterate over detunits
  for (TrackerGeometry::DetContainer::const_iterator it = geom->dets().begin(); it != geom->dets().end(); it++) 
    {
      DetId detId = ((*it)->geographicalId());
      unsigned int subid=detId.subdetId();
      
      if (! ((subid==1) || (subid==2))) continue;
      
      const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId) );
      
      SiPixelRecHitCollection::const_iterator pixeldet = recHitColl->find(detId);
      if (pixeldet == recHitColl->end()) continue;
      SiPixelRecHitCollection::DetSet pixelrechitRange = *pixeldet;
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.begin();
      SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorEnd   = pixelrechitRange.end();
      SiPixelRecHitCollection::DetSet::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
      std::vector<PSimHit> matched;
      
      //----Loop over rechits for this detId
      for ( ; pixeliter != pixelrechitRangeIteratorEnd; pixeliter++) 
	{
	  matched.clear();
	  matched = associate.associateHit(*pixeliter);
	  
	  if ( !matched.empty() ) 
	    {
	      float closest = 9999.9;
	      std::vector<PSimHit>::const_iterator closestit = matched.begin();
	      LocalPoint lp = pixeliter->localPosition();
	      float rechit_x = lp.x();
	      float rechit_y = lp.y();

	      //loop over sim hits and fill closet
	      for (std::vector<PSimHit>::const_iterator m = matched.begin(); m<matched.end(); m++) 
		{
		  float sim_x1 = (*m).entryPoint().x();
		  float sim_x2 = (*m).exitPoint().x();
		  float sim_xpos = 0.5*(sim_x1+sim_x2);

		  float sim_y1 = (*m).entryPoint().y();
		  float sim_y2 = (*m).exitPoint().y();
		  float sim_ypos = 0.5*(sim_y1+sim_y2);
		  
		  float x_res = fabs(sim_xpos - rechit_x);
		  float y_res = fabs(sim_ypos - rechit_y);
		  
		  float dist = sqrt(x_res*x_res + y_res*y_res);

		  if ( dist < closest ) 
		    {
		      closest = dist;
		      closestit = m;
		    }
		} // end sim hit loop
	      
	      if (subid==1) 
		{ //<----------barrel
		  fillBarrel(*pixeliter, *closestit, detId, theGeomDet,tTopo);	
		} // end barrel
	      if (subid==2) 
		{ // <-------forward
		  fillForward(*pixeliter, *closestit, detId, theGeomDet,tTopo);
		}
	      
	    } // end matched emtpy

	  int NsimHit = matched.size();
	  if (subid==1)
	    { //<----------barrel
	      for (unsigned int i=0; i<3; i++)
		if (tTopo->pxbLayer(detId) == i+1)
		  recHitNsimHitLayer[i]->Fill(NsimHit);
	    } // end barrel
	  if (subid==2)
	    { // <-------forward
	      if (tTopo->pxfDisk(detId) == 1)
		recHitNsimHitDisk1->Fill(NsimHit);
	      else 
		recHitNsimHitDisk2->Fill(NsimHit);
	    }
	} // <-----end rechit loop 
    } // <------ end detunit loop
}

void SiPixelRecHitsValid::fillBarrel(const SiPixelRecHit& recHit, const PSimHit& simHit, 
				     DetId detId, const PixelGeomDetUnit* theGeomDet,
				     const TrackerTopology *tTopo) 
{
  const float cmtomicron = 10000.0; 

  int bunch = simHit.eventId().bunchCrossing();
  int event = simHit.eventId().event();

  recHitBunchB->Fill(bunch);
  if (bunch == 0) recHitEventB->Fill(event);
  
  LocalPoint lp = recHit.localPosition();
  float lp_y = lp.y();  
  float lp_x = lp.x();

  LocalError lerr = recHit.localPositionError();
  float lerr_x = sqrt(lerr.xx());
  float lerr_y = sqrt(lerr.yy());
  
  recHitYAllModules->Fill(lp_y);
  
  float sim_x1 = simHit.entryPoint().x();
  float sim_x2 = simHit.exitPoint().x();
  float sim_xpos = 0.5*(sim_x1 + sim_x2);
  float res_x = (lp.x() - sim_xpos)*cmtomicron;
  
  recHitXResAllB->Fill(res_x);
  
  float sim_y1 = simHit.entryPoint().y();
  float sim_y2 = simHit.exitPoint().y();
  float sim_ypos = 0.5*(sim_y1 + sim_y2);
  float res_y = (lp.y() - sim_ypos)*cmtomicron;
  
  recHitYResAllB->Fill(res_y);
  
  float pull_x = ( lp_x - sim_xpos ) / lerr_x;
  float pull_y = ( lp_y - sim_ypos ) / lerr_y;

  recHitXPullAllB->Fill(pull_x);  
  recHitYPullAllB->Fill(pull_y);

  int rows = theGeomDet->specificTopology().nrows();
  
  if (rows == 160) 
    {
      recHitXFullModules->Fill(lp_x);
    } 
  else if (rows == 80) 
    {
      recHitXHalfModules->Fill(lp_x);
    }
  
  float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  
  if (tmp2<tmp1) 
    { // flipped
      for (unsigned int i=0; i<3; i++) 
	{
	  if (tTopo->pxbLayer(detId) == i+1) 
	    {
	      recHitXResFlippedLadderLayers[i]->Fill(res_x);
	      recHitXPullFlippedLadderLayers[i]->Fill(pull_x);
	    }
	}
    } 
  else 
    {
      for (unsigned int i=0; i<3; i++) 
	{
	  if (tTopo->pxbLayer(detId) == i+1) 
	    {
	      recHitXResNonFlippedLadderLayers[i]->Fill(res_x);
	      recHitXPullNonFlippedLadderLayers[i]->Fill(pull_x);
	    }
	}
    }
  
  //get cluster
  SiPixelRecHit::ClusterRef const& clust = recHit.cluster();
  
  // fill module dependent info
  for (unsigned int i=0; i<8; i++) 
    {
      if (tTopo->pxbModule(detId) == i+1) 
	{
	  int sizeY = (*clust).sizeY();
	  clustYSizeModule[i]->Fill(sizeY);
	  
	  if (tTopo->pxbLayer(detId) == 1) 
	    {
	      float charge = (*clust).charge();
	      clustChargeLayer1Modules[i]->Fill(charge);
	      recHitYResLayer1Modules[i]->Fill(res_y);
	      recHitYPullLayer1Modules[i]->Fill(pull_y);
	    }
	  else if (tTopo->pxbLayer(detId) == 2) 
	    {
	      float charge = (*clust).charge();
	      clustChargeLayer2Modules[i]->Fill(charge);
	      recHitYResLayer2Modules[i]->Fill(res_y);
	      recHitYPullLayer2Modules[i]->Fill(pull_y);
	    }
	  else if (tTopo->pxbLayer(detId) == 3) 
	    {
	      float charge = (*clust).charge();
	      clustChargeLayer3Modules[i]->Fill(charge);
	      recHitYResLayer3Modules[i]->Fill(res_y);
	      recHitYPullLayer3Modules[i]->Fill(pull_y);
	    }
	}
    }
  int sizeX = (*clust).sizeX();
  if (tTopo->pxbLayer(detId) == 1) clustXSizeLayer[0]->Fill(sizeX);
  if (tTopo->pxbLayer(detId) == 2) clustXSizeLayer[1]->Fill(sizeX);
  if (tTopo->pxbLayer(detId) == 3) clustXSizeLayer[2]->Fill(sizeX);
}

void SiPixelRecHitsValid::fillForward(const SiPixelRecHit & recHit, const PSimHit & simHit, 
				      DetId detId,const PixelGeomDetUnit * theGeomDet,
				      const TrackerTopology *tTopo) 
{
  int rows = theGeomDet->specificTopology().nrows();
  int cols = theGeomDet->specificTopology().ncolumns();
  
  const float cmtomicron = 10000.0;

  int bunch = simHit.eventId().bunchCrossing();
  int event = simHit.eventId().event();

  recHitBunchF->Fill(bunch);
  if (bunch == 0) recHitEventF->Fill(event);
  
  LocalPoint lp = recHit.localPosition();
  float lp_x = lp.x();
  float lp_y = lp.y();
  
  LocalError lerr = recHit.localPositionError();
  float lerr_x = sqrt(lerr.xx());
  float lerr_y = sqrt(lerr.yy());

  float sim_x1 = simHit.entryPoint().x();
  float sim_x2 = simHit.exitPoint().x();
  float sim_xpos = 0.5*(sim_x1 + sim_x2);
  
  float sim_y1 = simHit.entryPoint().y();
  float sim_y2 = simHit.exitPoint().y();
  float sim_ypos = 0.5*(sim_y1 + sim_y2);
  
  float pull_x = ( lp_x - sim_xpos ) / lerr_x;
  float pull_y = ( lp_y - sim_ypos ) / lerr_y;


  if (rows == 80) 
    {
      recHitXPlaquetteSize1->Fill(lp_x);
    } 
  else if (rows == 160) 
    {
      recHitXPlaquetteSize2->Fill(lp_x);
    }
  
  if (cols == 104) 
    {
      recHitYPlaquetteSize2->Fill(lp_y);
    } 
  else if (cols == 156) 
    {
      recHitYPlaquetteSize3->Fill(lp_y);
    } 
  else if (cols == 208) 
    {
      recHitYPlaquetteSize4->Fill(lp_y);
    } 
  else if (cols == 260) 
    {
      recHitYPlaquetteSize5->Fill(lp_y);
    }
  
  float res_x = (lp.x() - sim_xpos)*cmtomicron;
  
  recHitXResAllF->Fill(res_x);
  recHitXPullAllF->Fill(pull_x);

  float res_y = (lp.y() - sim_ypos)*cmtomicron;
  
  recHitYPullAllF->Fill(pull_y);
  
  // get cluster
  SiPixelRecHit::ClusterRef const& clust = recHit.cluster();
  
  // fill plaquette dependent info
  for (unsigned int i=0; i<7; i++) 
    {
      if (tTopo->pxfModule(detId) == i+1) 
	{
	  if (tTopo->pxfDisk(detId) == 1) 
	    {
	      int sizeX = (*clust).sizeX();
	      clustXSizeDisk1Plaquettes[i]->Fill(sizeX);
	      
	      int sizeY = (*clust).sizeY();
	      clustYSizeDisk1Plaquettes[i]->Fill(sizeY);
	      
	      float charge = (*clust).charge();
	      clustChargeDisk1Plaquettes[i]->Fill(charge);
	      
	      recHitXResDisk1Plaquettes[i]->Fill(res_x);
	      recHitYResDisk1Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk1Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk1Plaquettes[i]->Fill(pull_y);
	    }
	  else 
	    {
	      int sizeX = (*clust).sizeX();
	      clustXSizeDisk2Plaquettes[i]->Fill(sizeX);
	      
	      int sizeY = (*clust).sizeY();
	      clustYSizeDisk2Plaquettes[i]->Fill(sizeY);
	      
	      float charge = (*clust).charge();
	      clustChargeDisk2Plaquettes[i]->Fill(charge);
	      
	      recHitXResDisk2Plaquettes[i]->Fill(res_x);
	      recHitYResDisk2Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk2Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk2Plaquettes[i]->Fill(pull_y);
	      
	    } // end else
	} // end if module
      else if (tTopo->pxfPanel(detId) == 2 && (tTopo->pxfModule(detId)+4) == i+1) 
	{
	  if (tTopo->pxfDisk(detId) == 1) 
	    {
	      int sizeX = (*clust).sizeX();
	      clustXSizeDisk1Plaquettes[i]->Fill(sizeX);
	      
	      int sizeY = (*clust).sizeY();
	      clustYSizeDisk1Plaquettes[i]->Fill(sizeY);
	      
	      float charge = (*clust).charge();
	      clustChargeDisk1Plaquettes[i]->Fill(charge);
	      
	      recHitXResDisk1Plaquettes[i]->Fill(res_x);
	      recHitYResDisk1Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk1Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk1Plaquettes[i]->Fill(pull_y);
	    }
	  else 
	    {
	      int sizeX = (*clust).sizeX();
	      clustXSizeDisk2Plaquettes[i]->Fill(sizeX);
	      
	      int sizeY = (*clust).sizeY();
	      clustYSizeDisk2Plaquettes[i]->Fill(sizeY);
	      
	      float charge = (*clust).charge();
	      clustChargeDisk2Plaquettes[i]->Fill(charge);
	      
	      recHitXResDisk2Plaquettes[i]->Fill(res_x);
	      recHitYResDisk2Plaquettes[i]->Fill(res_y);

	      recHitXPullDisk2Plaquettes[i]->Fill(pull_x);
	      recHitYPullDisk2Plaquettes[i]->Fill(pull_y);

	    } // end else
        } // end else
    } // end for
}
DEFINE_FWK_MODULE(SiPixelRecHitsValid);
