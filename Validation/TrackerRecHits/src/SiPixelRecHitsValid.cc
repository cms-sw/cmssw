// SiPixelRecHitsValid.cc
// Description: see SiPixelRecHitsValid.h
// Author: Jason Shaev, JHU
// Created 6/7/06
//--------------------------------

#include "Validation/TrackerRecHits/interface/SiPixelRecHitsValid.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include <math.h>

SiPixelRecHitsValid::SiPixelRecHitsValid(const ParameterSet& ps):dbe_(0) {

   outputFile_ = ps.getUntrackedParameter<string>("outputFile", "pixelrechitshisto.root");
   dbe_ = Service<DaqMonitorBEInterface>().operator->();
   dbe_->showDirStructure();
   dbe_->setCurrentFolder("clustBPIX");

   Char_t histo[200];

   //Cluster y-size by module number for barrel
   for (int i=0; i<8; i++) {
	sprintf(histo, "Clust_y_size_Module%d", i+1);
	clustYSizeModule[i] = dbe_->book1D(histo,"Cluster y-size by Module", 20, 0.5, 20.5); 
   } // end for

   //Cluster x-size by layer for barrel
   for (int i=0; i<3; i++) {
	sprintf(histo, "Clust_x_size_Layer%d", i+1);
	clustXSizeLayer[i] = dbe_->book1D(histo,"Cluster x-size by Layer", 20, 0.5, 20.5);
   } // end for

   //Cluster charge by module for 3 layers of barrel
   for (int i=0; i<8; i++) {
	//Cluster charge by module for Layer1
	sprintf(histo, "Clust_charge_Layer1_Module%d", i+1);
	clustChargeLayer1Modules[i] = dbe_->book1D(histo, "Cluster charge Layer 1 by Module", 100, 0., 200000.);

	//Cluster charge by module for Layer2
	sprintf(histo, "Clust_charge_Layer2_Module%d", i+1);
	clustChargeLayer2Modules[i] = dbe_->book1D(histo, "Cluster charge Layer 2 by Module", 100, 0., 200000.);

	//Cluster charge by module for Layer3
	sprintf(histo, "Clust_charge_Layer3_Module%d", i+1);
	clustChargeLayer3Modules[i] = dbe_->book1D(histo, "Cluster charge Layer 3 by Module",100, 0., 200000.);	
   } // end for

   dbe_->setCurrentFolder("clustFPIX");
   //Cluster x-size, y-size, and charge by plaquette for Disks in Forward
   for (int i=0; i<7; i++) {
	//Cluster x-size for Disk1 by Plaquette
	sprintf(histo, "Clust_x_size_Disk1_Plaquette%d", i+1);
	clustXSizeDisk1Plaquettes[i] = dbe_->book1D(histo, "Cluster X-size for Disk1 by Plaquette", 20, 0.5, 20.5);

	//Cluster x-size for Disk2 by Plaquette
	sprintf(histo, "Clust_x_size_Disk2_Plaquette%d", i+1);
	clustXSizeDisk2Plaquettes[i] = dbe_->book1D(histo, "Cluster X-size for Disk2 by Plaquette", 20, 0.5, 20.5);

	//Cluster y-size for Disk1 by Plaquette
	sprintf(histo, "Clust_y_size_Disk1_Plaquette%d", i+1);
	clustYSizeDisk1Plaquettes[i] = dbe_->book1D(histo, "Cluster Y-size for Disk1 by Plaquette", 20, 0.5, 20.5);

	//Cluster y-size for Disk2 by Plaquette
	sprintf(histo, "Clust_y_size_Disk2_Plaquette%d", i+1);
	clustYSizeDisk2Plaquettes[i] = dbe_->book1D(histo, "Cluster Y-size for Disk2 by Plaquette", 20, 0.5, 20.5);

	//Cluster charge for Disk1 by Plaquette
	sprintf(histo, "Clust_charge_Disk1_Plaquette%d", i+1);
	clustChargeDisk1Plaquettes[i] = dbe_->book1D(histo, "Cluster charge for Disk1 by Plaquette", 100, 0., 200000.);

	//Cluster charge for Disk2 by Plaquette
	sprintf(histo, "Clust_charge_Disk2_Plaquette%d", i+1);
	clustChargeDisk2Plaquettes[i] = dbe_->book1D(histo, "Cluster charge for Disk2 by Plaquette", 100, 0., 200000.);
   } // end for

   dbe_->setCurrentFolder("recHitBPIX");
   //RecHit X distribution for full modules for barrel
   recHitXFullModules = dbe_->book1D("RecHit_x_FullModules", "RecHit X distribution for full modules", 100,-2., 2.);

   //RecHit X distribution for half modules for barrel
   recHitXHalfModules = dbe_->book1D("RecHit_x_HalfModules", "RecHit X distribution for half modules", 100, -1., 1.);

   //RecHit Y distribution all modules for barrel
   recHitYAllModules = dbe_->book1D("RecHit_y_AllModules", "RecHit Y distribution for all modules", 100, -4., 4.);

   //RecHit X resolution for flipped and unflipped ladders by layer for barrel
   for (int i=0; i<3; i++) {
	//RecHit X resolution for flipped ladders by layer
	sprintf(histo, "RecHit_XRes_FlippedLadder_Layer%d", i+1);
	recHitXResFlippedLadderLayers[i] = dbe_->book1D(histo, "RecHit XRes Flipped Ladders by Layer", 40, -20., 20.);

	//RecHit X resolution for unflipped ladders by layer
	sprintf(histo, "RecHit_XRes_UnFlippedLadder_Layer%d", i+1);
	recHitXResNonFlippedLadderLayers[i] = dbe_->book1D(histo, "RecHit XRes NonFlipped Ladders by Layer", 40, -20., 20.);
   } // end for

   //RecHit Y resolutions for layers by module for barrel
   for (int i=0; i<8; i++) {
	//Rec Hit Y resolution by module for Layer1
	sprintf(histo, "RecHit_YRes_Layer1_Module%d", i+1);
	recHitYResLayer1Modules[i] = dbe_->book1D(histo, "RecHit YRes Layer1 by module", 40, -20., 20.);

	//RecHit Y resolution by module for Layer2
	sprintf(histo, "RecHit_YRes_Layer2_Module%d", i+1);
	recHitYResLayer2Modules[i] = dbe_->book1D(histo, "RecHit YRes Layer2 by module", 40, -20., 20.);

	//RecHit Y resolution by module for Layer3
	sprintf(histo, "RecHit_YRes_Layer3_Module%d", i+1);
	recHitYResLayer3Modules[i] = dbe_->book1D(histo, "RecHit YRes Layer3 by module", 40, -20., 20.); 
   } // end for

   dbe_->setCurrentFolder("recHitFPIX");
   //RecHit X distribution for plaquette with x-size 1 in forward
   recHitXPlaquetteSize1 = dbe_->book1D("RecHit_x_Plaquette_xsize1", "RecHit X Distribution for plaquette x-size1", 100, -2., 2.);

   //RecHit X distribution for plaquette with x-size 2 in forward
   recHitXPlaquetteSize2 = dbe_->book1D("RecHit_x_Plaquette_xsize2", "RecHit X Distribution for plaquette x-size2", 100, -2., 2.);

   //RecHit Y distribution for plaquette with y-size 2 in forward
   recHitYPlaquetteSize2 = dbe_->book1D("RecHit_y_Plaquette_ysize2", "RecHit Y Distribution for plaquette y-size2", 100, -4., 4.);

   //RecHit Y distribution for plaquette with y-size 3 in forward
   recHitYPlaquetteSize3 = dbe_->book1D("RecHit_y_Plaquette_ysize3", "RecHit Y Distribution for plaquette y-size3", 100, -4., 4.);

   //RecHit Y distribution for plaquette with y-size 4 in forward
   recHitYPlaquetteSize4 = dbe_->book1D("RecHit_y_Plaquette_ysize4", "RecHit Y Distribution for plaquette y-size4", 100, -4., 4.);

   //RecHit Y distribution for plaquette with y-size 5 in forward
   recHitYPlaquetteSize5 = dbe_->book1D("RecHit_y_Plaquette_ysize5", "RecHit Y Distribution for plaquette y-size5", 100, -4., 4.);

   //X and Y resolutions for both disks by plaquette in forward
   for (int i=0; i<7; i++) {
	//X resolution for Disk1 by plaquette
	sprintf(histo, "RecHit_XRes_Disk1_Plaquette%d", i+1);
	recHitXResDisk1Plaquettes[i] = dbe_->book1D(histo, "RecHit XRes Disk1 by plaquette", 40, -20., 20.); 
	//X resolution for Disk2 by plaquette
	sprintf(histo, "RecHit_XRes_Disk2_Plaquette%d", i+1);
	recHitXResDisk2Plaquettes[i] = dbe_->book1D(histo, "RecHit XRes Disk2 by plaquette", 40, -20., 20.);  
 
	//Y resolution for Disk1 by plaquette
	sprintf(histo, "RecHit_YRes_Disk1_Plaquette%d", i+1);
	recHitYResDisk1Plaquettes[i] = dbe_->book1D(histo, "RecHit YRes Disk1 by plaquette", 40, -20., 20.);
	//Y resolution for Disk2 by plaquette
	sprintf(histo, "RecHit_YRes_Disk2_Plaquette%d", i+1);
	recHitYResDisk2Plaquettes[i] = dbe_->book1D(histo, "RecHit YRes Disk2 by plaquette", 40, -20., 20.);
  
   }
}

SiPixelRecHitsValid::~SiPixelRecHitsValid() {
}

void SiPixelRecHitsValid::beginJob(const EventSetup& c) {

}

void SiPixelRecHitsValid::endJob() {
   if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

void SiPixelRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es) {
   LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();
   cout << " Run = " << e.id().run() << " Event = " << e.id().event() << endl;


   //Get RecHits
   std::string recHitProducer = conf_.getUntrackedParameter<std::string>("recHitProducer","pixRecHitConverter");
   edm::Handle<SiPixelRecHitCollection> recHitColl;
   e.getByLabel(recHitProducer, recHitColl);

   //Get event setup
   edm::ESHandle<TrackerGeometry> geom;
   es.get<TrackerDigiGeometryRecord>().get(geom); 
   const TrackerGeometry& theTracker(*geom);

   TrackerHitAssociator associate(e); 

   //iterate over detunits
   for (TrackerGeometry::DetContainer::const_iterator it = geom->dets().begin(); it != geom->dets().end(); it++) {
	DetId detId = ((*it)->geographicalId());
	unsigned int subid=detId.subdetId();

	if (! ((subid==1) || (subid==2))) continue;

	const PixelGeomDetUnit * theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId) );

	SiPixelRecHitCollection::range pixelrechitRange = (recHitColl.product())->get(detId);
	SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.first;
	SiPixelRecHitCollection::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.second;
	SiPixelRecHitCollection::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
	std::vector<PSimHit> matched;

	//----Loop over rechits for this detId
	for ( ; pixeliter != pixelrechitRangeIteratorEnd; pixeliter++) {

	   matched.clear();
	   matched = associate.associateHit(*pixeliter);

	   if (!matched.empty()) {
		float closest = 9999.;
		std::vector<PSimHit>::const_iterator closestit = matched.begin();
		LocalPoint lp = pixeliter->localPosition();
		float rechit_x = lp.x();

		//loop over sim hits and fill closet
		for (std::vector<PSimHit>::const_iterator m = matched.begin(); m<matched.end(); m++) {
		   float sim_x1 = (*m).entryPoint().x();
		   float sim_x2 = (*m).exitPoint().x();
		   float sim_xpos = 0.5*(sim_x1+sim_x2);

		   float x_res = fabs(sim_xpos - rechit_x);
	
		   if (x_res < closest) {
			closest = x_res;
			closestit = m;
		   }
		} // end sim hit loop
		
		if (subid==1) { //<----------barrel
		   fillBarrel(*pixeliter, *closestit, detId, theGeomDet);	
		} // end barrel
		if (subid==2) { // <-------forward
		   fillForward(*pixeliter, *closestit, detId, theGeomDet);
		}

	   } // end matched emtpy
	} // <-----end rechit loop 
   } // <------ end detunit loop
}

void SiPixelRecHitsValid::fillBarrel(const SiPixelRecHit & recHit,const PSimHit & simHit, DetId detId, const PixelGeomDetUnit * theGeomDet) {

   const float cmtomicron=10000.; 

   LocalPoint lp = recHit.localPosition();
   float lp_y = lp.y();  
   float lp_x = lp.x();
 
   recHitYAllModules->Fill(lp_y);

   float sim_x1 = simHit.entryPoint().x();
   float sim_x2 = simHit.exitPoint().x();
   float sim_xpos = 0.5*(sim_x1 + sim_x2);
   float res_x = (lp.x() - sim_xpos)*cmtomicron;

   float sim_y1 = simHit.entryPoint().y();
   float sim_y2 = simHit.exitPoint().y();
   float sim_ypos = 0.5*(sim_y1 + sim_y2);
   float res_y = (lp.y() - sim_ypos)*cmtomicron;

   int rows = theGeomDet->specificTopology().nrows();

   if (rows == 160) {
	recHitXFullModules->Fill(lp_x);
   } else if (rows == 80) {
	recHitXHalfModules->Fill(lp_x);
   }

   float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
   float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
 
   if (tmp2<tmp1) { // flipped
	for (int i=0; i<3; i++) {
	   if (PXBDetId::PXBDetId(detId).layer() == i+1) {
		recHitXResFlippedLadderLayers[i]->Fill(res_x);
	   }
	}
   } 
   else {
	for (int i=0; i<3; i++) {
	   if (PXBDetId::PXBDetId(detId).layer() == i+1) {
		recHitXResNonFlippedLadderLayers[i]->Fill(res_x);
	   }
	}
   }

   //get cluster
   edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = recHit.cluster();

   // fill module dependent info
   for (int i=0; i<8; i++) {
      if (PXBDetId::PXBDetId(detId).module() == i+1) {
	int sizeY = (*clust).sizeY();
	clustYSizeModule[i]->Fill(sizeY);

	if (PXBDetId::PXBDetId(detId).layer() == 1) {
	   float charge = (*clust).charge();
	   clustChargeLayer1Modules[i]->Fill(charge);
	   recHitYResLayer1Modules[i]->Fill(res_y);
	}
	else if (PXBDetId::PXBDetId(detId).layer() == 2) {
	   float charge = (*clust).charge();
	   clustChargeLayer2Modules[i]->Fill(charge);
	   recHitYResLayer2Modules[i]->Fill(res_y);
	}
	else if (PXBDetId::PXBDetId(detId).layer() == 3) {
	   float charge = (*clust).charge();
	   clustChargeLayer3Modules[i]->Fill(charge);
	   recHitYResLayer3Modules[i]->Fill(res_y);
	}
      }
   }
   int sizeX = (*clust).sizeX();
   if (PXBDetId::PXBDetId(detId).layer() == 1) clustXSizeLayer[0]->Fill(sizeX);
   if (PXBDetId::PXBDetId(detId).layer() == 2) clustXSizeLayer[1]->Fill(sizeX);
   if (PXBDetId::PXBDetId(detId).layer() == 3) clustXSizeLayer[2]->Fill(sizeX);
}

void SiPixelRecHitsValid::fillForward(const SiPixelRecHit & recHit, const PSimHit & simHit, DetId detId,const PixelGeomDetUnit * theGeomDet ) {

   int rows = theGeomDet->specificTopology().nrows();
   int cols = theGeomDet->specificTopology().ncolumns();

   const float cmtomicron=10000.;

   LocalPoint lp = recHit.localPosition();
   float lp_x = lp.x();
   float lp_y = lp.y();

   if (rows == 80) {
	recHitXPlaquetteSize1->Fill(lp_x);
   } else if (rows == 160) {
	recHitXPlaquetteSize2->Fill(lp_x);
   }
 
   if (cols == 104) {
	recHitYPlaquetteSize2->Fill(lp_y);
   } else if (cols == 156) {
	recHitYPlaquetteSize3->Fill(lp_y);
   } else if (cols == 208) {
	recHitYPlaquetteSize4->Fill(lp_y);
   } else if (cols == 260) {
	recHitYPlaquetteSize5->Fill(lp_y);
   }

   float sim_x1 = simHit.entryPoint().x();
   float sim_x2 = simHit.exitPoint().x();
   float sim_xpos = 0.5*(sim_x1 + sim_x2);
   float res_x = (lp.x() - sim_xpos)*cmtomicron;

   float sim_y1 = simHit.entryPoint().y();
   float sim_y2 = simHit.exitPoint().y();
   float sim_ypos = 0.5*(sim_y1 + sim_y2);
   float res_y = (lp.y() - sim_ypos)*cmtomicron;

   // get cluster
   edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = recHit.cluster();

   // fill module dependent info
   for (int i=0; i<7; i++) {
      if (PXFDetId::PXFDetId(detId).module() == i+1) {
	if (PXFDetId::PXFDetId(detId).disk() == 1) {

	   int sizeX = (*clust).sizeX();
	   clustXSizeDisk1Plaquettes[i]->Fill(sizeX);

	   int sizeY = (*clust).sizeY();
	   clustYSizeDisk1Plaquettes[i]->Fill(sizeY);

	   float charge = (*clust).charge();
	   clustChargeDisk1Plaquettes[i]->Fill(charge);

	   recHitXResDisk1Plaquettes[i]->Fill(res_x);
	   recHitYResDisk1Plaquettes[i]->Fill(res_y);
	}
	else {
	   int sizeX = (*clust).sizeX();
	   clustXSizeDisk2Plaquettes[i]->Fill(sizeX);

	   int sizeY = (*clust).sizeY();
	   clustYSizeDisk2Plaquettes[i]->Fill(sizeY);

	   float charge = (*clust).charge();
	   clustChargeDisk2Plaquettes[i]->Fill(charge);

	   recHitXResDisk2Plaquettes[i]->Fill(res_x);
	   recHitYResDisk2Plaquettes[i]->Fill(res_y);
	}
      }
   }
}
