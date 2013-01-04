
// File: SiPixelTrackingRecHitsValid.cc
// Authors:  Xingtao Huang (Puerto Rico Univ.)
//           Gavril Giurgiu (JHU)
// Creation Date: Oct. 2006

#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "Validation/RecoTrack/interface/SiPixelTrackingRecHitsValid.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"


#include <TTree.h>
#include <TFile.h>

using namespace std;
using namespace edm;

// End job: write and close the ntuple file
void SiPixelTrackingRecHitsValid::endJob() 
{
  if(debugNtuple_.size()!=0){
  tfile_->Write();
  tfile_->Close();
  }
}

void SiPixelTrackingRecHitsValid::beginJob()
{

  if(debugNtuple_.size()!=0){
    tfile_ = new TFile (debugNtuple_.c_str() , "RECREATE");
    
    t_ = new TTree("Ntuple", "Ntuple");
    int bufsize = 64000;
  
    t_->Branch("subdetId", &subdetId, "subdetId/I", bufsize);
  
    t_->Branch("layer" , &layer , "layer/I" , bufsize);
    t_->Branch("ladder", &ladder, "ladder/I", bufsize);
    t_->Branch("mod"   , &mod   , "mod/I"   , bufsize);
    t_->Branch("side"  , &side  , "side/I"  , bufsize);
    t_->Branch("disk"  , &disk  , "disk/I"  , bufsize);
    t_->Branch("blade" , &blade , "blade/I" , bufsize);
    t_->Branch("panel" , &panel , "panel/I" , bufsize);
    t_->Branch("plaq"  , &plaq  , "plaq/I"  , bufsize);
    
    t_->Branch("rechitx"    , &rechitx    , "rechitx/F"    , bufsize);
    t_->Branch("rechity"    , &rechity    , "rechity/F"    , bufsize);
    t_->Branch("rechitz"    , &rechitz    , "rechitz/F"    , bufsize);
    t_->Branch("rechiterrx" , &rechiterrx , "rechiterrx/F" , bufsize);
    t_->Branch("rechiterry" , &rechiterry , "rechiterry/F" , bufsize);
    t_->Branch("rechitresx" , &rechitresx , "rechitresx/F" , bufsize);
    t_->Branch("rechitresy" , &rechitresy , "rechitresy/F" , bufsize);
    t_->Branch("rechitpullx", &rechitpullx, "rechitpullx/F", bufsize);
    t_->Branch("rechitpully", &rechitpully, "rechitpully/F", bufsize);
    
    t_->Branch("npix"  , &npix  , "npix/I"  , bufsize);
    t_->Branch("nxpix" , &nxpix , "nxpix/I" , bufsize);
    t_->Branch("nypix" , &nypix , "nypix/I" , bufsize);
    t_->Branch("charge", &charge, "charge/F", bufsize);
    
    t_->Branch("alpha", &alpha, "alpha/F", bufsize);
    t_->Branch("beta" , &beta , "beta/F" , bufsize);
    
    t_->Branch("phi", &phi, "phi/F", bufsize);
    t_->Branch("eta", &eta, "eta/F", bufsize);
    
    t_->Branch("half"   , &half   , "half/I"   , bufsize);
    t_->Branch("flipped", &flipped, "flipped/I", bufsize);
    
    t_->Branch("simhitx", &simhitx, "simhitx/F", bufsize);
    t_->Branch("simhity", &simhity, "simhity/F", bufsize);

    t_->Branch("nsimhit", &nsimhit, "nsimhit/I", bufsize);
    t_->Branch("pidhit" , &pidhit , "pidhit/I" , bufsize);
    
    t_->Branch("evt", &evt, "evt/I", bufsize);
    t_->Branch("run", &run, "run/I", bufsize);
  }
  
}

SiPixelTrackingRecHitsValid::SiPixelTrackingRecHitsValid(const ParameterSet& ps):conf_(ps), dbe_(0), tfile_(0), t_(0)
{
  //Read config file
  MTCCtrack_ = ps.getParameter<bool>("MTCCtrack");
  outputFile_ = ps.getUntrackedParameter<string>("outputFile", "pixeltrackingrechitshisto.root");
  src_ = ps.getUntrackedParameter<std::string>( "src" );
  builderName_ = ps.getParameter<std::string>("TTRHBuilder");   
  checkType_ = ps.getParameter<bool>("checkType");
  genType_ = ps.getParameter<int>("genType");
  debugNtuple_=ps.getUntrackedParameter<string>("debugNtuple", "SiPixelTrackingRecHitsValid_Ntuple.root");

  // Book histograms
  dbe_ = Service<DQMStore>().operator->();
  //dbe_->showDirStructure();

  //float math_pi = 3.14159265;
  //float radtodeg = 180.0 / math_pi;

  // Histogram ranges (low and high)

  float xl = -1.0; 
  float xh =  1.0;
  float errxl = 0.0;
  float errxh = 0.003;
  float resxl = -0.02;
  float resxh =  0.02;
  float pullxl = -10.0;
  float pullxh =  10.0;
  
  float yl = -4.0;
  float yh =  4.0;
  float erryl = 0.0;
  float erryh = 0.010;
  float resyl = -0.04;
  float resyh =  0.04;
  float pullyl = -10.0;
  float pullyh =  10.0;

  float barrel_alphal =  80.0;
  float barrel_alphah = 100.0;
  float barrel_betal =  10.0;
  float barrel_betah = 170.0;
  //float barrel_phil = -180.0;
  //float barrel_phih =  180.0;
  //float barrel_etal = -2.5;
  //float barrel_etah =  2.5;

  float forward_p1_alphal = 100.0; 
  float forward_p1_alphah = 115.0;
  float forward_p2_alphal =  65.0; 
  float forward_p2_alphah =  80.0;
  float forward_neg_betal = 67.0; 
  float forward_neg_betah = 73.0;
  float forward_pos_betal = 107.0;
  float forward_pos_betah = 113.0;
  //float forward_phil = -180.0;
  //float forward_phih =  180.0;
  //float forward_neg_etal = -2.5;
  //float forward_neg_etah = -1.5;
  //float forward_pos_etal = 1.5;
  //float forward_pos_etah = 2.5;

  // special ranges for pulls
  float pull_barrel_alphal =  80.0;
  float pull_barrel_alphah = 100.0;
  float pull_barrel_betal =  10.0;
  float pull_barrel_betah = 170.0;
  float pull_barrel_phil = -180.0;
  float pull_barrel_phih =  180.0;
  float pull_barrel_etal = -2.4;
  float pull_barrel_etah =  2.4;

  float pull_forward_p1_alphal = 100.0; 
  float pull_forward_p1_alphah = 112.0;
  float pull_forward_p2_alphal =  68.0; 
  float pull_forward_p2_alphah =  80.0;
  float pull_forward_neg_betal = 68.0; 
  float pull_forward_neg_betah = 72.0;
  float pull_forward_pos_betal = 108.0;
  float pull_forward_pos_betah = 112.0;
  float pull_forward_phil = -180.0;
  float pull_forward_phih =  180.0;
  float pull_forward_neg_etal = -2.4;
  float pull_forward_neg_etah = -1.4;
  float pull_forward_pos_etal = 1.5;
  float pull_forward_pos_etah = 2.5;
    
  int npixl = 0;
  int npixh = 20;
  int nxpixl = 0;
  int nxpixh = 10;
  int nypixl = 0;
  int nypixh = 20;

  float barrel_chargel = 0.0;
  float barrel_chargeh = 250000.0;
  float forward_chargel = 0.0;
  float forward_chargeh = 100000.0;

  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Pixel/Histograms_per_ring-layer_or_disk-plaquette");

  // Pixel barrel has 3 layers and 8 rings; book a histogram for each module given by the (layer, ring) pair 
  for (int i=0; i<3 ; i++) // loop over layers
    {
      Char_t chisto[100];

      sprintf(chisto, "meResxBarrelLayer_%d", i+1);
      meResxBarrelLayer[i] = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
      sprintf(chisto, "meResyBarrelLayer_%d", i+1);
      meResyBarrelLayer[i] = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
      sprintf(chisto, "mePullxBarrelLayer_%d", i+1);
      mePullxBarrelLayer[i] = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
      sprintf(chisto, "mePullyBarrelLayer_%d", i+1);
      mePullyBarrelLayer[i] = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	

      sprintf(chisto, "meResXvsAlphaBarrelFlippedLaddersLayer_%d", i+1);
      meResXvsAlphaBarrelFlippedLaddersLayer[i] = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resxh, "");
      sprintf(chisto, "meResYvsAlphaBarrelFlippedLaddersLayer_%d", i+1);
      meResYvsAlphaBarrelFlippedLaddersLayer[i] = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resyh, "");
      sprintf(chisto, "meResXvsBetaBarrelFlippedLaddersLayer_%d", i+1);
      meResXvsBetaBarrelFlippedLaddersLayer[i] = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resxh, "");
      sprintf(chisto, "meResYvsBetaBarrelFlippedLaddersLayer_%d", i+1);
      meResYvsBetaBarrelFlippedLaddersLayer[i] = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resyh, ""); 
      
      sprintf(chisto, "meResXvsAlphaBarrelNonFlippedLaddersLayer_%d", i+1);
      meResXvsAlphaBarrelNonFlippedLaddersLayer[i] 
	= dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resxh, "");
      sprintf(chisto, "meResYvsAlphaBarrelNonFlippedLaddersLayer_%d", i+1);
      meResYvsAlphaBarrelNonFlippedLaddersLayer[i] 
	= dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resyh, "");
      sprintf(chisto, "meResXvsBetaBarrelNonFlippedLaddersLayer_%d", i+1);
      meResXvsBetaBarrelNonFlippedLaddersLayer[i] 
	= dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resxh, "");
      sprintf(chisto, "meResYvsBetaBarrelNonFlippedLaddersLayer_%d", i+1);
      meResYvsBetaBarrelNonFlippedLaddersLayer[i] 
	= dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resyh, ""); 

      for (int j=0; j<8; j++) // loop over rings
	{
	  sprintf(chisto, "mePosxBarrelLayerModule_%d_%d", i+1, j+1);
	  mePosxBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, xl, xh);
	  sprintf(chisto, "mePosyBarrelLayerModule_%d_%d", i+1, j+1); 
	  mePosyBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, yl, yh);
	  sprintf(chisto, "meErrxBarrelLayerModule_%d_%d", i+1, j+1);
	  meErrxBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
	  sprintf(chisto, "meErryBarrelLayerModule_%d_%d", i+1, j+1);
	  meErryBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
	  sprintf(chisto, "meResxBarrelLayerModule_%d_%d", i+1, j+1);
	  meResxBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
	  sprintf(chisto, "meResyBarrelLayerModule_%d_%d", i+1, j+1);
	  meResyBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
	  sprintf(chisto, "mePullxBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullxBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
	  sprintf(chisto, "mePullyBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullyBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
	  sprintf(chisto, "meNpixBarrelLayerModule_%d_%d", i+1, j+1);
	  meNpixBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, npixl, npixh);
	  sprintf(chisto, "meNxpixBarrelLayerModule_%d_%d", i+1, j+1);
	  meNxpixBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);
	  sprintf(chisto, "meNypixBarrelLayerModule_%d_%d", i+1, j+1);
	  meNypixBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);
	  sprintf(chisto, "meChargeBarrelLayerModule_%d_%d", i+1, j+1);
	  meChargeBarrelLayerModule[i][j] = dbe_->book1D(chisto, chisto, 100, barrel_chargel, barrel_chargeh);
	  
	  sprintf(chisto, "meResXvsAlphaBarrelLayerModule_%d_%d", i+1, j+1);
	  meResXvsAlphaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resxh, "");
	  sprintf(chisto, "meResYvsAlphaBarrelLayerModule_%d_%d", i+1, j+1);
	  meResYvsAlphaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resyh, "");
	  sprintf(chisto, "meResXvsBetaBarrelLayerModule_%d_%d", i+1, j+1);
	  meResXvsBetaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resxh, "");
	  sprintf(chisto, "meResYvsBetaBarrelLayerModule_%d_%d", i+1, j+1);
	  meResYvsBetaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resyh, ""); 
	  
	  sprintf(chisto, "mePullXvsAlphaBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullXvsAlphaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 
								    100, pullxl, pullxh, "");
	  sprintf(chisto, "mePullYvsAlphaBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullYvsAlphaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 
								    100, pullyl, pullyh, "");
	  sprintf(chisto, "mePullXvsBetaBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullXvsBetaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 
								   100, pullxl, pullxh, "");
	  sprintf(chisto, "mePullYvsBetaBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullYvsBetaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 
								   100, pullyl, pullyh, ""); 
	  sprintf(chisto, "mePullXvsPhiBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullXvsPhiBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 100, 
								  pullxl, pullxh, "");
	  sprintf(chisto, "mePullYvsPhiBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullYvsPhiBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 
								  100, pullyl, pullyh, "");
	  sprintf(chisto, "mePullXvsEtaBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullXvsEtaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 
								  100, pullxl, pullxh, "");
	  sprintf(chisto, "mePullYvsEtaBarrelLayerModule_%d_%d", i+1, j+1);
	  mePullYvsEtaBarrelLayerModule[i][j] = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 
								  100, pullyl, pullyh, ""); 
	} //  for (int j=0; j<8; j++) // loop over rings
      
    } // for (int i=0; i<3 ; i++) // loop over layers
  
  // Pixel forward detector has 2 disks, 2 panels and either 3 or 4 plaquettes
  // Panel 1 has 4 plaquettes
  // Panel 2 has 3 plaquettes

  // Panel 1: 2 disks, 4 plaquets
  // Panel 2: 2 disks, 3 plaquets
  for (int i=0; i<2 ; i++) // loop over disks 
    for (int j=0; j<4; j++) // loop over plaquettes
      {
	Char_t chisto[100];

	sprintf(chisto, "mePosxZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePosxZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, xl, xh);
	sprintf(chisto, "mePosyZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePosyZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, yl, yh);
	sprintf(chisto, "meErrxZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meErrxZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
	sprintf(chisto, "meErryZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meErryZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
	sprintf(chisto, "meResxZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResxZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
	sprintf(chisto, "meResyZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResyZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
	sprintf(chisto, "mePullxZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullxZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
	sprintf(chisto, "mePullyZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullyZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
	sprintf(chisto, "meNpixZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meNpixZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
	sprintf(chisto, "meNxpixZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meNxpixZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
	sprintf(chisto, "meNypixZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meNypixZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
	sprintf(chisto, "meChargeZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meChargeZmPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	

	sprintf(chisto, "meResXvsAlphaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResXvsAlphaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsAlphaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResYvsAlphaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resyh, "");
	sprintf(chisto, "meResXvsBetaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResXvsBetaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsBetaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResYvsBetaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resyh, "");

	sprintf(chisto, "mePullXvsAlphaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsAlphaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsAlphaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsAlphaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsBetaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsBetaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsBetaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsBetaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsPhiZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsPhiZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsPhiZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsPhiZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsEtaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsEtaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsEtaZmPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsEtaZmPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullyl,pullyh, "");

	sprintf(chisto, "mePosxZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePosxZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, xl, xh);
	sprintf(chisto, "mePosyZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePosyZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, yl, yh);
	sprintf(chisto, "meErrxZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meErrxZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
	sprintf(chisto, "meErryZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meErryZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
	sprintf(chisto, "meResxZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResxZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
	sprintf(chisto, "meResyZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResyZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
	sprintf(chisto, "mePullxZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullxZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
	sprintf(chisto, "mePullyZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullyZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
	sprintf(chisto, "meNpixZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meNpixZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
	sprintf(chisto, "meNxpixZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meNxpixZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
	sprintf(chisto, "meNypixZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meNypixZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
	sprintf(chisto, "meChargeZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meChargeZpPanel1DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	
	sprintf(chisto, "meResXvsAlphaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResXvsAlphaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsAlphaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResYvsAlphaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resyh, "");
	sprintf(chisto, "meResXvsBetaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResXvsBetaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsBetaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	meResYvsBetaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resyh, "");

	sprintf(chisto, "mePullXvsAlphaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsAlphaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsAlphaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsAlphaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsBetaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsBetaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsBetaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsBetaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsPhiZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsPhiZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsPhiZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsPhiZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsEtaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsEtaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsEtaZpPanel1DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsEtaZpPanel1DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullyl,pullyh, "");
	
	if ( j>2 ) continue; // panel 2 has only 3 plaquettes
	
	sprintf(chisto, "mePosxZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePosxZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, xl, xh);
	sprintf(chisto, "mePosyZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePosyZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, yl, yh);
	sprintf(chisto, "meErrxZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meErrxZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
	sprintf(chisto, "meErryZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meErryZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
	sprintf(chisto, "meResxZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResxZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
	sprintf(chisto, "meResyZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResyZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
	sprintf(chisto, "mePullxZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullxZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
	sprintf(chisto, "mePullyZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullyZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
      	sprintf(chisto, "meNpixZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meNpixZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
	sprintf(chisto, "meNxpixZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meNxpixZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
	sprintf(chisto, "meNypixZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meNypixZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
	sprintf(chisto, "meChargeZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meChargeZmPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	
	sprintf(chisto, "meResXvsAlphaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResXvsAlphaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsAlphaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResYvsAlphaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resyh, "");
	sprintf(chisto, "meResXvsBetaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResXvsBetaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsBetaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResYvsBetaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resyh, ""); 

	sprintf(chisto, "mePullXvsAlphaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsAlphaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsAlphaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsAlphaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsBetaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsBetaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsBetaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsBetaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullyl,pullyh, ""); 
	sprintf(chisto, "mePullXvsPhiZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsPhiZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsPhiZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsPhiZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsEtaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsEtaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullxl, pullxh, "");
	sprintf(chisto, "mePullYvsEtaZmPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsEtaZmPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullyl, pullyh, "");
 
	sprintf(chisto, "mePosxZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePosxZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, xl, xh);
	sprintf(chisto, "mePosyZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePosyZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, yl, yh);
	sprintf(chisto, "meErrxZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meErrxZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
	sprintf(chisto, "meErryZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meErryZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
	sprintf(chisto, "meResxZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResxZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
	sprintf(chisto, "meResyZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResyZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
	sprintf(chisto, "mePullxZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullxZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
	sprintf(chisto, "mePullyZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullyZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
	sprintf(chisto, "meNpixZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meNpixZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
	sprintf(chisto, "meNxpixZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meNxpixZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
	sprintf(chisto, "meNypixZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meNypixZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
	sprintf(chisto, "meChargeZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meChargeZpPanel2DiskPlaq[i][j] = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	
	sprintf(chisto, "meResXvsAlphaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResXvsAlphaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsAlphaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResYvsAlphaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resyh, "");
	sprintf(chisto, "meResXvsBetaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResXvsBetaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resxh, "");
	sprintf(chisto, "meResYvsBetaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	meResYvsBetaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resyh, "");
    
  	sprintf(chisto, "mePullXvsAlphaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsAlphaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsAlphaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsAlphaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsBetaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsBetaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsBetaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsBetaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullyl,pullyh, "");
      	sprintf(chisto, "mePullXvsPhiZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsPhiZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsPhiZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsPhiZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,pullyh, "");
	sprintf(chisto, "mePullXvsEtaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullXvsEtaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullxl,pullxh, "");
	sprintf(chisto, "mePullYvsEtaZpPanel2DiskPlaq_%d_%d", i+1, j+1);
	mePullYvsEtaZpPanel2DiskPlaq[i][j] 
	  = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullyl,pullyh, "");
      
      } // for (int j=0; j<4; j++) // loop over plaquettes

  dbe_->setCurrentFolder("Tracking/TrackingRecHits/Pixel/Histograms_all");

  Char_t chisto[100];
  sprintf(chisto, "mePosxBarrel");
  mePosxBarrel = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyBarrel"); 
  mePosyBarrel = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "meErrxBarrel");
  meErrxBarrel = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
  sprintf(chisto, "meErryBarrel");
  meErryBarrel = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
  sprintf(chisto, "meResxBarrel");
  meResxBarrel = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
  sprintf(chisto, "meResyBarrel");
  meResyBarrel = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
  sprintf(chisto, "mePullxBarrel");
  mePullxBarrel = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
  sprintf(chisto, "mePullyBarrel");
  mePullyBarrel = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
  sprintf(chisto, "meNpixBarrel");
  meNpixBarrel = dbe_->book1D(chisto, chisto, 100, npixl, npixh);
  sprintf(chisto, "meNxpixBarrel");
  meNxpixBarrel = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);
  sprintf(chisto, "meNypixBarrel");
  meNypixBarrel = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);
  sprintf(chisto, "meChargeBarrel");
  meChargeBarrel = dbe_->book1D(chisto, chisto, 100, barrel_chargel, barrel_chargeh);
  sprintf(chisto, "meResXvsAlphaBarrel");
  meResXvsAlphaBarrel = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resxh, "");
  sprintf(chisto, "meResYvsAlphaBarrel");
  meResYvsAlphaBarrel = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resyh, "");	
  sprintf(chisto, "meResXvsBetaBarrel");
  meResXvsBetaBarrel = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resxh, "");
  sprintf(chisto, "meResYvsBetaBarrel");
  meResYvsBetaBarrel = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resyh, "");	
 
  sprintf(chisto, "mePullXvsAlphaBarrel");
  mePullXvsAlphaBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsAlphaBarrel");
  mePullYvsAlphaBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsBetaBarrel");
  mePullXvsBetaBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsBetaBarrel");
  mePullYvsBetaBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsPhiBarrel");
  mePullXvsPhiBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsPhiBarrel");
  mePullYvsPhiBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsEtaBarrel");
  mePullXvsEtaBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsEtaBarrel");
  mePullYvsEtaBarrel = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 100, pullyl, pullyh, "");	

  sprintf(chisto, "mePosxBarrelHalfModule");
  mePosxBarrelHalfModule = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosxBarrelFullModule");
  mePosxBarrelFullModule = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosxBarrelFlippedLadders");
  mePosxBarrelFlippedLadders = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosxBarrelNonFlippedLadders");
  mePosxBarrelNonFlippedLadders = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyBarrelHalfModule");
  mePosyBarrelHalfModule = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "mePosyBarrelFullModule");
  mePosyBarrelFullModule = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "mePosyBarrelFlippedLadders");
  mePosyBarrelFlippedLadders = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "mePosyBarrelNonFlippedLadders");
  mePosyBarrelNonFlippedLadders = dbe_->book1D(chisto, chisto, 100, yl, yh);
  
  sprintf(chisto, "meResXvsAlphaBarrelFlippedLadders");
  meResXvsAlphaBarrelFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resxh, "");
  sprintf(chisto, "meResYvsAlphaBarrelFlippedLadders");
  meResYvsAlphaBarrelFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resyh, "");	
  sprintf(chisto, "meResXvsBetaBarrelFlippedLadders");
  meResXvsBetaBarrelFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resxh, "");
  sprintf(chisto, "meResYvsBetaBarrelFlippedLadders");
  meResYvsBetaBarrelFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resyh, "");	

  sprintf(chisto, "mePullXvsAlphaBarrelFlippedLadders");
  mePullXvsAlphaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsAlphaBarrelFlippedLadders");
  mePullYvsAlphaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsBetaBarrelFlippedLadders");
  mePullXvsBetaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsBetaBarrelFlippedLadders");
  mePullYvsBetaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsPhiBarrelFlippedLadders");
  mePullXvsPhiBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsPhiBarrelFlippedLadders");
  mePullYvsPhiBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsEtaBarrelFlippedLadders");
  mePullXvsEtaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsEtaBarrelFlippedLadders");
  mePullYvsEtaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 100, pullyl, pullyh, "");	

  
  sprintf(chisto, "meWPullXvsAlphaBarrelFlippedLadders");
  meWPullXvsAlphaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullxl, pullxh, "");
  sprintf(chisto, "meWPullYvsAlphaBarrelFlippedLadders");
  meWPullYvsAlphaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "meWPullXvsBetaBarrelFlippedLadders");
  meWPullXvsBetaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullxl, pullxh, "");
  sprintf(chisto, "meWPullYvsBetaBarrelFlippedLadders");
  meWPullYvsBetaBarrelFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullyl, pullyh, "");	
 
  sprintf(chisto, "meResXvsAlphaBarrelNonFlippedLadders");
  meResXvsAlphaBarrelNonFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resxh, "");
  sprintf(chisto, "meResYvsAlphaBarrelNonFlippedLadders");
  meResYvsAlphaBarrelNonFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_alphal, barrel_alphah, 100, 0.0, resyh, "");	
  sprintf(chisto, "meResXvsBetaBarrelNonFlippedLadders");
  meResXvsBetaBarrelNonFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resxh, "");
  sprintf(chisto, "meResYvsBetaBarrelNonFlippedLadders");
  meResYvsBetaBarrelNonFlippedLadders = dbe_->bookProfile(chisto, chisto, 20, barrel_betal, barrel_betah, 100, 0.0, resyh, "");	

  sprintf(chisto, "mePullXvsAlphaBarrelNonFlippedLadders");
  mePullXvsAlphaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsAlphaBarrelNonFlippedLadders");
  mePullYvsAlphaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsBetaBarrelNonFlippedLadders");
  mePullXvsBetaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsBetaBarrelNonFlippedLadders");
  mePullYvsBetaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsPhiBarrelNonFlippedLadders");
  mePullXvsPhiBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsPhiBarrelNonFlippedLadders");
  mePullYvsPhiBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_phil, pull_barrel_phih, 100, pullyl, pullyh, "");	
  sprintf(chisto, "mePullXvsEtaBarrelNonFlippedLadders");
  mePullXvsEtaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 100, pullxl, pullxh, "");
  sprintf(chisto, "mePullYvsEtaBarrelNonFlippedLadders");
  mePullYvsEtaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_etal, pull_barrel_etah, 100, pullyl, pullyh, "");	


  sprintf(chisto, "meWPullXvsAlphaBarrelNonFlippedLadders");
  meWPullXvsAlphaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullxl, pullxh, "");
  sprintf(chisto, "meWPullYvsAlphaBarrelNonFlippedLadders");
  meWPullYvsAlphaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_alphal, pull_barrel_alphah, 100, pullyl, pullyh, "");	
  sprintf(chisto, "meWPullXvsBetaBarrelNonFlippedLadders");
  meWPullXvsBetaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullxl, pullxh, "");
  sprintf(chisto, "meWPullYvsBetaBarrelNonFlippedLadders");
  meWPullYvsBetaBarrelNonFlippedLadders 
    = dbe_->bookProfile(chisto, chisto, 20, pull_barrel_betal, pull_barrel_betah, 100, pullyl, pullyh, "");	


  sprintf(chisto, "mePosxZmPanel1");
  mePosxZmPanel1 = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZmPanel1");
  mePosyZmPanel1 = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "meErrxZmPanel1");
  meErrxZmPanel1 = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
  sprintf(chisto, "meErryZmPanel1");
  meErryZmPanel1 = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
  sprintf(chisto, "meResxZmPanel1");
  meResxZmPanel1 = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
  sprintf(chisto, "meResyZmPanel1");
  meResyZmPanel1 = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
  sprintf(chisto, "mePullxZmPanel1");
  mePullxZmPanel1 = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
  sprintf(chisto, "mePullyZmPanel1");
  mePullyZmPanel1 = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
  sprintf(chisto, "meNpixZmPanel1");
  meNpixZmPanel1 = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
  sprintf(chisto, "meNxpixZmPanel1");
  meNxpixZmPanel1 = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
  sprintf(chisto, "meNypixZmPanel1");
  meNypixZmPanel1 = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
  sprintf(chisto, "meChargeZmPanel1");
  meChargeZmPanel1 = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	
  sprintf(chisto, "meResXvsAlphaZmPanel1");
  meResXvsAlphaZmPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsAlphaZmPanel1");
  meResYvsAlphaZmPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resyh, "");	
  sprintf(chisto, "meResXvsBetaZmPanel1");
  meResXvsBetaZmPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsBetaZmPanel1");
  meResYvsBetaZmPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resyh, "");	

  sprintf(chisto, "mePullXvsAlphaZmPanel1");
  mePullXvsAlphaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsAlphaZmPanel1");
  mePullYvsAlphaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsBetaZmPanel1");
  mePullXvsBetaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsBetaZmPanel1");
  mePullYvsBetaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsPhiZmPanel1");
  mePullXvsPhiZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsPhiZmPanel1");
  mePullYvsPhiZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsEtaZmPanel1");
  mePullXvsEtaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsEtaZmPanel1");
  mePullYvsEtaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullyl,  pullyh, "");

  sprintf(chisto, "meWPullXvsAlphaZmPanel1");
  meWPullXvsAlphaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsAlphaZmPanel1");
  meWPullYvsAlphaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "meWPullXvsBetaZmPanel1");
  meWPullXvsBetaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsBetaZmPanel1");
  meWPullYvsBetaZmPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullyl,  pullyh, "");	

  sprintf(chisto, "mePosxZpPanel1");
  mePosxZpPanel1 = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZpPanel1");
  mePosyZpPanel1 = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "meErrxZpPanel1");
  meErrxZpPanel1 = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
  sprintf(chisto, "meErryZpPanel1");
  meErryZpPanel1 = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
  sprintf(chisto, "meResxZpPanel1");
  meResxZpPanel1 = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
  sprintf(chisto, "meResyZpPanel1");
  meResyZpPanel1 = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
  sprintf(chisto, "mePullxZpPanel1");
  mePullxZpPanel1 = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
  sprintf(chisto, "mePullyZpPanel1");
  mePullyZpPanel1 = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
  sprintf(chisto, "meNpixZpPanel1");
  meNpixZpPanel1 = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
  sprintf(chisto, "meNxpixZpPanel1");
  meNxpixZpPanel1 = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
  sprintf(chisto, "meNypixZpPanel1");
  meNypixZpPanel1 = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
  sprintf(chisto, "meChargeZpPanel1");
  meChargeZpPanel1 = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	
  sprintf(chisto, "meResXvsAlphaZpPanel1");
  meResXvsAlphaZpPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsAlphaZpPanel1");
  meResYvsAlphaZpPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_p1_alphal, forward_p1_alphah, 100, 0.0,  resyh, "");	
  sprintf(chisto, "meResXvsBetaZpPanel1");
  meResXvsBetaZpPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsBetaZpPanel1");
  meResYvsBetaZpPanel1 = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resyh, "");	
 
  sprintf(chisto, "mePullXvsAlphaZpPanel1");
  mePullXvsAlphaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsAlphaZpPanel1");
  mePullYvsAlphaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsBetaZpPanel1");
  mePullXvsBetaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsBetaZpPanel1");
  mePullYvsBetaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsPhiZpPanel1");
  mePullXvsPhiZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsPhiZpPanel1");
  mePullYvsPhiZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsEtaZpPanel1");
  mePullXvsEtaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsEtaZpPanel1");
  mePullYvsEtaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullyl,  pullyh, "");
 
  sprintf(chisto, "meWPullXvsAlphaZpPanel1");
  meWPullXvsAlphaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsAlphaZpPanel1");
  meWPullYvsAlphaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p1_alphal, pull_forward_p1_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "meWPullXvsBetaZpPanel1");
  meWPullXvsBetaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsBetaZpPanel1");
  meWPullYvsBetaZpPanel1 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullyl,  pullyh, "");	

  sprintf(chisto, "mePosxZmPanel2");
  mePosxZmPanel2 = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZmPanel2");
  mePosyZmPanel2 = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "meErrxZmPanel2");
  meErrxZmPanel2 = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
  sprintf(chisto, "meErryZmPanel2");
  meErryZmPanel2 = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
  sprintf(chisto, "meResxZmPanel2");
  meResxZmPanel2 = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
  sprintf(chisto, "meResyZmPanel2");
  meResyZmPanel2 = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
  sprintf(chisto, "mePullxZmPanel2");
  mePullxZmPanel2 = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
  sprintf(chisto, "mePullyZmPanel2");
  mePullyZmPanel2 = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
  sprintf(chisto, "meNpixZmPanel2");
  meNpixZmPanel2 = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
  sprintf(chisto, "meNxpixZmPanel2");
  meNxpixZmPanel2 = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
  sprintf(chisto, "meNypixZmPanel2");
  meNypixZmPanel2 = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
  sprintf(chisto, "meChargeZmPanel2");
  meChargeZmPanel2 = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	
  sprintf(chisto, "meResXvsAlphaZmPanel2");
  meResXvsAlphaZmPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsAlphaZmPanel2");
  meResYvsAlphaZmPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resyh, "");	
  sprintf(chisto, "meResXvsBetaZmPanel2");
  meResXvsBetaZmPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsBetaZmPanel2");
  meResYvsBetaZmPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_neg_betal, forward_neg_betah, 100, 0.0,  resyh, "");	
 
  sprintf(chisto, "mePullXvsAlphaZmPanel2");
  mePullXvsAlphaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsAlphaZmPanel2");
  mePullYvsAlphaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsBetaZmPanel2");
  mePullXvsBetaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsBetaZmPanel2");
  mePullYvsBetaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsPhiZmPanel2");
  mePullXvsPhiZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsPhiZmPanel2");
  mePullYvsPhiZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsEtaZmPanel2");
  mePullXvsEtaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsEtaZmPanel2");
  mePullYvsEtaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_etal, pull_forward_neg_etah, 100, pullyl,  pullyh, "");

  sprintf(chisto, "meWPullXvsAlphaZmPanel2");
  meWPullXvsAlphaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsAlphaZmPanel2");
  meWPullYvsAlphaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "meWPullXvsBetaZmPanel2");
  meWPullXvsBetaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsBetaZmPanel2");
  meWPullYvsBetaZmPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_neg_betal, pull_forward_neg_betah, 100, pullyl,  pullyh, "");	


  sprintf(chisto, "mePosxZpPanel2");
  mePosxZpPanel2 = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZpPanel2");
  mePosyZpPanel2 = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "meErrxZpPanel2");
  meErrxZpPanel2 = dbe_->book1D(chisto, chisto, 100, errxl, errxh);
  sprintf(chisto, "meErryZpPanel2");
  meErryZpPanel2 = dbe_->book1D(chisto, chisto, 100, erryl, erryh);	
  sprintf(chisto, "meResxZpPanel2");
  meResxZpPanel2 = dbe_->book1D(chisto, chisto, 100, resxl, resxh);
  sprintf(chisto, "meResyZpPanel2");
  meResyZpPanel2 = dbe_->book1D(chisto, chisto, 100, resyl, resyh);	
  sprintf(chisto, "mePullxZpPanel2");
  mePullxZpPanel2 = dbe_->book1D(chisto, chisto, 100, pullxl, pullxh);
  sprintf(chisto, "mePullyZpPanel2");
  mePullyZpPanel2 = dbe_->book1D(chisto, chisto, 100, pullyl, pullyh);	
  sprintf(chisto, "meNpixZpPanel2");
  meNpixZpPanel2 = dbe_->book1D(chisto, chisto, 100, npixl, npixh);	
  sprintf(chisto, "meNxpixZpPanel2");
  meNxpixZpPanel2 = dbe_->book1D(chisto, chisto, 100, nxpixl, nxpixh);	
  sprintf(chisto, "meNypixZpPanel2");
  meNypixZpPanel2 = dbe_->book1D(chisto, chisto, 100, nypixl, nypixh);	
  sprintf(chisto, "meChargeZpPanel2");
  meChargeZpPanel2 = dbe_->book1D(chisto, chisto, 100, forward_chargel, forward_chargeh);	
  sprintf(chisto, "meResXvsAlphaZpPanel2");
  meResXvsAlphaZpPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsAlphaZpPanel2");
  meResYvsAlphaZpPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_p2_alphal, forward_p2_alphah, 100, 0.0,  resyh, "");	
  sprintf(chisto, "meResXvsBetaZpPanel2");
  meResXvsBetaZpPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resxh, "");
  sprintf(chisto, "meResYvsBetaZpPanel2");
  meResYvsBetaZpPanel2 = dbe_->bookProfile(chisto, chisto, 20, forward_pos_betal, forward_pos_betah, 100, 0.0,  resyh, "");	
 
  sprintf(chisto, "mePullXvsAlphaZpPanel2");
  mePullXvsAlphaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsAlphaZpPanel2");
  mePullYvsAlphaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsBetaZpPanel2");
  mePullXvsBetaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsBetaZpPanel2");
  mePullYvsBetaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullyl,  pullyh, "");	  
  sprintf(chisto, "mePullXvsPhiZpPanel2");
  mePullXvsPhiZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsPhiZpPanel2");
  mePullYvsPhiZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_phil, pull_forward_phih, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "mePullXvsEtaZpPanel2");
  mePullXvsEtaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "mePullYvsEtaZpPanel2");
  mePullYvsEtaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_etal, pull_forward_pos_etah, 100, pullyl,  pullyh, "");

  sprintf(chisto, "meWPullXvsAlphaZpPanel2");
  meWPullXvsAlphaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsAlphaZpPanel2");
  meWPullYvsAlphaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_p2_alphal, pull_forward_p2_alphah, 100, pullyl,  pullyh, "");	
  sprintf(chisto, "meWPullXvsBetaZpPanel2");
  meWPullXvsBetaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullxl,  pullxh, "");
  sprintf(chisto, "meWPullYvsBetaZpPanel2");
  meWPullYvsBetaZpPanel2 
    = dbe_->bookProfile(chisto, chisto, 20, pull_forward_pos_betal, pull_forward_pos_betah, 100, pullyl,  pullyh, "");

  // all hits (not only from tracks) 
  sprintf(chisto, "mePosxBarrel_all_hits");
  mePosxBarrel_all_hits = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyBarrel_all_hits"); 
  mePosyBarrel_all_hits = dbe_->book1D(chisto, chisto, 100, yl, yh);

  sprintf(chisto, "mePosxZmPanel1_all_hits");
  mePosxZmPanel1_all_hits = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZmPanel1_all_hits");
  mePosyZmPanel1_all_hits = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "mePosxZmPanel2_all_hits");
  mePosxZmPanel2_all_hits = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZmPanel2_all_hits");
  mePosyZmPanel2_all_hits = dbe_->book1D(chisto, chisto, 100, yl, yh);

  sprintf(chisto, "mePosxZpPanel1_all_hits");
  mePosxZpPanel1_all_hits = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZpPanel1_all_hits");
  mePosyZpPanel1_all_hits = dbe_->book1D(chisto, chisto, 100, yl, yh);
  sprintf(chisto, "mePosxZpPanel2_all_hits");
  mePosxZpPanel2_all_hits = dbe_->book1D(chisto, chisto, 100, xl, xh);
  sprintf(chisto, "mePosyZpPanel2_all_hits");
  mePosyZpPanel2_all_hits = dbe_->book1D(chisto, chisto, 100, yl, yh);

  // control histograms
  meTracksPerEvent     = dbe_->book1D("meTracksPerEvent"    , "meTracksPerEvent"    , 200, 0.0, 200.0);
  mePixRecHitsPerTrack = dbe_->book1D("mePixRecHitsPerTrack", "mePixRecHitsPerTrack",  6, 0.0,  6.0);

}

// Virtual destructor needed.
SiPixelTrackingRecHitsValid::~SiPixelTrackingRecHitsValid() 
{  
  if ( outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}  

// Functions that gets called by framework every event
void SiPixelTrackingRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  es.get<IdealGeometryRecord>().get(tTopo);


  run = e.id().run();
  evt = e.id().event();

  //  if ( evt%1000 == 0 ) 
    //cout << "evt = " << evt << endl;
  
  float math_pi = 3.14159265;
  float radtodeg = 180.0 / math_pi;
    
  DetId detId;

  LocalPoint position;
  LocalError error;
  float mindist = 999999.9;

  std::vector<PSimHit> matched;
  TrackerHitAssociator associate(e,conf_);

  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry* tracker = &(* pDD);

  if ( !MTCCtrack_ )
    {
      // --------------------------------------- all hits -----------------------------------------------------------
      //--- Fetch Pixel RecHits
      edm::Handle<SiPixelRecHitCollection> recHitColl;
      e.getByLabel( "siPixelRecHits", recHitColl);
      
      //cout <<" ----- Found " 
      //   << const_cast<SiPixelRecHitCollection*>(recHitColl.product())->size()
      //   << " Pixel RecHits" << std::endl;
  
      //-----Iterate over detunits
      for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) 
	{
	  DetId detId = ((*it)->geographicalId());
	 
	  unsigned int subid = detId.subdetId();
	  if ( !((subid==1) || (subid==2)) ) 
	    continue; // end subid if

          SiPixelRecHitCollection::const_iterator match = recHitColl->find(detId);
          if (match == recHitColl->end()) continue;

          SiPixelRecHitCollection::DetSet pixelrechitRange = *match;
          SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorBegin = pixelrechitRange.begin();
          SiPixelRecHitCollection::DetSet::const_iterator pixelrechitRangeIteratorEnd = pixelrechitRange.end();
          SiPixelRecHitCollection::DetSet::const_iterator pixeliter = pixelrechitRangeIteratorBegin;
	  std::vector<PSimHit> matched;
	  
	  //----Loop over rechits for this detId
	  for ( ; pixeliter != pixelrechitRangeIteratorEnd; ++pixeliter) 
	    {
	      LocalPoint lp = pixeliter->localPosition();
	      float rechitx = lp.x();
	      float rechity = lp.y();
	     
	      detId = (*it)->geographicalId();
	      subdetId = (int)detId.subdetId();
	      if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelBarrel ) 
		{
		  mePosxBarrel_all_hits->Fill( rechitx );
		  mePosyBarrel_all_hits->Fill( rechity );
		}
	      else if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelEndcap )
		{
		  
		  side  = tTopo->pxfSide(detId);
		  disk  = tTopo->pxfDisk(detId);
		  blade = tTopo->pxfBlade(detId);
		  panel = tTopo->pxfPanel(detId);
		  plaq  = tTopo->pxfModule(detId); // also known as plaquette
		  
		  if ( side==1 ) 
		    {
		      if ( panel==1 )
			{
			  mePosxZmPanel1_all_hits->Fill( rechitx );
			  mePosyZmPanel1_all_hits->Fill( rechity );
			}
		      else if ( panel==2 )
			{
			  mePosxZmPanel2_all_hits->Fill( rechitx );
			  mePosyZmPanel2_all_hits->Fill( rechity );
			}
		      else LogWarning("SiPixelTrackingRecHitsValid") << "..............................................Wrong panel number !"; 
		    } // if ( side==1 ) 
		  else if ( side==2 )
		    {
		      if ( panel==1 )
			{
			  mePosxZpPanel1_all_hits->Fill( rechitx );
			  mePosyZpPanel1_all_hits->Fill( rechity );
			}
		       else if ( panel==2 )
			 {
			   mePosxZpPanel2_all_hits->Fill( rechitx );
			   mePosyZpPanel2_all_hits->Fill( rechity );
			 }
		       else  LogWarning("SiPixelTrackingRecHitsValid")<< "..............................................Wrong panel number !";
		    } //else if ( side==2 )
		  else LogWarning("SiPixelTrackingRecHitsValid") << ".......................................................Wrong side !" ;
		  
		} // else if ( detId.subdetId()==PixelSubdetector::PixelEndcap )
	      else LogWarning("SiPixelTrackingRecHitsValid") << "Pixel rechit collection but we are not in the pixel detector" << (int)detId.subdetId() ;
	      
	    }
	}
      // ------------------------------------------------ all hits ---------------------------------------------------------------
       
      // Get tracks
      edm::Handle<reco::TrackCollection> trackCollection;
      e.getByLabel(src_, trackCollection);
      const reco::TrackCollection *tracks = trackCollection.product();
      reco::TrackCollection::const_iterator tciter;

      int n_tracks = (int)tracks->size(); // number of tracks in this event
      meTracksPerEvent->Fill( n_tracks );

      if ( tracks->size() > 0 )
	{
	  // Loop on tracks
	  for ( tciter=tracks->begin(); tciter!=tracks->end(); tciter++)
	    {
	      phi = tciter->momentum().phi() / math_pi*180.0;
	      eta = tciter->momentum().eta();
	      
	      int n_hits = 0;
	      // First loop on hits: find matched hits
	      for ( trackingRecHit_iterator it = tciter->recHitsBegin(); it != tciter->recHitsEnd(); it++) 
		{
		  const TrackingRecHit &thit = **it;
		  // Is it a matched hit?
		  const SiPixelRecHit* matchedhit = dynamic_cast<const SiPixelRecHit*>(&thit);
		  
		  if ( matchedhit ) 
		    {
		      ++n_hits;
		      
		      layer  = -9999; 
		      ladder = -9999; 
		      mod    = -9999; 
		      side   = -9999;  
		      disk   = -9999;  
		      blade  = -9999; 
		      panel  = -9999; 
		      plaq   = -9999; 

		      rechitx = -9999.9;
		      rechity = -9999.9;
		      rechitz = -9999.9;
		      rechiterrx = -9999.9;
		      rechiterry = -9999.9;		      
		      rechitresx = -9999.9;
		      rechitresy = -9999.9;
		      rechitpullx = -9999.9;
		      rechitpully = -9999.9;
		      
		      npix = -9999;
		      nxpix = -9999;
		      nypix = -9999;
		      charge = -9999.9;
		      
		      alpha = -9999.9;
		      beta  = -9999.9;

		      half = -9999;
		      flipped = -9999;
		 
		      nsimhit = -9999;
		         
		      simhitx = -9999.9;
		      simhity = -9999.9;

		      position = (*it)->localPosition();
		      error = (*it)->localPositionError();

		      rechitx = position.x();
		      rechity = position.y();
		      rechitz = position.z();
		      rechiterrx = sqrt(error.xx());
		      rechiterry = sqrt(error.yy());

		      npix = (*matchedhit).cluster()->size();
		      nxpix = (*matchedhit).cluster()->sizeX();
		      nypix = (*matchedhit).cluster()->sizeY();
		      charge = (*matchedhit).cluster()->charge();

		      //Association of the rechit to the simhit
		      matched.clear();
		      matched = associate.associateHit(*matchedhit);

		      nsimhit = (int)matched.size();

		      if ( !matched.empty() ) 
			{
			  mindist = 999999.9;
			  float distx, disty, dist;
			  bool found_hit_from_generated_particle = false;
			  
			  int n_assoc_muon = 0;

			  vector<PSimHit>::const_iterator closestit = matched.begin();
			  for (vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++)
			    {
			      if ( checkType_ )
				{
				  int pid = (*m).particleType();
				  if ( abs(pid) != genType_ )
				    continue;
				  
				}
			      
			      float simhitx = 0.5 * ( (*m).entryPoint().x() + (*m).exitPoint().x() );
			      float simhity = 0.5 * ( (*m).entryPoint().y() + (*m).exitPoint().y() );
			      
			      distx = fabs(rechitx - simhitx);
			      disty = fabs(rechity - simhity);
			      dist = sqrt( distx*distx + disty*disty );
	
			      if ( dist < mindist )
				{
				  n_assoc_muon++;

				  mindist = dist;
				  closestit = m;
				  found_hit_from_generated_particle = true;
				}
			    } // for (vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++)
			  
			  // This recHit does not have any simHit with the same particleType as the particles generated
			  // Ignore it as most probably come from delta rays.
			  if ( checkType_ && !found_hit_from_generated_particle )
			    continue; 
			  
			  if ( n_assoc_muon > 1 )
			    {
			      LogWarning("SiPixelTrackingRecHitsValid") << " ----- This is not good: n_assoc_muon = " << n_assoc_muon ;
			      LogWarning("SiPixelTrackingRecHitsValid") << "evt = " << evt ;
			    }

			  pidhit = (*closestit).particleType();

			  simhitx = 0.5*( (*closestit).entryPoint().x() + (*closestit).exitPoint().x() );
			  simhity = 0.5*( (*closestit).entryPoint().y() + (*closestit).exitPoint().y() );
			  
			  rechitresx = rechitx - simhitx;
			  rechitresy = rechity - simhity;
			  rechitpullx = ( rechitx - simhitx ) / sqrt(error.xx());
			  rechitpully = ( rechity - simhity ) / sqrt(error.yy());

			  float simhitpx = (*closestit).momentumAtEntry().x();
			  float simhitpy = (*closestit).momentumAtEntry().y();
			  float simhitpz = (*closestit).momentumAtEntry().z();
			  			  
			  //beta  = atan2(simhitpz, simhitpy) * radtodeg;
			  //alpha = atan2(simhitpz, simhitpx) * radtodeg;
			  
			  beta  = fabs(atan2(simhitpz, simhitpy)) * radtodeg;
			  alpha = fabs(atan2(simhitpz, simhitpx)) * radtodeg;
		
			  detId = (*it)->geographicalId();

			  subdetId = (int)detId.subdetId();

			  if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelBarrel ) 
			    {
			      mePosxBarrel->Fill( rechitx );
			      mePosyBarrel->Fill( rechity );
			      meErrxBarrel->Fill( rechiterrx );			      
			      meErryBarrel->Fill( rechiterry );
			      meResxBarrel->Fill( rechitresx );
			      meResyBarrel->Fill( rechitresy );
			      mePullxBarrel->Fill( rechitpullx );
			      mePullyBarrel->Fill( rechitpully );
			      meNpixBarrel->Fill( npix );
			      meNxpixBarrel->Fill( nxpix );
			      meNypixBarrel->Fill( nypix );
			      meChargeBarrel->Fill( charge );
			      meResXvsAlphaBarrel->Fill( alpha, fabs(rechitresx) );
			      meResYvsAlphaBarrel->Fill( alpha, fabs(rechitresy) );
			      meResXvsBetaBarrel->Fill( beta, fabs(rechitresx) );
			      meResYvsBetaBarrel->Fill( beta, fabs(rechitresy) );
			      mePullXvsAlphaBarrel->Fill( alpha, rechitpullx );
			      mePullYvsAlphaBarrel->Fill( alpha, rechitpully );
			      mePullXvsBetaBarrel->Fill( beta, rechitpullx );
			      mePullYvsBetaBarrel->Fill( beta, rechitpully );
			      mePullXvsPhiBarrel->Fill( phi, rechitpullx );
			      mePullYvsPhiBarrel->Fill( phi, rechitpully );
			      mePullXvsEtaBarrel->Fill( eta, rechitpullx );
			      mePullYvsEtaBarrel->Fill( eta, rechitpully );
			      
			      const PixelGeomDetUnit * theGeomDet 
				= dynamic_cast<const PixelGeomDetUnit*> ( tracker->idToDet(detId) );
			      //const PixelTopology * topol = (&(theGeomDet->specificTopology()));
			      
			      int tmp_nrows = theGeomDet->specificTopology().nrows();
			      
			      if ( tmp_nrows == 80 ) 
				{
				  mePosxBarrelHalfModule->Fill( rechitx );
				  mePosyBarrelHalfModule->Fill( rechity );
				  half = 1;
				}
			      else if ( tmp_nrows == 160 ) 
				{
				  mePosxBarrelFullModule->Fill( rechitx );
				  mePosyBarrelFullModule->Fill( rechity );
				  half = 0;
				}
			      else 
				LogWarning("SiPixelTrackingRecHitsValid") << "-------------------------------------------------- Wrong module size !!!";

			      float tmp1 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
			      float tmp2 = theGeomDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
			      
			      if ( tmp2<tmp1 ) 
				{ // flipped
				  mePosxBarrelFlippedLadders->Fill( rechitx );
				  mePosyBarrelFlippedLadders->Fill( rechity );
				  flipped = 1;
				
				  meResXvsAlphaBarrelFlippedLadders->Fill( alpha, fabs(rechitresx) );
				  meResYvsAlphaBarrelFlippedLadders->Fill( alpha, fabs(rechitresy) );
				  meResXvsBetaBarrelFlippedLadders->Fill( beta, fabs(rechitresx) );
				  meResYvsBetaBarrelFlippedLadders->Fill( beta, fabs(rechitresy) );
				  mePullXvsAlphaBarrelFlippedLadders->Fill( alpha, rechitpullx );
				  mePullYvsAlphaBarrelFlippedLadders->Fill( alpha, rechitpully );
				  mePullXvsBetaBarrelFlippedLadders->Fill( beta, rechitpullx );
				  mePullYvsBetaBarrelFlippedLadders->Fill( beta, rechitpully );
				  mePullXvsPhiBarrelFlippedLadders->Fill( phi, rechitpullx );
				  mePullYvsPhiBarrelFlippedLadders->Fill( phi, rechitpully );
				  mePullXvsEtaBarrelFlippedLadders->Fill( eta, rechitpullx );
				  mePullYvsEtaBarrelFlippedLadders->Fill( eta, rechitpully );
				
				  meWPullXvsAlphaBarrelFlippedLadders->Fill( alpha, fabs(rechitpullx) );
				  meWPullYvsAlphaBarrelFlippedLadders->Fill( alpha, fabs(rechitpully) );
				  meWPullXvsBetaBarrelFlippedLadders->Fill( beta, fabs(rechitpullx) );
				  meWPullYvsBetaBarrelFlippedLadders->Fill( beta, fabs(rechitpully) );
				}
			      else 
				{ // not flipped
				  mePosxBarrelNonFlippedLadders->Fill( rechitx );
				  mePosyBarrelNonFlippedLadders->Fill( rechity );
				  flipped = 0;
				
				  meResXvsAlphaBarrelNonFlippedLadders->Fill( alpha, fabs(rechitresx) );
				  meResYvsAlphaBarrelNonFlippedLadders->Fill( alpha, fabs(rechitresy) );
				  meResXvsBetaBarrelNonFlippedLadders->Fill( beta, fabs(rechitresx) );
				  meResYvsBetaBarrelNonFlippedLadders->Fill( beta, fabs(rechitresy) );
				  mePullXvsAlphaBarrelNonFlippedLadders->Fill( alpha, rechitpullx );
				  mePullYvsAlphaBarrelNonFlippedLadders->Fill( alpha, rechitpully );
				  mePullXvsBetaBarrelNonFlippedLadders->Fill( beta, rechitpullx );
				  mePullYvsBetaBarrelNonFlippedLadders->Fill( beta, rechitpully );
				  mePullXvsPhiBarrelNonFlippedLadders->Fill( phi, rechitpullx );
				  mePullYvsPhiBarrelNonFlippedLadders->Fill( phi, rechitpully );
				  mePullXvsEtaBarrelNonFlippedLadders->Fill( eta, rechitpullx );
				  mePullYvsEtaBarrelNonFlippedLadders->Fill( eta, rechitpully );

				  meWPullXvsAlphaBarrelNonFlippedLadders->Fill( alpha, fabs(rechitpullx) );
				  meWPullYvsAlphaBarrelNonFlippedLadders->Fill( alpha, fabs(rechitpully) );
				  meWPullXvsBetaBarrelNonFlippedLadders->Fill( beta, fabs(rechitpullx) );
				  meWPullYvsBetaBarrelNonFlippedLadders->Fill( beta, fabs(rechitpully) );
				}
			          
			      
			      layer  = tTopo->pxbLayer(detId);   // Layer: 1,2,3.
			      ladder = tTopo->pxbLadder(detId);  // Ladder: 1-20, 32, 44. 
			      mod   = tTopo->pxbModule(detId);  // Mod: 1-8.
			      
			      mePosxBarrelLayerModule[layer-1][mod-1]->Fill( rechitx );
			      mePosyBarrelLayerModule[layer-1][mod-1]->Fill( rechity );
			      meErrxBarrelLayerModule[layer-1][mod-1]->Fill( rechiterrx );
			      meErryBarrelLayerModule[layer-1][mod-1]->Fill( rechiterry );
			      meResxBarrelLayerModule[layer-1][mod-1]->Fill( rechitresx );
			      meResyBarrelLayerModule[layer-1][mod-1]->Fill( rechitresy );
			      mePullxBarrelLayerModule[layer-1][mod-1]->Fill( rechitpullx );
			      mePullyBarrelLayerModule[layer-1][mod-1]->Fill( rechitpully );
			      meNpixBarrelLayerModule[layer-1][mod-1]->Fill( npix );
			      meNxpixBarrelLayerModule[layer-1][mod-1]->Fill( nxpix );
			      meNypixBarrelLayerModule[layer-1][mod-1]->Fill( nypix );
			      meChargeBarrelLayerModule[layer-1][mod-1]->Fill( charge );
			      meResXvsAlphaBarrelLayerModule[layer-1][mod-1]->Fill( alpha, fabs(rechitresx) );
			      meResYvsAlphaBarrelLayerModule[layer-1][mod-1]->Fill( alpha, fabs(rechitresy) );
			      meResXvsBetaBarrelLayerModule[layer-1][mod-1]->Fill( beta, fabs(rechitresx) );
			      meResYvsBetaBarrelLayerModule[layer-1][mod-1]->Fill( beta, fabs(rechitresy) );
			      mePullXvsAlphaBarrelLayerModule[layer-1][mod-1]->Fill( alpha, rechitpullx );
			      mePullYvsAlphaBarrelLayerModule[layer-1][mod-1]->Fill( alpha, rechitpully );
			      mePullXvsBetaBarrelLayerModule[layer-1][mod-1]->Fill( beta, rechitpullx );
			      mePullYvsBetaBarrelLayerModule[layer-1][mod-1]->Fill( beta, rechitpully );
			      mePullXvsPhiBarrelLayerModule[layer-1][mod-1]->Fill( phi, rechitpullx );
			      mePullYvsPhiBarrelLayerModule[layer-1][mod-1]->Fill( phi, rechitpully );
			      mePullXvsEtaBarrelLayerModule[layer-1][mod-1]->Fill( eta, rechitpullx );
			      mePullYvsEtaBarrelLayerModule[layer-1][mod-1]->Fill( eta, rechitpully );
			    
			      meResxBarrelLayer[layer-1]->Fill( rechitresx );
			      meResyBarrelLayer[layer-1]->Fill( rechitresy );
			      mePullxBarrelLayer[layer-1]->Fill( rechitpullx );
			      mePullyBarrelLayer[layer-1]->Fill( rechitpully );

			      if ( tmp2<tmp1 ) 
				{ // flipped
				  meResXvsAlphaBarrelFlippedLaddersLayer[layer-1]->Fill( alpha, fabs(rechitresx) );
				  meResYvsAlphaBarrelFlippedLaddersLayer[layer-1]->Fill( alpha, fabs(rechitresy) );
				  meResXvsBetaBarrelFlippedLaddersLayer[layer-1]->Fill( beta, fabs(rechitresx) );
				  meResYvsBetaBarrelFlippedLaddersLayer[layer-1]->Fill( beta, fabs(rechitresy) );
				}
			      else
				{ // not flipped
				  meResXvsAlphaBarrelNonFlippedLaddersLayer[layer-1]->Fill( alpha, fabs(rechitresx) );
				  meResYvsAlphaBarrelNonFlippedLaddersLayer[layer-1]->Fill( alpha, fabs(rechitresy) );
				  meResXvsBetaBarrelNonFlippedLaddersLayer[layer-1]->Fill( beta, fabs(rechitresx) );
				  meResYvsBetaBarrelNonFlippedLaddersLayer[layer-1]->Fill( beta, fabs(rechitresy) );
				}
			      
			    }
			  else if ( (int)detId.subdetId() == (int)PixelSubdetector::PixelEndcap )
			    {
			      
			      side  = tTopo->pxfSide(detId);
			      disk  = tTopo->pxfDisk(detId);
			      blade = tTopo->pxfBlade(detId);
			      panel = tTopo->pxfPanel(detId);
			      plaq  = tTopo->pxfModule(detId); // also known as plaquette

			      if ( side==1 ) 
				{
				  if ( panel==1 )
				    {
				      mePosxZmPanel1->Fill( rechitx );
				      mePosyZmPanel1->Fill( rechity );
				      meErrxZmPanel1->Fill( rechiterrx );
				      meErryZmPanel1->Fill( rechiterry );
				      meResxZmPanel1->Fill( rechitresx );
				      meResyZmPanel1->Fill( rechitresy );
				      mePullxZmPanel1->Fill( rechitpullx );
				      mePullyZmPanel1->Fill( rechitpully );
				      meNpixZmPanel1->Fill( npix );
				      meNxpixZmPanel1->Fill( nxpix );
				      meNypixZmPanel1->Fill( nypix );
				      meChargeZmPanel1->Fill( charge );
				      meResXvsAlphaZmPanel1->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZmPanel1->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZmPanel1->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZmPanel1->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZmPanel1->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZmPanel1->Fill( alpha, rechitpully );
				      mePullXvsBetaZmPanel1->Fill( beta, rechitpullx );
				      mePullYvsBetaZmPanel1->Fill( beta, rechitpully );
				      mePullXvsPhiZmPanel1->Fill( phi, rechitpullx );
				      mePullYvsPhiZmPanel1->Fill( phi, rechitpully );
				      mePullXvsEtaZmPanel1->Fill( eta, rechitpullx );
				      mePullYvsEtaZmPanel1->Fill( eta, rechitpully );
				      
				      meWPullXvsAlphaZmPanel1->Fill( alpha, fabs(rechitpullx) );
				      meWPullYvsAlphaZmPanel1->Fill( alpha, fabs(rechitpully) );
				      meWPullXvsBetaZmPanel1->Fill( beta, fabs(rechitpullx) );
				      meWPullYvsBetaZmPanel1->Fill( beta, fabs(rechitpully) );
      
				      mePosxZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitx );
				      mePosyZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechity );
				      meErrxZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechiterrx );
				      meErryZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechiterry );
				      meResxZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitresx );
				      meResyZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitresy );
				      mePullxZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitpullx );
				      mePullyZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitpully );
				      meNpixZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( npix );
				      meNxpixZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( nxpix );
				      meNypixZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( nypix );
				      meChargeZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( charge );
				      meResXvsAlphaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpully );
				      mePullXvsBetaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpullx );
				      mePullYvsBetaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpully );
				      mePullXvsPhiZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpullx );
				      mePullYvsPhiZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpully );
				      mePullXvsEtaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpullx );
				      mePullYvsEtaZmPanel1DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpully );
				      
				    }
				  else if ( panel==2 )
				    {
				      mePosxZmPanel2->Fill( rechitx );
				      mePosyZmPanel2->Fill( rechity );
				      meErrxZmPanel2->Fill( rechiterrx );
				      meErryZmPanel2->Fill( rechiterry );
				      meResxZmPanel2->Fill( rechitresx );
				      meResyZmPanel2->Fill( rechitresy );
				      mePullxZmPanel2->Fill( rechitpullx );
				      mePullyZmPanel2->Fill( rechitpully );
				      meNpixZmPanel2->Fill( npix );
				      meNxpixZmPanel2->Fill( nxpix );
				      meNypixZmPanel2->Fill( nypix );
				      meChargeZmPanel2->Fill( charge );
				      meResXvsAlphaZmPanel2->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZmPanel2->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZmPanel2->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZmPanel2->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZmPanel2->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZmPanel2->Fill( alpha, rechitpully );
				      mePullXvsBetaZmPanel2->Fill( beta, rechitpullx );
				      mePullYvsBetaZmPanel2->Fill( beta, rechitpully );
				      mePullXvsPhiZmPanel2->Fill( phi, rechitpullx );
				      mePullYvsPhiZmPanel2->Fill( phi, rechitpully );
				      mePullXvsEtaZmPanel2->Fill( eta, rechitpullx );
				      mePullYvsEtaZmPanel2->Fill( eta, rechitpully );

				      meWPullXvsAlphaZmPanel2->Fill( alpha, fabs(rechitpullx) );
				      meWPullYvsAlphaZmPanel2->Fill( alpha, fabs(rechitpully) );
				      meWPullXvsBetaZmPanel2->Fill( beta, fabs(rechitpullx) );
				      meWPullYvsBetaZmPanel2->Fill( beta, fabs(rechitpully) );

				      mePosxZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitx );
				      mePosyZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechity );
				      meErrxZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechiterrx );
				      meErryZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechiterry );
				      meResxZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitresx );
				      meResyZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitresy );
				      mePullxZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitpullx );
				      mePullyZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitpully );
				      meNpixZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( npix );
				      meNxpixZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( nxpix );
				      meNypixZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( nypix );
				      meChargeZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( charge );
				      meResXvsAlphaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpully );
				      mePullXvsBetaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpullx );
				      mePullYvsBetaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpully );
				      mePullXvsPhiZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpullx );
				      mePullYvsPhiZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpully );
				      mePullXvsEtaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpullx );
				      mePullYvsEtaZmPanel2DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpully );

				    }
				  else LogWarning("SiPixelTrackingRecHitsValid") << "..............................................Wrong panel number !"; 
				} // if ( side==1 ) 
			      else if ( side==2 )
				{
				  if ( panel==1 )
				    {
				      mePosxZpPanel1->Fill( rechitx );
				      mePosyZpPanel1->Fill( rechity );
				      meErrxZpPanel1->Fill( rechiterrx );
				      meErryZpPanel1->Fill( rechiterry );
				      meResxZpPanel1->Fill( rechitresx );
				      meResyZpPanel1->Fill( rechitresy );
				      mePullxZpPanel1->Fill( rechitpullx );
				      mePullyZpPanel1->Fill( rechitpully );
				      meNpixZpPanel1->Fill( npix );
				      meNxpixZpPanel1->Fill( nxpix );
				      meNypixZpPanel1->Fill( nypix );
				      meChargeZpPanel1->Fill( charge );
				      meResXvsAlphaZpPanel1->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZpPanel1->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZpPanel1->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZpPanel1->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZpPanel1->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZpPanel1->Fill( alpha, rechitpully );
				      mePullXvsBetaZpPanel1->Fill( beta, rechitpullx );
				      mePullYvsBetaZpPanel1->Fill( beta, rechitpully );
				      mePullXvsPhiZpPanel1->Fill( phi, rechitpullx );
				      mePullYvsPhiZpPanel1->Fill( phi, rechitpully );
				      mePullXvsEtaZpPanel1->Fill( eta, rechitpullx );
				      mePullYvsEtaZpPanel1->Fill( eta, rechitpully );

				      meWPullXvsAlphaZpPanel1->Fill( alpha, fabs(rechitpullx) );
				      meWPullYvsAlphaZpPanel1->Fill( alpha, fabs(rechitpully) );
				      meWPullXvsBetaZpPanel1->Fill( beta, fabs(rechitpullx) );
				      meWPullYvsBetaZpPanel1->Fill( beta, fabs(rechitpully) );

				      mePosxZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitx );
				      mePosyZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechity );
				      meErrxZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechiterrx );
				      meErryZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechiterry );
				      meResxZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitresx );
				      meResyZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitresy );
				      mePullxZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitpullx );
				      mePullyZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( rechitpully );
				      meNpixZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( npix );
				      meNxpixZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( nxpix );
				      meNypixZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( nypix );
				      meChargeZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( charge );
				      meResXvsAlphaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpully );
				      mePullXvsBetaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpullx );
				      mePullYvsBetaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpully );
				      mePullXvsPhiZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpullx );
				      mePullYvsPhiZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpully );
				      mePullXvsEtaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpullx );
				      mePullYvsEtaZpPanel1DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpully );
				      
				    }
				  else if ( panel==2 )
				    {
				      mePosxZpPanel2->Fill( rechitx );
				      mePosyZpPanel2->Fill( rechity );
				      meErrxZpPanel2->Fill( rechiterrx );
				      meErryZpPanel2->Fill( rechiterry );
				      meResxZpPanel2->Fill( rechitresx );
				      meResyZpPanel2->Fill( rechitresy );
				      mePullxZpPanel2->Fill( rechitpullx );
				      mePullyZpPanel2->Fill( rechitpully );
				      meNpixZpPanel2->Fill( npix );
				      meNxpixZpPanel2->Fill( nxpix );
				      meNypixZpPanel2->Fill( nypix );
				      meChargeZpPanel2->Fill( charge );
				      meResXvsAlphaZpPanel2->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZpPanel2->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZpPanel2->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZpPanel2->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZpPanel2->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZpPanel2->Fill( alpha, rechitpully );
				      mePullXvsBetaZpPanel2->Fill( beta, rechitpullx );
				      mePullYvsBetaZpPanel2->Fill( beta, rechitpully );
				      mePullXvsPhiZpPanel2->Fill( phi, rechitpullx );
				      mePullYvsPhiZpPanel2->Fill( phi, rechitpully );
				      mePullXvsEtaZpPanel2->Fill( eta, rechitpullx );
				      mePullYvsEtaZpPanel2->Fill( eta, rechitpully );
				    
				      meWPullXvsAlphaZpPanel2->Fill( alpha, fabs(rechitpullx) );
				      meWPullYvsAlphaZpPanel2->Fill( alpha, fabs(rechitpully) );
				      meWPullXvsBetaZpPanel2->Fill( beta, fabs(rechitpullx) );
				      meWPullYvsBetaZpPanel2->Fill( beta, fabs(rechitpully) );

				      mePosxZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitx );
				      mePosyZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechity );
				      meErrxZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechiterrx );
				      meErryZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechiterry );
				      meResxZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitresx );
				      meResyZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitresy );
				      mePullxZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitpullx );
				      mePullyZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( rechitpully );
				      meNpixZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( npix );
				      meNxpixZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( nxpix );
				      meNypixZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( nypix );
				      meChargeZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( charge );
				      meResXvsAlphaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresx) );
				      meResYvsAlphaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, fabs(rechitresy) );
				      meResXvsBetaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresx) );
				      meResYvsBetaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, fabs(rechitresy) );
				      mePullXvsAlphaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpullx );
				      mePullYvsAlphaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( alpha, rechitpully );
				      mePullXvsBetaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpullx );
				      mePullYvsBetaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( beta, rechitpully );
				      mePullXvsPhiZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpullx );
				      mePullYvsPhiZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( phi, rechitpully );
				      mePullXvsEtaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpullx );
				      mePullYvsEtaZpPanel2DiskPlaq[disk-1][plaq-1]->Fill( eta, rechitpully );

				    }
				  else LogWarning("SiPixelTrackingRecHitsValid") << "..............................................Wrong panel number !"; 
				} //else if ( side==2 )
			      else LogWarning("SiPixelTrackingRecHitsValid") << ".......................................................Wrong side !" ;
			      
			    } // else if ( detId.subdetId()==PixelSubdetector::PixelEndcap )
			  else LogWarning("SiPixelTrackingRecHitsValid") << "Pixel rechit but we are not in the pixel detector" << (int)detId.subdetId() ;
			  
			  if(debugNtuple_.size()!=0)t_->Fill();

			} // if ( !matched.empty() )
		      //else
		      //cout << "---------------- RecHit with no associated SimHit !!! -------------------------- " << endl;
		      
		    } // matchedhit.
		  
		} // end of loop on hits
	      
	      mePixRecHitsPerTrack->Fill( n_hits );
	      //cout << "n_hits = " << n_hits << endl;
	      
	    } //end of loop on track 
	  
	} // tracks > 0.
      
    } //end of MTCCTrack

}
