// File: StdHitNtuplizer.cc
// Description: see StdHitNtuplizer.h
// Authors: H. Cheung
//--------------------------------------------------------------


#include "SLHCUpgradeSimulations/Geometry/test/StdHitNtuplizer.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DataFormats
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

// Geometry
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

// For ROOT
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

// STD
#include <memory>
#include <string>
#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;

StdHitNtuplizer::StdHitNtuplizer(edm::ParameterSet const& conf) : 
  conf_(conf), 
  src_( conf.getParameter<edm::InputTag>( "src" ) ),
  rphiRecHits_( conf.getParameter<edm::InputTag>("rphiRecHits") ),
  stereoRecHits_( conf.getParameter<edm::InputTag>("stereoRecHits") ),
  matchedRecHits_( conf.getParameter<edm::InputTag>("matchedRecHits") ),
  tfile_(0), 
  pixeltree_(0), 
  striptree_(0),
  pixeltree2_(0)
{
}


StdHitNtuplizer::~StdHitNtuplizer() { }  

void StdHitNtuplizer::endJob() 
{
  std::cout << " StdHitNtuplizer::endJob" << std::endl;
  tfile_->Write();
  tfile_->Close();
}



void StdHitNtuplizer::beginJob()
{
  std::cout << " StdHitNtuplizer::beginJob" << std::endl;
  std::string outputFile = conf_.getParameter<std::string>("OutputFile");
 
  tfile_ = new TFile ( outputFile.c_str() , "RECREATE" );
  pixeltree_ = new TTree("PixelNtuple","Pixel hit analyzer ntuple");
  striptree_ = new TTree("StripNtuple","Strip hit analyzer ntuple");
  pixeltree2_ = new TTree("Pixel2Ntuple","Track Pixel hit analyzer ntuple");

  int bufsize = 64000;

  //Common Branch
  pixeltree_->Branch("evt",    &evt_,      "run/I:evtnum/I", bufsize);
  pixeltree_->Branch("pixel_recHit", &recHit_, 
    "x/F:y:xx:xy:yy:row:col:gx:gy:gz:subid/I:module:layer:ladder:disk:blade:panel:side:nsimhit:spreadx:spready:hx/F:hy:tx:ty:theta:phi", bufsize);
  pixeltree2_->Branch("evt",    &evt_,      "run/I:evtnum/I", bufsize);
  pixeltree2_->Branch("pixel_recHit", &recHit_, 
    "x/F:y:xx:xy:yy:row:col:gx:gy:gz:subid/I:module:layer:ladder:disk:blade:panel:side:nsimhit:spreadx:spready:hx/F:hy:tx:ty:theta:phi", bufsize);
  // Strip Branches 
  striptree_->Branch("evt",    &evt_,      "run/I:evtnum/I", bufsize);
  striptree_->Branch("strip_recHit", &striprecHit_,
    "x/F:y:xx:xy:yy:row:col:gx:gy:gz:subid/I:layer:nsimhit:hx/F:hy:tx:ty:theta:phi", bufsize);

}

// Functions that gets called by framework every event
void StdHitNtuplizer::analyze(const edm::Event& e, const edm::EventSetup& es)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // geometry setup
  edm::ESHandle<TrackerGeometry>        geometry;

  es.get<TrackerDigiGeometryRecord>().get(geometry);
  const TrackerGeometry*  theGeometry = &(*geometry);

  // fastsim rechits
  //edm::Handle<SiTrackerGSRecHit2DCollection> theGSRecHits;
  //edm::InputTag hitProducer;
  //hitProducer = conf_.getParameter<edm::InputTag>("HitProducer");
  //e.getByLabel(hitProducer, theGSRecHits);

  edm::Handle<SiPixelRecHitCollection> recHitColl;
  e.getByLabel( src_, recHitColl);

  // for finding matched simhit
  TrackerHitAssociator associate( e, conf_ );

//  std::cout << " Step A: Standard RecHits found " << (recHitColl.product())->dataSize() << std::endl;
  if((recHitColl.product())->dataSize() > 0) {
    SiPixelRecHitCollection::const_iterator recHitIdIterator      = (recHitColl.product())->begin();
    SiPixelRecHitCollection::const_iterator recHitIdIteratorEnd   = (recHitColl.product())->end();

    std::string detname ;
    std::vector<PSimHit> matched;
    std::vector<PSimHit>::const_iterator closest_simhit;

// Loop over Detector IDs
    for ( ; recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++) {
      SiPixelRecHitCollection::DetSet detset = *recHitIdIterator;

      if( detset.empty() ) continue;
      DetId detId = DetId(detset.detId()); // Get the Detid object

      const GeomDet* geomDet( theGeometry->idToDet(detId) );
      
      // Loop over rechits for this detid
      SiPixelRecHitCollection::DetSet::const_iterator rechitRangeIteratorBegin = detset.begin();
      SiPixelRecHitCollection::DetSet::const_iterator rechitRangeIteratorEnd   = detset.end();
      SiPixelRecHitCollection::DetSet::const_iterator iterRecHit;
      for ( iterRecHit = rechitRangeIteratorBegin; 
           iterRecHit != rechitRangeIteratorEnd; ++iterRecHit) {
        // get matched simhit
        matched.clear();
        matched = associate.associateHit(*iterRecHit);
        if ( !matched.empty() ) {
          float closest = 9999.9;
          std::vector<PSimHit>::const_iterator closestit = matched.begin();
          LocalPoint lp = iterRecHit->localPosition();
          float rechit_x = lp.x();
          float rechit_y = lp.y();
          //loop over simhits and find closest
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
            if ( dist < closest ) {
                  closest = dist;
                  closestit = m;
            }
          } // end of simhit loop
          closest_simhit = closestit;
        } // end matched emtpy
        unsigned int subid = detId.subdetId();
        int layer_num = -99,ladder_num=-99,module_num=-99,disk_num=-99,blade_num=-99,panel_num=-99,side_num=-99;
        if ( (subid==1)||(subid==2) ) {
          // 1 = PXB, 2 = PXF
          if ( subid ==  PixelSubdetector::PixelBarrel ) {
            layer_num   = tTopo->pxbLayer(detId.rawId());
	    ladder_num	= tTopo->pxbLadder(detId.rawId());
	    module_num	= tTopo->pxbModule(detId.rawId());
	    std::cout <<"\ndetId = "<<subid<<" : "<<tTopo->pxbLayer(detId.rawId())<<" , "<<tTopo->pxbLadder(detId.rawId())<<" , "<< tTopo->pxbModule(detId.rawId());
          } else if ( subid ==  PixelSubdetector::PixelEndcap ) {
	    module_num	= tTopo->pxfModule(detId());
	    disk_num	= tTopo->pxfDisk(detId());
	    blade_num	= tTopo->pxfBlade(detId());
	    panel_num	= tTopo->pxfPanel(detId());
	    side_num	= tTopo->pxfSide(detId());
          }
          int num_simhit = matched.size();
          fillPRecHit(	subid,layer_num,ladder_num,module_num,disk_num,blade_num,panel_num,side_num,
			iterRecHit, num_simhit, closest_simhit, geomDet );
          fillEvt(e);
          pixeltree_->Fill();
          init();
        }
      } // end of rechit loop
    } // end of detid loop
  } // end of loop test on recHitColl size

// Now loop over recotracks

  edm::Handle<View<reco::Track> >  trackCollection;
  edm::InputTag trackProducer;
  trackProducer = conf_.getParameter<edm::InputTag>("trackProducer");
  e.getByLabel(trackProducer, trackCollection);

/*
  std::cout << " num of reco::Tracks with "
            << trackProducer.process()<<":"
            << trackProducer.label()<<":"
            << trackProducer.instance()
            << ": " << trackCollection->size() << "\n";
*/
  int rT = 0;
  for(View<reco::Track>::size_type i=0; i<trackCollection->size(); ++i){
      ++rT;
      RefToBase<reco::Track> track(trackCollection, i);
//      std::cout << " num of hits for track " << rT << " = " << track->recHitsSize() << std::endl;
      for(trackingRecHit_iterator ih=track->recHitsBegin(); ih != track->recHitsEnd(); ++ih) {
        TrackingRecHit * hit = (*ih)->clone();
        const DetId& detId =  hit->geographicalId();
        const GeomDet* geomDet( theGeometry->idToDet(detId) );

/////comment out begin
/*
        unsigned int subdetId = detId.subdetId();
        int layerNumber=0;
        int ringNumber = 0;
        int stereo = 0;
        std::string detname;
        if ( subdetId == StripSubdetector::TIB) {
          detname = "TIB";
          
          layerNumber = tTopo->tibLayer(detId.rawId);
          stereo = tTopo->tibStereo(detId.rawId);
        } else if ( subdetId ==  StripSubdetector::TOB ) {
          detname = "TOB";
	  
          layerNumber = tTopo->tobLayer(detId.rawId);
          stereo = tTopo->tobStereo(detId.rawId);
        } else if ( subdetId ==  StripSubdetector::TID) {
          detname = "TID";
          
          layerNumber = tTopo->tidWheel(detId.rawId);
          ringNumber = tTopo->tidRing(detId.rawId);
          stereo = tTopo->tidStereo(detId.rawId);
        } else if ( subdetId ==  StripSubdetector::TEC ) {
          detname = "TEC";
          
          layerNumber = tTopo->tecWheel(detId.rawId);
          ringNumber = tTopo->tecRing(detId.rawId);
          stereo = tTopo->tecStereo(detId.rawId);
        } else if ( subdetId ==  PixelSubdetector::PixelBarrel ) {
          detname = "PXB";
          
          layerNumber = tTopo->pxbLayer(detId.rawId);
          stereo = 1;
        } else if ( subdetId ==  PixelSubdetector::PixelEndcap ) {
          detname = "PXF";
          
          layerNumber = tTopo->pxfDisk(detId.rawId);
          stereo = 1;
        }
*/
//        std::cout << "RecHit in " << detname << " from detid " << detId.rawId()
//                  << " subdet = " << subdetId
//                  << " layer = " << layerNumber
//                  << " Stereo = " << stereo
//                  << std::endl;
        if(hit->isValid()) {
          unsigned int subid = detId.subdetId();
          if ( (subid==1)||(subid==2) ) {
            // 1 = PXB, 2 = PXF
            fillPRecHit(subid, ih, geomDet);
            fillEvt(e);
            pixeltree2_->Fill();
            init();
/*
            TrackingRecHit * hit = (*ih)->clone();
            LocalPoint lp = hit->localPosition();
            LocalError le = hit->localPositionError();
//            std::cout << "   lp x,y = " << lp.x() << " " << lp.y() << " lpe xx,xy,yy = "
//                  << le.xx() << " " << le.xy() << " " << le.yy() << std::endl;
            std::cout << "Found RecHit in " << detname << " from detid " << detId.rawId()
		<< " subdet = " << subdetId
		<< " layer = " << layerNumber
                << "global x/y/z/r = "
                 << geomDet->surface().toGlobal(lp).x() << " " 
                 << geomDet->surface().toGlobal(lp).y() << " " 
                 << geomDet->surface().toGlobal(lp).z() << " " 
                 << geomDet->surface().toGlobal(lp).perp() 
                << " err x/y = " << sqrt(le.xx()) << " " << sqrt(le.yy()) << std::endl;
*/
          }
        }
	delete hit;
      } //end of loop on tracking rechits
  } // end of loop on recotracks

  // now for strip rechits
  edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
  edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  e.getByLabel(rphiRecHits_, rechitsrphi);
  e.getByLabel(stereoRecHits_, rechitsstereo);
  e.getByLabel(matchedRecHits_, rechitsmatched);

  //std::cout << " Step A: Standard Strip RPHI RecHits found " << rechitsrphi.product()->dataSize() << std::endl;
  //std::cout << " Step A: Standard Strip Stereo RecHits found " << rechitsstereo.product()->dataSize() << std::endl;
  //std::cout << " Step A: Standard Strip Matched RecHits found " << rechitsmatched.product()->dataSize() << std::endl;
  if(rechitsrphi->size() > 0) {
    //Loop over all rechits in RPHI collection (can also loop only over DetId)
//    SiStripRecHit2DCollectionOld::const_iterator theRecHitRangeIteratorBegin = rechitsrphi->begin();
//    SiStripRecHit2DCollectionOld::const_iterator theRecHitRangeIteratorEnd   = rechitsrphi->end();
//    SiStripRecHit2DCollectionOld::const_iterator iterRecHit;
    SiStripRecHit2DCollection::const_iterator recHitIdIterator      = (rechitsrphi.product())->begin();
    SiStripRecHit2DCollection::const_iterator recHitIdIteratorEnd   = (rechitsrphi.product())->end();


    std::string detname ;

//    for ( iterRecHit = theRecHitRangeIteratorBegin; 
//          iterRecHit != theRecHitRangeIteratorEnd; ++iterRecHit) {
// Loop over Detector IDs
    for ( ; recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++) {
      SiStripRecHit2DCollection::DetSet detset = *recHitIdIterator;

      if( detset.empty() ) continue;
      DetId detId = DetId(detset.detId()); // Get the Detid object

//      const DetId& detId =  iterRecHit->geographicalId();
      const GeomDet* geomDet( theGeometry->idToDet(detId) );

      // Loop over rechits for this detid
      SiStripRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorBegin = detset.begin();
      SiStripRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorEnd   = detset.end();
      SiStripRecHit2DCollection::DetSet::const_iterator iterRecHit;
      for ( iterRecHit = rechitRangeIteratorBegin;
           iterRecHit != rechitRangeIteratorEnd; ++iterRecHit) {

/////comment out begin
/*
	unsigned int subdetId = detId.subdetId();
	int layerNumber=0;
	int ringNumber = 0;
	int stereo = 0;
	if ( subdetId == StripSubdetector::TIB) {
	  detname = "TIB";
	  
	  layerNumber = tTopo->tibLayer(detId.rawId);
	  stereo = tTopo->tibStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TOB ) {
	  detname = "TOB";
	  
	  layerNumber = tTopo->tobLayer(detId.rawId);
	  stereo = tTopo->tobStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TID) {
	  detname = "TID";
	  
	  layerNumber = tTopo->tidWheel(detId.rawId);
	  ringNumber = tTopo->tidRing(detId.rawId);
	  stereo = tTopo->tidStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TEC ) {
	  detname = "TEC";
	  
	  layerNumber = tTopo->tecWheel(detId.rawId);
	  ringNumber = tTopo->tecRing(detId.rawId);
	  stereo = tTopo->tecStereo(detId.rawId);
	} else if ( subdetId ==  PixelSubdetector::PixelBarrel ) {
	  detname = "PXB";
	  
	  layerNumber = tTopo->pxbLayer(detId.rawId);
	  stereo = 1;
	} else if ( subdetId ==  PixelSubdetector::PixelEndcap ) {
	  detname = "PXF";
	  
	  layerNumber = tTopo->pxfDisk(detId.rawId);
	  stereo = 1;
	}
*/	
//      std::cout << "Found SiStripRPhiRecHit in " << detname << " from detid " << detId.rawId()
//                << " subdet = " << subdetId
//                << " layer = " << layerNumber
//                << " Stereo = " << stereo
//                << std::endl;
//      std::cout << "Rechit global x/y/z/r : "
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).x() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).y() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).z() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).perp() << std::endl;
//comment out end
        unsigned int subid = detId.subdetId();
        fillSRecHit(subid, iterRecHit, geomDet);
        fillEvt(e);
        striptree_->Fill();
        init();
      } // end of rechit loop
    } // end of detid loop
  } // end of loop test on rechit size

  // now stereo hits
  if(rechitsstereo.product()->dataSize() > 0) {
    //Loop over all rechits in RPHI collection (can also loop only over DetId)
//    SiStripRecHit2DCollectionOld::const_iterator theRecHitRangeIteratorBegin = rechitsstereo->begin();
//    SiStripRecHit2DCollectionOld::const_iterator theRecHitRangeIteratorEnd   = rechitsstereo->end();
//    SiStripRecHit2DCollectionOld::const_iterator iterRecHit;
    SiStripRecHit2DCollection::const_iterator recHitIdIterator      = (rechitsstereo.product())->begin();
    SiStripRecHit2DCollection::const_iterator recHitIdIteratorEnd   = (rechitsstereo.product())->end();

    std::string detname ;

// Loop over Detector IDs
    for ( ; recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++) {
      SiStripRecHit2DCollection::DetSet detset = *recHitIdIterator;
//    for ( iterRecHit = theRecHitRangeIteratorBegin; 
//          iterRecHit != theRecHitRangeIteratorEnd; ++iterRecHit) {

      if( detset.empty() ) continue;
      DetId detId = DetId(detset.detId()); // Get the Detid object

//      const DetId& detId =  iterRecHit->geographicalId();
      const GeomDet* geomDet( theGeometry->idToDet(detId) );

      // Loop over rechits for this detid
      SiStripRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorBegin = detset.begin();
      SiStripRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorEnd   = detset.end();
      SiStripRecHit2DCollection::DetSet::const_iterator iterRecHit;
      for ( iterRecHit = rechitRangeIteratorBegin;
           iterRecHit != rechitRangeIteratorEnd; ++iterRecHit) {
/////comment out begin

/*	unsigned int subdetId = detId.subdetId();
	int layerNumber=0;
	int ringNumber = 0;
	int stereo = 0;
	if ( subdetId == StripSubdetector::TIB) {
	  detname = "TIB";
	  
	  layerNumber = tTopo->tibLayer(detId.rawId);
	  stereo = tTopo->tibStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TOB ) {
	  detname = "TOB";
	  
	  layerNumber = tTopo->tobLayer(detId.rawId);
	  stereo = tTopo->tobStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TID) {
	  detname = "TID";
	  
	  layerNumber = tTopo->tidWheel(detId.rawId);
	  ringNumber = tTopo->tidRing(detId.rawId);
	  stereo = tTopo->tidStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TEC ) {
	  detname = "TEC";
	  
	  layerNumber = tTopo->tecWheel(detId.rawId);
	  ringNumber = tTopo->tecRing(detId.rawId);
	  stereo = tTopo->tecStereo(detId.rawId);
	} else if ( subdetId ==  PixelSubdetector::PixelBarrel ) {
	  detname = "PXB";
	  
	  layerNumber = tTopo->pxbLayer(detId.rawId);
	  stereo = 1;
	} else if ( subdetId ==  PixelSubdetector::PixelEndcap ) {
	  detname = "PXF";
	  
	  layerNumber = tTopo->pxfDisk(detId.rawId);
	  stereo = 1;
	}
*/
//      std::cout << "Found SiStripStereoRecHit in " << detname << " from detid " << detId.rawId()
//                << " subdet = " << subdetId
//                << " layer = " << layerNumber
//                << " Stereo = " << stereo
//                << std::endl;
//      std::cout << "Rechit global x/y/z/r : "
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).x() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).y() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).z() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).perp() << std::endl;
//comment out end
        unsigned int subid = detId.subdetId();
        fillSRecHit(subid, iterRecHit, geomDet);
        fillEvt(e);
        striptree_->Fill();
        init();
      } // end of rechit loop
    } // end of detid loop
  } // end of loop test on rechit size

  // now matched hits
  if(rechitsmatched.product()->dataSize() > 0) {
    //Loop over all rechits in RPHI collection (can also loop only over DetId)
    //SiStripMatchedRecHit2DCollectionOld::const_iterator theRecHitRangeIteratorBegin = rechitsmatched->begin();
    //SiStripMatchedRecHit2DCollectionOld::const_iterator theRecHitRangeIteratorEnd   = rechitsmatched->end();
    //SiStripMatchedRecHit2DCollectionOld::const_iterator iterRecHit;
    SiStripMatchedRecHit2DCollection::const_iterator recHitIdIterator      = (rechitsmatched.product())->begin();
    SiStripMatchedRecHit2DCollection::const_iterator recHitIdIteratorEnd   = (rechitsmatched.product())->end();

    std::string detname ;

// Loop over Detector IDs
    for ( ; recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++) {
      SiStripMatchedRecHit2DCollection::DetSet detset = *recHitIdIterator;

      if( detset.empty() ) continue;
      DetId detId = DetId(detset.detId()); // Get the Detid object

//    for ( iterRecHit = theRecHitRangeIteratorBegin; 
//          iterRecHit != theRecHitRangeIteratorEnd; ++iterRecHit) {

//      const DetId& detId =  iterRecHit->geographicalId();
      const GeomDet* geomDet( theGeometry->idToDet(detId) );

      // Loop over rechits for this detid
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorBegin = detset.begin();
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorEnd   = detset.end();
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator iterRecHit;
      for ( iterRecHit = rechitRangeIteratorBegin;
           iterRecHit != rechitRangeIteratorEnd; ++iterRecHit) {
/////comment out begin
/*
	unsigned int subdetId = detId.subdetId();
	int layerNumber=0;
	int ringNumber = 0;
	int stereo = 0;
	if ( subdetId == StripSubdetector::TIB) {
	  detname = "TIB";
	  
	  layerNumber = tTopo->tibLayer(detId.rawId);
	  stereo = tTopo->tibStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TOB ) {
	  detname = "TOB";
	  
	  layerNumber = tTopo->tobLayer(detId.rawId);
	  stereo = tTopo->tobStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TID) {
	  detname = "TID";
	  
	  layerNumber = tTopo->tidWheel(detId.rawId);
	  ringNumber = tTopo->tidRing(detId.rawId);
	  stereo = tTopo->tidStereo(detId.rawId);
	} else if ( subdetId ==  StripSubdetector::TEC ) {
	  detname = "TEC";
	  
	  layerNumber = tTopo->tecWheel(detId.rawId);
	  ringNumber = tTopo->tecRing(detId.rawId);
	  stereo = tTopo->tecStereo(detId.rawId);
	} else if ( subdetId ==  PixelSubdetector::PixelBarrel ) {
	  detname = "PXB";
	  
	  layerNumber = tTopo->pxbLayer(detId.rawId);
	  stereo = 1;
	} else if ( subdetId ==  PixelSubdetector::PixelEndcap ) {
	  detname = "PXF";
	  
	  layerNumber = tTopo->pxfDisk(detId.rawId);
	  stereo = 1;
	}
	
*/
//      std::cout << "Found SiStripMatchedRecHit in " << detname << " from detid " << detId.rawId()
//                << " subdet = " << subdetId
//                << " layer = " << layerNumber
//                << " Stereo = " << stereo
//                << std::endl;
//      std::cout << "Rechit global x/y/z/r : "
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).x() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).y() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).z() << " " 
//                << geomDet->surface().toGlobal(iterRecHit->localPosition()).perp() << std::endl;
//comment out end
        unsigned int subid = detId.subdetId();
        fillSRecHit(subid, iterRecHit, geomDet);
        fillEvt(e);
        striptree_->Fill();
        init();
      } // end of rechit loop
    } // end of detidt loop
  } // end of loop test on rechit size
            
} // end analyze function

void StdHitNtuplizer::fillSRecHit(const int subid, 
                                   SiStripRecHit2DCollection::DetSet::const_iterator pixeliter,
                                   const GeomDet* theGeom)
{
  LocalPoint lp = pixeliter->localPosition();
  LocalError le = pixeliter->localPositionError();

  striprecHit_.x = lp.x();
  striprecHit_.y = lp.y();
  striprecHit_.xx = le.xx();
  striprecHit_.xy = le.xy();
  striprecHit_.yy = le.yy();
  GlobalPoint GP = theGeom->surface().toGlobal(pixeliter->localPosition());
  striprecHit_.gx = GP.x();
  striprecHit_.gy = GP.y();
  striprecHit_.gz = GP.z();
  striprecHit_.subid = subid;
}
void StdHitNtuplizer::fillSRecHit(const int subid, 
                                   SiStripMatchedRecHit2DCollection::DetSet::const_iterator pixeliter,
                                   const GeomDet* theGeom)
{
  LocalPoint lp = pixeliter->localPosition();
  LocalError le = pixeliter->localPositionError();

  striprecHit_.x = lp.x();
  striprecHit_.y = lp.y();
  striprecHit_.xx = le.xx();
  striprecHit_.xy = le.xy();
  striprecHit_.yy = le.yy();
  GlobalPoint GP = theGeom->surface().toGlobal(pixeliter->localPosition());
  striprecHit_.gx = GP.x();
  striprecHit_.gy = GP.y();
  striprecHit_.gz = GP.z();
  striprecHit_.subid = subid;
}
void StdHitNtuplizer::fillSRecHit(const int subid, 
                                   SiTrackerGSRecHit2DCollection::const_iterator pixeliter,
                                   const GeomDet* theGeom)
{
  LocalPoint lp = pixeliter->localPosition();
  LocalError le = pixeliter->localPositionError();

  striprecHit_.x = lp.x();
  striprecHit_.y = lp.y();
  striprecHit_.xx = le.xx();
  striprecHit_.xy = le.xy();
  striprecHit_.yy = le.yy();
  //MeasurementPoint mp = topol->measurementPosition(LocalPoint(striprecHit_.x, striprecHit_.y));
  //striprecHit_.row = mp.x();
  //striprecHit_.col = mp.y();
  GlobalPoint GP = theGeom->surface().toGlobal(pixeliter->localPosition());
  striprecHit_.gx = GP.x();
  striprecHit_.gy = GP.y();
  striprecHit_.gz = GP.z();
  striprecHit_.subid = subid;
}
// I know it is lazy to pass everything, but I'm doing it anyway. -EB
void StdHitNtuplizer::fillPRecHit(const int subid, 
                                  const int layer_num,const int ladder_num,const int module_num,
				  const int disk_num,const int blade_num,const int panel_num,const int side_num,
                                  SiPixelRecHitCollection::DetSet::const_iterator pixeliter,
                                  const int num_simhit,
                                  std::vector<PSimHit>::const_iterator closest_simhit,
                                  const GeomDet* PixGeom)
{
  LocalPoint lp = pixeliter->localPosition();
  LocalError le = pixeliter->localPositionError();

  recHit_.x = lp.x();
  recHit_.y = lp.y();
  recHit_.xx = le.xx();
  recHit_.xy = le.xy();
  recHit_.yy = le.yy();
  //MeasurementPoint mp = topol->measurementPosition(LocalPoint(recHit_.x, recHit_.y));
  //recHit_.row = mp.x();
  //recHit_.col = mp.y();
  GlobalPoint GP = PixGeom->surface().toGlobal(pixeliter->localPosition());
  recHit_.gx = GP.x();
  recHit_.gy = GP.y();
  recHit_.gz = GP.z();

  SiPixelRecHit::ClusterRef const& Cluster =  pixeliter->cluster();
  recHit_.spreadx = Cluster->sizeX();
  recHit_.spready = Cluster->sizeY();

  recHit_.subid = subid;
  recHit_.nsimhit = num_simhit;
 
  recHit_.layer	= layer_num;
  recHit_.ladder= ladder_num;
  recHit_.module= module_num;
  recHit_.module= module_num;
  recHit_.disk	= disk_num;
  recHit_.blade	= blade_num;
  recHit_.panel	= panel_num;
  recHit_.side	= side_num;

  //std::cout << "num_simhit = " << num_simhit << std::endl;
  if(num_simhit > 0) {
    float sim_x1 = (*closest_simhit).entryPoint().x();
    float sim_x2 = (*closest_simhit).exitPoint().x();
    recHit_.hx = 0.5*(sim_x1+sim_x2);
    float sim_y1 = (*closest_simhit).entryPoint().y();
    float sim_y2 = (*closest_simhit).exitPoint().y();
    recHit_.hy = 0.5*(sim_y1+sim_y2);
    //std::cout << "num_simhit x, y = " << 0.5*(sim_x1+sim_x2) << " " << 0.5*(sim_y1+sim_y2) << std::endl;
  }
/*
       std::cout << "Found RecHit in " << subid
                 << " global x/y/z : "
                 << PixGeom->surface().toGlobal(pixeliter->localPosition()).x() << " " 
                 << PixGeom->surface().toGlobal(pixeliter->localPosition()).y() << " " 
                 << PixGeom->surface().toGlobal(pixeliter->localPosition()).z() << std::endl;
*/
}
void StdHitNtuplizer::fillPRecHit(const int subid, 
                                  trackingRecHit_iterator ih,
                                  const GeomDet* PixGeom)
{
  TrackingRecHit * pixeliter = (*ih)->clone();
  LocalPoint lp = pixeliter->localPosition();
  LocalError le = pixeliter->localPositionError();

  recHit_.x = lp.x();
  recHit_.y = lp.y();
  recHit_.xx = le.xx();
  recHit_.xy = le.xy();
  recHit_.yy = le.yy();
  GlobalPoint GP = PixGeom->surface().toGlobal(pixeliter->localPosition());
  recHit_.gx = GP.x();
  recHit_.gy = GP.y();
  recHit_.gz = GP.z();
  delete pixeliter;
  recHit_.subid = subid;
}

void
StdHitNtuplizer::fillEvt(const edm::Event& E)
{
   evt_.run = E.id().run();
   evt_.evtnum = E.id().event();
}

void StdHitNtuplizer::init()
{
  evt_.init();
  recHit_.init();
  striprecHit_.init();
}

void StdHitNtuplizer::evt::init()
{
  int dummy_int = 9999;
  run = dummy_int;
  evtnum = dummy_int;
}

void StdHitNtuplizer::RecHit::init()
{
  float dummy_float = 9999.0;

  x = dummy_float;
  y = dummy_float;
  xx = dummy_float;
  xy = dummy_float; 
  yy = dummy_float;
  row = dummy_float;
  col = dummy_float;
  gx = dummy_float;
  gy = dummy_float;
  gz = dummy_float;
  nsimhit = 0;
  subid=-99;
  module=-99;
  layer=-99;
  ladder=-99;
  disk=-99;
  blade=-99;
  panel=-99;
  side=-99;
  spreadx=0;
  spready=0;
  hx = dummy_float;
  hy = dummy_float;
  tx = dummy_float;
  ty = dummy_float;
  theta = dummy_float;
  phi = dummy_float;
}

//define this as a plug-in
DEFINE_FWK_MODULE(StdHitNtuplizer);

