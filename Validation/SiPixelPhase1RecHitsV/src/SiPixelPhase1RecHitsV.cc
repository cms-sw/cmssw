// -*- C++ -*-
//
// Package:     SiPixelPhase1RecHitsV
// Class:       SiPixelPhase1RecHitsV
//

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

#include "Validation/SiPixelPhase1RecHitsV/interface/SiPixelPhase1RecHitsV.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

SiPixelPhase1RecHitsV::SiPixelPhase1RecHitsV(const edm::ParameterSet& iConfig) :
  SiPixelPhase1Base(iConfig),
  trackerHitAssociatorConfig_( iConfig, consumesCollector() ),
  srcToken_ ( consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("src")) )
{}

void SiPixelPhase1RecHitsV::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<SiPixelRecHitCollection> input;
  iEvent.getByToken(srcToken_, input);
  if (!input.isValid()) return;

  TrackerHitAssociator associate(iEvent, trackerHitAssociatorConfig_);

  SiPixelRecHitCollection::const_iterator it;
  for (it = input->begin(); it != input->end(); ++it) {
    auto id = DetId(it->detId());

    for(SiPixelRecHit const& rechit : *it) {
      SiPixelRecHit::ClusterRef const& clust = rechit.cluster();

      std::vector<PSimHit> associateSimHit;
      associateSimHit = associate.associateHit(rechit);
      std::vector<PSimHit>::const_iterator closestIt = associateSimHit.begin();

      LocalPoint lp = rechit.localPosition();
      float rechit_x = lp.x();
      float rechit_y = lp.y();
      
      LocalError lerr = rechit.localPositionError();
      float lerr_x = sqrt(lerr.xx());
      float lerr_y = sqrt(lerr.yy());

      // loop over associated sim hits and find the closest
      if ( !associateSimHit.empty() ) {
      float closestSimHit = 9999.9;

       for (std::vector<PSimHit>::const_iterator m = associateSimHit.begin(); m < associateSimHit.end(); m++) {
	  float sim_x1 ( (*m).entryPoint().x() ), sim_x2 ( (*m).exitPoint().x() ), sim_xpos ( 0.5*(sim_x1+sim_x2) );
	  float sim_y1 ( (*m).entryPoint().y() ), sim_y2 ( (*m).exitPoint().y() ), sim_ypos ( 0.5*(sim_y1+sim_y2) );

          float xres ( std::abs(sim_xpos - rechit_x) ), yres ( std::abs(sim_ypos - rechit_y) );
          float dist = std::sqrt( xres*xres + yres*yres );

          if ( dist < closestSimHit ) {
            closestSimHit = dist;
            closestIt = m;
          }
        }
      }

      // Sim Hit stuff
      
      if ( !associateSimHit.empty() ) {

        const PSimHit& simHit = *closestIt;
      
        int bunch = simHit.eventId().bunchCrossing();
        int event = simHit.eventId().event();

        float sim_x1 ( simHit.entryPoint().x() ), sim_x2 ( simHit.exitPoint().x() ), sim_xpos ( 0.5*(sim_x1 + sim_x2) );
        float sim_y1 ( simHit.entryPoint().y() ), sim_y2 ( simHit.exitPoint().y() ), sim_ypos ( 0.5*(sim_y1 + sim_y2) );

        float res_x = (rechit_x - sim_xpos) * 10000.0;
        float res_y = (rechit_y - sim_ypos) * 10000.0;

        float pull_x = ( rechit_x - sim_xpos ) / lerr_x;
        float pull_y = ( rechit_y - sim_ypos ) / lerr_y;
  
        // Now Plotting stuff
        if ( bunch == 0 ) histo[IN_TIME_BUNCH].fill(bunch, id, &iEvent);
        if ( bunch != 0 ) histo[OUT_TIME_BUNCH].fill(bunch, id, &iEvent);
 
        histo[NSIMHITS].fill(event, id, &iEvent);

        histo[RECHIT_X].fill(rechit_x, id, &iEvent);
        histo[RECHIT_Y].fill(rechit_y, id, &iEvent);

        histo[RES_X].fill(res_x, id, &iEvent);
        histo[RES_Y].fill(res_y, id, &iEvent);

        histo[ERROR_X].fill(lerr_x, id, &iEvent);
        histo[ERROR_Y].fill(lerr_y, id, &iEvent);

        histo[PULL_X].fill(pull_x, id, &iEvent);
        histo[PULL_Y].fill(pull_y, id, &iEvent); 
      }
    }
  }
}

DEFINE_FWK_MODULE(SiPixelPhase1RecHitsV);

