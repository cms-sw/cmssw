// File: TestAssociator.cc
// Author:  P. Azzi
// Creation Date:  PA May 2006 Initial version.
//                 Pixel RecHits added by V.Chiochia - 18/5/06
// 25/9/17 (W.T.Ford) Add Phase 2 Outer Tracker, common template function
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "SimTracker/TrackerHitAssociation/test/TestAssociator.h"

//--- for Geometry:
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//--- for RecHits
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

//--- for SimHits
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

using namespace std;
using namespace edm;

void TestAssociator::analyze(const edm::Event& e, const edm::EventSetup& es) {
  using namespace edm;
  int pixelcounter = 0;
  int stripcounter = 0;

  edm::LogVerbatim("TrackAssociator") << " === TestAssociator ";

  // Get inputs
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
  edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
  edm::Handle<SiPixelRecHitCollection> pixelrechits;
  edm::Handle<Phase2TrackerRecHit1DCollectionNew> phase2rechits;

  // Construct the associator object
  TrackerHitAssociator associate(e, trackerHitAssociatorConfig_);

  // Process each RecHit collection in turn
  if (doPixel_) {
    e.getByToken(siPixelRecHitsToken, pixelrechits);
    if (pixelrechits.isValid())
      printRechitSimhit(pixelrechits, "Pixel        ", pixelcounter, associate);
  }
  if (doStrip_) {
    if (useOTph2_) {
      e.getByToken(siPhase2RecHitsToken, phase2rechits);
      if (phase2rechits.isValid())
        printRechitSimhit(phase2rechits, "Phase 2 OT   ", stripcounter, associate);
    } else {
      e.getByToken(rphiRecHitToken, rechitsrphi);
      if (rechitsrphi.isValid())
        printRechitSimhit(rechitsrphi, "Strip rphi   ", stripcounter, associate);
      e.getByToken(stereoRecHitToken, rechitsstereo);
      if (rechitsstereo.isValid())
        printRechitSimhit(rechitsstereo, "Strip stereo ", stripcounter, associate);
      e.getByToken(matchedRecHitToken, rechitsmatched);
      if (rechitsmatched.isValid())
        printRechitSimhit(rechitsmatched, "Strip matched", stripcounter, associate);
    }
  }
  if (!doPixel_ && !doStrip_)
    throw edm::Exception(errors::Configuration, "Strip and pixel association disabled");

  edm::LogVerbatim("TrackAssociator") << " === TestAssociator end\n ";
}

template <typename rechitType>
void TestAssociator::printRechitSimhit(const edm::Handle<edmNew::DetSetVector<rechitType>> rechitCollection,
                                       const char* rechitName,
                                       int hitCounter,
                                       TrackerHitAssociator& associate) const {
  std::vector<PSimHit> matched;
  // Loop over sensors with detected rechits of type rechitType
  for (auto const& theDetSet : *rechitCollection) {
    DetId detid = theDetSet.detId();
    uint32_t myid = detid.rawId();
    // Loop over the RecHits in this sensor
    for (auto const& rechit : theDetSet) {
      hitCounter++;
      edm::LogVerbatim("TrackAssociator") << hitCounter << ") " << rechitName << " RecHit subDet, DetId " << detid.subdetId() << ", " << myid << " Pos = " << rechit.localPosition();
      bool isPixel = false;
      float mindist = 999999;
      float dist, distx, disty;
      PSimHit closest;
      // Find the vector of SimHits matching this RecHit
      matched.clear();
      matched = associate.associateHit(rechit);
      if (!matched.empty()) {
        // Print out the SimHit positions and residuals
        for (auto const& m : matched) {
          edm::LogVerbatim("TrackAssociator") << " simtrack ID = " << m.trackId() << "                            Simhit Pos = " << m.localPosition();
          // Seek the smallest residual
          if (const SiPixelRecHit* dummy = dynamic_cast<const SiPixelRecHit*>(&rechit)) {
            isPixel = true;
            dist = (rechit.localPosition() - m.localPosition()).mag();  // pixels measure 2 dimensions
          } else {
            isPixel = false;
            dist = fabs(rechit.localPosition().x() - m.localPosition().x());
          }
          if (dist < mindist) {
            mindist = dist;
            closest = m;
          }
        }
	std::ostringstream st1;
        st1 << " Closest Simhit = " << closest.localPosition();
        if (isPixel) {
          distx = fabs(rechit.localPosition().x() - closest.localPosition().x());
          disty = fabs(rechit.localPosition().y() - closest.localPosition().y());
          st1 << ", diff(x,y) = (" << distx << ", " << disty << ")";
        }
        edm::LogVerbatim("TrackAssociator") << st1.str() << ", |diff| = " << mindist;
      }
    }
  }  // end loop on detSets
}

//---------------
// Constructor --
//---------------

TestAssociator::TestAssociator(edm::ParameterSet const& conf)
    : trackerHitAssociatorConfig_(conf, consumesCollector()),
      doPixel_(conf.getParameter<bool>("associatePixel")),
      doStrip_(conf.getParameter<bool>("associateStrip")),
      useOTph2_(conf.getParameter<bool>("usePhase2Tracker")) {
  matchedRecHitToken =
      consumes<edmNew::DetSetVector<SiStripMatchedRecHit2D>>(conf.getParameter<edm::InputTag>("matchedRecHit"));
  rphiRecHitToken = consumes<edmNew::DetSetVector<SiStripRecHit2D>>(conf.getParameter<edm::InputTag>("rphiRecHit"));
  stereoRecHitToken = consumes<edmNew::DetSetVector<SiStripRecHit2D>>(conf.getParameter<edm::InputTag>("stereoRecHit"));
  siPixelRecHitsToken =
      consumes<edmNew::DetSetVector<SiPixelRecHit>>(conf.getParameter<edm::InputTag>("siPixelRecHits"));
  siPhase2RecHitsToken =
      consumes<edmNew::DetSetVector<Phase2TrackerRecHit1D>>(conf.getParameter<edm::InputTag>("siPhase2RecHits"));
}

//---------------
// Destructor --
//---------------

TestAssociator::~TestAssociator() {}
