#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"

#include "PixelBarrelLayerBuilder.h"
#include "Phase2OTBarrelLayerBuilder.h"
#include "PixelForwardLayerBuilder.h"
#include "Phase2EndcapLayerBuilder.h"
#include "Phase2EndcapLayerDoubleDiskBuilder.h"
#include "TIBLayerBuilder.h"
#include "TOBLayerBuilder.h"
#include "TIDLayerBuilder.h"
#include "TECLayerBuilder.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

using namespace std;

GeometricSearchTracker *GeometricSearchTrackerBuilder::build(const GeometricDet *theGeometricTracker,
                                                             const TrackerGeometry *theGeomDetGeometry,
                                                             const TrackerTopology *tTopo,
                                                             const bool usePhase2Stacks) {
  PixelBarrelLayerBuilder aPixelBarrelLayerBuilder;
  Phase2OTBarrelLayerBuilder aPhase2OTBarrelLayerBuilder;
  PixelForwardLayerBuilder<PixelBlade, PixelForwardLayer> aPixelForwardLayerBuilder;
  PixelForwardLayerBuilder<Phase1PixelBlade, PixelForwardLayerPhase1> aPhase1PixelForwardLayerBuilder;
  Phase2EndcapLayerBuilder aPhase2EndcapLayerBuilder;
  Phase2EndcapLayerDoubleDiskBuilder aPhase2EndcapLayerDoubleDiskBuilder;
  TIBLayerBuilder aTIBLayerBuilder;
  TOBLayerBuilder aTOBLayerBuilder;
  TIDLayerBuilder aTIDLayerBuilder;
  TECLayerBuilder aTECLayerBuilder;

  vector<BarrelDetLayer const *> thePxlBarLayers;
  vector<BarrelDetLayer const *> theTIBLayers;
  vector<BarrelDetLayer const *> theTOBLayers;
  vector<ForwardDetLayer const *> theNegPxlFwdLayers;
  vector<ForwardDetLayer const *> thePosPxlFwdLayers;
  vector<ForwardDetLayer const *> theNegTIDLayers;
  vector<ForwardDetLayer const *> thePosTIDLayers;
  vector<ForwardDetLayer const *> theNegTECLayers;
  vector<ForwardDetLayer const *> thePosTECLayers;
  bool useBrothers = !usePhase2Stacks;

  auto const &theGeometricDetLayers = theGeometricTracker->components();
  for (auto const &theGeomDetLayer : theGeometricDetLayers) {
    if (theGeomDetLayer->type() == GeometricDet::PixelBarrel) {
      auto const &thePxlBarGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : thePxlBarGeometricDetLayers) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::PixelPhase1Barrel) {
      auto const &thePxlBarGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : thePxlBarGeometricDetLayers) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::PixelPhase2Barrel) {
      auto const &thePxlBarGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : thePxlBarGeometricDetLayers) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::TIB) {
      auto const &theTIBGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : theTIBGeometricDetLayers) {
        theTIBLayers.push_back(aTIBLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::TOB) {
      auto const &theTOBGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : theTOBGeometricDetLayers) {
        theTOBLayers.push_back(aTOBLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::OTPhase2Barrel) {
      auto const &theTOBGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : theTOBGeometricDetLayers) {
        theTOBLayers.push_back(aPhase2OTBarrelLayerBuilder.build(thisGeomDet, theGeomDetGeometry, useBrothers));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::PixelEndCap) {
      auto const &thePxlFwdGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : thePxlFwdGeometricDetLayers) {
        if (thisGeomDet->positionBounds().z() < 0)
          theNegPxlFwdLayers.push_back(aPixelForwardLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
        else
          thePosPxlFwdLayers.push_back(aPixelForwardLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::PixelPhase1EndCap) {
      auto const &thePxlFwdGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : thePxlFwdGeometricDetLayers) {
        if (thisGeomDet->positionBounds().z() < 0)
          theNegPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
        else
          thePosPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::PixelPhase2EndCap) {
      auto const &thePxlFwdGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : thePxlFwdGeometricDetLayers) {
        //FIXME: this is just to keep the compatibility with the PixelPhase1 extension layout
        //hopefully we can get rid of it soon
        if (thisGeomDet->positionBounds().z() < 0) {
          if (thisGeomDet->type() == GeometricDet::PixelPhase2FullDisk ||
              thisGeomDet->type() == GeometricDet::PixelPhase2ReducedDisk)
            theNegPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
          else if (thisGeomDet->type() == GeometricDet::PixelPhase2TDRDisk)
            theNegPxlFwdLayers.push_back(aPhase2EndcapLayerBuilder.build(thisGeomDet, theGeomDetGeometry, false));
          else if (thisGeomDet->type() == GeometricDet::PixelPhase2DoubleDisk)
            theNegPxlFwdLayers.push_back(aPhase2EndcapLayerDoubleDiskBuilder.build(thisGeomDet, theGeomDetGeometry));
        } else if (thisGeomDet->positionBounds().z() > 0) {
          if (thisGeomDet->type() == GeometricDet::PixelPhase2FullDisk ||
              thisGeomDet->type() == GeometricDet::PixelPhase2ReducedDisk)
            thePosPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
          else if (thisGeomDet->type() == GeometricDet::PixelPhase2TDRDisk)
            thePosPxlFwdLayers.push_back(aPhase2EndcapLayerBuilder.build(thisGeomDet, theGeomDetGeometry, false));
          else if (thisGeomDet->type() == GeometricDet::PixelPhase2DoubleDisk)
            thePosPxlFwdLayers.push_back(aPhase2EndcapLayerDoubleDiskBuilder.build(thisGeomDet, theGeomDetGeometry));

        } else {
          edm::LogError("TkDetLayers") << "In PixelPhase2EndCap the disks are neither PixelPhase2FullDisk nor "
                                          "PixelPhase2ReducedDisk nor PixelPhase2TDRDisk...";
        }
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::TID) {
      auto const &theTIDGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : theTIDGeometricDetLayers) {
        if (thisGeomDet->positionBounds().z() < 0)
          theNegTIDLayers.push_back(aTIDLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
        else
          thePosTIDLayers.push_back(aTIDLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::OTPhase2EndCap) {
      auto const &theTIDGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : theTIDGeometricDetLayers) {
        if (thisGeomDet->positionBounds().z() < 0)
          theNegTIDLayers.push_back(aPhase2EndcapLayerBuilder.build(thisGeomDet, theGeomDetGeometry, useBrothers));
        else
          thePosTIDLayers.push_back(aPhase2EndcapLayerBuilder.build(thisGeomDet, theGeomDetGeometry, useBrothers));
      }
    }

    if (theGeomDetLayer->type() == GeometricDet::TEC) {
      auto const &theTECGeometricDetLayers = theGeomDetLayer->components();
      for (auto const &thisGeomDet : theTECGeometricDetLayers) {
        if (thisGeomDet->positionBounds().z() < 0)
          theNegTECLayers.push_back(aTECLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
        else
          thePosTECLayers.push_back(aTECLayerBuilder.build(thisGeomDet, theGeomDetGeometry));
      }
    }
  }

  return new GeometricSearchTracker(thePxlBarLayers,
                                    theTIBLayers,
                                    theTOBLayers,
                                    theNegPxlFwdLayers,
                                    theNegTIDLayers,
                                    theNegTECLayers,
                                    thePosPxlFwdLayers,
                                    thePosTIDLayers,
                                    thePosTECLayers,
                                    tTopo);
}

GeometricSearchTracker *GeometricSearchTrackerBuilder::build(const GeometricDet *theGeometricTracker,
                                                             const TrackerGeometry *theGeomDetGeometry,
                                                             const TrackerTopology *tTopo,
                                                             const MTDGeometry *mtd,
                                                             const MTDTopology *mTopo,
                                                             const bool usePhase2Stacks) {
  //Tracker part
  GeometricSearchTracker *theSearchTrack = this->build(theGeometricTracker, theGeomDetGeometry, tTopo, usePhase2Stacks);

  theSearchTrack->addDetLayerGeometry();
  theSearchTrack->mtdDetLayerGeometry->buildLayers(mtd, mTopo);
  theSearchTrack->mtdDetLayerGeometry->sortLayers();

  std::vector<const BarrelDetLayer *> barrel;
  for (auto &&e : theSearchTrack->mtdDetLayerGeometry->allBarrelLayers()) {
    auto p = dynamic_cast<const BarrelDetLayer *>(e);
    if (p) {
      barrel.push_back(p);
    }
  }
  std::vector<const ForwardDetLayer *> backward;
  for (auto &&e : theSearchTrack->mtdDetLayerGeometry->allBackwardLayers()) {
    auto p = dynamic_cast<const ForwardDetLayer *>(e);
    if (p) {
      backward.push_back(p);
    }
  }
  std::vector<const ForwardDetLayer *> forward;
  for (auto &&e : theSearchTrack->mtdDetLayerGeometry->allForwardLayers()) {
    auto p = dynamic_cast<const ForwardDetLayer *>(e);
    if (p) {
      forward.push_back(p);
    }
  }
  //Include the MTD layers in the TrackerSearchGeometry
  theSearchTrack->addMTDLayers(barrel, backward, forward);
  return theSearchTrack;
}
