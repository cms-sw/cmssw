#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"

#include "PixelBarrelLayerBuilder.h"
#include "Phase2OTBarrelLayerBuilder.h"
#include "PixelForwardLayerBuilder.h"
#include "Phase2EndcapLayerBuilder.h"
#include "TIBLayerBuilder.h"
#include "TOBLayerBuilder.h"
#include "TIDLayerBuilder.h"
#include "TECLayerBuilder.h"

#include "Geometry/TrackerGeometryBuilder/interface/trackerHierarchy.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "DataFormats/Common/interface/Trie.h"
#include <boost/function.hpp>
#include <boost/bind.hpp>

using namespace std;

GeometricSearchTracker* GeometricSearchTrackerBuilder::build(const GeometricDet* theGeometricTracker,
                                                             const TrackerGeometry* theGeomDetGeometry,
                                                             const TrackerTopology* tTopo) {
  PixelBarrelLayerBuilder aPixelBarrelLayerBuilder;
  Phase2OTBarrelLayerBuilder aPhase2OTBarrelLayerBuilder;
  PixelForwardLayerBuilder<PixelBlade, PixelForwardLayer> aPixelForwardLayerBuilder;
  PixelForwardLayerBuilder<Phase1PixelBlade, PixelForwardLayerPhase1> aPhase1PixelForwardLayerBuilder;
  Phase2EndcapLayerBuilder aPhase2EndcapLayerBuilder;
  TIBLayerBuilder aTIBLayerBuilder;
  TOBLayerBuilder aTOBLayerBuilder;
  TIDLayerBuilder aTIDLayerBuilder;
  TECLayerBuilder aTECLayerBuilder;

  vector<BarrelDetLayer const*> thePxlBarLayers;
  vector<BarrelDetLayer const*> theTIBLayers;
  vector<BarrelDetLayer const*> theTOBLayers;
  vector<ForwardDetLayer const*> theNegPxlFwdLayers;
  vector<ForwardDetLayer const*> thePosPxlFwdLayers;
  vector<ForwardDetLayer const*> theNegTIDLayers;
  vector<ForwardDetLayer const*> thePosTIDLayers;
  vector<ForwardDetLayer const*> theNegTECLayers;
  vector<ForwardDetLayer const*> thePosTECLayers;

  vector<const GeometricDet*> theGeometricDetLayers = theGeometricTracker->components();
  for (auto theGeometricDetLayer : theGeometricDetLayers) {
    if (theGeometricDetLayer->type() == GeometricDet::PixelBarrel) {
      vector<const GeometricDet*> thePxlBarGeometricDetLayers = theGeometricDetLayer->components();
      for (auto thePxlBarGeometricDetLayer : thePxlBarGeometricDetLayers) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(thePxlBarGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::PixelPhase1Barrel) {
      vector<const GeometricDet*> thePxlBarGeometricDetLayers = theGeometricDetLayer->components();
      for (auto thePxlBarGeometricDetLayer : thePxlBarGeometricDetLayers) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(thePxlBarGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::PixelPhase2Barrel) {
      vector<const GeometricDet*> thePxlBarGeometricDetLayers = theGeometricDetLayer->components();
      for (auto thePxlBarGeometricDetLayer : thePxlBarGeometricDetLayers) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(thePxlBarGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::TIB) {
      vector<const GeometricDet*> theTIBGeometricDetLayers = theGeometricDetLayer->components();
      for (auto theTIBGeometricDetLayer : theTIBGeometricDetLayers) {
        theTIBLayers.push_back(aTIBLayerBuilder.build(theTIBGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::TOB) {
      vector<const GeometricDet*> theTOBGeometricDetLayers = theGeometricDetLayer->components();
      for (auto theTOBGeometricDetLayer : theTOBGeometricDetLayers) {
        theTOBLayers.push_back(aTOBLayerBuilder.build(theTOBGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::OTPhase2Barrel) {
      vector<const GeometricDet*> theTOBGeometricDetLayers = theGeometricDetLayer->components();
      for (auto theTOBGeometricDetLayer : theTOBGeometricDetLayers) {
        theTOBLayers.push_back(aPhase2OTBarrelLayerBuilder.build(theTOBGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::PixelEndCap) {
      vector<const GeometricDet*> thePxlFwdGeometricDetLayers = theGeometricDetLayer->components();
      for (auto thePxlFwdGeometricDetLayer : thePxlFwdGeometricDetLayers) {
        if (thePxlFwdGeometricDetLayer->positionBounds().z() < 0)
          theNegPxlFwdLayers.push_back(aPixelForwardLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry));
        if (thePxlFwdGeometricDetLayer->positionBounds().z() > 0)
          thePosPxlFwdLayers.push_back(aPixelForwardLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::PixelPhase1EndCap) {
      vector<const GeometricDet*> thePxlFwdGeometricDetLayers = theGeometricDetLayer->components();
      for (auto thePxlFwdGeometricDetLayer : thePxlFwdGeometricDetLayers) {
        if (thePxlFwdGeometricDetLayer->positionBounds().z() < 0)
          theNegPxlFwdLayers.push_back(
              aPhase1PixelForwardLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry));
        if (thePxlFwdGeometricDetLayer->positionBounds().z() > 0)
          thePosPxlFwdLayers.push_back(
              aPhase1PixelForwardLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::PixelPhase2EndCap) {
      vector<const GeometricDet*> thePxlFwdGeometricDetLayers = theGeometricDetLayer->components();
      for (auto thePxlFwdGeometricDetLayer : thePxlFwdGeometricDetLayers) {
        //FIXME: this is just to keep the compatibility with the PixelPhase1 extension layout
        //hopefully we can get rid of it soon
        if (thePxlFwdGeometricDetLayer->positionBounds().z() < 0) {
          if (thePxlFwdGeometricDetLayer->type() == GeometricDet::PixelPhase2FullDisk ||
              thePxlFwdGeometricDetLayer->type() == GeometricDet::PixelPhase2ReducedDisk)
            theNegPxlFwdLayers.push_back(
                aPhase1PixelForwardLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry));
          else if (thePxlFwdGeometricDetLayer->type() == GeometricDet::PixelPhase2TDRDisk)
            theNegPxlFwdLayers.push_back(
                aPhase2EndcapLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry, false));
        } else if (thePxlFwdGeometricDetLayer->positionBounds().z() > 0) {
          if (thePxlFwdGeometricDetLayer->type() == GeometricDet::PixelPhase2FullDisk ||
              thePxlFwdGeometricDetLayer->type() == GeometricDet::PixelPhase2ReducedDisk)
            thePosPxlFwdLayers.push_back(
                aPhase1PixelForwardLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry));
          else if (thePxlFwdGeometricDetLayer->type() == GeometricDet::PixelPhase2TDRDisk)
            thePosPxlFwdLayers.push_back(
                aPhase2EndcapLayerBuilder.build(thePxlFwdGeometricDetLayer, theGeomDetGeometry, false));
        } else {
          edm::LogError("TkDetLayers") << "In PixelPhase2EndCap the disks are neither PixelPhase2FullDisk nor "
                                          "PixelPhase2ReducedDisk nor PixelPhase2TDRDisk...";
        }
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::TID) {
      vector<const GeometricDet*> theTIDGeometricDetLayers = theGeometricDetLayer->components();
      for (auto theTIDGeometricDetLayer : theTIDGeometricDetLayers) {
        if (theTIDGeometricDetLayer->positionBounds().z() < 0)
          theNegTIDLayers.push_back(aTIDLayerBuilder.build(theTIDGeometricDetLayer, theGeomDetGeometry));
        if (theTIDGeometricDetLayer->positionBounds().z() > 0)
          thePosTIDLayers.push_back(aTIDLayerBuilder.build(theTIDGeometricDetLayer, theGeomDetGeometry));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::OTPhase2EndCap) {
      vector<const GeometricDet*> theTIDGeometricDetLayers = theGeometricDetLayer->components();
      for (auto theTIDGeometricDetLayer : theTIDGeometricDetLayers) {
        if (theTIDGeometricDetLayer->positionBounds().z() < 0)
          theNegTIDLayers.push_back(aPhase2EndcapLayerBuilder.build(theTIDGeometricDetLayer, theGeomDetGeometry, true));
        if (theTIDGeometricDetLayer->positionBounds().z() > 0)
          thePosTIDLayers.push_back(aPhase2EndcapLayerBuilder.build(theTIDGeometricDetLayer, theGeomDetGeometry, true));
      }
    }

    if (theGeometricDetLayer->type() == GeometricDet::TEC) {
      vector<const GeometricDet*> theTECGeometricDetLayers = theGeometricDetLayer->components();
      for (auto theTECGeometricDetLayer : theTECGeometricDetLayers) {
        if (theTECGeometricDetLayer->positionBounds().z() < 0)
          theNegTECLayers.push_back(aTECLayerBuilder.build(theTECGeometricDetLayer, theGeomDetGeometry));
        if (theTECGeometricDetLayer->positionBounds().z() > 0)
          thePosTECLayers.push_back(aTECLayerBuilder.build(theTECGeometricDetLayer, theGeomDetGeometry));
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
