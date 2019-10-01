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

using namespace std;

GeometricSearchTracker* GeometricSearchTrackerBuilder::build(const GeometricDet* theGeometricTracker,
                                                             const TrackerGeometry* theGeomDetGeometry,
                                                             const TrackerTopology* tTopo,
                                                             const bool usePhase2Stacks) {
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
  bool useBrothers = !usePhase2Stacks;

  vector<const GeometricDet*> theGeometricDetLayers = theGeometricTracker->components();
  for (vector<const GeometricDet*>::const_iterator it = theGeometricDetLayers.begin();
       it != theGeometricDetLayers.end();
       it++) {
    if ((*it)->type() == GeometricDet::PixelBarrel) {
      vector<const GeometricDet*> thePxlBarGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = thePxlBarGeometricDetLayers.begin();
           it2 != thePxlBarGeometricDetLayers.end();
           it2++) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::PixelPhase1Barrel) {
      vector<const GeometricDet*> thePxlBarGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = thePxlBarGeometricDetLayers.begin();
           it2 != thePxlBarGeometricDetLayers.end();
           it2++) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::PixelPhase2Barrel) {
      vector<const GeometricDet*> thePxlBarGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = thePxlBarGeometricDetLayers.begin();
           it2 != thePxlBarGeometricDetLayers.end();
           it2++) {
        thePxlBarLayers.push_back(aPixelBarrelLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::TIB) {
      vector<const GeometricDet*> theTIBGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = theTIBGeometricDetLayers.begin();
           it2 != theTIBGeometricDetLayers.end();
           it2++) {
        theTIBLayers.push_back(aTIBLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::TOB) {
      vector<const GeometricDet*> theTOBGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = theTOBGeometricDetLayers.begin();
           it2 != theTOBGeometricDetLayers.end();
           it2++) {
        theTOBLayers.push_back(aTOBLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::OTPhase2Barrel) {
      vector<const GeometricDet*> theTOBGeometricDetLayers = (*it)->components();

      for (vector<const GeometricDet*>::const_iterator it2 = theTOBGeometricDetLayers.begin();
           it2 != theTOBGeometricDetLayers.end();
           it2++) {
        theTOBLayers.push_back(aPhase2OTBarrelLayerBuilder.build(*it2, theGeomDetGeometry, useBrothers));
      }
    }

    if ((*it)->type() == GeometricDet::PixelEndCap) {
      vector<const GeometricDet*> thePxlFwdGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = thePxlFwdGeometricDetLayers.begin();
           it2 != thePxlFwdGeometricDetLayers.end();
           it2++) {
        if ((*it2)->positionBounds().z() < 0)
          theNegPxlFwdLayers.push_back(aPixelForwardLayerBuilder.build(*it2, theGeomDetGeometry));
        if ((*it2)->positionBounds().z() > 0)
          thePosPxlFwdLayers.push_back(aPixelForwardLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::PixelPhase1EndCap) {
      vector<const GeometricDet*> thePxlFwdGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = thePxlFwdGeometricDetLayers.begin();
           it2 != thePxlFwdGeometricDetLayers.end();
           it2++) {
        if ((*it2)->positionBounds().z() < 0)
          theNegPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(*it2, theGeomDetGeometry));
        if ((*it2)->positionBounds().z() > 0)
          thePosPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::PixelPhase2EndCap) {
      vector<const GeometricDet*> thePxlFwdGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = thePxlFwdGeometricDetLayers.begin();
           it2 != thePxlFwdGeometricDetLayers.end();
           it2++) {
        //FIXME: this is just to keep the compatibility with the PixelPhase1 extension layout
        //hopefully we can get rid of it soon
        if ((*it2)->positionBounds().z() < 0) {
          if ((*it2)->type() == GeometricDet::PixelPhase2FullDisk ||
              (*it2)->type() == GeometricDet::PixelPhase2ReducedDisk)
            theNegPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(*it2, theGeomDetGeometry));
          else if ((*it2)->type() == GeometricDet::PixelPhase2TDRDisk)
            theNegPxlFwdLayers.push_back(aPhase2EndcapLayerBuilder.build(*it2, theGeomDetGeometry, false));
        } else if ((*it2)->positionBounds().z() > 0) {
          if ((*it2)->type() == GeometricDet::PixelPhase2FullDisk ||
              (*it2)->type() == GeometricDet::PixelPhase2ReducedDisk)
            thePosPxlFwdLayers.push_back(aPhase1PixelForwardLayerBuilder.build(*it2, theGeomDetGeometry));
          else if ((*it2)->type() == GeometricDet::PixelPhase2TDRDisk)
            thePosPxlFwdLayers.push_back(aPhase2EndcapLayerBuilder.build(*it2, theGeomDetGeometry, false));
        } else {
          edm::LogError("TkDetLayers") << "In PixelPhase2EndCap the disks are neither PixelPhase2FullDisk nor "
                                          "PixelPhase2ReducedDisk nor PixelPhase2TDRDisk...";
        }
      }
    }

    if ((*it)->type() == GeometricDet::TID) {
      vector<const GeometricDet*> theTIDGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = theTIDGeometricDetLayers.begin();
           it2 != theTIDGeometricDetLayers.end();
           it2++) {
        if ((*it2)->positionBounds().z() < 0)
          theNegTIDLayers.push_back(aTIDLayerBuilder.build(*it2, theGeomDetGeometry));
        if ((*it2)->positionBounds().z() > 0)
          thePosTIDLayers.push_back(aTIDLayerBuilder.build(*it2, theGeomDetGeometry));
      }
    }

    if ((*it)->type() == GeometricDet::OTPhase2EndCap) {
      vector<const GeometricDet*> theTIDGeometricDetLayers = (*it)->components();

      bool useBrothers = !usePhase2Stacks;
      for (vector<const GeometricDet*>::const_iterator it2 = theTIDGeometricDetLayers.begin();
           it2 != theTIDGeometricDetLayers.end();
           it2++) {
        if ((*it2)->positionBounds().z() < 0)
          theNegTIDLayers.push_back(aPhase2EndcapLayerBuilder.build(*it2, theGeomDetGeometry, useBrothers));
        if ((*it2)->positionBounds().z() > 0)
          thePosTIDLayers.push_back(aPhase2EndcapLayerBuilder.build(*it2, theGeomDetGeometry, useBrothers));
      }
    }

    if ((*it)->type() == GeometricDet::TEC) {
      vector<const GeometricDet*> theTECGeometricDetLayers = (*it)->components();
      for (vector<const GeometricDet*>::const_iterator it2 = theTECGeometricDetLayers.begin();
           it2 != theTECGeometricDetLayers.end();
           it2++) {
        if ((*it2)->positionBounds().z() < 0)
          theNegTECLayers.push_back(aTECLayerBuilder.build(*it2, theGeomDetGeometry));
        if ((*it2)->positionBounds().z() > 0)
          thePosTECLayers.push_back(aTECLayerBuilder.build(*it2, theGeomDetGeometry));
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
