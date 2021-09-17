#ifndef TrackAssociator_MuonDetIdAssociator_h
#define TrackAssociator_MuonDetIdAssociator_h 1
// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      MuonDetIdAssociator
//
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Fri Apr 21 10:59:41 PDT 2006
//
//

#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TAMuonChamberMatch.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixStateInfo.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"

class MuonDetIdAssociator : public DetIdAssociator {
public:
  MuonDetIdAssociator()
      : DetIdAssociator(48, 48, 0.125), geometry_(nullptr), cscbadchambers_(nullptr), includeBadChambers_(false){};
  MuonDetIdAssociator(const int nPhi, const int nEta, const double etaBinSize)
      : DetIdAssociator(nPhi, nEta, etaBinSize),
        geometry_(nullptr),
        cscbadchambers_(nullptr),
        includeBadChambers_(false){};

  MuonDetIdAssociator(int nPhi,
                      int nEta,
                      double etaBinSize,
                      const GlobalTrackingGeometry* geom,
                      const CSCBadChambers* badChambers,
                      bool includeBadChambers,
                      bool includeGEM,
                      bool includeME0)
      : DetIdAssociator(nPhi, nEta, etaBinSize),
        geometry_(geom),
        cscbadchambers_(badChambers),
        includeBadChambers_(includeBadChambers),
        includeGEM_(includeGEM),
        includeME0_(includeME0){};

  virtual void setGeometry(const GlobalTrackingGeometry* ptr) { geometry_ = ptr; }

  virtual void setCSCBadChambers(const CSCBadChambers* ptr) { cscbadchambers_ = ptr; }

  const GeomDet* getGeomDet(const DetId& id) const override;

  const char* name() const override { return "AllMuonDetectors"; }

protected:
  void check_setup() const override;

  GlobalPoint getPosition(const DetId& id) const override;

  void getValidDetIds(unsigned int, std::vector<DetId>&) const override;

  std::pair<const_iterator, const_iterator> getDetIdPoints(const DetId& id,
                                                           std::vector<GlobalPoint>& points) const override;

  bool insideElement(const GlobalPoint& point, const DetId& id) const override;

  const GlobalTrackingGeometry* geometry_;

  const CSCBadChambers* cscbadchambers_;
  bool includeBadChambers_;
  bool includeGEM_;
  bool includeME0_;
};
#endif
