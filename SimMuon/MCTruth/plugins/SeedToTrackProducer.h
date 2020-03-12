// -*- C++ -*-
//
// Package:    SeedToTrackProducer
// Class:      SeedToTrackProducer
//
/**\class SeedToTrackProducer SeedToTrackProducer.cc
 hugues/SeedToTrackProducer/plugins/SeedToTrackProducer.cc

 Description:

*/
//
// Original Author:  Hugues Brun
//         Created:  Tue, 05 Nov 2013 13:42:04 GMT
// $Id$
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

//
// class declaration
//

typedef math::Error<5>::type CovarianceMatrix;

class SeedToTrackProducer : public edm::one::EDProducer<> {
public:
  explicit SeedToTrackProducer(const edm::ParameterSet &);
  ~SeedToTrackProducer() override;

private:
  void beginJob() override;
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  virtual TrajectoryStateOnSurface seedTransientState(const TrajectorySeed &);
  // ----------member data ---------------------------

  edm::EDGetTokenT<TrajectorySeedCollection> L2seedsTagT_;
  edm::EDGetTokenT<edm::View<TrajectorySeed>> L2seedsTagS_;

  edm::ESHandle<MagneticField> theMGField;
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
};
