#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include <iostream>
#include <string>

using namespace edm;

class TrackAnalyzer : public edm::EDAnalyzer {
public:
  TrackAnalyzer(const edm::ParameterSet& pset) {}

  ~TrackAnalyzer() override {}

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    //
    // extract tracker geometry
    //
    edm::ESHandle<TrackerGeometry> theG;
    setup.get<TrackerDigiGeometryRecord>().get(theG);

    using namespace std;

    std::cout << "\nEvent ID = " << event.id() << std::endl;

    edm::Handle<reco::TrackCollection> trackCollection;
    event.getByLabel("ctfWithMaterialTracks", trackCollection);
    //event.getByType(trackCollection);

    const reco::TrackCollection tC = *(trackCollection.product());

    std::cout << "Reconstructed " << tC.size() << " tracks" << std::endl;

    int i = 1;
    for (reco::TrackCollection::const_iterator track = tC.begin(); track != tC.end(); track++) {
      std::cout << "Track number " << i << std::endl;
      std::cout << "\tmomentum: " << track->momentum() << std::endl;
      std::cout << "\tPT: " << track->pt() << std::endl;
      std::cout << "\tvertex: " << track->vertex() << std::endl;
      std::cout << "\timpact parameter: " << track->d0() << std::endl;
      std::cout << "\tcharge: " << track->charge() << std::endl;
      std::cout << "\tnormalizedChi2: " << track->normalizedChi2() << std::endl;

      i++;
      cout << "\tFrom EXTRA : " << endl;
      cout << "\t\touter PT " << track->outerPt() << endl;
      std::cout << "\t direction: " << track->seedDirection() << std::endl;
      if (!track->seedRef().isNull())
        std::cout << "\t direction from seedRef: " << track->seedRef()->direction() << std::endl;
      //
      // try and access Hits
      //
      cout << "\t\tNumber of RecHits " << track->recHitsSize() << endl;
      for (trackingRecHit_iterator it = track->recHitsBegin(); it != track->recHitsEnd(); it++) {
        if ((*it)->isValid()) {
          cout << "\t\t\tRecHit on det " << (*it)->geographicalId().rawId() << endl;
          cout << "\t\t\tRecHit in LP " << (*it)->localPosition() << endl;
          cout << "\t\t\tRecHit in GP "
               << theG->idToDet((*it)->geographicalId())->surface().toGlobal((*it)->localPosition()) << endl;
        } else {
          cout << "\t\t Invalid Hit On " << (*it)->geographicalId().rawId() << endl;
        }
      }
    }
  }
};

DEFINE_FWK_MODULE(TrackAnalyzer);
