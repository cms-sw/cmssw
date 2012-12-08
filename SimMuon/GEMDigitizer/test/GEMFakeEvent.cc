/** \class GEMFakeEvent
 *   Class for testing creation of fake digis in every GEM strip
 *
 *  \author Khotilovich Vadim
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;


class GEMFakeEvent : public edm::EDProducer
{
public:

  GEMFakeEvent(const edm::ParameterSet& config);

  ~GEMFakeEvent(){}

  void produce(edm::Event& e, const edm::EventSetup& c);

private:

  int nEtaPartitions_;
};


GEMFakeEvent::GEMFakeEvent(const edm::ParameterSet& config) 
{
  cout <<"Initialize the Event Dump"<<endl;
  produces<GEMDigiCollection>();

  nEtaPartitions_ = config.getUntrackedParameter<int>("nEtaPartitions", 6); 
}

void
GEMFakeEvent::produce(edm::Event & ev, const edm::EventSetup& es)
{
  cout <<"RUN "<<ev.id().run() <<" EVENT "<<ev.id().event()<<endl;

  cout <<"Getting the gem geometry"<<endl;
  edm::ESHandle<GEMGeometry> gemGeom;
  es.get<MuonGeometryRecord>().get(gemGeom);

  auto_ptr<GEMDigiCollection> pDigis(new GEMDigiCollection());

  for (int e = -1; e <= 1; e += 2)
  {
    for (int c = 1; c <= 36; ++c)
    {
      for (int l = 1; l <= 2; ++l)
      {
        for (int p = 1; p <= nEtaPartitions_; ++p)
        {
          GEMDetId d(e, 1, 1, l, c, p);
          const GEMEtaPartition* ep = gemGeom->etaPartition(d);
          int ntrips = ep->nstrips();
          cout <<"----- adding digis in GEM "<<d<<" with number of strips "<< ntrips<<endl;
          for (int s = 1; s <= ntrips; ++s)
          {
            if (s == 1 || s == ntrips) cout<<" s="<<s<<endl;
            if (s == 2) cout<<"..."<<endl;
            GEMDigi gemDigi(s, 0);
            pDigis->insertDigi(d, gemDigi);  
          }
        }
      }
    }
  }

  cout<<"Will put GEMDigiCollection into event..."<<endl;
  ev.put(pDigis);
  cout<<"Done with event!"<<endl;
}

DEFINE_FWK_MODULE(GEMFakeEvent);

