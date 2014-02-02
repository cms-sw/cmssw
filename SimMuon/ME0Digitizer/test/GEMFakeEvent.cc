/** \class ME0FakeEvent
 *   Class for testing creation of fake digis in every ME0 strip
 *
 *  \author Khotilovich Vadim
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "Geometry/ME0Geometry/interface/ME0EtaPartition.h"
#include "Geometry/ME0Geometry/interface/ME0EtaPartitionSpecs.h"
#include "Geometry/ME0Geometry/interface/ME0Geometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/ME0Digi/interface/ME0DigiCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

using namespace std;


class ME0FakeEvent : public edm::EDProducer
{
public:

  ME0FakeEvent(const edm::ParameterSet& config);

  ~ME0FakeEvent(){}

  void produce(edm::Event& e, const edm::EventSetup& c);

private:

  int nEtaPartitions_;
};


ME0FakeEvent::ME0FakeEvent(const edm::ParameterSet& config) 
{
  cout <<"Initialize the Event Dump"<<endl;
  produces<ME0DigiCollection>();

  nEtaPartitions_ = config.getUntrackedParameter<int>("nEtaPartitions", 6); 
}

void
ME0FakeEvent::produce(edm::Event & ev, const edm::EventSetup& es)
{
  cout <<"RUN "<<ev.id().run() <<" EVENT "<<ev.id().event()<<endl;

  cout <<"Getting the me0 geometry"<<endl;
  edm::ESHandle<ME0Geometry> me0Geom;
  es.get<MuonGeometryRecord>().get(me0Geom);

  auto_ptr<ME0DigiCollection> pDigis(new ME0DigiCollection());

  for (int e = -1; e <= 1; e += 2)
  {
    for (int c = 1; c <= 36; ++c)
    {
      for (int l = 1; l <= 2; ++l)
      {
        for (int p = 1; p <= nEtaPartitions_; ++p)
        {
          ME0DetId d(e, 1, 1, l, c, p);
          const ME0EtaPartition* ep = me0Geom->etaPartition(d);
          int ntrips = ep->nstrips();
          cout <<"----- adding digis in ME0 "<<d<<" with number of strips "<< ntrips<<endl;
          for (int s = 1; s <= ntrips; ++s)
          {
            if (s == 1 || s == ntrips) cout<<" s="<<s<<endl;
            if (s == 2) cout<<"..."<<endl;
            ME0Digi me0Digi(s, 0);
            pDigis->insertDigi(d, me0Digi);  
          }
        }
      }
    }
  }

  cout<<"Will put ME0DigiCollection into event..."<<endl;
  ev.put(pDigis);
  cout<<"Done with event!"<<endl;
}

DEFINE_FWK_MODULE(ME0FakeEvent);

