#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class SimMuFilter : public edm::stream::EDFilter<>
{
 public:

   explicit SimMuFilter(const edm::ParameterSet&);
   ~SimMuFilter() override;

 private:

   virtual void beginJob();
   virtual void endJob();
   bool filter(edm::Event&, const edm::EventSetup&) override;
   
 private:
   
   edm::EDGetTokenT<std::vector<SimTrack> > simTracksToken_;
   edm::EDGetTokenT<edm::PSimHitContainer> simHitsMuonRPCToken_;
   edm::EDGetTokenT<edm::PSimHitContainer> simHitsMuonCSCToken_;
   edm::EDGetTokenT<edm::PSimHitContainer> simHitsMuonDTToken_;
   
   edm::Handle<std::vector<SimTrack> > simTracksHandle;
   edm::Handle<edm::PSimHitContainer> simHitsMuonRPCHandle;
   edm::Handle<edm::PSimHitContainer> simHitsMuonCSCHandle;
   edm::Handle<edm::PSimHitContainer> simHitsMuonDTHandle;
   
   int nMuSel_;
};

SimMuFilter::SimMuFilter(const edm::ParameterSet& iConfig)
{
   simTracksToken_    = consumes<std::vector<SimTrack> >(iConfig.getParameter<edm::InputTag>("simTracksInput"));
   simHitsMuonRPCToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("simHitsMuonRPCInput"));
   simHitsMuonCSCToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("simHitsMuonCSCInput"));
   simHitsMuonDTToken_ = consumes<edm::PSimHitContainer>(iConfig.getParameter<edm::InputTag>("simHitsMuonDTInput"));
   nMuSel_             = iConfig.getParameter<int>("nMuSel");
}

SimMuFilter::~SimMuFilter()
{
}

bool SimMuFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   iEvent.getByToken(simTracksToken_,simTracksHandle);
   iEvent.getByToken(simHitsMuonRPCToken_,simHitsMuonRPCHandle);
   iEvent.getByToken(simHitsMuonCSCToken_,simHitsMuonCSCHandle);
   iEvent.getByToken(simHitsMuonDTToken_,simHitsMuonDTHandle);
      
   const std::vector<SimTrack> &simTracks = *simTracksHandle.product();
   
   int nTracks = simTracks.size();

   int nPass = 0;
   
   for(int it=0;it<nTracks;it++)
     {
	SimTrack simTrk = simTracks[it];
	
	int pdgId = simTrk.type();
	float pt = simTrk.momentum().pt();
	
	if( abs(pdgId) != 13 ) continue;
	if( pt < 3. ) continue;
	
	int nSimHitRPC = 0;
	int nSimHitCSC = 0;
	int nSimHitDT = 0;

	for( PSimHitContainer::const_iterator simHitIt = simHitsMuonRPCHandle->begin();simHitIt!=simHitsMuonRPCHandle->end();simHitIt++ )
	  {
	     if( simHitIt->trackId() != simTrk.trackId() ) continue;
	     
	     nSimHitRPC++;
	  }	     
	
	for( PSimHitContainer::const_iterator simHitIt = simHitsMuonCSCHandle->begin();simHitIt!=simHitsMuonCSCHandle->end();simHitIt++ )
	  {
	     if( simHitIt->trackId() != simTrk.trackId() ) continue;
	     
	     nSimHitCSC++;
	  }
	
	for( PSimHitContainer::const_iterator simHitIt = simHitsMuonDTHandle->begin();simHitIt!=simHitsMuonDTHandle->end();simHitIt++ )
	  {
	     if( simHitIt->trackId() != simTrk.trackId() ) continue;
	     
	     nSimHitDT++;
	  }
	
	if( nSimHitRPC+nSimHitCSC+nSimHitDT > 0 ) nPass++;
     }   
   
   return (nPass >= nMuSel_);
}

void SimMuFilter::beginJob()
{
}

void SimMuFilter::endJob()
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimMuFilter);

