#include <memory>
#include "SimMuon/MCTruth/plugins/MuonAssociatorEDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

MuonAssociatorEDProducer::MuonAssociatorEDProducer(const edm::ParameterSet& parset):
  tracksTag(parset.getParameter< edm::InputTag >("tracksTag")),
  tpTag(parset.getParameter< edm::InputTag >("tpTag")),
  ignoreMissingTrackCollection(parset.getUntrackedParameter<bool>("ignoreMissingTrackCollection",false)),
  parset_(parset)
{
  LogTrace("MuonAssociatorEDProducer") << "constructing  MuonAssociatorEDProducer" << parset_.dump();
  produces<reco::RecoToSimCollection>();
  produces<reco::SimToRecoCollection>();
}

MuonAssociatorEDProducer::~MuonAssociatorEDProducer() {}

void MuonAssociatorEDProducer::beginJob(const edm::EventSetup& setup) {
  LogTrace("MuonAssociatorEDProducer") << "MuonAssociatorEDProducer::beginJob : constructing MuonAssociatorByHits";
  associatorByHits = new MuonAssociatorByHits::MuonAssociatorByHits(parset_);
}

void MuonAssociatorEDProducer::endJob() {}

void MuonAssociatorEDProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
   using namespace edm;

   Handle<TrackingParticleCollection>  TPCollection ;
   LogTrace("MuonAssociatorEDProducer") <<"getting TrackingParticle collection - "<<tpTag;
   event.getByLabel(tpTag, TPCollection);
   LogTrace("MuonAssociatorEDProducer") <<"\t... size = "<<TPCollection->size();

   Handle<edm::View<reco::Track> > trackCollection;
   LogTrace("MuonAssociatorEDProducer") <<"getting reco::Track collection - "<<tracksTag;
   bool trackAvailable = event.getByLabel (tracksTag, trackCollection);
   if (trackAvailable) LogTrace("MuonAssociatorEDProducer") <<"\t... size = "<<trackCollection->size();
   else LogTrace("MuonAssociatorEDProducer") <<"\t... NOT FOUND.";

   std::auto_ptr<reco::RecoToSimCollection> rts;
   std::auto_ptr<reco::SimToRecoCollection> str;

   if (ignoreMissingTrackCollection && !trackAvailable) {
     //the track collection is not in the event and we're being told to ignore this.
     //do not output anything to the event, other wise this would be considered as inefficiency.
     LogTrace("MuonAssociatorEDProducer") << "\n ignoring missing track collection." << "\n";
   }   
   else {
     LogTrace("MuonAssociatorEDProducer") << "\n >>> Calling associateRecoToSim method <<<" << "\n";
     reco::RecoToSimCollection recSimColl = associatorByHits->associateRecoToSim(trackCollection,TPCollection,&event,&setup);
     
     LogTrace("MuonAssociatorEDProducer") << "\n >>> Calling associateSimToReco method <<<" << "\n";
     reco::SimToRecoCollection simRecColl = associatorByHits->associateSimToReco(trackCollection,TPCollection,&event,&setup);
     
     rts.reset(new reco::RecoToSimCollection(recSimColl));
     str.reset(new reco::SimToRecoCollection(simRecColl));
     
     event.put(rts);
     event.put(str);
   }
}
