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

  /// Perform some sanity checks of the configuration
  edm::LogVerbatim("MuonAssociatorByHits") << "constructing  MuonAssociatorByHits" << parset_.dump();
  edm::LogVerbatim("MuonAssociatorByHits") << "\n MuonAssociatorByHits will associate reco::Tracks with "<<tracksTag
					   << "\n\t\t and TrackingParticles with "<<tpTag;
  const std::string recoTracksLabel = tracksTag.label();
  const std::string recoTracksInstance = tracksTag.instance();

  // check and fix inconsistent input settings
  // tracks with hits only on muon detectors
  if (recoTracksLabel == "standAloneMuons" || recoTracksLabel == "standAloneSETMuons" ||
      recoTracksLabel == "cosmicMuons" || recoTracksLabel == "hltL2Muons") {
    if (parset_.getParameter<bool>("UseTracker")) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseTracker = true"<<"\n ---> setting UseTracker = false ";
      parset_.addParameter<bool>("UseTracker",false);
    }
    if (!parset_.getParameter<bool>("UseMuon")) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseMuon = false"<<"\n ---> setting UseMuon = true ";
      parset_.addParameter<bool>("UseMuon",true);
    }
  }
  // tracks with hits only on tracker
  if (recoTracksLabel == "generalTracks" || recoTracksLabel == "ctfWithMaterialTracksP5LHCNavigation" ||
      recoTracksLabel == "hltL3TkTracksFromL2" || 
      (recoTracksLabel == "hltL3Muons" && recoTracksInstance == "L2Seeded")) {
    if (parset_.getParameter<bool>("UseMuon")) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseMuon = true"<<"\n ---> setting UseMuon = false ";
      parset_.addParameter<bool>("UseMuon",false);
    }
    if (!parset_.getParameter<bool>("UseTracker")) {
      edm::LogWarning("MuonAssociatorByHits") 
	<<"\n*** WARNING : inconsistent input tracksTag = "<<tracksTag
	<<"\n with UseTracker = false"<<"\n ---> setting UseTracker = true ";
      parset_.addParameter<bool>("UseTracker",true);
    }
  }

}

MuonAssociatorEDProducer::~MuonAssociatorEDProducer() {}

void MuonAssociatorEDProducer::beginJob() {
  LogTrace("MuonAssociatorEDProducer") << "MuonAssociatorEDProducer::beginJob : constructing MuonAssociatorByHits";
  associatorByHits = new MuonAssociatorByHits(parset_);
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
     edm::LogVerbatim("MuonAssociatorEDProducer") 
       <<"\n >>> RecoToSim association <<< \n"
       <<"     Track collection : "
       <<tracksTag.label()<<":"<<tracksTag.instance()<<" (size = "<<trackCollection->size()<<") \n"
       <<"     TrackingParticle collection : "
       <<tpTag.label()<<":"<<tpTag.instance()<<" (size = "<<TPCollection->size()<<")";
     
     reco::RecoToSimCollection recSimColl = associatorByHits->associateRecoToSim(trackCollection,TPCollection,&event,&setup);
     
     edm::LogVerbatim("MuonAssociatorEDProducer") 
       <<"\n >>> SimToReco association <<< \n"
       <<"     TrackingParticle collection : "
       <<tpTag.label()<<":"<<tpTag.instance()<<" (size = "<<TPCollection->size()<<") \n"
       <<"     Track collection : "
       <<tracksTag.label()<<":"<<tracksTag.instance()<<" (size = "<<trackCollection->size()<<")";
     
     reco::SimToRecoCollection simRecColl = associatorByHits->associateSimToReco(trackCollection,TPCollection,&event,&setup);
     
     rts.reset(new reco::RecoToSimCollection(recSimColl));
     str.reset(new reco::SimToRecoCollection(simRecColl));
     
     event.put(rts);
     event.put(str);
   }
}
