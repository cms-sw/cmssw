#include <memory>
#include "SimMuon/MCTruth/plugins/MuonAssociatorEDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

MuonAssociatorEDProducer::MuonAssociatorEDProducer(const edm::ParameterSet& parset):
  tracksTag(parset.getParameter< edm::InputTag >("tracksTag")),
  tpTag(parset.getParameter< edm::InputTag >("tpTag")),
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

   Handle<edm::View<reco::Track> > trackCollection;
   LogTrace("MuonAssociatorEDProducer") <<"getting reco::Track collection - "<<tracksTag;
   event.getByLabel (tracksTag, trackCollection );
   LogTrace("MuonAssociatorEDProducer") <<"... size = "<<trackCollection->size();

   Handle<TrackingParticleCollection>  TPCollection ;
   LogTrace("MuonAssociatorEDProducer") <<"getting TrackingParticle collection - "<<tpTag;
   event.getByLabel(tpTag, TPCollection);
   LogTrace("MuonAssociatorEDProducer") <<"... size = "<<TPCollection->size();

   LogTrace("MuonAssociatorEDProducer") << "\n >>> Calling associateRecoToSim method <<<" << "\n";
   reco::RecoToSimCollection recSimColl = 
	associatorByHits->associateRecoToSim(trackCollection,TPCollection,&event,&setup);

   LogTrace("MuonAssociatorEDProducer") << "\n >>> Calling associateSimToReco method <<<" << "\n";
   reco::SimToRecoCollection simRecColl = 
	associatorByHits->associateSimToReco(trackCollection,TPCollection,&event,&setup);

   std::auto_ptr<reco::RecoToSimCollection> rts(new reco::RecoToSimCollection(recSimColl));
   std::auto_ptr<reco::SimToRecoCollection> str(new reco::SimToRecoCollection(simRecColl));

   event.put(rts);
   event.put(str);
}
