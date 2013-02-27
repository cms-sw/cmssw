#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"

//
// class declaration
//
class UpdatedMuonInnerTrackRef : public edm::EDProducer {
   public:
      explicit UpdatedMuonInnerTrackRef(const edm::ParameterSet&);
      ~UpdatedMuonInnerTrackRef();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      reco::TrackRef findNewRef(reco::TrackRef oldTrackRef, edm::Handle<reco::TrackCollection>& newTrackCollection);

      edm::InputTag muonTag_;
      edm::InputTag oldTrackTag_;
      edm::InputTag newTrackTag_;

      double maxInvPtDiff;
      double minDR;
};


/////////////////////////////////////////////////////////////////////////////////////
UpdatedMuonInnerTrackRef::UpdatedMuonInnerTrackRef(const edm::ParameterSet& pset)
{
   // What is being produced
   produces<std::vector<reco::Muon> >();

    // Input products
   muonTag_     = pset.getUntrackedParameter<edm::InputTag> ("MuonTag"    , edm::InputTag("muons"));
   oldTrackTag_ = pset.getUntrackedParameter<edm::InputTag> ("OldTrackTag", edm::InputTag("generalTracks"));
   newTrackTag_ = pset.getUntrackedParameter<edm::InputTag> ("NewTrackTag", edm::InputTag("generalTracksSkim"));

    // matching criteria products
   maxInvPtDiff=pset.getUntrackedParameter<double>("maxInvPtDiff", 0.005);
   minDR=pset.getUntrackedParameter<double>("minDR", 0.1);
} 

/////////////////////////////////////////////////////////////////////////////////////
UpdatedMuonInnerTrackRef::~UpdatedMuonInnerTrackRef(){
}

/////////////////////////////////////////////////////////////////////////////////////
void UpdatedMuonInnerTrackRef::beginJob() {
}

/////////////////////////////////////////////////////////////////////////////////////
void UpdatedMuonInnerTrackRef::endJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
void UpdatedMuonInnerTrackRef::produce(edm::Event& ev, const edm::EventSetup& iSetup)
{
      // Muon collection
      edm::Handle<edm::View<reco::Muon> > muonCollectionHandle;
      if (!ev.getByLabel(muonTag_, muonCollectionHandle)) {
            edm::LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }


      edm::Handle<reco::TrackCollection> oldTrackCollection;
      if (!ev.getByLabel(oldTrackTag_, oldTrackCollection)) {
            edm::LogError("") << ">>> Old Track collection does not exist !!!";
            return;
      }

      edm::Handle<reco::TrackCollection> newTrackCollection;
      if (!ev.getByLabel(newTrackTag_, newTrackCollection)) {
            edm::LogError("") << ">>> New Track collection does not exist !!!";
            return;
      }

      unsigned int muonCollectionSize = muonCollectionHandle->size();
      std::auto_ptr<reco::MuonCollection> newmuons (new reco::MuonCollection);


      for (unsigned int i=0; i<muonCollectionSize; i++) {
            edm::RefToBase<reco::Muon> mu = muonCollectionHandle->refAt(i);
            reco::Muon* newmu = mu->clone();

            if(mu->innerTrack().isNonnull()){ 
               reco::TrackRef newTrackRef = findNewRef(mu->innerTrack(), newTrackCollection);
/*               printf(" %6.2f %+6.2f %+6.2f --> ",mu->innerTrack()->pt (), mu->innerTrack()->eta(), mu->innerTrack()->phi());
               if(newTrackRef.isNonnull()){
                  printf(" %6.2f %+6.2f %+6.2f\n",newTrackRef->pt (), newTrackRef->eta(), newTrackRef->phi());
               }else{
                  printf("\n");
               }
*/
               newmu->setInnerTrack(newTrackRef);
            }

            newmuons->push_back(*newmu);
      }

      ev.put(newmuons);
}

reco::TrackRef UpdatedMuonInnerTrackRef::findNewRef(reco::TrackRef oldTrackRef, edm::Handle<reco::TrackCollection>& newTrackCollection){
   float dRMin=1000; int found = -1;   
   for(unsigned int i=0;i<newTrackCollection->size();i++){
      reco::TrackRef newTrackRef  = reco::TrackRef( newTrackCollection, i );
      if(newTrackRef.isNull())continue;

      if( fabs( (1.0/newTrackRef->pt())-(1.0/oldTrackRef->pt())) > maxInvPtDiff) continue;
      float dR = deltaR(newTrackRef->momentum(), oldTrackRef->momentum());
      if(dR <= minDR && dR < dRMin){ dRMin=dR; found = i;}
   }

   if(found>=0){     
      return reco::TrackRef( newTrackCollection, found );
   }else{
      return reco::TrackRef();
   }
}

DEFINE_FWK_MODULE(UpdatedMuonInnerTrackRef);




