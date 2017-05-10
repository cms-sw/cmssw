#include <map>
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
//#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
/*#include <string>
#include <sstream>
#include <cmath>
#include <Math/Functions.h>
#include <Math/SVector.h>
#include <Math/SMatrix.h>
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TMath.h"
#include "TTree.h"
#include <DataFormats/PatCandidates/interface/Jet.h>
#include <DataFormats/JetReco/interface/PFJet.h>
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "Validation/RecoVertex/interface/TrackParameterAnalyzer.h"
*/
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

/*#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
*/

#include "SimTracker/TrackHistory/interface/VertexClassifierByProxy.h"
#include "SimTracker/TrackHistory/interface/TrackClassifier.h"
//#include <SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h>
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
/*#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/Math/interface/Vector.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "TROOT.h"
#include "Math/VectorUtil.h"
#include <TVector3.h>
#include <Math/GenVector/PxPyPzE4D.h>
#include <Math/GenVector/PxPyPzM4D.h>
#include "DataFormats/Math/interface/LorentzVector.h"
*/
#include <SimDataFormats/GeneratorProducts/interface/HepMCProduct.h>
#include <DataFormats/HepMCCandidate/interface/GenParticle.h> 
/*#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <DataFormats/SiPixelCluster/interface/SiPixelCluster.h>  
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include <DataFormats/Common/interface/DetSetNew.h>
*/
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"
/*#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaR.h"
*/
//
// class decleration
//
using namespace reco;
using namespace std;
using namespace edm;
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

class BTracksProducer : public edm::EDProducer
{

	public:
		typedef math::XYZPoint 	Point;
		explicit BTracksProducer(const edm::ParameterSet&);
		~BTracksProducer();

	private:

		virtual void beginJob() ;		
		virtual void produce(edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;
		// Member data

		//VertexClassifierByProxy<reco::SecondaryVertexTagInfoCollection> classifier_;
		TrackClassifier trackclassifier_;	

		//Int_t numberVertexClassifier_;
		edm::InputTag trackingTruth_;
		// edm::InputTag svTagInfoProducer_;
		edm::InputTag tracks_;
		std::string dqmLabel;
		edm::InputTag simG4_;
	        bool allSim_;

};


BTracksProducer::BTracksProducer(const edm::ParameterSet& config) : 
trackclassifier_(config.getParameter<edm::ParameterSet>("trackConfig")),
simG4_( config.getParameter<edm::InputTag>( "simG4" ) ),
trackingTruth_(config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" )),
tracks_(config.getUntrackedParameter<edm::InputTag> ( "trackInputTag" )),
allSim_(config.getUntrackedParameter<bool> ( "allSim" ))
{

  produces<reco::TrackCollection>("bTracks");
  produces<reco::TrackCollection>("fakeTracks");
  produces<TrackingParticleCollection>("bTrackingParticles");
  produces<std::vector<SimTrack> >("bSimTracks");
  produces<TrackingParticleCollection>("bNotReconstructed");
  produces<std::vector<SimTrack> >("bNotReconstructed");
  if(allSim_)
  {
    produces<TrackingParticleCollection>("all");
    produces<std::vector<SimTrack> >("all");
  produces<TrackingParticleCollection>("allNotReconstructed");
  produces<std::vector<SimTrack> >("allNotReconstructed");

  }
//  produces<std::vector<HepMC::GenParticle> >("bNotReconstructed");
}

BTracksProducer::~BTracksProducer()
{
}
 

void BTracksProducer::produce(edm::Event& event, const edm::EventSetup& setup)
{
   Handle<HepMCProduct> MCEvt;
   event.getByLabel("generator", MCEvt);
   const HepMC::GenEvent* evt = MCEvt->GetEvent();


  //RECO TO SIM 
  trackclassifier_.newEvent(event, setup);  

  TrackHistory const & flavortracer = trackclassifier_.history();
  edm::Handle<reco::GenParticleCollection> genParticles;
  event.getByLabel("genParticles", genParticles);
 

   edm::Handle<TrackingParticleCollection>  TPCollection;
   event.getByLabel(trackingTruth_, TPCollection);
	
  //get tracks collection
  edm::Handle<edm::View<reco::Track> > Tracks; 
  event.getByLabel(tracks_, Tracks);
	
  edm::ESHandle<TransientTrackBuilder> builder;
  setup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);

   auto_ptr<reco::TrackCollection> bTracks(new reco::TrackCollection);
   auto_ptr<reco::TrackCollection> fakeTracks(new reco::TrackCollection);
  
  
   auto_ptr<TrackingParticleCollection> allNotReconstructedTP(new TrackingParticleCollection);
   auto_ptr<std::vector<SimTrack> > allNotReconstructedST(new std::vector<SimTrack>);
   auto_ptr<TrackingParticleCollection> bNotReconstructedTP(new TrackingParticleCollection);
   auto_ptr<std::vector<SimTrack> > bNotReconstructedST(new std::vector<SimTrack>);
   auto_ptr<TrackingParticleCollection> bRecoedTP(new TrackingParticleCollection);
   auto_ptr<std::vector<SimTrack> > bRecoedST(new std::vector<SimTrack>);
   auto_ptr<TrackingParticleCollection> allTP(new TrackingParticleCollection);
   auto_ptr<std::vector<SimTrack> > allST(new std::vector<SimTrack>);
  
 auto_ptr<reco::GenParticleCollection> bNotReconstructedGP(new reco::GenParticleCollection);


   for(unsigned int  index = 0; index < Tracks->size(); ++index)
    {
  	  reco::TrackBaseRef trkID((Tracks), index);	 
	  trackclassifier_.evaluate(trkID);//reco::TrackBaseRef((Tracks), index));

	     		
	if (trackclassifier_.is(TrackCategories::Fake) ) {
	  std::cout<<trkID.key()<< " is fake" << std::endl;  	
	  fakeTracks->push_back(*trkID);
	}
        else
        {
		TrackingParticleRef trackingParticle=trackclassifier_.history().simParticle();
                allTP->push_back(*trackingParticle);
                 for( TrackingParticle::g4t_iterator g4T=(*trackingParticle).g4Track_begin(); g4T != (*trackingParticle).g4Track_end(); ++g4T ){
 	                allST->push_back(*g4T);
 		}

	}
	if( trackclassifier_.is(TrackCategories::Bottom)){
	  std::cout<<trkID.key()<< " is B" << trackclassifier_.is(TrackCategories::BWeakDecay) << std::endl;  	
	  bTracks->push_back(*trkID);

	}
   }
     event.put(bTracks, "bTracks");
     event.put(fakeTracks, "fakeTracks");

// std::cout<<"  index reco  "<<indexr<<std::endl;

 //SIM TO RECO

  
   Handle<edm::SimVertexContainer> simVtcs;
   event.getByLabel( simG4_, simVtcs);
  //jet association
   int indexs =0;
   for (std::size_t index = 0; index < TPCollection->size(); ++index){
     TrackingParticleRef trackingParticle(TPCollection, index);
     trackclassifier_.evaluate(trackingParticle);
     for( TrackingParticle::g4t_iterator g4T=(*trackingParticle).g4Track_begin(); g4T != (*trackingParticle).g4Track_end(); ++g4T ){
      
       /*   if(g4T->vertIndex()>0){
		     if(sqrt((*simVtcs)[g4T->vertIndex()].position().perp2())>2||TMath::Abs((*simVtcs)[g4T->vertIndex()].position().z())>20.) continue;
		     	GlobalVector vtx((*simVtcs)[g4T->vertIndex()].position().x(),(*simVtcs)[g4T->vertIndex()].position().y(),(*simVtcs)[g4T->vertIndex()].position().z());
	//trk_type_a[indexs] = g4T->type();
	     }
	 */
          const TrackingVertexRef&  pv =  (*trackingParticle).parentVertex() ;


            TrackingVertexRefVector::iterator  iTV  = (*trackingParticle).decayVertices_begin();
	     if( iTV != (*trackingParticle).decayVertices_end()){
/*     trk_nDecayVtx_a[indexs]=(*trackingParticle).decayVertices().size();
		     trk_DecayVtx_rho_a[indexs] = (**iTV).position().r();
		     trk_DecayVtx_z_a[indexs] = (**iTV).position().z();*/
              }
	 //std::cout << "SimTrackID: " << g4T->trackId()  << " TP index " << index << " GenParticleID " << g4T->genpartIndex() << " flags  : " <<  trackclassifier_.is(TrackCategories::Bottom) << trackclassifier_.is(TrackCategories::Light) << trackclassifier_.is(TrackCategories::BWeakDecay) <<  std::endl;
	 if(trackclassifier_.is(TrackCategories::Reconstructed) && (trackclassifier_.is(TrackCategories::Bottom)  ||  trackclassifier_.is(TrackCategories::BWeakDecay) ))
         {
		bRecoedTP->push_back(*trackingParticle);
	        bRecoedST->push_back(*g4T);

	 }

	 if(!trackclassifier_.is(TrackCategories::Reconstructed)) {
		 if(trackingParticle->pt() > 1 && trackingParticle->numberOfTrackerLayers()>8) {
			 allNotReconstructedTP->push_back(*trackingParticle);
			 allNotReconstructedST->push_back(*g4T);
		 }
		 if( (trackclassifier_.is(TrackCategories::Bottom) || trackclassifier_.is(TrackCategories::BWeakDecay) ) )
		 {
			 std::cout << "Missing SimTrackID: " << g4T->trackId()  << " TP index " << index << " GenParticleID " << g4T->genpartIndex() << std::endl;
			 bNotReconstructedTP->push_back(*trackingParticle);
			 bNotReconstructedST->push_back(*g4T);
			 // 	        bNotReconstructedGP->push_back(*evt->barcode_to_particle( g4T->genpartIndex() ));
		 }
	 }

     }
   }

   event.put(bNotReconstructedTP, "bNotReconstructed");
   event.put(bRecoedTP, "bTrackingParticles");
   event.put(bRecoedST, "bSimTracks");
   event.put(bNotReconstructedST, "bNotReconstructed");
   if(allSim_)
   {
	   event.put(allNotReconstructedTP, "allNotReconstructed");
	   event.put(allNotReconstructedST, "allNotReconstructed");
	   event.put(allTP,"all");
	   event.put(allST,"all");
   }

}

// ------------ method called once each job just before starting event loop  ------------
	void 
BTracksProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BTracksProducer::endJob() {

	std::cout << std::endl;

} 

DEFINE_FWK_MODULE(BTracksProducer);
