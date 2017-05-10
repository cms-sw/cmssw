#include <map>
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
//#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TNtuple.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/JetReco/interface/CaloJet.h"


//#include "SimTracker/TrackHistory/interface/VertexClassifierByProxy.h"
#include "SimTracker/TrackHistory/interface/TrackClassifier.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include <SimDataFormats/GeneratorProducts/interface/HepMCProduct.h>
#include <DataFormats/HepMCCandidate/interface/GenParticle.h> 
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackHistory/interface/TrackHistory.h"
//
// class decleration
//
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

using namespace reco;
using namespace std;
using namespace edm;
#include "DataFormats/GeometryVector/interface/VectorUtil.h"

class LightTrackNtuple : public edm::EDProducer
{

	public:
		typedef math::XYZPoint 	Point;
		explicit LightTrackNtuple(const edm::ParameterSet&);
		~LightTrackNtuple();

	private:

		virtual void beginJob() ;		
		virtual void produce(edm::Event&, const edm::EventSetup&);
		virtual void endJob() ;
		// Member data

		//VertexClassifierByProxy<reco::SecondaryVertexTagInfoCollection> classifier_;
//		TrackClassifier trackclassifier_;	

		//Int_t numberVertexClassifier_;
		edm::InputTag trackingTruth_;
		// edm::InputTag svTagInfoProducer_;
		edm::InputTag tracks_;
		std::string dqmLabel;
		edm::InputTag simG4_;
	        bool allSim_;
		float v[60];
		TNtuple *nt;
		TNtuple *nt1;
};


LightTrackNtuple::LightTrackNtuple(const edm::ParameterSet& config) : 
//trackclassifier_(config.getParameter<edm::ParameterSet>("trackConfig")),
simG4_( config.getParameter<edm::InputTag>( "simG4" ) ),
trackingTruth_(config.getUntrackedParameter<edm::InputTag> ( "trackingTruth" )),
tracks_(config.getUntrackedParameter<edm::InputTag> ( "trackInputTag" )),
allSim_(config.getUntrackedParameter<bool> ( "allSim" ))
{

//  produces<std::vector<HepMC::GenParticle> >("bNotReconstructed");
}

LightTrackNtuple::~LightTrackNtuple()
{
}
 

void LightTrackNtuple::produce(edm::Event& event, const edm::EventSetup& setup)
{
   Handle<HepMCProduct> MCEvt;
   event.getByLabel("generator", MCEvt);
   const HepMC::GenEvent* evt = MCEvt->GetEvent();


  //RECO TO SIM 
//  trackclassifier_.newEvent(event, setup);  

//  TrackHistory const & flavortracer = trackclassifier_.history();
  edm::Handle<reco::GenParticleCollection> genParticles;
  event.getByLabel("genParticles", genParticles);
 

   edm::Handle<TrackingParticleCollection>  TPCollection;
   event.getByLabel(trackingTruth_, TPCollection);
	
  //get tracks collection
 // edm::Handle<edm::View<reco::Track> > Tracks2; 
//  event.getByLabel(edm::InputTag("generalTracks","","RECO"), Tracks2);
  edm::Handle<edm::View<reco::Track> > Tracks; 
  event.getByLabel(tracks_, Tracks);
	
  edm::ESHandle<TransientTrackBuilder> builder;
  setup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);

   auto_ptr<reco::TrackCollection> bTracks(new reco::TrackCollection);
   auto_ptr<reco::TrackCollection> fakeTracks(new reco::TrackCollection);
  
   ESHandle<TrackAssociatorBase> myAssociator;
   setup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits", myAssociator); 
   auto_ptr<TrackingParticleCollection> bNotReconstructedTP(new TrackingParticleCollection);
   auto_ptr<std::vector<SimTrack> > bNotReconstructedST(new std::vector<SimTrack>);
   auto_ptr<TrackingParticleCollection> bRecoedTP(new TrackingParticleCollection);
   auto_ptr<std::vector<SimTrack> > bRecoedST(new std::vector<SimTrack>);
   auto_ptr<TrackingParticleCollection> allTP(new TrackingParticleCollection);
   auto_ptr<std::vector<SimTrack> > allST(new std::vector<SimTrack>);
  
 auto_ptr<reco::GenParticleCollection> bNotReconstructedGP(new reco::GenParticleCollection);

   Handle<edm::View<reco::Jet> > jH;
   event.getByLabel("ak5CaloJets",jH);
   const edm::View<reco::Jet> & jets = *jH.product();

   int indexs =0;


     edm::RefVector<TrackingParticleCollection> tpc(TPCollection.id());
     edm::RefToBaseVector<reco::Track> tc(Tracks);
     for (unsigned int j=0; j<Tracks->size();j++)
       tc.push_back(edm::RefToBase<reco::Track>(Tracks,j)); 

//   reco::SimToRecoCollection p = myAssociator->associateSimToReco(Tracks,TPCollection,&event,&setup );

   for (std::size_t index = 0; index < TPCollection->size(); ++index){
	   TrackingParticleRef trackingParticle(TPCollection, index);
	   if(trackingParticle->p4().Pt() < 1 || trackingParticle->numberOfTrackerLayers() < 5 || trackingParticle->status() < 0) continue;
           tpc.push_back(edm::Ref<TrackingParticleCollection>(TPCollection,index));
    }

   reco::SimToRecoCollection p = myAssociator->associateSimToReco(tc,tpc,&event,&setup );
   for (std::size_t index = 0; index < TPCollection->size(); ++index){
	   TrackingParticleRef trackingParticle(TPCollection, index);
	   if(trackingParticle->p4().Pt() < 1 || trackingParticle->numberOfTrackerLayers() < 5 || trackingParticle->status() < 0) continue;
	   float dr=99; 
	   float jetpt=0;	
	   for(edm::View<reco::Jet>::const_iterator it = jets.begin() ; it != jets.end() ; it++)
	   {
		   if(it->pt() > 30 )
		   { 	
			   GlobalVector jetDir(it->momentum().x(),it->momentum().y(),it->momentum().z());
			   GlobalVector tkDir(trackingParticle->p4().x(),trackingParticle->p4().y(),trackingParticle->p4().z());
			   float deltar=Geom::deltaR(jetDir,tkDir);
			   if(deltar<dr) {dr=deltar; jetpt=it->pt(); }
		   }
	   }

	   std::vector<std::pair<edm::RefToBase<reco::Track>, double> > simRecAsso;
	   if(p.find(trackingParticle) != p.end()) { 
		   simRecAsso = (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >) p[trackingParticle];
		   for (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >::const_iterator IT = simRecAsso.begin();    IT != simRecAsso.end(); ++IT) {
			   edm::RefToBase<reco::Track> track = IT->first;
			   double quality = IT->second;
				v[0]=trackingParticle->p4().Pt();
				v[1]=trackingParticle->p4().Eta();
				v[2]=trackingParticle->p4().Phi();
				v[3]=track->pt();
				v[4]=track->eta();
				v[5]=track->phi();
				v[6]=track->d0();
				v[7]=track->dz();
				v[8]=track->dxy(trackingParticle->vertex());
				v[9]=track->dz(trackingParticle->vertex());
				v[10]=track->hitPattern().numberOfValidPixelHits();
				v[11]=track->numberOfValidHits();
				v[12]=dr;
				v[13]=jetpt;
				v[14]=trackingParticle->numberOfTrackerLayers();
				v[15]=track->numberOfLostHits();
				v[16]=trackingParticle->vertex().rho();
				v[17]=event.eventAuxiliary().event();
				v[18]=event.eventAuxiliary().run();
				v[19]=event.eventAuxiliary().luminosityBlock();
				v[20]=trackingParticle->vertex().x();
				v[21]=trackingParticle->vertex().y();
				v[22]=trackingParticle->vertex().z();
				v[23]=track->d0Error();
				v[24]=track->dzError();
				v[25]=trackingParticle->status();
				v[26]=trackingParticle->pdgId();
				nt->Fill(v);
		   }

	   }else{


		   v[0]=trackingParticle->p4().Pt();
		   v[1]= trackingParticle->p4().Eta();
		   v[2]= trackingParticle->p4().Phi();
		   v[3]= -999;
		   v[4]= -999;
		   v[5]= -999;
		   v[6]= -999;
		   v[7]= -999;
		   v[8]= -999;
		   v[9]= -999;
		   v[10]=-999;
		   v[11]=-999;
		   v[12]=   dr;
		   v[13]=jetpt;
		   v[14]=trackingParticle->numberOfTrackerLayers();
		   v[15]=-999;
		   v[16]=trackingParticle->vertex().rho();
				v[17]=event.eventAuxiliary().event();
				v[18]=event.eventAuxiliary().run();
				v[19]=event.eventAuxiliary().luminosityBlock();
		   v[20]=trackingParticle->vertex().x();
		   v[21]=trackingParticle->vertex().y();
		   v[22]=trackingParticle->vertex().z();
		   v[23]=-999;
		   v[24]=-999;
	 	   v[25]=trackingParticle->status();
                   v[26]=trackingParticle->pdgId();

		   nt->Fill(v);

	   }
   }



}

// ------------ method called once each job just before starting event loop  ------------
	void 
LightTrackNtuple::beginJob()
{
	edm::Service<TFileService> fs;
	nt = fs->make<TNtuple>( "nt"  , "nt", "pt:eta:phi:pt_r:eta_r:phi_r:d0_r:z0_r:resD0:resZ0:npix:nvalid:jetdR:jetpt:matched:nlost:rho:evid:evrun:evlumi:svx:svy:svz:d0err:z0err:status:pdgid" );
}

// ------------ method called once each job just after ending the event loop  ------------
void 
LightTrackNtuple::endJob() {

	std::cout << std::endl;

} 

DEFINE_FWK_MODULE(LightTrackNtuple);
