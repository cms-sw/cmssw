#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//using namespace reco;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PrimaryVertexProducer::PrimaryVertexProducer(const edm::ParameterSet& conf)
  : theAlgo(conf), theConfig(conf)
{
  edm::LogInfo("PVDebugInfo") 
    << "Initializing PV producer " << "\n";
  fVerbose=conf.getUntrackedParameter<bool>("verbose", false);
  trackLabel = conf.getParameter<edm::InputTag>("TrackLabel");
  beamSpotLabel = conf.getParameter<edm::InputTag>("beamSpotLabel");
 
  produces<reco::VertexCollection>();

}


PrimaryVertexProducer::~PrimaryVertexProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
PrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  std::auto_ptr<reco::VertexCollection> result(new reco::VertexCollection);
  reco::VertexCollection vColl;

  // get the BeamSpot, it will alwys be needed, even when not used as a constraint
  reco::BeamSpot vertexBeamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotLabel,recoBeamSpotHandle);
  if (recoBeamSpotHandle.isValid()){
    vertexBeamSpot = *recoBeamSpotHandle;
  }else{
    edm::LogError("UnusableBeamSpot") << "No beam spot available from EventSetup";
  }



  // get RECO tracks from the event
  // `tks` can be used as a ptr to a reco::TrackCollection
  edm::Handle<reco::TrackCollection> tks;
  iEvent.getByLabel(trackLabel, tks);


  // interface RECO tracks to vertex reconstruction
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
  std::vector<reco::TransientTrack> t_tks = (*theB).build(tks, vertexBeamSpot);
  if(fVerbose) {std::cout << "RecoVertex/PrimaryVertexProducer"
		     << "Found: " << t_tks.size() << " reconstructed tracks" << "\n";
  }


  // call vertex reconstruction
  std::vector<TransientVertex> t_vts = theAlgo.vertices(t_tks, vertexBeamSpot);
  if(fVerbose){
    std::cout <<"RecoVertex/PrimaryVertexProducer: "
	      << " found " << t_vts.size() << " reconstructed vertices" << "\n";
  }
   
  // convert transient vertices returned by the theAlgo to (reco) vertices
  for (std::vector<TransientVertex>::const_iterator iv = t_vts.begin();
       iv != t_vts.end(); iv++) {
    reco::Vertex v = *iv;
    vColl.push_back(v);
  }

  if (vColl.empty()) {
    GlobalError bse(vertexBeamSpot.rotatedCovariance3D());
    if ( (bse.cxx() <= 0.) || 
  	(bse.cyy() <= 0.) ||
  	(bse.czz() <= 0.) ) {
      AlgebraicSymMatrix33 we;
      we(0,0)=10000; we(1,1)=10000; we(2,2)=10000;
      vColl.push_back(reco::Vertex(vertexBeamSpot.position(), we,0.,0.,0));
      if(fVerbose){
	std::cout <<"RecoVertex/PrimaryVertexProducer: "
		  << "Beamspot with invalid errors "<<bse.matrix()<<std::endl;
	std::cout << "Will put Vertex derived from dummy-fake BeamSpot into Event.\n";
      }
    } else {
      vColl.push_back(reco::Vertex(vertexBeamSpot.position(), 
				 vertexBeamSpot.rotatedCovariance3D(),0.,0.,0));
      if(fVerbose){
	std::cout <<"RecoVertex/PrimaryVertexProducer: "
		  << " will put Vertex derived from BeamSpot into Event.\n";
      }
    }
  }

  if(fVerbose){
    int ivtx=0;
    for(reco::VertexCollection::const_iterator v=vColl.begin(); 
	v!=vColl.end(); ++v){
      std::cout << "recvtx "<< ivtx++ 
		<< "#trk " << std::setw(3) << v->tracksSize()
		<< " chi2 " << std::setw(4) << v->chi2() 
		<< " ndof " << std::setw(3) << v->ndof() 
		<< " x "  << std::setw(6) << v->position().x() 
		<< " dx " << std::setw(6) << v->xError()
		<< " y "  << std::setw(6) << v->position().y() 
		<< " dy " << std::setw(6) << v->yError()
		<< " z "  << std::setw(6) << v->position().z() 
		<< " dz " << std::setw(6) << v->zError()
		<< std::endl;
    }
  }

  
  *result = vColl;
  //  iEvent.put(result, "PrimaryVertex");
  iEvent.put(result);
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducer);
