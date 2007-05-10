\#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "VertexReco/VertexPrimitives/BeamSpot.h"

using namespace reco;

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
  edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
    << "Initializing PV producer " << "\n";
  fVerbose=conf.getUntrackedParameter<bool>("verbose", false);

  //  produces<VertexCollection>("PrimaryVertex");
  produces<VertexCollection>();

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

  try {
    edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Reconstructing event number: " << iEvent.id() << "\n";
    
    // get RECO tracks from the event
    // `tks` can be used as a ptr to a reco::TrackCollection
    edm::Handle<reco::TrackCollection> tks;
    iEvent.getByLabel(trackLabel(), tks);

    // interface RECO tracks to vertex reconstruction
    edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Found: " << (*tks).size() << " reconstructed tracks" << "\n";
    if (fVerbose) {cout << "RecoVertex/PrimaryVertexProducer:got " 
			<< (*tks).size() << " tracks " << endl;}
    edm::ESHandle<TransientTrackBuilder> theB;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
    vector<TransientTrack> t_tks = (*theB).build(tks);

   edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Found: " << t_tks.size() << " reconstructed tracks" << "\n";


   // get the BeamSpot, it will alwys be needed, even when not used as a constraint
   BeamSpot vertexBeamSpot;   // the beamspot from VertexReco/VertexPrimitives
   edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
   try{
     iEvent.getByType(recoBeamSpotHandle);
     vertexBeamSpot=BeamSpot(*recoBeamSpotHandle);
   }catch(std::exception & err)
     edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
       << "Exception occured retrieving BeamSpot in event: " << iEvent.id() 
       << "\n" << err.what() << "\n"
       << "continue using default BeamSpot" << endl;
  }


    // here call vertex reconstruction
    
    vector<TransientVertex> t_vts = theAlgo.vertices(t_tks, vertexBeamSpot);
    edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Found: " << t_vts.size() << " reconstructed vertices" << "\n";

    // convert transient vertices returned by the theAlgo to (reco) vertices
    for (vector<TransientVertex>::const_iterator iv = t_vts.begin();
	 iv != t_vts.end(); iv++) {
      Vertex v = *iv;
      vColl.push_back(v);
    }


    if(fVerbose){
      cout << "RecoVertex/PrimaryVertexProducer:   nv=" <<vColl.size()<< endl;
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

  }

  catch (std::exception & err) {
    edm::LogInfo("RecoVertex/PrimaryVertexProducer") 
      << "Exception during event number: " << iEvent.id() 
      << "\n" << err.what() << "\n";
  }

  *result = vColl;
  //  iEvent.put(result, "PrimaryVertex");
  iEvent.put(result);
  
}


std::string PrimaryVertexProducer::trackLabel() const
{
  return config().getParameter<std::string>("TrackLabel");
}


//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducer);
