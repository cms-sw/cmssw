#include "CLHEP/Vector/LorentzVector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include <map>

using namespace edm;
using namespace std; 
using CLHEP::HepLorentzVector;

typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;
typedef std::vector<TrackingParticle> TrackingParticleCollection;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >       GenVertexRef;

string MessageCategory = "TrackingTruthProducer";

TrackingTruthProducer::TrackingTruthProducer(const edm::ParameterSet &conf) {
  produces<TrackingVertexCollection>();
  produces<TrackingParticleCollection>();
  conf_ = conf;
  distanceCut_ = conf_.getParameter<double>("distanceCut");
  dataLabels_  = conf_.getParameter<vector<string> >("dataLabels");
  edm::LogInfo (MessageCategory) << "Vertex distance cut set to " << distanceCut_ << " mm";
}

void TrackingTruthProducer::produce(Event &event, const EventSetup &) {

  // Get information out of event record
  edm::Handle<edm::HepMCProduct>           hepMC;
  for (vector<string>::const_iterator source = dataLabels_.begin(); source !=
      dataLabels_.end(); ++source) {
    try {
      event.getByLabel(*source,hepMC);
      edm::LogInfo (MessageCategory) << "Using HepMC source " << *source;
      break;
    } catch (std::exception &e) {
      
    }    
  }
  const edm::HepMCProduct *mcp = hepMC.product();
  
  
  edm::Handle<EmbdSimVertexContainer>      G4VtxContainer;
  edm::Handle<edm::EmbdSimTrackContainer>  G4TrkContainer;
  event.getByType(G4VtxContainer);
  event.getByType(G4TrkContainer);
  
  const HepMC::GenEvent            &genEvent = hepMC -> getHepMCData();
  // Is const HepMC::GenEvent *hme = mcp -> GetEvent(); // faster?

  const edm::EmbdSimTrackContainer      *etc = G4TrkContainer.product();

  if (mcp == 0) {
    edm::LogWarning (MessageCategory) << "No HepMC source found";
    return;
  }  
   
//  genEvent.print();

//Put TrackingParticle here... need charge, momentum, vertex position, time, pdg id
  auto_ptr<TrackingParticleCollection> tPC(new TrackingParticleCollection);
  std::map<int,int> productionVertex;
  int iG4Track = 0;
  for (edm::EmbdSimTrackContainer::const_iterator itP = G4TrkContainer->begin();
       itP !=  G4TrkContainer->end(); ++itP){
       TrackingParticle::Charge q = 0;
       CLHEP::HepLorentzVector p = itP -> momentum();
       const TrackingParticle::LorentzVector theMomentum(p.x(), p.y(), p.z(), p.t());
       double time =  0; 
       int pdgId = 0; 
       const HepMC::GenParticle * gp = 0;       
       int genPart = itP -> genpartIndex();
       if (genPart >= 0) {
           gp = genEvent.barcode_to_particle(genPart);  //pointer to the generating part.
	   pdgId = gp -> pdg_id();
       }
        math::XYZPoint theVertex;
       // = Point(0, 0, 0);
       int genVert = itP -> vertIndex(); // Is this a HepMC vertex # or GenVertex #?
       if (genVert >= 0){
           const EmbdSimVertex &gv = (*G4VtxContainer)[genVert];
	   const CLHEP::HepLorentzVector &v = gv.position();
	   theVertex = math::XYZPoint(v.x(), v.y(), v.z());
	   time = v.t(); 
       }
       TrackingParticle tp(q, theMomentum, theVertex, time, pdgId);
       tp.addG4Track(EmbdSimTrackRef(G4VtxContainer,iG4Track));
       tp.addGenParticle(GenParticleRef(hepMC,genPart));
       productionVertex.insert(pair<int,int>(tPC->size(),genVert));
       tPC -> push_back(tp);
       ++iG4Track;
  }
  edm::OrphanHandle<TrackingParticleCollection> tpcHandle = event.put(tPC);
  TrackingParticleCollection trackCollection = *tpcHandle;
  edm::LogInfo (MessageCategory) << "Put "<< trackCollection.size() << " tracks in event";
       
// Find and loop over EmbdSimVertex vertices
    
  auto_ptr<TrackingVertexCollection> tVC( new TrackingVertexCollection );  

  int index = 0;
  for (edm::EmbdSimVertexContainer::const_iterator itVtx = G4VtxContainer->begin(); 
       itVtx != G4VtxContainer->end(); 
       ++itVtx) {
    bool InVolume = false;
         
    CLHEP::HepLorentzVector position = itVtx -> position();  // Get position of ESV

    if (position.perp() < 1200 && abs(position.z()) < 3000) { // In or out of Tracker
      InVolume = true;
    }
    
// Figure out the barcode of the HepMC Vertex if there is one
    int vtxParent = itVtx -> parentIndex(); // Get incoming track (EST)
    int partHepMC = -1;
    int vb = 0;       
    if (vtxParent >= 0) {                     // Is there a parent track 
      EmbdSimTrack est = etc->at(vtxParent);  // Pull track out from vector
      partHepMC =     est.genpartIndex();     // Get HepMC particle barcode
      HepMC::GenParticle *hmp = genEvent.barcode_to_particle(partHepMC); // Convert barcode
      if (hmp != 0) {
        HepMC::GenVertex *hmpv = hmp -> production_vertex(); 
        if (hmpv != 0) {
          vb = hmpv  -> barcode();
        }  
      }  
    }  

// Find closest vertex to this one

    double closest = 9e99;
    TrackingVertexCollection::iterator nearestVertex;

    for (TrackingVertexCollection::iterator tV = tVC -> begin(); tV != tVC ->end(); ++tV) {
//      CLHEP::HepLorentzVector tPosition = tV -> position();  // Get position of ESV
//      double distance = sqrt(pow(tPosition.x()-position.x(),2) +  
//                             pow(tPosition.y()-position.y(),2) + 
//                             pow(tPosition.z()-position.z(),2)); 
      double distance = HepLorentzVector(tV -> position() - position).v().r();
      if (distance < closest) { // flag which one so we can associate them
        closest = distance;
        nearestVertex = tV; 
      }   
    }

// If outside cutoff, create another TrackingVertex,
    
    if (closest > distanceCut_) {
      tVC -> push_back(TrackingVertex(position));
      nearestVertex = tVC -> end();
      --nearestVertex;
    } 
     
// Add data to closest vertex
    (*nearestVertex).addG4Vertex(EmbdSimVertexRef(G4VtxContainer, index) ); // Add G4 vertex
    if (vb) {
      (*nearestVertex).addGenVertex(GenVertexRef(hepMC,vb)); // Add HepMC vertex
    }

// Identify and add child tracks       
    for (std::map<int,int>::iterator mapIndex = productionVertex.begin(); 
         mapIndex != productionVertex.end();
         ++mapIndex) {
      if (mapIndex -> second == index) {
//        edm::LogInfo (MessageCategory) << "Adding track "<< mapIndex->first << " to vertex "<<tVC->size()-1;
        (*nearestVertex).add(TrackingParticleRef(tpcHandle,mapIndex -> first));
      }
    }
    ++index;     
  }

  edm::LogInfo (MessageCategory) << "TrackingTruth found " << tVC->size() << " unique vertices";
  

// Dump out the results  
  
  index = 0;
  for (TrackingVertexCollection::const_iterator v =
       tVC -> begin();
       v != tVC ->end(); ++v) {
    edm::LogInfo (MessageCategory) << "TrackingVertex " << index << " has " 
      << (v -> g4Vertices()).size()  << " G4 vertices, " 
      << (v -> genVertices()).size() << " HepMC vertices and " 
      << (v -> trackingParticles()).size() << " tracks";
    ++index;  
  }        
  
// Put new info into event record  
  
  event.put(tVC);
}
  
DEFINE_FWK_MODULE(TrackingTruthProducer)
