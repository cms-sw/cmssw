#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"

#include <map>

using namespace edm;
using namespace std; 
using CLHEP::HepLorentzVector;

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >   GenVertexRef;

string MessageCategory = "TrackingTruthProducer";

TrackingTruthProducer::TrackingTruthProducer(const edm::ParameterSet &conf) {
  produces<TrackingVertexCollection>("VertexTruth");
  produces<TrackingParticleCollection>("TrackTruth");

  conf_ = conf;
  distanceCut_      = conf_.getParameter<double>("vertexDistanceCut");
  dataLabels_       = conf_.getParameter<vector<string> >("HepMCDataLabels");
  simHitLabel_	    = conf_.getParameter<string>("simHitLabel");
  hitLabelsVector_  = conf_.getParameter<vector<string> >("TrackerHitLabels");
  volumeRadius_     = conf_.getParameter<double>("volumeRadius");   
  volumeZ_          = conf_.getParameter<double>("volumeZ");   
  discardOutVolume_ = conf_.getParameter<bool>("discardOutVolume");     

  edm::LogInfo (MessageCategory) << "Setting up TrackingTruthProducer";
  edm::LogInfo (MessageCategory) << "Vertex distance cut set to " << distanceCut_  << " mm";
  edm::LogInfo (MessageCategory) << "Volume radius set to "       << volumeRadius_ << " mm";
  edm::LogInfo (MessageCategory) << "Volume Z      set to "       << volumeZ_      << " mm";
  edm::LogInfo (MessageCategory) << "Discard out of volume? "     << discardOutVolume_;
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
  if (mcp == 0) {
    edm::LogWarning (MessageCategory) << "No HepMC source found";
    return;
  }  
  const HepMC::GenEvent *genEvent = mcp -> GetEvent();
   
  edm::Handle<SimVertexContainer>      G4VtxContainer;
  edm::Handle<edm::SimTrackContainer>  G4TrkContainer;
  try {
    event.getByType(G4VtxContainer);
    event.getByType(G4TrkContainer);
  } catch (std::exception &e) {
    edm::LogWarning (MessageCategory) << "Geant tracks and/or vertices not found.";
    return;
  }    

  const edm::SimTrackContainer      *etc = G4TrkContainer.product();

  vector<edm::Handle<edm::PSimHitContainer> > AlltheConteiners;
  for (vector<string>::const_iterator source = hitLabelsVector_.begin(); source !=
      hitLabelsVector_.end(); ++source){
      try{
      edm::Handle<edm::PSimHitContainer> HitContainer;
      event.getByLabel(simHitLabel_,*source, HitContainer);
      AlltheConteiners.push_back(HitContainer);
      } catch (std::exception &e) {
      
    }
      
  }   
   
  
  
//  genEvent.print();
//  genEvent ->  signal_process_id();
  // 13 cosmic muons
  // 20 particle 
  // Others from Pythia, begin on page 132. Hope there is a flag somewhere else
  // Don't want to figure out minBias vs. other things.
  
//Put TrackingParticle here... need charge, momentum, vertex position, time, pdg id
  auto_ptr<TrackingParticleCollection> tPC(new TrackingParticleCollection);
  auto_ptr<TrackingVertexCollection>   tVC(new TrackingVertexCollection  );  

  edm::RefProd<TrackingParticleCollection> refTPC =
      event.getRefBeforePut<TrackingParticleCollection>("TrackTruth");
  edm::RefProd<TrackingVertexCollection>   refTVC =
      event.getRefBeforePut<TrackingVertexCollection>("VertexTruth");
  
  map<int,int> g4T_TP;        // Map of SimTrack index to TrackingParticle index
  map<int,int> g4T_G4SourceV; // Map of SimTrack to (source) SimVertex index
  
  int iG4Track = 0;
  edm::SimTrackContainer::const_iterator itP;
  for (itP = G4TrkContainer->begin(); itP !=  G4TrkContainer->end(); ++itP){
    TrackingParticle::Charge q = 0;
    CLHEP::HepLorentzVector p = itP -> momentum();
    const TrackingParticle::LorentzVector theMomentum(p.x(), p.y(), p.z(), p.t());
    double time =  0; 
    int pdgId = 0;
    EncodedEventId trackEventId = itP->eventId(); 
    const HepMC::GenParticle * gp = 0;       
    int genPart = itP -> genpartIndex();
    if (genPart >= 0) {
      gp = genEvent -> barcode_to_particle(genPart);  //pointer to the generating part.
      pdgId = gp -> pdg_id();
    }
    math::XYZPoint theVertex;
    // = Point(0, 0, 0);
    int genVert = itP -> vertIndex(); // Is this a HepMC vertex # or GenVertex #?
    if (genVert >= 0){
      g4T_G4SourceV.insert(pair<int,int>(iG4Track,genVert));
      const SimVertex &gv = (*G4VtxContainer)[genVert];
      const CLHEP::HepLorentzVector &v = gv.position();
      theVertex = math::XYZPoint(v.x(), v.y(), v.z());
      time = v.t(); 
    }
    TrackingParticle tp(q, theMomentum, theVertex, time, pdgId, trackEventId);
    
    typedef vector<edm::Handle<edm::PSimHitContainer> >::const_iterator cont_iter;
//    map<int, TrackPSimHitRefVector> trackIdPSimHitMap;
    unsigned int simtrackId = itP -> trackId();
    for( cont_iter allCont = AlltheConteiners.begin(); allCont != AlltheConteiners.end(); ++allCont ){
      int  index = 0;
      for (edm::PSimHitContainer::const_iterator hit = (*allCont)->begin(); hit != (*allCont)->end(); ++hit, ++index) {
     	if (simtrackId == hit->trackId() && trackEventId == hit->eventId() ){
     	  tp.addPSimHit(TrackPSimHitRef(*allCont, index));
//     	   trackIdPSimHitMap[hit->trackId()].push_back(TrackPSimHitRef(*allCont, index));
        }
      }
    }

    tp.addG4Track(SimTrackRef(G4TrkContainer,iG4Track));
    if (genPart >= 0) {
      tp.addGenParticle(GenParticleRef(hepMC,genPart));
    }
    g4T_TP.insert(pair<int,int>(iG4Track,tPC->size()));
    tPC -> push_back(tp);
    ++iG4Track;
  }

// Find and loop over EmbdSimVertex vertices
    
  int indexG4V = 0;
  for (edm::SimVertexContainer::const_iterator itVtx = G4VtxContainer->begin(); 
       itVtx != G4VtxContainer->end(); 
       ++itVtx,++indexG4V) {

    CLHEP::HepLorentzVector position = itVtx -> position();  // Get position of ESV
    bool inVolume = (position.perp() < volumeRadius_ && abs(position.z()) < volumeZ_); // In or out of Tracker
    if (!inVolume && discardOutVolume_) { continue; }        // Skip if desired
    
    EncodedEventId vertEvtId = itVtx -> eventId();     // May not be right one, get from HepMC?
    
// Figure out the barcode of the HepMC Vertex if there is one by
// getting incoming SimTtrack (if any), finding corresponding HepMC track and
// then decay (HepMC) vertex of that track    
    int vertexBarcode = 0;       
    int vtxParent = itVtx -> parentIndex();    
    if (vtxParent >= 0) {                      
      SimTrack parentTrack = etc->at(vtxParent);       
      int parentBC = parentTrack.genpartIndex();  
      HepMC::GenParticle *parentParticle = genEvent -> barcode_to_particle(parentBC);
      if (parentParticle != 0) {
        HepMC::GenVertex *hmpv = parentParticle -> end_vertex(); 
        if (hmpv != 0) {
          vertexBarcode = hmpv  -> barcode();
        }  
      }  
    }  

// Find closest vertex to this one in same sub-event, save in nearestVertex
    int indexTV = 0;
    double closest = 9e99;
    TrackingVertexCollection::iterator nearestVertex;

    int tmpTV = 0;
    for (TrackingVertexCollection::iterator iTrkVtx = tVC -> begin(); iTrkVtx != tVC ->end(); ++iTrkVtx, ++tmpTV) {
      double distance = HepLorentzVector(iTrkVtx -> position() - position).v().mag();
      if (distance <= closest && vertEvtId == iTrkVtx -> eventId()) { // flag which one so we can associate them
        closest = distance;
        nearestVertex = iTrkVtx;
        indexTV = tmpTV; 
      }   
    }

// If outside cutoff, create another TrackingVertex, set nearestVertex to it
    
    if (closest > distanceCut_) {
      indexTV = tVC -> size();
      tVC -> push_back(TrackingVertex(position,inVolume,vertEvtId));
      nearestVertex = --(tVC -> end());  // Last entry of vector
    } 
     
// Add data to closest vertex
    
    (*nearestVertex).addG4Vertex(SimVertexRef(G4VtxContainer, indexG4V) ); // Add G4 vertex
    if (vertexBarcode != 0) {
      (*nearestVertex).addGenVertex(GenVertexRef(hepMC,vertexBarcode)); // Add HepMC vertex
    }

// Identify and add child and parent tracks     

    for (std::map<int,int>::iterator mapIndex = g4T_G4SourceV.begin(); 
         mapIndex != g4T_G4SourceV.end(); ++mapIndex) {
      if (mapIndex -> second == indexG4V) {
        int indexTP = g4T_TP[mapIndex -> first];
        (*nearestVertex).addDaughterTrack(TrackingParticleRef(refTPC,indexTP));
        (tPC->at(indexTP)).setParentVertex(TrackingVertexRef(refTVC,indexTV));
      }
    }
    if (vtxParent >= 0) {
      int indexTP = g4T_TP[vtxParent];
      (tPC->at(indexTP)).setDecayVertex(TrackingVertexRef(refTVC,indexTV));
      (*nearestVertex).addParentTrack(TrackingParticleRef(refTPC,indexTP));
    }  
  }

  edm::LogInfo(MessageCategory) << "TrackingTruth found "  << tVC -> size() 
                                << " unique vertices and " << tPC -> size() << " tracks.";
// Put TrackingParticles and TrackingVertices in event
  event.put(tPC,"TrackTruth");
  event.put(tVC,"VertexTruth");
}
  
DEFINE_FWK_MODULE(TrackingTruthProducer)
