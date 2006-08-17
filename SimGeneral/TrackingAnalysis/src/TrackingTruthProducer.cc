#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackVertexMap.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"

#include <map>

#include "DataFormats/Common/interface/RefVectorBase.h"

using namespace edm;
using namespace std; 
using CLHEP::HepLorentzVector;

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >   GenVertexRef;

string MessageCategory = "TrackingTruthProducer";

TrackingTruthProducer::TrackingTruthProducer(const edm::ParameterSet &conf) {
  produces<TrackingVertexCollection>("VertexTruth");
  produces<TrackingParticleCollection>("TrackTruth");
  produces<TrackVertexAssociationCollection>();
  produces<VertexTrackAssociationCollection>();
  conf_ = conf;
  distanceCut_      = conf_.getParameter<double>("vertexDistanceCut");
  dataLabels_       = conf_.getParameter<vector<string> >("HepMCDataLabels");
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
  
  auto_ptr<TrackVertexAssociationCollection> trackVertexMap(new TrackVertexAssociationCollection);
  auto_ptr<VertexTrackAssociationCollection> vertexTrackMap(new VertexTrackAssociationCollection);
  
  
  edm::Handle<SimVertexContainer>      G4VtxContainer;
  edm::Handle<edm::SimTrackContainer>  G4TrkContainer;
  event.getByType(G4VtxContainer);
  event.getByType(G4TrkContainer);
  
  edm::Handle<edm::PSimHitContainer> TIBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TIDHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TIDHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TOBHitsHighTof;
  edm::Handle<edm::PSimHitContainer> TECHitsLowTof;
  edm::Handle<edm::PSimHitContainer> TECHitsHighTof;
  edm::Handle<edm::PSimHitContainer> PixelBarrelHitsHighTof;
  edm::Handle<edm::PSimHitContainer> PixelBarrelHitsLowTof;
  edm::Handle<edm::PSimHitContainer> PixelEndcapHitsHighTof;
  edm::Handle<edm::PSimHitContainer> PixelEndcapHitsLowTof;

  event.getByLabel("g4SimHits","TrackerHitsTIBLowTof", TIBHitsLowTof);
  event.getByLabel("g4SimHits","TrackerHitsTIBHighTof", TIBHitsHighTof);
  event.getByLabel("g4SimHits","TrackerHitsTIDLowTof", TIDHitsLowTof);
  event.getByLabel("g4SimHits","TrackerHitsTIDHighTof", TIDHitsHighTof);
  event.getByLabel("g4SimHits","TrackerHitsTOBLowTof", TOBHitsLowTof);
  event.getByLabel("g4SimHits","TrackerHitsTOBHighTof", TOBHitsHighTof);
  event.getByLabel("g4SimHits","TrackerHitsTECLowTof", TECHitsLowTof);
  event.getByLabel("g4SimHits","TrackerHitsTECHighTof", TECHitsHighTof);
  event.getByLabel("g4SimHits","TrackerHitsPixelBarrelHighTof", PixelBarrelHitsHighTof);
  event.getByLabel("g4SimHits","TrackerHitsPixelBarrelLowTof", PixelBarrelHitsLowTof);  
  event.getByLabel("g4SimHits","TrackerHitsPixelEndcapHighTof", PixelEndcapHitsHighTof);  
  event.getByLabel("g4SimHits","TrackerHitsPixelEndcapLowTof", PixelEndcapHitsLowTof);
  
//  const HepMC::GenEvent            &genEvent = hepMC -> getHepMCData();
  const HepMC::GenEvent *genEvent = mcp -> GetEvent(); // faster?

  const edm::SimTrackContainer      *etc = G4TrkContainer.product();

  if (mcp == 0) {
    edm::LogWarning (MessageCategory) << "No HepMC source found";
    return;
  }  
   
   
  vector<edm::Handle<edm::PSimHitContainer> > AlltheConteiners;
   
  AlltheConteiners.push_back(TIBHitsLowTof);
  AlltheConteiners.push_back(TIBHitsHighTof);
  AlltheConteiners.push_back(TIDHitsLowTof);
  AlltheConteiners.push_back(TIDHitsHighTof);
  AlltheConteiners.push_back(TOBHitsLowTof);
  AlltheConteiners.push_back(TOBHitsHighTof);
  AlltheConteiners.push_back(TECHitsLowTof);
  AlltheConteiners.push_back(TECHitsHighTof);
//  AlltheConteiners.push_back(PixelBarrelHitsHighTof);
//  AlltheConteiners.push_back(PixelBarrelHitsLowTof);
//  AlltheConteiners.push_back(PixelEndcapHitsHighTof);
//  AlltheConteiners.push_back(PixelEndcapHitsLowTof);
  
  
//  genEvent.print();
//  genEvent ->  signal_process_id();
  // 13 cosmic muons
  // 20 particle 
  // Others from Pythia, begin on page 132. Hope there is a flag somewhere else
  // Don't want to figure out minBias vs. other things.
  
//Put TrackingParticle here... need charge, momentum, vertex position, time, pdg id
  auto_ptr<TrackingParticleCollection> tPC(new TrackingParticleCollection);

  edm::RefProd<TrackingParticleCollection> refTPC =
      event.getRefBeforePut<TrackingParticleCollection>("TrackTruth");
  edm::RefProd<TrackingVertexCollection> refTVC =
      event.getRefBeforePut<TrackingVertexCollection>("VertexTruth");
  
  std::map<int,int> productionVertex;
  std::multimap<int,int> tmpTrackVertexMap;
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
//      int index = 0;
//      for(edm::PSimHitContainer::const_iterator itSimHit =  TIBHitsLowTof->begin(); itSimHit !=TIBHitsLowTof->end();
//	 ++itSimHit, ++index){
//	 tp.addPSimHit(TrackPSimHitRef(TIBHitsLowTof, index));
//      }
	
    tp.addG4Track(SimTrackRef(G4TrkContainer,iG4Track));
    if (genPart >= 0) {
      tp.addGenParticle(GenParticleRef(hepMC,genPart));
    }
    productionVertex.insert(pair<int,int>(tPC->size(),genVert));
    tPC -> push_back(tp);
    ++iG4Track;
  }

// Put TrackingParticles in event and get handle to access them    
  
  edm::OrphanHandle<TrackingParticleCollection> tpcHandle =
      event.put(tPC,"TrackTruth");
//  TrackingParticleCollection trackCollection = *tpcHandle;
       
// Find and loop over EmbdSimVertex vertices
    
  auto_ptr<TrackingVertexCollection> tVC( new TrackingVertexCollection );  

  int index = 0;
  for (edm::SimVertexContainer::const_iterator itVtx = G4VtxContainer->begin(); 
       itVtx != G4VtxContainer->end(); 
       ++itVtx,++index) {

    CLHEP::HepLorentzVector position = itVtx -> position();  // Get position of ESV
    bool inVolume = (position.perp() < volumeRadius_ && abs(position.z()) < volumeZ_); // In or out of Tracker
    if (!inVolume && discardOutVolume_) { continue; }        // Skip if desired
    
    EncodedEventId vertEvtId = itVtx -> eventId();     // May not be right one, get from HepMC?
    
// Figure out the barcode of the HepMC Vertex if there is one
    
    int vertexBarcode = 0;       
    int vtxParent = itVtx -> parentIndex();    // Get incoming SimTtrack
    if (vtxParent >= 0) {                      // Is there a parent track? 
      SimTrack est = etc->at(vtxParent);       // Pull track out from vector
      int partHepMC =     est.genpartIndex();  // Get HepMC particle barcode
      HepMC::GenParticle *hmp = genEvent -> barcode_to_particle(partHepMC); // Convert barcode
      if (hmp != 0) {
        HepMC::GenVertex *hmpv = hmp -> production_vertex(); 
        if (hmpv != 0) {
          vertexBarcode = hmpv  -> barcode();
        }  
      }  
    }  

// Find closest vertex to this one in same sub-event, save in nearestVertex

    double closest = 9e99;
    TrackingVertexCollection::iterator nearestVertex;

    for (TrackingVertexCollection::iterator iTrkVtx = tVC -> begin(); iTrkVtx != tVC ->end(); ++iTrkVtx) {
      double distance = HepLorentzVector(iTrkVtx -> position() - position).v().mag();
      if (distance <= closest && vertEvtId == iTrkVtx -> eventId()) { // flag which one so we can associate them
        closest = distance;
        nearestVertex = iTrkVtx; 
      }   
    }

// If outside cutoff, create another TrackingVertex, set nearest to it
    
    if (closest > distanceCut_) {
      tVC -> push_back(TrackingVertex(position,inVolume,vertEvtId));
      nearestVertex = --(tVC -> end());  // Last entry of vector
    } 
     
// Add data to closest vertex
    
    (*nearestVertex).addG4Vertex(SimVertexRef(G4VtxContainer, index) ); // Add G4 vertex
    if (vertexBarcode != 0) {
      (*nearestVertex).addGenVertex(GenVertexRef(hepMC,vertexBarcode)); // Add HepMC vertex
    }

// Identify and add child tracks       

    for (std::map<int,int>::iterator mapIndex = productionVertex.begin(); 
         mapIndex != productionVertex.end();
         ++mapIndex) {
      if (mapIndex -> second == index) {
//        (*nearestVertex).add(TrackingParticleRef(tpcHandle,mapIndex -> first));
        tmpTrackVertexMap.insert(pair<int,int>(mapIndex -> first,tVC->size()-1));
      }
    }
         
  }

  edm::LogInfo (MessageCategory) << "TrackingTruth found " << tVC->size() << " unique vertices";

// Dump out the results  
  
  index = 0;
  for (TrackingVertexCollection::const_iterator v =
       tVC -> begin();
       v != tVC ->end(); ++v) {
    edm::LogInfo (MessageCategory) << "TrackingVertex " << index << " has " 
      << (v -> g4Vertices()).size()  << " G4 vertices, " 
      << (v -> genVertices()).size() << " HepMC vertices";
    ++index;  
  }        
  
// Put TrackingVertices in event and get handle to access them    
  
  edm::OrphanHandle<TrackingVertexCollection> tvcHandle =
      event.put(tVC,"VertexTruth");
//  TrackingVertexCollection vertexCollection = *tvcHandle;
  for (multimap<int,int>::const_iterator a = tmpTrackVertexMap.begin();
      a !=   tmpTrackVertexMap.end(); ++a) {
     int iVertex = a -> second;
     int iTrack  = a -> first;
    (*trackVertexMap).insert(TrackingParticleRef(tpcHandle,iTrack),
                             TrackingVertexRef(tvcHandle,iVertex));
    (*vertexTrackMap).insert(TrackingVertexRef(tvcHandle,iVertex),
                             TrackingParticleRef(tpcHandle,iTrack));
  }
  
  event.put(trackVertexMap);
  event.put(vertexTrackMap);
}
  
DEFINE_FWK_MODULE(TrackingTruthProducer)
