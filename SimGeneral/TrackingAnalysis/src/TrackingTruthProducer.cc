#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"


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
  distanceCut_           = conf_.getParameter<double>("vertexDistanceCut");
  dataLabels_            = conf_.getParameter<vector<string> >("HepMCDataLabels");
  simHitLabel_	         = conf_.getParameter<string>("simHitLabel");
  hitLabelsVector_       = conf_.getParameter<vector<string> >("TrackerHitLabels");
  volumeRadius_          = conf_.getParameter<double>("volumeRadius");   
  volumeZ_               = conf_.getParameter<double>("volumeZ");   
  discardOutVolume_      = conf_.getParameter<bool>("discardOutVolume");     
  discardHitsFromDeltas_ = conf_.getParameter<bool>("DiscardHitsFromDeltas");

  edm::LogInfo (MessageCategory) << "Setting up TrackingTruthProducer";
  edm::LogInfo (MessageCategory) << "Vertex distance cut set to " << distanceCut_  << " mm";
  edm::LogInfo (MessageCategory) << "Volume radius set to "       << volumeRadius_ << " mm";
  edm::LogInfo (MessageCategory) << "Volume Z      set to "       << volumeZ_      << " mm";
  edm::LogInfo (MessageCategory) << "Discard out of volume? "     << discardOutVolume_;
  edm::LogInfo (MessageCategory) << "Discard Hits from Deltas? "     << discardHitsFromDeltas_;

  /* Uncommenting will print out the various hit collections that will be scanned  

  for (vector<string>::const_iterator name = hitLabelsVector_.begin(); name != hitLabelsVector_.end(); ++name) {
    edm::LogInfo (MessageCategory) << "Use hits with label(s) "   << *name;
  }  

  */
}

void TrackingTruthProducer::produce(Event &event, const EventSetup &) {

  // Get information out of event record
  edm::Handle<edm::HepMCProduct> hepMC;
  for (vector<string>::const_iterator source = dataLabels_.begin(); 
       source != dataLabels_.end(); ++source) {
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
   
  edm::Handle<CrossingFrame> cf;
  try {
    event.getByType(cf);      
  } catch (std::exception &e) {
    edm::LogWarning (MessageCategory) << "Crossing frame not found.";
    return;
  }    
  std::auto_ptr<MixCollection<SimTrack> >   trackCollection (new MixCollection<SimTrack>(cf.product()));
  std::auto_ptr<MixCollection<SimVertex> > vertexCollection (new MixCollection<SimVertex>(cf.product()));
  std::auto_ptr<MixCollection<PSimHit> >      hitCollection (new MixCollection<PSimHit>(cf.product(),hitLabelsVector_));

// Create collections of things we will put in event,
// get references before put so we can cross reference  
  
  auto_ptr<TrackingParticleCollection> tPC(new TrackingParticleCollection);
  auto_ptr<TrackingVertexCollection>   tVC(new TrackingVertexCollection  );  

  edm::RefProd<TrackingParticleCollection> refTPC =
      event.getRefBeforePut<TrackingParticleCollection>("TrackTruth");
  edm::RefProd<TrackingVertexCollection>   refTVC =
      event.getRefBeforePut<TrackingVertexCollection>("VertexTruth");
  
  map<EncodedTruthId,EncodedTruthId> simTrack_sourceV; // Encoded SimTrack to encoded source vertex
  map<EncodedTruthId,int> simTrack_tP;                 // Encoded SimTrack to TrackingParticle index
               
  for (MixCollection<SimTrack>::MixItr itP = trackCollection->begin(); 
       itP !=  trackCollection->end(); ++itP){
    char                      q = (char)itP -> charge(); // Check this
    CLHEP::HepLorentzVector   p = itP -> momentum();
    unsigned int     simtrackId = itP -> trackId();
    EncodedEventId trackEventId = itP -> eventId(); 
    EncodedTruthId trackId      = EncodedTruthId(trackEventId,simtrackId);
    int                 genPart = itP -> genpartIndex(); // The HepMC particle number
    int                 genVert = itP -> vertIndex();    // The SimVertex #
    
    bool signalEvent = (trackEventId.event() == 0 && trackEventId.bunchCrossing() == 0);
    const TrackingParticle::LorentzVector theMomentum(p.x(), p.y(), p.z(), p.t());
    double  time = 0; 
    int    pdgId = 0;

    const HepMC::GenParticle *gp = 0;       

    if (genPart >= 0 && signalEvent) {
      gp = genEvent -> barcode_to_particle(genPart);  // Pointer to the generating particle.
      pdgId = gp -> pdg_id();
    }
    
    math::XYZPoint theVertex;

    if (genVert >= 0){ // Add to useful maps
      EncodedTruthId vertexId = EncodedTruthId(trackEventId,genVert);                    
      simTrack_sourceV.insert(make_pair(trackId,vertexId));
    }

    TrackingParticle tp(q, theMomentum, theVertex, time, pdgId, trackEventId);
    
// Counting the TP hits using the layers (as in ORCA). 
// Does seem to find less hits. maybe b/c layer is a number now, not a pointer
    int totsimhit = 0; 
    int oldlay = 0;
    int newlay = 0;
    int olddet = 0;
    int newdet = 0;

    for (MixCollection<PSimHit>::MixItr hit = hitCollection -> begin(); 
         hit != hitCollection -> end(); ++hit) {
      if (simtrackId == hit->trackId() && trackEventId == hit->eventId() ) {
	float pratio = hit->pabs()/(itP->momentum().v().mag());
        
// Discard hits from delta rays if requested        
	
        if (!discardHitsFromDeltas_ || ( discardHitsFromDeltas_ &&  0.5 < pratio && pratio < 2) ) {  
	  tp.addPSimHit(*hit);
	  unsigned int detid = hit->detUnitId();      
	  DetId detId = DetId(detid);
	  oldlay = newlay;
	  olddet = newdet;
	  newlay = LayerFromDetid(detid);
	  newdet = detId.subdetId();

// Count hits using layers for glued detectors
           
	  if (oldlay != newlay || (oldlay==newlay && olddet!=newdet) ) {
	    totsimhit++;
	  }
        }
      }
    }

    tp.setMatchedHit(totsimhit);
   
    tp.addG4Track(*itP);
    if (genPart >= 0 && signalEvent) {
      tp.addGenParticle(GenParticleRef(hepMC,genPart));
    }

// Add indices to map and add to collection    
    
    simTrack_tP.insert(make_pair(trackId,tPC->size()));
    tPC -> push_back(tp);
  }

// Find and loop over EmbdSimVertex vertices
    
  int indexG4V = 0;
  unsigned int simVertexId = 0;
  EncodedEventId lastEventID;
  for (MixCollection<SimVertex>::MixItr itVtx = vertexCollection->begin(); 
       itVtx != vertexCollection->end(); 
       ++itVtx,++indexG4V) {

    CLHEP::HepLorentzVector position = itVtx -> position();  // Get position of ESV
    bool inVolume = (position.perp() < volumeRadius_ && abs(position.z()) < volumeZ_); // In or out of Tracker
    if (!inVolume && discardOutVolume_) { continue; }        // Skip if desired
    
    EncodedEventId vertEvtId = itVtx -> eventId();     
    EncodedTruthId vertexId  = EncodedTruthId(vertEvtId,simVertexId);

//    unsigned int     simVertexId = 0; // Would like accessor in SimTrack. Do this kludge instead
// This may not be correct. May have to detect a bX change or a change from singal to pileup    
    ++simVertexId;
    if (lastEventID != vertEvtId) {
      lastEventID = vertEvtId;
      simVertexId = 0;
    }  
// Figure out the barcode of the HepMC Vertex if there is one by
// getting incoming SimTtrack (if any), finding corresponding HepMC track and
// then decay (HepMC) vertex of that track    
    int vertexBarcode = 0;       
    unsigned int vtxParent = itVtx -> parentIndex();    
    if (vtxParent >= 0) {                      
      for (MixCollection<SimTrack>::MixItr itP = trackCollection->begin(); itP != trackCollection->end(); ++itP){
	if(vtxParent==itP->trackId() && itP.bunch() == itVtx.bunch()){
	  int parentBC = itP->genpartIndex();  
	  HepMC::GenParticle *parentParticle = genEvent -> barcode_to_particle(parentBC);
	  if (parentParticle != 0) {
	    HepMC::GenVertex *hmpv = parentParticle -> end_vertex(); 
	    if (hmpv != 0) {
	      vertexBarcode = hmpv  -> barcode();
	    }  
	  }
	  break;
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
    
    (*nearestVertex).addG4Vertex(*itVtx); // Add G4 vertex
    if (vertexBarcode != 0) {
      (*nearestVertex).addGenVertex(GenVertexRef(hepMC,vertexBarcode)); // Add HepMC vertex
    }

// Identify and add child and parent tracks     

    for (std::map<EncodedTruthId,EncodedTruthId>::iterator mapIndex = simTrack_sourceV.begin(); 
         mapIndex != simTrack_sourceV.end(); ++mapIndex) {
      if (mapIndex -> second == vertexId) {
        int indexTP = simTrack_tP[mapIndex -> first];
        (*nearestVertex).addDaughterTrack(TrackingParticleRef(refTPC,indexTP));
        (tPC->at(indexTP)).setParentVertex(TrackingVertexRef(refTVC,indexTV));
        const CLHEP::HepLorentzVector &v = (*nearestVertex).position();
        math::XYZPoint xyz = math::XYZPoint(v.x(), v.y(), v.z());
        double t = v.t(); 
        (tPC->at(indexTP)).setVertex(xyz,t);
      }
    }

    if (vtxParent > 0) {
      EncodedTruthId trackId =  EncodedTruthId(vertEvtId,vtxParent);
      int indexTP = simTrack_tP[trackId];
      (tPC->at(indexTP)).addDecayVertex(TrackingVertexRef(refTVC,indexTV));
      if (indexTP > 0) {
        (*nearestVertex).addParentTrack(TrackingParticleRef(refTPC,indexTP-1));
      }  
    }  
  }

  edm::LogInfo(MessageCategory) << "TrackingTruth found "  << tVC -> size() 
                                << " unique vertices and " << tPC -> size() << " tracks.";

// Put TrackingParticles and TrackingVertices in event
  event.put(tPC,"TrackTruth");
  event.put(tVC,"VertexTruth");
}

int TrackingTruthProducer::LayerFromDetid(const unsigned int& detid )
{
  DetId detId = DetId(detid);
  int layerNumber=0;
  unsigned int subdetId = static_cast<unsigned int>(detId.subdetId()); 
  if ( subdetId == StripSubdetector::TIB) 
    { 
      TIBDetId tibid(detId.rawId()); 
      layerNumber = tibid.layer();
    }
  else if ( subdetId ==  StripSubdetector::TOB )
    { 
      TOBDetId tobid(detId.rawId()); 
      layerNumber = tobid.layer();
    }
  else if ( subdetId ==  StripSubdetector::TID) 
    { 
      TIDDetId tidid(detId.rawId());
      layerNumber = tidid.wheel();
    }
  else if ( subdetId ==  StripSubdetector::TEC )
    { 
      TECDetId tecid(detId.rawId()); 
      layerNumber = tecid.wheel(); 
    }
  else if ( subdetId ==  PixelSubdetector::PixelBarrel ) 
    { 
      PXBDetId pxbid(detId.rawId()); 
      layerNumber = pxbid.layer();  
    }
  else if ( subdetId ==  PixelSubdetector::PixelEndcap ) 
    { 
      PXFDetId pxfid(detId.rawId()); 
      layerNumber = pxfid.disk();  
    }
  else
    edm::LogVerbatim("TrackingTruthProducer") << "Unknown subdetid: " <<  subdetId;
  
  return layerNumber;
} 
  
DEFINE_FWK_MODULE(TrackingTruthProducer);
