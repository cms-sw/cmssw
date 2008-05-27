
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"


using namespace edm;
using namespace std;


typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >   GenVertexRef;
typedef math::XYZTLorentzVectorD    LorentzVector;
typedef math::XYZPoint Vector;

TrackingTruthProducer::TrackingTruthProducer(const edm::ParameterSet &conf) 
{
  conf_ = conf;
  distanceCut_           = conf_.getParameter<double>("vertexDistanceCut");
  dataLabels_            = conf_.getParameter<vector<string> >("HepMCDataLabels");
  simHitLabel_           = conf_.getParameter<string>("simHitLabel");
  hitLabelsVector_       = conf_.getParameter<vector<string> >("TrackerHitLabels");
  volumeRadius_          = conf_.getParameter<double>("volumeRadius");
  volumeZ_               = conf_.getParameter<double>("volumeZ");
  discardOutVolume_      = conf_.getParameter<bool>("discardOutVolume");
  discardHitsFromDeltas_ = conf_.getParameter<bool>("DiscardHitsFromDeltas");
  mergedBremsstrahlung_  = conf_.getParameter<bool>("mergedBremsstrahlung");
  
  MessageCategory_       = "TrackingTruthProducer";

  edm::LogInfo (MessageCategory_) << "Setting up TrackingTruthProducer";
  edm::LogInfo (MessageCategory_) << "Vertex distance cut set to " << distanceCut_  << " mm";
  edm::LogInfo (MessageCategory_) << "Volume radius set to "       << volumeRadius_ << " mm";
  edm::LogInfo (MessageCategory_) << "Volume Z      set to "       << volumeZ_      << " mm";
  edm::LogInfo (MessageCategory_) << "Discard out of volume? "     << discardOutVolume_;
  edm::LogInfo (MessageCategory_) << "Discard Hits from Deltas? "  << discardHitsFromDeltas_;

  produces<TrackingVertexCollection>();
  produces<TrackingParticleCollection>();
}

void TrackingTruthProducer::produce(Event &event, const EventSetup &)
{
//  TimerStack timers;  // Don't need the timers now, left for example
//  timers.push("TrackingTruth:Producer");
//  timers.push("TrackingTruth:Setup");
  
  
  bool foundHepMC = false;
  for (vector<string>::const_iterator source = dataLabels_.begin(); source != dataLabels_.end(); ++source)
  {
    foundHepMC = event.getByLabel(*source, hepmc_);
    if (foundHepMC)
    {
      edm::LogInfo (MessageCategory_) << "Using HepMC source " << *source;
      break;
    }
  }

  if (!foundHepMC)
  {
    edm::LogWarning (MessageCategory_) << "No HepMC source found";
    return;
  }
  
  // Grab all the PSimHit from the different sencitive volumes
  edm::Handle<CrossingFrame<PSimHit> > cfPSimHits;
  std::vector<const CrossingFrame<PSimHit> *> cfPSimHitProductPointers;

  // Collect the product pointers to the different psimhit collection
  for(std::size_t i = 0; i< hitLabelsVector_.size(); ++i)
  {
    event.getByLabel("mix", hitLabelsVector_[i], cfPSimHits);
    cfPSimHitProductPointers.push_back(cfPSimHits.product());
  }

  // Create a mix collection from the different psimhit collections
  pSimHits_ = std::auto_ptr<MixCollection<PSimHit> >(new MixCollection<PSimHit>(cfPSimHitProductPointers));

  // Collect all the simtracks from the crossing frame
  edm::Handle<CrossingFrame<SimTrack> > cfSimTracks;
  event.getByLabel("mix", simHitLabel_, cfSimTracks);
  
  // Create a mix collection from one simtrack collection
  simTracks_ = std::auto_ptr<MixCollection<SimTrack> >( new MixCollection<SimTrack>(cfSimTracks.product()) );

  // Collect all the simvertex from the crossing frame
  edm::Handle<CrossingFrame<SimVertex> > cfSimVertexes;
  event.getByLabel("mix", simHitLabel_, cfSimVertexes);
  
  // Create a mix collection from one simvertex collection
  simVertexes_ = std::auto_ptr<MixCollection<SimVertex> >( new MixCollection<SimVertex>(cfSimVertexes.product()) );

  // Create collections of things we will put in event
  trackingParticles_ = std::auto_ptr<TrackingParticleCollection>( new TrackingParticleCollection );
  trackingVertexes_ = std::auto_ptr<TrackingVertexCollection>( new TrackingVertexCollection );

  // Get references before put so we can cross reference
  refTrackingParticles_ = event.getRefBeforePut<TrackingParticleCollection>();
  refTrackingVertexes_ = event.getRefBeforePut<TrackingVertexCollection>();

  //  timers.pop();
  
  // Create a map between trackId and track index
  associator(simTracks_, trackIdToIndex_);
     
  // Create a multimap between trackId and hit indices
  associator(pSimHits_, trackIdToHits_);
  
  createTrackingTruth();

  cout << "Saving in the event" << std::endl << std::endl;

  event.put(trackingParticles_);
  event.put(trackingVertexes_);

  //  timers.pop();
  //  timers.pop();
}


void TrackingTruthProducer::createTrackingTruth()
{	
  // Reset the event counter (use for define vertexId)	
  eventIdCounter_.clear();
  
  // Define a container of vetoed traks
  std::map<int,std::size_t> vetoedTracks;

  // Define map between parent simtrack and tv indexes
  std::map<int,std::size_t> vetoedSimVertexes; 
    
  for (int simTrackIndex = 0; simTrackIndex != simTracks_->size(); ++simTrackIndex)  {
  	// Check if the simTrack is excluded (includes non traceable and recovered by history)
    if( vetoedTracks.find(simTrackIndex) != vetoedTracks.end() ) continue;

  	SimTrack const & simTrack = simTracks_->getObject(simTrackIndex);
  	
  	TrackingParticle trackingParticle;
  	
  	// Set a bare tp (only with the psimhit) with a given simtrack
  	// the function return true if it is tracable
    if ( setTrackingParticle(simTrack, trackingParticle) )
    {
      cout << std::endl << "Found a tracebla track with #SimHits : " << trackingParticle. matchedHit() << std::endl;

      // Follows the path upward recovering the history of the particle      
      SimTrack const * currentSimTrack = & simTrack;

      // Initial condition for the tp and tv indexes
      int trackingParticleIndex = -1;
      int trackingVertexIndex = -1;

      do 
      {
      	// Set a new tracking particle for the current simtrack
      	// and add it to the list of parent tracks of previous vertex
      	if (trackingParticleIndex >= 0)
      	{
      	  setTrackingParticle(*currentSimTrack, trackingParticle);

      	  // Set the tp index to its new value
          trackingParticleIndex = trackingParticles_->size();
          // Push the tp in to the collection 
          trackingParticles_->push_back(trackingParticle);
          
          // Add the previous track to the list of decay vertexes of the new tp
          trackingParticles_->at(trackingParticleIndex).addDecayVertex(
            TrackingVertexRef(refTrackingVertexes_, trackingVertexIndex)
          );  
          
          // Add the new tp to the list of parent tracks of the previous tv
          trackingVertexes_->at(trackingVertexIndex).addParentTrack(
            TrackingParticleRef(refTrackingParticles_, trackingParticleIndex)
          );
          
      	  cout << "Adding secondary tp with index : " << trackingParticleIndex << std::endl;
      	}
      	else 
      	{
      	  // Set the tp index to its new value
          trackingParticleIndex = trackingParticles_->size();
          // Push the tp in to the collection 
          trackingParticles_->push_back(trackingParticle);
          
     	  cout << "Adding primaty tp with index : " << trackingParticleIndex << std::endl;
      	}
      	
      	// Verify if the parent simVertex has a simTrack or if the source is a vetoSimVertex
        if (currentSimTrack->noVertex()) break;
      	
      	// Get the simTrack parent index
      	unsigned int parentSimVertexIndex = currentSimTrack->vertIndex();

      	cout << "Getting parent simtrack index: " << parentSimVertexIndex << std::endl;
      	      	      	  
        // Create a new tv 	
        TrackingVertex trackingVertex;
        // Get the parent simVertex associated to the current simTrack
        SimVertex const * parentSimVertex = & simVertexes_->getObject(parentSimVertexIndex);       

        bool vetoSimVertex = vetoedSimVertexes.find(parentSimVertexIndex) != vetoedSimVertexes.end();

        // Check for a already visited parent simTrack
      	if ( !vetoSimVertex )
        {
          cout << "Vertex was not visited before: " << parentSimVertexIndex << std::endl;
                 	 
          // Set the tv by using simvertex
          trackingVertexIndex = setTrackingVertex(*parentSimVertex, trackingVertex);

          cout << "Vertex setted return tv index: " << trackingVertexIndex << std::endl;

          // Check if a new vertex needs to be created
          if (trackingVertexIndex < 0)
          {
            // Set the tv index ot its new value
            trackingVertexIndex = trackingVertexes_->size();
            // Push the new tv in to the collection
            trackingVertexes_->push_back(trackingVertex);

            cout << "Vertex added with tv index: " << trackingVertexIndex << std::endl;
          }
          else
          {
            // Get the postion and time of the vertex
            const LorentzVector & position = trackingVertexes_->at(trackingVertexIndex).position();
            Vector xyz = Vector(position.x(), position.y(), position.z());
            double t = position.t();            // Set the vertex postion of the tp to the closest vertex        	
            trackingParticles_->at(trackingParticleIndex).setVertex(xyz, t);
            
            cout << "Merged vertex with index: " << trackingVertexIndex << std::endl;
          }
          
          vetoedSimVertexes.insert( make_pair(parentSimVertexIndex, trackingVertexIndex) );
        }
        else
        {
          trackingVertexIndex = vetoedSimVertexes[parentSimVertexIndex];          
          cout << "Visited vertex with index: " << trackingVertexIndex << std::endl;
        }

        cout << "Setting up vertex reference" << std::endl;

        // Set the newly created tv as parent vertex
        trackingParticles_->at(trackingParticleIndex).setParentVertex(
          TrackingVertexRef(refTrackingVertexes_, trackingVertexIndex)
        );

        // Add the newly created tp to the tv daughter list
        trackingVertexes_->at(trackingVertexIndex).addDaughterTrack(
          TrackingParticleRef(refTrackingParticles_, trackingParticleIndex)
        );
        
        // Verify if the parent simVertex has a simTrack or if the source is a vetoSimVertex
        if (parentSimVertex->noParent() || vetoSimVertex) break;
        
        // Get the next simTrack index (it is implicit should be in the same event as current).
        unsigned int nextSimTrackIndex = trackIdToIndex_[
          EncodedTruthId(
            currentSimTrack->eventId(),
            parentSimVertex->parentIndex()
          )
        ];
          
        // Check if the next track exist
        if( vetoedTracks.find(nextSimTrackIndex) != vetoedTracks.end() )
        {
          // Add to the newly created tv the existent next simtrack in to parent list. 
          trackingVertexes_->at(trackingVertexIndex).addParentTrack(
            TrackingParticleRef(refTrackingParticles_, vetoedTracks[nextSimTrackIndex])
          );
          break;
        }
          
        // Vetoed the next simTrack
        vetoedTracks.insert( make_pair(nextSimTrackIndex, trackingParticleIndex) );
          
        // Set the current simTrack as the next simTrack
        currentSimTrack = & simTracks_->getObject(nextSimTrackIndex);

        cout << "SimTrack with history" << std::endl;
 
      } while(!currentSimTrack->noVertex());
    }
  }
}


bool TrackingTruthProducer::setTrackingParticle(
  SimTrack const & simTrack,
  TrackingParticle & trackingParticle
)
{
  // Get the eventid associated to the track
  EncodedEventId trackEventId = simTrack.eventId();	
  // Get the simtrack id
  EncodedTruthId simTrackId = EncodedTruthId( trackEventId, simTrack.trackId() );
		
  // Location of the parent vertex
  LorentzVector position;
  // If not parent then location is (0,0,0,0)
  if (simTrack.noVertex())
     position = LorentzVector(0, 0, 0, 0);
  else
     position = simVertexes_->getObject(simTrack.vertIndex()). position();   

  // Define the default status and pdgid
  int status = -99;
  int pdgId = simTrack.type();

  int genParticleIndex = simTrack.genpartIndex();
  bool signalEvent = (trackEventId.event() == 0 && trackEventId.bunchCrossing() == 0);

  // Get the generated particle
  const HepMC::GenParticle * genParticle = hepmc_->GetEvent()->barcode_to_particle(genParticleIndex); 

  // In the case of a existing generated particle and track 
  // event is signal redefine status a pdgId  
  if (genParticleIndex >= 0 && signalEvent)    if (genParticle)    {      status = genParticle->status();      pdgId  = genParticle->pdg_id();    }

  // Create a tp from the simtrack
  trackingParticle = TrackingParticle(
    (char) simTrack.charge(),
    simTrack.momentum(),
    Vector(position.x(), position.y(), position.z()),
    position.t(),
    pdgId,
    status,
    trackEventId
  );
  
  bool init = true;
  
  int processType = 0;
  int particleType = 0;
  
  // Counting the TP hits using the layers (as in ORCA).  // Does seem to find less hits. maybe b/c layer is a number now, not a pointer  int totalSimHits = 0;  int oldLayer = 0;  int newLayer = 0;  int oldDetector = 0;  int newDetector = 0;  
    
  // Loop over the associated hits per track
  for (
    EncodedTruthIdToIndexes::const_iterator iEntry = trackIdToHits_.lower_bound(simTrackId);	iEntry != trackIdToHits_.upper_bound(simTrackId); 	++iEntry
  )   {
  	// Get a constant reference to the simhit    PSimHit const & pSimHit = pSimHits_->getObject(iEntry->second);      
    // Initial condition for consistent simhit selection      if(init)    {      processType = pSimHit.processType();
      particleType = pSimHit.particleType();      init = false;
    }    // Check for delta and interaction products discards    if(processType == pSimHit.processType() && particleType == pSimHit.particleType() && pdgId == pSimHit.particleType() )    {      trackingParticle.addPSimHit(pSimHit);
            unsigned int detectorIdIndex = pSimHit.detUnitId();      DetId detectorId = DetId(detectorIdIndex);      oldLayer = newLayer;      oldDetector = newDetector;      newLayer = LayerFromDetid(detectorIdIndex);      newDetector = detectorId.subdetId();	  // Count hits using layers for glued detectors      if (oldLayer != newLayer || (oldLayer==newLayer && oldDetector!=newDetector) ) totalSimHits++;    }
  }
  
  // Set the number of matched simhits
  trackingParticle.setMatchedHit(totalSimHits);

  // Add the simtrack associated to the tp
  trackingParticle.addG4Track(simTrack);
    
  // Add the generator information
  if (genParticleIndex >= 0 && signalEvent)    trackingParticle.addGenParticle( GenParticleRef(hepmc_, genParticleIndex) );
      
  if (totalSimHits) return true;
  
  return false;
}


int TrackingTruthProducer::setTrackingVertex(
  SimVertex const & simVertex,
  TrackingVertex & trackingVertex
)
{
  LorentzVector const & position = simVertex.position();
	
  // Look for close by vertexes
  for(std::size_t trackingVertexIndex = 0; trackingVertexIndex < trackingVertexes_->size(); ++trackingVertexIndex)
  {
  	// Calculate the distance
    double distance = (position - trackingVertexes_->at(trackingVertexIndex).position()).P();
    // If the distance is under a given cut return the trackingVertex index (vertex merging)
    if (distance <= distanceCut_)
    {
      // Add simvertex to the pre existent tv	
      trackingVertexes_->at(trackingVertexIndex).addG4Vertex(simVertex);
      // return tv index 
      return trackingVertexIndex;
    }
  }

  // Get the event if from the simvertex	
  EncodedEventId simVertexEventId = simVertex.eventId();
  
  // Initialize the event counter	
  if (eventIdCounter_.find(simVertexEventId) == eventIdCounter_.end())
    eventIdCounter_[simVertexEventId] = 0;

  // Get the simVertex id
  EncodedTruthId simVertexId = EncodedTruthId(simVertexEventId, eventIdCounter_[simVertexEventId]);

  // Calculate if the vertex is in the tracker volume (it needs to be review for other detectors)
  bool inVolume = (position.Pt() < volumeRadius_ && abs(position.z()) < volumeZ_); // In or out of Tracker

  // Initialize the new vertex
  trackingVertex = TrackingVertex(position, inVolume, simVertexId);

  // Find the the closest GenVertexes to the created tv
  addCloseGenVertexes(trackingVertex);

  // Add the g4 vertex to the tv
  trackingVertex.addG4Vertex(simVertex);

  // Initialize the event counter   
  eventIdCounter_[simVertexEventId]++;
  
  return -1;
}


void TrackingTruthProducer::addCloseGenVertexes(TrackingVertex & trackingVertex)
{
  // Get the generated particle
  const HepMC::GenEvent * genEvent = hepmc_->GetEvent();

  // Get the postion of the tv
  Vector tvPosition(trackingVertex.position().x(), trackingVertex.position().y(), trackingVertex.position().z());
	
  // Find HepMC vertices, put them in a close TrackingVertex (this could conceivably add the same GenVertex to multiple TrackingVertices)  for (
    HepMC::GenEvent::vertex_const_iterator iGenVertex = genEvent->vertices_begin(); 
    iGenVertex != genEvent->vertices_end();
    ++iGenVertex
  )   {
  	// Get the position of the genVertex
    HepMC::ThreeVector rawPosition = (*iGenVertex)->position();

    // Convert to cm    Vector genPosition(rawPosition.x()/10.0, rawPosition.y()/10.0, rawPosition.z()/10.0);
 
    // Calculate the dis
    double distance = sqrt( (tvPosition - genPosition).mag2() );    
    if (distance <= distanceCut_)      trackingVertex.addGenVertex( GenVertexRef(hepmc_, (*iGenVertex)->barcode()) );
   }
}


int TrackingTruthProducer::LayerFromDetid(const unsigned int& detid)
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

//DEFINE_FWK_MODULE(TrackingTruthProducer);
