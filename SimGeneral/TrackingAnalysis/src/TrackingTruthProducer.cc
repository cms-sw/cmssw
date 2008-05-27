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

#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"


using namespace edm;
using namespace std;

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >   GenVertexRef;
typedef math::XYZTLorentzVectorD    LorentzVector;

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

  if (!mergedBremsstrahlung_)
  {
    produces<TrackingVertexCollection>();
    produces<TrackingParticleCollection>();
  }
  else
  {
    produces<TrackingVertexCollection>();
    produces<TrackingParticleCollection>();
    produces<TrackingVertexCollection>("MergedTrackTruth");
    produces<TrackingParticleCollection>("MergedTrackTruth");  
  }
}

void TrackingTruthProducer::produce(Event &event, const EventSetup &)
{
//  TimerStack timers;  // Don't need the timers now, left for example
//  timers.push("TrackingTruth:Producer");
//  timers.push("TrackingTruth:Setup");

  // Get HepMC out of event record
  edm::Handle<edm::HepMCProduct> hepMC;
  bool foundHepMC = false;
  for (vector<string>::const_iterator source = dataLabels_.begin(); source != dataLabels_.end(); ++source) 
  {
    foundHepMC = event.getByLabel(*source,hepMC);
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

  const edm::HepMCProduct * mcp = hepMC.product();

  if (mcp == 0)
  {
    edm::LogWarning (MessageCategory_) << "Null HepMC pointer";
    return;
  }
  
  // New Templated CF
  // Grab all the PSimHit
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  for(uint32_t i = 0; i< hitLabelsVector_.size();i++)
  {
    event.getByLabel("mix",hitLabelsVector_[i],cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }
  std::auto_ptr<MixCollection<PSimHit> > hitCollection(new MixCollection<PSimHit>(cf_simhitvec));

  // Get all the simtracks from the crossing frame 
  edm::Handle<CrossingFrame<SimTrack> > cf_simtrack;
  event.getByLabel("mix", simHitLabel_, cf_simtrack);
  std::auto_ptr<MixCollection<SimTrack> > trackCollection(new MixCollection<SimTrack>(cf_simtrack.product()));

  // Get all the simvertex from the crossing frame 
  edm::Handle<CrossingFrame<SimVertex> > cf_simvertex;
  event.getByLabel("mix", simHitLabel_, cf_simvertex);
  std::auto_ptr<MixCollection<SimVertex> > vertexCollection(new MixCollection<SimVertex>(cf_simvertex.product()));

  // Create collections of things we will put in event
  auto_ptr<TrackingParticleCollection> tPC(new TrackingParticleCollection);
  auto_ptr<TrackingVertexCollection>   tVC(new TrackingVertexCollection  );

  // Get references before put so we can cross reference
  TrackingParticleRefProd refTPC = event.getRefBeforePut<TrackingParticleCollection>();
  TrackingVertexRefProd   refTVC = event.getRefBeforePut<TrackingVertexCollection>();

  //  timers.pop();
  
  // Create a one to many association between simtracks and hits
  simTrackHitsAssociator(hitCollection);

  // Assamble the tracking particles in function of the simtrack collection
  trackingParticleAssambler(tPC, trackCollection, hepMC);
  
  // Assamble the tracking vertexes including parents-daughters relations
  trackingVertexAssambler(tPC, tVC, trackCollection, vertexCollection, refTPC, refTVC, hepMC);

  edm::LogInfo(MessageCategory_) << "TrackingTruthProducer found " << tVC -> size()
                                 << " unique vertices and "        << tPC -> size() 
                                 << " tracks in the unmerged collection.";

  if (mergedBremsstrahlung_)
  {
  	// Create collections of things we will put in event,
    auto_ptr<TrackingParticleCollection> mergedTPC(new TrackingParticleCollection);
    auto_ptr<TrackingVertexCollection>   mergedTVC(new TrackingVertexCollection  );
    
    // Get references before put so we can cross reference
    TrackingParticleRefProd refMergedTPC = event.getRefBeforePut<TrackingParticleCollection>("MergedTrackTruth");
    TrackingVertexRefProd   refMergedTVC = event.getRefBeforePut<TrackingVertexCollection>("MergedTrackTruth");

    // Merged Bremsstrahlung and copy the new collection into mergedTPC and mergedTVC
    mergeBremsstrahlung(tPC, tVC, mergedTPC, mergedTVC, refMergedTPC, refMergedTVC);  

    edm::LogInfo(MessageCategory_) << "TrackingTruthProducer found " << tVC -> size()
                                   << " unique vertices and "        << tPC -> size() 
                                   << " tracks in the merged collection.";

    // Put TrackingParticles and TrackingVertices in event
	event.put(mergedTPC,"MergedTrackTruth");
    event.put(mergedTVC,"MergedTrackTruth");
    event.put(tPC);
    event.put(tVC);
  }
  else
  {
    // Put TrackingParticles and TrackingVertices in event
    event.put(tPC);
    event.put(tVC);
  }
  
  //  timers.pop();
  //  timers.pop();
}


void TrackingTruthProducer::simTrackHitsAssociator(
  std::auto_ptr<MixCollection<PSimHit> > & hits
)
{
  simTrack_hit.clear();
  for (MixCollection<PSimHit>::MixItr hit = hits->begin(); hit != hits->end(); ++hit)
  {
    EncodedTruthId simTrackId = EncodedTruthId(hit->eventId(),hit->trackId());
    simTrack_hit.insert(make_pair(simTrackId,*hit));
  }
}


void TrackingTruthProducer::trackingParticleAssambler(
  auto_ptr<TrackingParticleCollection> & tPC,
  auto_ptr<MixCollection<SimTrack> > & tracks,
  Handle<edm::HepMCProduct> const & hepMC
)
{
  simTrack_sourceV.clear();
  simTrack_tP.clear();
  
  const HepMC::GenEvent * genEvent = hepMC->GetEvent();

  for (MixCollection<SimTrack>::MixItr itP = tracks->begin(); itP != tracks->end(); ++itP)
  {
    int                           q = (int)(itP->charge()); // Check this
    const LorentzVector theMomentum = itP->momentum();
    unsigned int         simtrackId = itP->trackId();
    int                     genPart = itP->genpartIndex(); // The HepMC particle number
    int                     genVert = itP->vertIndex();    // The SimVertex #
    int                       pdgId = itP->type();
    int                      status = -99;

    EncodedEventId trackEventId     = itP->eventId();
    EncodedTruthId      trackId     = EncodedTruthId(trackEventId,simtrackId);

    bool signalEvent = (trackEventId.event() == 0 && trackEventId.bunchCrossing() == 0);

    double  time = 0;

    const HepMC::GenParticle * gp = 0;

    if (genPart >= 0 && signalEvent)
    {
      gp = genEvent->barcode_to_particle(genPart);  // Pointer to the generating particle.
      if (gp != 0)
      {
        status = gp->status();
        pdgId  = gp->pdg_id();
      }
    }

    math::XYZPoint theVertex;

    if (genVert >= 0)
    { // Add to useful maps
      EncodedTruthId vertexId = EncodedTruthId(trackEventId,genVert);
      simTrack_sourceV.insert(make_pair(trackId,vertexId));
    }

    TrackingParticle tp(q, theMomentum, theVertex, time, pdgId, status, trackEventId);

    // Counting the TP hits using the layers (as in ORCA).
    // Does seem to find less hits. maybe b/c layer is a number now, not a pointer
    int totsimhit = 0;
    int oldlay = 0;
    int newlay = 0;
    int olddet = 0;
    int newdet = 0;

    // Using simTrack_hit map makes this very fast
    //now check the process ID
    bool checkProc = true;
    unsigned short procType = 0;
    int partType = 0;
    int hitcount = 0;
     
    for (
      multimap<EncodedTruthId,PSimHit>::const_iterator iHit = simTrack_hit.lower_bound(trackId);
	  iHit != simTrack_hit.upper_bound(trackId); 
	  ++iHit
	) 
	{
      PSimHit hit = iHit->second;
      hitcount++;
      
      if(checkProc)
      {
	    procType = hit.processType();
	    partType = hit.particleType();
	    checkProc = false; //check only the procType of the first hit
	    //std::cout << "First Hit (proc, part) = " << procType << ", " << partType << std::endl;
      }

      //Check for delta and interaction products discards
      //std::cout << hitcount << " Hit (proc, part) = " << hit.processType() << ", " << hit.particleType() << std::endl;
      if(procType == hit.processType() && partType == hit.particleType() && pdgId == hit.particleType() )
      {
	    //std::cout << "PASSED" << std::endl;
        tp.addPSimHit(hit);

        unsigned int detid = hit.detUnitId();
        DetId detId = DetId(detid);
        oldlay = newlay;
        olddet = newdet;
        newlay = LayerFromDetid(detid);
        newdet = detId.subdetId();

	    // Count hits using layers for glued detectors

	    if (oldlay != newlay || (oldlay==newlay && olddet!=newdet) )
	    {
	      totsimhit++;
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
    tPC->push_back(tp);
  }
}


void TrackingTruthProducer::trackingVertexAssambler(
  auto_ptr<TrackingParticleCollection> & tPC,
  auto_ptr<TrackingVertexCollection> & tVC,
  auto_ptr<MixCollection<SimTrack> > & tracks,  
  auto_ptr<MixCollection<SimVertex> > & vertexes,
  TrackingParticleRefProd & refTPC,
  TrackingVertexRefProd & refTVC,
  Handle<edm::HepMCProduct> const & hepMC
)
{

  const HepMC::GenEvent * genEvent = hepMC->GetEvent();

  // Find and loop over EmbdSimVertex vertices
  
  int vertexIndex = 0;        // Needed for
  int oldTrigger = -1;        // renumbering
  int oldBX      = -999999;   // of vertices
  
  for (MixCollection<SimVertex>::MixItr itV = vertexes->begin(); itV != vertexes->end(); ++itV)
  {
    // LorentzVector position = itV -> position();  // Get position of ESV
    LorentzVector position(itV->position().x(),itV->position().y(),itV->position().z(),itV->position().t());

    bool inVolume = (position.Pt() < volumeRadius_ && abs(position.z()) < volumeZ_); // In or out of Tracker

    if (!inVolume && discardOutVolume_) { continue; }        // Skip if desired

    EncodedEventId vertEvtId = itV -> eventId();

    // Begin renumbering vertices if we move from signal to pileup or change bunch crossings
    if (oldTrigger !=  itV.getTrigger() || oldBX !=  vertEvtId.bunchCrossing())
    {
      vertexIndex = 0;
      oldTrigger =  itV.getTrigger();
      oldBX =  vertEvtId.bunchCrossing();
    }

    EncodedTruthId vertexId  = EncodedTruthId(vertEvtId,vertexIndex);

    // Figure out the barcode of the HepMC Vertex if there is one by
    // getting incoming SimTtrack (if any), finding corresponding HepMC track and
    // then decay (HepMC) vertex of that track.  HepMC data only exists for signal sub-event

    int vertexBarcode = 0;
    unsigned int vtxParent = itV -> parentIndex();
    if (vtxParent >= 0 && itV.getTrigger() ) 
    {
      for (MixCollection<SimTrack>::MixItr itP = tracks->begin(); itP != tracks->end(); ++itP)
      {
        if (vtxParent==itP->trackId() && itP->eventId() == vertEvtId)
        {
          int parentBC = itP->genpartIndex();
          HepMC::GenParticle *parentParticle = genEvent -> barcode_to_particle(parentBC);
          if (parentParticle != 0) 
          {
            HepMC::GenVertex * hmpv = parentParticle -> end_vertex();
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
    for (TrackingVertexCollection::iterator iTrkVtx = tVC -> begin(); iTrkVtx != tVC ->end(); ++iTrkVtx, ++tmpTV)
    {
      double distance = (iTrkVtx -> position() - position).P();
      if (distance <= closest && vertEvtId == iTrkVtx -> eventId())
      { // flag which one so we can associate them
        closest = distance;
        nearestVertex = iTrkVtx;
        indexTV = tmpTV;
      }
    }

    // If outside cutoff, create another TrackingVertex, set nearestVertex to it
    if (closest > distanceCut_)
    {
      indexTV = tVC -> size();
      tVC -> push_back(TrackingVertex(position,inVolume,vertEvtId));
      nearestVertex = --(tVC -> end());  // Last entry of vector
    }

    // Add data to closest vertex

    (*nearestVertex).addG4Vertex(*itV); // Add G4 vertex
    if (vertexBarcode != 0)
    {
      (*nearestVertex).addGenVertex(GenVertexRef(hepMC,vertexBarcode)); // Add HepMC vertex
    }

    // Identify and add child tracks
    for (std::map<EncodedTruthId,EncodedTruthId>::iterator mapIndex = simTrack_sourceV.begin(); mapIndex != simTrack_sourceV.end(); ++mapIndex)
    {
      EncodedTruthId mapTrackId  = mapIndex -> first;
      EncodedTruthId mapVertexId = mapIndex -> second;
      if (mapVertexId == vertexId)
      {
        if (simTrack_tP.count(mapTrackId))
        {
          int indexTP = simTrack_tP[mapTrackId];
          (*nearestVertex).addDaughterTrack(TrackingParticleRef(refTPC,indexTP));
          (tPC->at(indexTP)).setParentVertex(TrackingVertexRef(refTVC,indexTV));
          const LorentzVector  &v = (*nearestVertex).position();

          math::XYZPoint xyz = math::XYZPoint(v.x(), v.y(), v.z());
          double t = v.t();
          (tPC->at(indexTP)).setVertex(xyz,t);
        }
      }
    }

    // Identify and add parent tracks
    if (vtxParent > 0) 
    {
      EncodedTruthId trackId =  EncodedTruthId(vertEvtId,vtxParent);
      if (simTrack_tP.count(trackId) > 0) 
      {
        int indexTP = simTrack_tP[trackId];
        (tPC->at(indexTP)).addDecayVertex(TrackingVertexRef(refTVC,indexTV));
        (*nearestVertex).addParentTrack(TrackingParticleRef(refTPC,indexTP));
      }
    }
    ++vertexIndex;
  } // Loop on MixCollection<SimVertex>

  // Find HepMC vertices, put them in a close TrackingVertex (this could conceivably add the same GenVertex to multiple TrackingVertices)
  for (HepMC::GenEvent::vertex_const_iterator genVIt = genEvent->vertices_begin(); genVIt != genEvent->vertices_end(); ++genVIt) 
  {
    HepMC::ThreeVector rawPos = (**genVIt).position();
    // Convert to cm
    math::XYZPoint genPos = math::XYZPoint(rawPos.x()/10.0,rawPos.y()/10.0,rawPos.z()/10.0);
    for (TrackingVertexCollection::iterator iTrkVtx = tVC -> begin(); iTrkVtx != tVC ->end(); ++iTrkVtx)
    {
      rawPos = iTrkVtx->position();
      math::XYZPoint simPos = math::XYZPoint(rawPos.x(),rawPos.y(),rawPos.z());
      double distance = sqrt((simPos-genPos).mag2());
      if (distance <= distanceCut_)
      {
        TrackingVertex::genv_iterator tvGenVIt;
        for (tvGenVIt = iTrkVtx->genVertices_begin(); tvGenVIt != iTrkVtx->genVertices_end(); ++tvGenVIt)
        {
          if ((**genVIt).barcode()  == (**tvGenVIt).barcode())
          {
            break;
          }
        }
        if (tvGenVIt== iTrkVtx->genVertices_end() )
        {
          iTrkVtx->addGenVertex(GenVertexRef(hepMC,(**genVIt).barcode())); // Add HepMC vertex
        }
      }
    }
  }
}



void TrackingTruthProducer::mergeBremsstrahlung(
  auto_ptr<TrackingParticleCollection> & tPC,
  auto_ptr<TrackingVertexCollection>   & tVC,
  auto_ptr<TrackingParticleCollection> & mergedTPC,
  auto_ptr<TrackingVertexCollection>   & mergedTVC,
  TrackingParticleRefProd & refMergedTPC,
  TrackingVertexRefProd & refMergedTVC
)
{     
  std::set<uint> excludedTV, excludedTP;
  
  uint index = 0;
  
  // Merge Bremsstrahlung vertexes
  for (TrackingVertexCollection::iterator iVC = tVC->begin(); iVC != tVC->end(); ++iVC, ++index)
  {
  	// Check Bremsstrahlung vertex
  	if ( isBremsstrahlungVertex(*iVC, tPC) )
  	{
  	  // Get a pointer to the source track (A Ref<> cannot be use with a product!)	
  	  TrackingParticle * track = &tPC->at(iVC->sourceTracks_begin()->key());
  	  // Get a Ref<> to the source track
  	  TrackingParticleRef trackRef = *iVC->sourceTracks_begin();

      // Pointer to electron daughter
      TrackingParticle * daughter = 0;
      // Ref<> to electron daughter
      TrackingParticleRef daughterRef;

      // Select the electron daughter and veto the photon
      for (TrackingVertex::tp_iterator idaughter = iVC->daughterTracks_begin(); idaughter != iVC->daughterTracks_end(); ++idaughter)
      {
      	TrackingParticle * pointer = &tPC->at(idaughter->key());
        if ( abs( pointer->pdgId() ) == 11 )
        {
          // Set pointer to the electron daughter
      	  daughter = pointer;
      	  // Set Ref<> to the electron daughter
      	  daughterRef = *idaughter;
        }
        else if ( pointer->pdgId() == 22 )
        {
          // Remove reference to the voted photon	
          for ( TrackingParticle::tv_iterator idecay = pointer->decayVertices_begin(); idecay != pointer->decayVertices_end(); ++idecay )
          {
            // Get a reference to decay vertex
            TrackingVertex * vertex = &tVC->at( idecay->key() );
            // Copy all the source tracks from of the decay vertex 
            TrackingParticleRefVector sources( vertex->sourceTracks() );    
            // Clear the source track references
            vertex->clearParentTracks();
            // Add the new source tracks by excluding the one with the segment merged 
            for(TrackingVertex::tp_iterator isource = sources.begin(); isource != sources.end(); ++isource)
              if (*isource != *idaughter)
                vertex->addParentTrack(*isource);
          }
          excludedTP.insert( idaughter->key() );          
        }
      }

      // Add the electron segments from the electron daughter
      for (TrackingParticle::g4t_iterator isegment = daughter->g4Track_begin(); isegment != daughter->g4Track_end(); ++isegment)
        track->addG4Track(*isegment);
      
      // Copy all the simhits to the new track  
      for (std::vector<PSimHit>::const_iterator ihit = daughter->pSimHit_begin(); ihit != daughter->pSimHit_end(); ++ihit)
        track->addPSimHit(*ihit);

      // Clear the decay vertex list 	  
      track->clearDecayVertices();

      // Redirect all the decay source vertexes to those in the electron daughter  
      for (TrackingParticle::tv_iterator idecay = daughter->decayVertices_begin(); idecay != daughter->decayVertices_end(); ++idecay)
      { 
      	// Add the vertexes to the decay list of the source particles
        track->addDecayVertex(*idecay);
        // Get a reference to decay vertex
        TrackingVertex * vertex = &tVC->at( idecay->key() );
        // Copy all the source tracks from of the decay vertex 
        TrackingParticleRefVector sources( vertex->sourceTracks() );
        // Clear the source track references
        vertex->clearParentTracks();
        // Add the new source tracks by excluding the one with the segment merged 
        for(TrackingVertex::tp_iterator isource = sources.begin(); isource != sources.end(); ++isource)
          if (*isource != daughterRef) 
            vertex->addParentTrack(*isource);
        // Add the track reference to the list of sources 
        vertex->addParentTrack(trackRef);
      } 
              
      // Adding the vertex to the exlusion list
      excludedTV.insert(index);
            
      // Adding the electron segment tp into the exlusion list
      excludedTP.insert( daughterRef.key() );
  	}
  }   	

  std::cout << "Generating the merged collection." << std::endl;
	
  // Reserved the same amount of memory for the merged collections	
  mergedTPC->reserve(tPC->size());
  mergedTVC->reserve(tVC->size());

  index = 0;
  map<uint, uint> vertexMap;

  // Copy non-excluded vertices discarding parent & child tracks 
  for (TrackingVertexCollection::const_iterator iVC = tVC->begin(); iVC != tVC->end(); ++iVC, ++index)
  {
  	if ( excludedTV.find(index) != excludedTV.end() ) continue;
    // Save the new location of the non excluded vertexes (take in consideration those were removed) 
    vertexMap.insert( make_pair(index, mergedTVC->size()) );
    // Copy those vertexes are not excluded
    TrackingVertex newVertex = (*iVC);
    newVertex.clearDaughterTracks();
    newVertex.clearParentTracks();
    mergedTVC->push_back(newVertex);
  }
  
  index = 0;
  
  // Copy and cross reference the non-excluded tp to the merged collection
  for (TrackingParticleCollection::const_iterator iTP = tPC->begin(); iTP != tPC->end(); ++iTP, ++index)
  {
  	if ( excludedTP.find(index) != excludedTP.end() ) continue;
    
    TrackingVertexRef       sourceV = iTP->parentVertex();
    TrackingVertexRefVector decayVs = iTP->decayVertices();
    TrackingParticle newTrack = *iTP;
 
    newTrack.clearParentVertex();
    newTrack.clearDecayVertices();

    // Set vertex indices for new vertex product and track references in those vertices
    
    // Index of parent vertex in vertex container
    uint parentIndex = vertexMap[sourceV.key()];
    // Index of this track in track container
    uint tIndex      = mergedTPC->size();

    // Add vertex to track
    newTrack.setParentVertex(TrackingVertexRef(refMergedTVC,parentIndex));
    // Add track to vertex
    (mergedTVC->at(parentIndex)).addDaughterTrack(TrackingParticleRef(refMergedTPC,tIndex));
    
    for (TrackingVertexRefVector::const_iterator iDecayV = decayVs.begin(); iDecayV != decayVs.end(); ++iDecayV)
    {
       // Index of decay vertex in vertex container
      uint daughterIndex = vertexMap[iDecayV->key()];
      // Add vertex to track
      newTrack.addDecayVertex(TrackingVertexRef(refMergedTVC,daughterIndex));
      // Add track to vertex
      (mergedTVC->at(daughterIndex)).addParentTrack(TrackingParticleRef(refMergedTPC,tIndex));
    }
    
    mergedTPC->push_back(newTrack);
  }
}
 

bool TrackingTruthProducer::isBremsstrahlungVertex(
  TrackingVertex const & vertex,
  auto_ptr<TrackingParticleCollection> & tPC
)
{
  const TrackingParticleRefVector parents(vertex.sourceTracks());
      
  // Check for the basic parent conditions
  if ( parents.size() != 1)
    return false;

  // Check for the parent particle is a |electron| (electron or positron)
  if ( abs( tPC->at(parents.begin()->key()).pdgId() ) != 11 ) 
    return false;
    
  int nElectrons = 0;
  int nPhotons = 0;
  int nOthers = 0;  

  // Loop over the daughter particles and counts the number of |electrons|, photons or others        
  for ( TrackingVertex::tp_iterator it = vertex.daughterTracks_begin(); it != vertex.daughterTracks_end(); ++it )
  {
    if ( abs( tPC->at(it->key()).pdgId() ) == 11 )
      nElectrons++;
    else if ( tPC->at(it->key()).pdgId() == 22 )
      nPhotons++;
    else
      nOthers++;
  }

  // Condition to be a Bremsstrahlung Vertex
  if (nElectrons == 1 && nPhotons == 1 && nOthers == 0) 
    return true;
  
  return false;   
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

//DEFINE_FWK_MODULE(TrackingTruthProducer);
