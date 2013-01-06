
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

typedef edm::Ref<edm::HepMCProduct, HepMC::GenParticle > GenParticleRef;
typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >   GenVertexRef;
typedef math::XYZTLorentzVectorD    LorentzVector;
typedef math::XYZPoint Vector;

TrackingTruthProducer::TrackingTruthProducer(const edm::ParameterSet & config) :
        pSimHitSelector_(config), pixelPSimHitSelector_(config), trackerPSimHitSelector_(config), muonPSimHitSelector_(config)
{
    // Initialize global parameters
    dataLabels_             = config.getParameter<std::vector<std::string> >("HepMCDataLabels");
    useMultipleHepMCLabels_ = config.getParameter<bool>("useMultipleHepMCLabels");

    distanceCut_            = config.getParameter<double>("vertexDistanceCut");
    volumeRadius_           = config.getParameter<double>("volumeRadius");
    volumeZ_                = config.getParameter<double>("volumeZ");
    mergedBremsstrahlung_   = config.getParameter<bool>("mergedBremsstrahlung");
    removeDeadModules_      = config.getParameter<bool>("removeDeadModules");
    simHitLabel_            = config.getParameter<std::string>("simHitLabel");

    // Initialize selection for building TrackingParticles
    if ( config.exists("select") )
    {
        edm::ParameterSet param = config.getParameter<edm::ParameterSet>("select");
        selector_ = TrackingParticleSelector(
                        param.getParameter<double>("ptMinTP"),
                        param.getParameter<double>("minRapidityTP"),
                        param.getParameter<double>("maxRapidityTP"),
                        param.getParameter<double>("tipTP"),
                        param.getParameter<double>("lipTP"),
                        param.getParameter<int>("minHitTP"),
                        param.getParameter<bool>("signalOnlyTP"),
                        param.getParameter<bool>("chargedOnlyTP"),
                        param.getParameter<bool>("stableOnlyTP"),
                        param.getParameter<std::vector<int> >("pdgIdTP")
                    );
        selectorFlag_ = true;
    }
    else
        selectorFlag_ = false;

    MessageCategory_       = "TrackingTruthProducer";

    edm::LogInfo (MessageCategory_) << "Setting up TrackingTruthProducer";
    edm::LogInfo (MessageCategory_) << "Vertex distance cut set to " << distanceCut_  << " mm";
    edm::LogInfo (MessageCategory_) << "Volume radius set to "       << volumeRadius_ << " mm";
    edm::LogInfo (MessageCategory_) << "Volume Z      set to "       << volumeZ_      << " mm";

    if (useMultipleHepMCLabels_) edm::LogInfo (MessageCategory_) << "Collecting generator information from pileup.";
    if (mergedBremsstrahlung_) edm::LogInfo (MessageCategory_) << "Merging electrom bremsstralung";
    if (removeDeadModules_) edm::LogInfo (MessageCategory_) << "Removing psimhit from dead modules";

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
    m_trackingVertexBinMins[ 0 ] = 0. ;
    m_trackingVertexBinMins[ 1 ] = 100. ;
    m_trackingVertexBinMins[ 2 ] = 192. ;
    m_trackingVertexBinMins[ 3 ] = 196. ;
    m_trackingVertexBinMins[ 4 ] = 198. ;
    m_trackingVertexBinMins[ 5 ] = 200. ;
    m_trackingVertexBinMins[ 6 ] = 203. ;
    m_trackingVertexBinMins[ 7 ] = 207. ;
    m_trackingVertexBinMins[ 8 ] = 220. ;
    m_trackingVertexBinMins[ 9 ] = 275. ;
}


void TrackingTruthProducer::produce(edm::Event &event, const edm::EventSetup & setup)
{
  edm::ESHandle<TrackerTopology> tTopoHand;
  setup.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

    // Clean the list of hepmc products
    hepMCProducts_.clear();

    // Collect all the HepMCProducts
    edm::Handle<edm::HepMCProduct> hepMCHandle;

    for (std::vector<std::string>::const_iterator source = dataLabels_.begin(); source != dataLabels_.end(); ++source)
    {
        if ( event.getByLabel(*source, hepMCHandle) )
        {
            hepMCProducts_.push_back(hepMCHandle);
            edm::LogInfo (MessageCategory_) << "Using HepMC source " << *source;
            if (!useMultipleHepMCLabels_) break;
        }
    }

    if ( hepMCProducts_.empty() )
    {
        edm::LogWarning (MessageCategory_) << "No HepMC source found";
        return;
    }
    else if (hepMCProducts_.size() > 1 || useMultipleHepMCLabels_)
    {
        edm::LogInfo (MessageCategory_) << "You are using more than one HepMC source.";
        edm::LogInfo (MessageCategory_) << "If the labels are not in the same order as the events in the crossing frame (i.e. signal, pileup(s) ) ";
        edm::LogInfo (MessageCategory_) << "or there are fewer labels than events in the crossing frame";
        edm::LogInfo (MessageCategory_) << MessageCategory_ << " may try to access data in the wrong HepMCProduct and crash.";
    }

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

    // Create a list of psimhits
    if (removeDeadModules_)
    {
        pSimHits_.clear();
        pixelPSimHitSelector_.select(pSimHits_, event, setup);
        trackerPSimHitSelector_.select(pSimHits_, event, setup);
        muonPSimHitSelector_.select(pSimHits_, event, setup);
    }
    else
    {
        pSimHits_.clear();
        pSimHitSelector_.select(pSimHits_, event, setup);
    }

    // Create a multimap between trackId and hit indices
    associator(pSimHits_, trackIdToHits_);

    // Create a map between trackId and track index
    associator(simTracks_, trackIdToIndex_);

    // Create a map between vertexId and vertex index
    associator(simVertexes_, vertexIdToIndex_);

    createTrackingTruth(tTopo);

    if (mergedBremsstrahlung_)
    {
        // Create collections of things we will put in event,
        mergedTrackingParticles_ = std::auto_ptr<TrackingParticleCollection>( new TrackingParticleCollection );
        mergedTrackingVertexes_ = std::auto_ptr<TrackingVertexCollection>( new TrackingVertexCollection );

        // Get references before put so we can cross reference
        refMergedTrackingParticles_ = event.getRefBeforePut<TrackingParticleCollection>("MergedTrackTruth");
        refMergedTrackingVertexes_ = event.getRefBeforePut<TrackingVertexCollection>("MergedTrackTruth");

        // Merged brem electrons
        mergeBremsstrahlung();

        // Put TrackingParticles and TrackingVertices in event
        event.put(mergedTrackingParticles_, "MergedTrackTruth");
        event.put(mergedTrackingVertexes_, "MergedTrackTruth");
        event.put(trackingParticles_);
        event.put(trackingVertexes_);
    }
    else
    {
        // Put TrackingParticles and TrackingVertices in event
        event.put(trackingParticles_);
        event.put(trackingVertexes_);
    }
}


void TrackingTruthProducer::associator(
    std::vector<PSimHit> const & pSimHits,
    EncodedTruthIdToIndexes & association
)
{
    // Clear the association map
    association.clear();

    // Create a association from simtracks to overall index in the mix collection
    for (std::size_t i = 0; i < pSimHits.size(); ++i)
    {
        EncodedTruthIdToIndexes::key_type objectId = EncodedTruthIdToIndexes::key_type(pSimHits[i].eventId(), pSimHits[i].trackId());
        association.insert( std::make_pair(objectId, i) );
    }
}


void TrackingTruthProducer::associator(
    std::auto_ptr<MixCollection<SimTrack> > const & mixCollection,
    EncodedTruthIdToIndex & association
)
{
    int index = 0;
    // Clear the association map
    association.clear();
    // Create a association from simtracks to overall index in the mix collection
    for (MixCollection<SimTrack>::MixItr iterator = mixCollection->begin(); iterator != mixCollection->end(); ++iterator, ++index)
    {
        EncodedTruthId objectId = EncodedTruthId(iterator->eventId(), iterator->trackId());
        association.insert( std::make_pair(objectId, index) );
    }
}


void TrackingTruthProducer::associator(
    std::auto_ptr<MixCollection<SimVertex> > const & mixCollection,
    EncodedTruthIdToIndex & association
)
{
    int index = 0;

    // Solution to the problem of not having vertexId
    bool useVertexId = true;
    EncodedEventIdToIndex vertexId;
    EncodedEventId oldEventId;
    unsigned int oldVertexId = 0;

    // Loop for finding repeated vertexId (vertexId problem hack)
    for (MixCollection<SimVertex>::MixItr iterator = mixCollection->begin(); iterator != mixCollection->end(); ++iterator, ++index)
    {
        if (!index || iterator->eventId() != oldEventId)
        {
            oldEventId = iterator->eventId();
            oldVertexId = iterator->vertexId();
            continue;
        }

        if ( iterator->vertexId() == oldVertexId )
        {
            edm::LogWarning(MessageCategory_) << "Multiple vertexId found, no using vertexId.";
            useVertexId = false;
            break;
        }
    }

    // Reset the index
    index = 0;

    // Clear the association map
    association.clear();

    // Create a association from simvertexes to overall index in the mix collection
    for (MixCollection<SimVertex>::MixItr iterator = mixCollection->begin(); iterator != mixCollection->end(); ++iterator, ++index)
    {
        EncodedTruthId objectId;
        if (useVertexId)
            objectId = EncodedTruthId(iterator->eventId(), iterator->vertexId());
        else
            objectId = EncodedTruthId(iterator->eventId(), vertexId[iterator->eventId()]++);
        association.insert( std::make_pair(objectId, index) );
    }
}


void TrackingTruthProducer::mergeBremsstrahlung()
{
    unsigned int index = 0;

    std::set<unsigned int> excludedTV, excludedTP;

    // Merge Bremsstrahlung vertexes
    for (TrackingVertexCollection::iterator iVC = trackingVertexes_->begin(); iVC != trackingVertexes_->end(); ++iVC, ++index)
    {
        // Check Bremsstrahlung vertex
        if ( isBremsstrahlungVertex(*iVC, trackingParticles_) )
        {
            // Get a pointer to the source track (A Ref<> cannot be use with a product!)
            TrackingParticle * track = &trackingParticles_->at(iVC->sourceTracks_begin()->key());
            // Get a Ref<> to the source track
            TrackingParticleRef trackRef = *iVC->sourceTracks_begin();
            // Pointer to electron daughter
            TrackingParticle * daughter = 0;
            // Ref<> to electron daughter
            TrackingParticleRef daughterRef;

            // Select the electron daughter and redirect the photon
            for (TrackingVertex::tp_iterator idaughter = iVC->daughterTracks_begin(); idaughter != iVC->daughterTracks_end(); ++idaughter)
            {
                TrackingParticle * pointer = &trackingParticles_->at(idaughter->key());
                if ( std::abs( pointer->pdgId() ) == 11 )
                {
                    // Set pointer to the electron daughter
                    daughter = pointer;
                    // Set Ref<> to the electron daughter
                    daughterRef = *idaughter;
                }
                else if ( pointer->pdgId() == 22 )
                {
                    // Delete the photon original parent vertex
                    pointer->clearParentVertex();
                    // Set the new parent vertex to the vertex of the source track
                    pointer->setParentVertex( track->parentVertex() );
                    // Get a non-const pointer to the parent vertex
                    TrackingVertex * vertex = &trackingVertexes_->at( track->parentVertex().key() );
                    // Add the photon to the doughter list of the parent vertex
                    vertex->addDaughterTrack( *idaughter );
                }
            }

            // Add the electron segments from the electron daughter
            // track must not be the same particle as daughter
            // if (track != daughter)
            for (TrackingParticle::g4t_iterator isegment = daughter->g4Track_begin(); isegment != daughter->g4Track_end(); ++isegment)
                track->addG4Track(*isegment);

            // Copy all the simhits to the new track
            for (std::vector<PSimHit>::const_iterator ihit = daughter->pSimHit_begin(); ihit != daughter->pSimHit_end(); ++ihit)
                track->addPSimHit(*ihit);

            // Make a copy of the decay vertexes of the track
            TrackingVertexRefVector decayVertices( track->decayVertices() );

            // Clear the decay vertex list
            track->clearDecayVertices();

            // Add the remaining vertexes
            for (TrackingVertexRefVector::const_iterator idecay = decayVertices.begin(); idecay != decayVertices.end(); ++idecay)
                if ( (*idecay).key() != index ) track->addDecayVertex(*idecay);

            // Redirect all the decay source vertexes to those in the electron daughter
            for (TrackingParticle::tv_iterator idecay = daughter->decayVertices_begin(); idecay != daughter->decayVertices_end(); ++idecay)
            {
                // Add the vertexes to the decay list of the source particles
                track->addDecayVertex(*idecay);
                // Get a reference to decay vertex
                TrackingVertex * vertex = &trackingVertexes_->at( idecay->key() );
                // Copy all the source tracks from of the decay vertex
                TrackingParticleRefVector sources( vertex->sourceTracks() );
                // Clear the source track references
                vertex->clearParentTracks();
                // Add the new source tracks by excluding the one with the segment merged
                for (TrackingVertex::tp_iterator isource = sources.begin(); isource != sources.end(); ++isource)
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

    edm::LogInfo(MessageCategory_) << "Generating the merged collection." << std::endl;

    // Reserved the same amount of memory for the merged collections
    mergedTrackingParticles_->reserve(trackingParticles_->size());
    mergedTrackingVertexes_->reserve(trackingVertexes_->size());

    index = 0;
    std::map<unsigned int, unsigned int> vertexMap;

    // Copy non-excluded vertices discarding parent & child tracks
    for (TrackingVertexCollection::const_iterator iVC = trackingVertexes_->begin(); iVC != trackingVertexes_->end(); ++iVC, ++index)
    {
        if ( excludedTV.find(index) != excludedTV.end() ) continue;
        // Save the new location of the non excluded vertexes (take in consideration those were removed)
        vertexMap.insert( std::make_pair(index, mergedTrackingVertexes_->size()) );
        // Copy those vertexes are not excluded
        TrackingVertex newVertex = (*iVC);
        newVertex.clearDaughterTracks();
        newVertex.clearParentTracks();
        mergedTrackingVertexes_->push_back(newVertex);
    }

    index = 0;

    // Copy and cross reference the non-excluded tp to the merged collection
    for (TrackingParticleCollection::const_iterator iTP = trackingParticles_->begin(); iTP != trackingParticles_->end(); ++iTP, ++index)
    {
        if ( excludedTP.find(index) != excludedTP.end() ) continue;

        TrackingVertexRef       sourceV = iTP->parentVertex();
        TrackingVertexRefVector decayVs = iTP->decayVertices();
        TrackingParticle newTrack = *iTP;

        newTrack.clearParentVertex();
        newTrack.clearDecayVertices();

        // Set vertex indices for new vertex product and track references in those vertices

        // Index of parent vertex in vertex container
        unsigned int parentIndex = vertexMap[sourceV.key()];
        // Index of this track in track container
        unsigned int tIndex = mergedTrackingParticles_->size();

        // Add vertex to track
        newTrack.setParentVertex( TrackingVertexRef(refMergedTrackingVertexes_, parentIndex) );
        // Add track to vertex
        (mergedTrackingVertexes_->at(parentIndex)).addDaughterTrack(TrackingParticleRef(refMergedTrackingParticles_, tIndex));

        for (TrackingVertexRefVector::const_iterator iDecayV = decayVs.begin(); iDecayV != decayVs.end(); ++iDecayV)
        {
            // Index of decay vertex in vertex container
            unsigned int daughterIndex = vertexMap[iDecayV->key()];
            // Add vertex to track
            newTrack.addDecayVertex( TrackingVertexRef(refMergedTrackingVertexes_, daughterIndex) );
            // Add track to vertex
            (mergedTrackingVertexes_->at(daughterIndex)).addParentTrack( TrackingParticleRef(refMergedTrackingParticles_, tIndex) );
        }

        mergedTrackingParticles_->push_back(newTrack);
    }
}


bool TrackingTruthProducer::isBremsstrahlungVertex(
    TrackingVertex const & vertex,
    std::auto_ptr<TrackingParticleCollection> & tPC
)
{
    const TrackingParticleRefVector parents(vertex.sourceTracks());

    // Check for the basic parent conditions
    if ( parents.size() != 1)
        return false;

    // Check for the parent particle is a |electron| (electron or positron)
    if ( std::abs( tPC->at(parents.begin()->key()).pdgId() ) != 11 )
        return false;

    unsigned int nElectrons = 0;
    unsigned int nOthers = 0;

    // Loop over the daughter particles and counts the number of |electrons|, others (non photons)
    for ( TrackingVertex::tp_iterator it = vertex.daughterTracks_begin(); it != vertex.daughterTracks_end(); ++it )
    {
        // Stronger rejection for looping particles
        if ( parents[0] == *it )
            return false;

        if ( std::abs( tPC->at(it->key()).pdgId() ) == 11 )
            nElectrons++;
        else if ( tPC->at(it->key()).pdgId() != 22 )
            nOthers++;
    }

    // Condition to be a Bremsstrahlung Vertex
    if (nElectrons == 1 && nOthers == 0)
        return true;

    return false;
}


void TrackingTruthProducer::createTrackingTruth(const TrackerTopology *tTopo)
{
    // Reset the event counter (use for define vertexId)
    eventIdCounter_.clear();

    // Define a container of vetoed traks
    std::map<int,std::size_t> vetoedTracks;

    // Define map between parent simtrack and tv indexes
    std::map<int,std::size_t> vetoedSimVertexes;

    //std::cout << "NUMBER SIMTRACKS " << simTracks_->size() << std::endl ;
    int setCount = 0 ;
    m_vertexCounter = 0 ;
    m_noMatchVertexCounter = 0 ;

    // Clear vertex bins
    for( int i = 0 ; i < 10 ; ++i )
      {
        m_trackingVertexBins[ i ].clear() ;
      }

    for (int simTrackIndex = 0; simTrackIndex != simTracks_->size(); ++simTrackIndex)
    {
        // Check if the simTrack is excluded (includes non traceable and recovered by history)
        if ( vetoedTracks.find(simTrackIndex) != vetoedTracks.end() ) continue;

        SimTrack const & simTrack = simTracks_->getObject(simTrackIndex);

        TrackingParticle trackingParticle;

        // Set a bare tp (only with the psimhit) with a given simtrack
        // the function return true if it is tracable
        if ( setTrackingParticle(simTrack, trackingParticle, tTopo) )
        {
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
		  setTrackingParticle(*currentSimTrack, trackingParticle,tTopo);

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
                }
                else
                {
                    // Set the tp index to its new value
                    trackingParticleIndex = trackingParticles_->size();
                    // Push the tp in to the collection
                    trackingParticles_->push_back(trackingParticle);
                    // Vetoed the simTrack
                    vetoedTracks.insert( std::make_pair(simTrackIndex, trackingParticleIndex) );
                }

                // Verify if the parent simVertex has a simTrack or if the source is a vetoSimVertex
                if (currentSimTrack->noVertex()) break;

                // Get the simTrack parent index (it is implicit should be in the same event as current)
                unsigned int parentSimVertexIndex = vertexIdToIndex_[
                                                        EncodedTruthId(
                                                            currentSimTrack->eventId(),
                                                            currentSimTrack->vertIndex()
                                                        )
                                                    ];
                // Create a new tv
                TrackingVertex trackingVertex;
                // Get the parent simVertex associated to the current simTrack
                SimVertex const * parentSimVertex = & simVertexes_->getObject(parentSimVertexIndex);

                bool vetoSimVertex = vetoedSimVertexes.find(parentSimVertexIndex) != vetoedSimVertexes.end();

                // Check for a already visited parent simTrack
                if ( !vetoSimVertex )
                {
                    // Set the tv by using simvertex
		    ++setCount ;
                    trackingVertexIndex = setTrackingVertex(*parentSimVertex, trackingVertex);

                    // Check if a new vertex needs to be created
                    if (trackingVertexIndex < 0)
                    {
                        // Set the tv index ot its new value
                        trackingVertexIndex = trackingVertexes_->size();
                        // Push the new tv in to the collection
                        trackingVertexes_->push_back(trackingVertex);

                        // Find the distance bin for this vertex
                        double distance = trackingVertex.position().P() ;
                        for( int i = 9 ; i >= 0 ; --i )
                          {
                            if( distance >= m_trackingVertexBinMins[ i ] )
                              {
                                m_trackingVertexBins[ i ].push_back( trackingVertexIndex ) ;
                                break ;
                              }
                          }

                    }
                    else
                    {
                        // Get the postion and time of the vertex
                        const LorentzVector & position = trackingVertexes_->at(trackingVertexIndex).position();
                        Vector xyz = Vector(position.x(), position.y(), position.z());
                        double t = position.t();
                        // Set the vertex postion of the tp to the closest vertex
                        trackingParticles_->at(trackingParticleIndex).setVertex(xyz, t);
                    }

                    vetoedSimVertexes.insert( std::make_pair(parentSimVertexIndex, trackingVertexIndex) );
                }
                else
                    trackingVertexIndex = vetoedSimVertexes[parentSimVertexIndex];

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
                if ( vetoedTracks.find(nextSimTrackIndex) != vetoedTracks.end() )
                {
                    // Add to the newly created tv the existent next simtrack in to parent list.
                    trackingVertexes_->at(trackingVertexIndex).addParentTrack(
                        TrackingParticleRef(refTrackingParticles_, vetoedTracks[nextSimTrackIndex])
                    );
                    // Add the vertex to list of decay vertexes of the new tp
                    trackingParticles_->at(vetoedTracks[nextSimTrackIndex]).addDecayVertex(
                        TrackingVertexRef(refTrackingVertexes_, trackingVertexIndex)
                    );
                    break;
                }

                // Vetoed the next simTrack
                vetoedTracks.insert( std::make_pair(nextSimTrackIndex, trackingParticleIndex) );

                // Set the current simTrack as the next simTrack
                currentSimTrack = & simTracks_->getObject(nextSimTrackIndex);
            }
            while (!currentSimTrack->noVertex());
        }
    }

    //std::cout << "NUMBER CALLS SETTRACKINGVERTEX " << setCount << std::endl ;
    //std::cout << "FINAL NUMBER VERTEXES " << trackingVertexes_->size() << std::endl ;
    //std::cout << "NUMBER VERTEX ITERATIONS " << m_vertexCounter << std::endl ;
    //std::cout << "NUMBER VERTEX NO MATCH " << m_noMatchVertexCounter << std::endl ;

    //for( int i = 0 ; i < 10 ; ++i )
    //  {
    //    std::cout << "NUMBER VERTEXES BIN " << i << " = " << m_trackingVertexBins[ i ].size() << std::endl ;
    //  }
}


bool TrackingTruthProducer::setTrackingParticle(
    SimTrack const & simTrack,
    TrackingParticle & trackingParticle,
    const TrackerTopology *tTopo
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

    // In the case of a existing generated particle and track
    // event is signal redefine status a pdgId

    edm::Handle<edm::HepMCProduct> hepmc;

    if (genParticleIndex >= 0 && (signalEvent || useMultipleHepMCLabels_) )
    {
        // Get the generated particle
        hepmc = (useMultipleHepMCLabels_) ? hepMCProducts_.at(trackEventId.rawId()) : hepmc = hepMCProducts_.at(0);

        const HepMC::GenParticle * genParticle = hepmc->GetEvent()->barcode_to_particle(genParticleIndex);

        if (genParticle)
        {
            status = genParticle->status();
            pdgId  = genParticle->pdg_id();
        }
    }

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

    // Counting the TP hits using the layers (as in ORCA).
    // Does seem to find less hits. maybe b/c layer is a number now, not a pointer
    int totalSimHits = 0;
    int oldLayer = 0;
    int newLayer = 0;
    int oldDetector = 0;
    int newDetector = 0;

    // Loop over the associated hits per track
    for (
        EncodedTruthIdToIndexes::const_iterator iEntry = trackIdToHits_.lower_bound(simTrackId);
        iEntry != trackIdToHits_.upper_bound(simTrackId);
        ++iEntry
    )
    {
        // Get a constant reference to the simhit
        PSimHit const & pSimHit = pSimHits_.at(iEntry->second);

        // Initial condition for consistent simhit selection
        if (init)
        {
            processType = pSimHit.processType();
            particleType = pSimHit.particleType();
            init = false;
        }

        // Check for delta and interaction products discards
        if (processType == pSimHit.processType() && particleType == pSimHit.particleType() && pdgId == pSimHit.particleType() )
        {
            trackingParticle.addPSimHit(pSimHit);

            unsigned int detectorIdIndex = pSimHit.detUnitId();
            DetId detectorId = DetId(detectorIdIndex);
            oldLayer = newLayer;
            oldDetector = newDetector;
            newLayer = 0;
	    if ( detectorId.det() == DetId::Tracker ) newLayer=tTopo->layer(detectorId);

            newDetector = detectorId.subdetId();

            // Count hits using layers for glued detectors
            // newlayer !=0 excludes Muon layers set to 0 by LayerFromDetid
            if ( ( oldLayer != newLayer || (oldLayer==newLayer && oldDetector!=newDetector ) ) && newLayer != 0 ) totalSimHits++;
        }
    }

    // Set the number of matched simhits
    trackingParticle.setMatchedHit(totalSimHits);

    // Add the simtrack associated to the tp
    trackingParticle.addG4Track(simTrack);

    // Add the generator information
    if ( genParticleIndex >= 0 && (signalEvent || useMultipleHepMCLabels_) )
        trackingParticle.addGenParticle( GenParticleRef(hepmc, genParticleIndex) );

    if (selectorFlag_) return selector_(trackingParticle);

    return true;
}


int TrackingTruthProducer::setTrackingVertex(
    SimVertex const & simVertex,
    TrackingVertex & trackingVertex
)
{
    LorentzVector const & position = simVertex.position();

    // Find tracking vertex bin
    double simDist = position.P() ;
    int bin = -1 ;
    for( int i = 9 ; i >= 0 ; --i )
      {
        if( simDist >= m_trackingVertexBinMins[ i ] )
          {
            bin = i ;
            break ;
          }
      }

    if( bin > -1 )
      {
        // Look for close by vertexes in this bin, starting at end
        for( std::size_t ivtx = m_trackingVertexBins[ bin ].size() ; ivtx > 0 ; --ivtx )
          {
            std::size_t trackingVertexIndex = m_trackingVertexBins[ bin ].at( ivtx-1 ) ;
            ++m_vertexCounter ;

            // Calculate the distance
            double distance = (position - trackingVertexes_->at(trackingVertexIndex).position()).P();
            // If the distance is under a given cut return the trackingVertex index (vertex merging)
            if (distance <= distanceCut_)
              {
                // Add simvertex to the pre existent tv
                trackingVertexes_->at(trackingVertexIndex).addG4Vertex(simVertex);

                // std::cout << "MATCHED VERTEX " << trackingVertexIndex-1 << " OF " << trackingVertexes_->size() << ", POSITION (" << position.x() << ", " << position.y() << ", " << position.z() << ") r = " << position.P() << std::endl ;

                // return tv index
                return trackingVertexIndex;
              }
          }
    
        // If no match found, check if we are close to a neighboring bin
        int nbin = -1 ;
        if( bin != 0 && fabs( simDist - m_trackingVertexBinMins[ bin ] ) <= distanceCut_ )
          {
            nbin = bin - 1 ;
          }
        else if( bin != 9 && fabs( simDist - m_trackingVertexBinMins[ bin + 1 ] ) <= distanceCut_ )
          {
            nbin = bin + 1 ;
          }

        if( nbin > -1 )
          {
            // std::cout << "CHECKING NEIGHBORING BIN " << bin << " -> " << nbin << std::endl ;

            // Look for close by vertexes in this bin, starting at end
            for( std::size_t ivtx = m_trackingVertexBins[ nbin ].size() ; ivtx > 0 ; --ivtx )
              {
                std::size_t trackingVertexIndex = m_trackingVertexBins[ nbin ].at( ivtx-1 ) ;
                ++m_vertexCounter ;

                // Calculate the distance
                double distance = (position - trackingVertexes_->at(trackingVertexIndex).position()).P();
                // If the distance is under a given cut return the trackingVertex index (vertex merging)
                if (distance <= distanceCut_)
                  {
                    // Add simvertex to the pre existent tv
                    trackingVertexes_->at(trackingVertexIndex).addG4Vertex(simVertex);

                    // std::cout << "MATCHED VERTEX " << trackingVertexIndex-1 << " OF " << trackingVertexes_->size() << ", POSITION (" << position.x() << ", " << position.y() << ", " << position.z() << ") r = " << position.P() << std::endl ;

                    // return tv index
                    return trackingVertexIndex;
                  }
              }
          }
      }

    ++m_noMatchVertexCounter ;

    // Get the event if from the simvertex
    EncodedEventId simVertexEventId = simVertex.eventId();

    // Initialize the event counter
    if (eventIdCounter_.find(simVertexEventId) == eventIdCounter_.end())
        eventIdCounter_[simVertexEventId] = 0;

    // Get the simVertex id
    EncodedTruthId simVertexId = EncodedTruthId(simVertexEventId, eventIdCounter_[simVertexEventId]);

    // Calculate if the vertex is in the tracker volume (it needs to be review for other detectors)
    bool inVolume = (position.Pt() < volumeRadius_ && std::abs(position.z()) < volumeZ_); // In or out of Tracker

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
    edm::Handle<edm::HepMCProduct> hepmc = (useMultipleHepMCLabels_) ? hepMCProducts_.at(trackingVertex.eventId().rawId()) : hepMCProducts_.at(0);
    const HepMC::GenEvent * genEvent = hepmc->GetEvent();

    // Get the postion of the tv
    Vector tvPosition(trackingVertex.position().x(), trackingVertex.position().y(), trackingVertex.position().z());

    // Find HepMC vertices, put them in a close TrackingVertex (this could conceivably add the same GenVertex to multiple TrackingVertices)
    for (
        HepMC::GenEvent::vertex_const_iterator iGenVertex = genEvent->vertices_begin();
        iGenVertex != genEvent->vertices_end();
        ++iGenVertex
    )
    {
        // Get the position of the genVertex
        HepMC::ThreeVector rawPosition = (*iGenVertex)->position();

        // Convert to cm
        Vector genPosition(rawPosition.x()/10.0, rawPosition.y()/10.0, rawPosition.z()/10.0);

        // Calculate the dis
        double distance = sqrt( (tvPosition - genPosition).mag2() );

        if (distance <= distanceCut_)
            trackingVertex.addGenVertex( GenVertexRef(hepmc, (*iGenVertex)->barcode()) );
    }
}



DEFINE_FWK_MODULE(TrackingTruthProducer);
