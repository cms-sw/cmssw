/** @file
 * @brief Definitions of the TrackingTruthAccumulator methods.
 *
 * There are a lot of utility classes and functions in this file. This makes it quite long but
 * I didn't want to confuse the directory structure with lots of extra files. My reasoning was
 * that lots of people look at the directory contents but only the really interested ones will
 * look in this particular file, and the utility stuff isn't used elsewhere.
 *
 * There are three main sections to this file, from top to bottom:
 * 1 - Declarations of extra utility stuff not in the header file. All in the unnamed namespace.
 * 2 - Definitions of the TrackingTruthAccumulator methods.
 * 3 - Definitions of the utility classes and functions declared in (1).
 *
 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
 * @date circa Oct/2012 to Feb/2013
 *
 * Changelog:
 * 17/Jul/2014 Dominik Nowatschin (dominik.nowatschin@cern.ch) - added SimVertex and a ref to
 * HepMC::Genvertex to TrackingVertex in TrackingParticleFactory::createTrackingVertex; handle to
 * edm::HepMCProduct is created directly in TrackingTruthAccumulator::accumulate and not in 
 * accumulateEvent as edm::Event and PileUpEventPrincipal have different getByLabel() functions
 * 
 * 07/Feb/2013 Mark Grimes - Reorganised and added a bit more documentation. Still not enough
 * though.
 * 12/Mar/2012 (branch NewTrackingParticle only) Mark Grimes - Updated TrackingParticle creation
 * to fit in with Subir Sarkar's re-implementation of TrackingParticle.
 */
#include "SimGeneral/TrackingAnalysis/interface/TrackingTruthAccumulator.h"

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"




//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//------                                                                     ------
//------  Declarations for utility classes and functions in the unnamed      ------
//------  namespace. The definitions are right at the bottom of the file.    ------
//------                                                                     ------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

namespace
{
	/** @brief Class to represent tracks in the decay chain.
	 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
	 * @date 30/Oct/2012
	 */
	struct DecayChainTrack
	{
		int simTrackIndex;
		struct DecayChainVertex* pParentVertex;
		// Some things have multiple decay vertices. Not sure how that works but it seems to be mostly electrons and some photons.
		std::vector<struct DecayChainVertex*> daughterVertices;
		DecayChainTrack* pMergedBremSource;
		DecayChainTrack() : simTrackIndex(-1), pParentVertex(NULL), pMergedBremSource(NULL) {}
		DecayChainTrack( int newSimTrackIndex ) : simTrackIndex(newSimTrackIndex), pParentVertex(NULL), pMergedBremSource() {}
	};

	/** @brief Class to represent a vertex in the decay chain, and it's relationship with parents and daughters.
	 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
	 * @date 30/Oct/2012
	 */
	struct DecayChainVertex
	{
		int simVertexIndex;
		DecayChainTrack* pParentTrack;
		std::vector<DecayChainTrack*> daughterTracks;
		DecayChainVertex* pMergedBremSource;
		DecayChainVertex() : simVertexIndex(-1), pParentTrack(NULL), pMergedBremSource(NULL) {}
		DecayChainVertex( int newIndex ) : simVertexIndex(newIndex), pParentTrack(NULL), pMergedBremSource(NULL) {}
	};

	/** @brief Intermediary class. Mainly here to handle memory safely.
	 *
	 * Reconstructs the parent and child information from SimVertex and SimTracks to create the decay chain.
	 * SimVertex and SimTrack only have information about their parent, so to get information about the children the
	 * whole collection has to be analysed. This function does that and returns a tree of DecayChainTrack and
	 * DecayChainVertex classes that represent the full decay chain.
	 *
	 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
	 * @date 31/Oct/2012
	 */
	struct DecayChain
	{
	public:
		DecayChain( const std::vector<SimTrack>& trackCollection, const std::vector<SimVertex>& vertexCollection );
		const size_t decayTracksSize;
		const size_t decayVerticesSize;

		/** @brief Testing check. Won't actually be called when the code becomes production.
		 *
		 * Checks that there are no dangling objects not associated in the decay chain.
		 */
		void integrityCheck();
		const SimTrack& getSimTrack( const ::DecayChainTrack* pDecayTrack ) const { return simTrackCollection_.at( pDecayTrack->simTrackIndex ); }
		const SimVertex& getSimVertex( const ::DecayChainVertex* pDecayVertex ) const { return simVertexCollection_.at( pDecayVertex->simVertexIndex ); }
	private:
		void findBrem( const std::vector<SimTrack>& trackCollection, const std::vector<SimVertex>& vertexCollection );
		std::unique_ptr< ::DecayChainTrack[]> decayTracks_;
		std::unique_ptr< ::DecayChainVertex[]> decayVertices_;

		/// The vertices at the top of the decay chain. These are a subset of the ones in the decayVertices member. Don't delete them.
		std::vector< ::DecayChainVertex*> rootVertices_;

		// Keep a record of the constructor parameters
		const std::vector<SimTrack>& simTrackCollection_;
		const std::vector<SimVertex>& simVertexCollection_;
	public:
		const std::unique_ptr< ::DecayChainTrack[]>& decayTracks; ///< Reference maps to decayTracks_ for easy external const access.
		const std::unique_ptr< ::DecayChainVertex[]>& decayVertices; ///< Reference maps to decayVertices_ for easy external const access.
		const std::vector< ::DecayChainVertex*>& rootVertices; ///< Reference maps to rootVertices_ for easy external const access.
	};

	/** @brief Class to create TrackingParticle and TrackingVertex objects.
	 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
	 * @date 12/Nov/2012
	 */
	class TrackingParticleFactory
	{
	public:
		TrackingParticleFactory( const ::DecayChain& decayChain, const edm::Handle< std::vector<reco::GenParticle> >& hGenParticles,
				const edm::Handle< edm::HepMCProduct >& hepMCproduct, const edm::Handle< std::vector<int> >& hHepMCGenParticleIndices,
				const std::vector<const PSimHit*>& simHits, double volumeRadius, double volumeZ, double vertexDistanceCut, bool allowDifferentProcessTypes );
		TrackingParticle createTrackingParticle( const DecayChainTrack* pTrack, const TrackerTopology *tTopo ) const;
		TrackingVertex createTrackingVertex( const DecayChainVertex* pVertex) const;
		bool vectorIsInsideVolume( const math::XYZTLorentzVectorD& vector ) const;
	private:
		const ::DecayChain& decayChain_;
		const edm::Handle< std::vector<reco::GenParticle> >& hGenParticles_;
		const edm::Handle< edm::HepMCProduct >& hepMCproduct_;
		std::vector<int> genParticleIndices_;
		const std::vector<const PSimHit*>& simHits_;
		const double volumeRadius_;
		const double volumeZ_;
		const double vertexDistanceCut2_; // distance based on which HepMC::GenVertexs are added to SimVertexs
		std::multimap<unsigned int, size_t> trackIdToHitIndex_; ///< A multimap linking SimTrack::trackId() to the hit index in pSimHits_
		bool allowDifferentProcessTypeForDifferentDetectors_; ///< See the comment for the same member in TrackingTruthAccumulator
	};

	/** @brief Handles adding objects to the output collections correctly, and checks to see if a particular object already exists.
	 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
	 * @date 09/Nov/2012
	 */
	class OutputCollectionWrapper
	{
	public:
		OutputCollectionWrapper( const DecayChain& decayChain, TrackingTruthAccumulator::OutputCollections& outputCollections );
		TrackingParticle* addTrackingParticle( const ::DecayChainTrack* pDecayTrack, const TrackingParticle& trackingParticle );
		TrackingVertex* addTrackingVertex( const ::DecayChainVertex* pDecayVertex, const TrackingVertex& trackingVertex );
		TrackingParticle* getTrackingParticle( const ::DecayChainTrack* pDecayTrack );
		/// @brief After this call, any call to getTrackingParticle or getRef with pOriginalTrack will return the TrackingParticle for pProxyTrack instead.
		void setProxy( const ::DecayChainTrack* pOriginalTrack, const ::DecayChainTrack* pProxyTrack );
		void setProxy( const ::DecayChainVertex* pOriginalVertex, const ::DecayChainVertex* pProxyVertex );
		TrackingVertex* getTrackingVertex( const ::DecayChainVertex* pDecayVertex );
		TrackingParticleRef getRef( const ::DecayChainTrack* pDecayTrack );
		TrackingVertexRef getRef( const ::DecayChainVertex* pDecayVertex );
	private:
		void associateToExistingObjects( const ::DecayChainVertex* pChainVertex );
		void associateToExistingObjects( const ::DecayChainTrack* pChainTrack );
		TrackingTruthAccumulator::OutputCollections& output_;
		std::vector<int> trackingParticleIndices_;
		std::vector<int> trackingVertexIndices_;
	};

	/** @brief Adds the supplied TrackingParticle and its parent TrackingVertex to the output collection. Checks to make sure they don't already exist first.
	 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
	 * @date 12/Nov/2012
	 */
	TrackingParticle* addTrackAndParentVertex( ::DecayChainTrack* pDecayTrack, const TrackingParticle& trackingParticle, ::OutputCollectionWrapper* pOutput );

	/** @brief Creates a TrackingParticle for the supplied DecayChainTrack and adds it to the output if it passes selection. Gets called recursively if adding parents.
	 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
	 * @date 05/Nov/2012
	 */
	void addTrack( ::DecayChainTrack* pDecayChainTrack, const TrackingParticleSelector* pSelector, ::OutputCollectionWrapper* pUnmergedOutput, ::OutputCollectionWrapper* pMergedOutput, const ::TrackingParticleFactory& objectFactory, bool addAncestors, const TrackerTopology *tTopo);

} // end of the unnamed namespace




//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//------                                                                     ------
//------  Definitions for the TrackingTruthAccumulator methods               ------
//------  are below.                                                         ------
//------                                                                     ------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------

TrackingTruthAccumulator::TrackingTruthAccumulator( const edm::ParameterSet & config, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC) :
		messageCategory_("TrackingTruthAccumulator"),
		volumeRadius_( config.getParameter<double>("volumeRadius") ),
		volumeZ_( config.getParameter<double>("volumeZ") ),
		vertexDistanceCut_( config.getParameter<double>("vertexDistanceCut") ),
		ignoreTracksOutsideVolume_( config.getParameter<bool>("ignoreTracksOutsideVolume") ),
		maximumPreviousBunchCrossing_( config.getParameter<unsigned int>("maximumPreviousBunchCrossing") ),
		maximumSubsequentBunchCrossing_( config.getParameter<unsigned int>("maximumSubsequentBunchCrossing") ),
		createUnmergedCollection_( config.getParameter<bool>("createUnmergedCollection") ),
		createMergedCollection_(config.getParameter<bool>("createMergedBremsstrahlung") ),
		addAncestors_( config.getParameter<bool>("alwaysAddAncestors") ),
		removeDeadModules_( config.getParameter<bool>("removeDeadModules") ),
		simTrackLabel_( config.getParameter<edm::InputTag>("simTrackCollection") ),
		simVertexLabel_( config.getParameter<edm::InputTag>("simVertexCollection") ),
		collectionTags_( ),
		genParticleLabel_( config.getParameter<edm::InputTag>("genParticleCollection") ),
		hepMCproductLabel_( config.getParameter<edm::InputTag>("HepMCProductLabel") ),
		allowDifferentProcessTypeForDifferentDetectors_( config.getParameter<bool>("allowDifferentSimHitProcesses") )
{
	//
	// Make sure at least one of the merged and unmerged collections have been set
	// to be created.
	//
	if( !createUnmergedCollection_ && !createMergedCollection_ )
		edm::LogError(messageCategory_) << "Both \"createUnmergedCollection\" and \"createMergedBremsstrahlung\" have been"
			<< "set to false, which means no collections will be created";

	// Initialize selection for building TrackingParticles
	//
	if( config.exists( "select" ) )
	{
		edm::ParameterSet param=config.getParameter<edm::ParameterSet>("select");
		selector_=TrackingParticleSelector( param.getParameter<double>( "ptMinTP" ),
				param.getParameter<double>( "minRapidityTP" ),
				param.getParameter<double>( "maxRapidityTP" ),
				param.getParameter<double>( "tipTP" ),
				param.getParameter<double>( "lipTP" ),
				param.getParameter<int>( "minHitTP" ),
				param.getParameter<bool>( "signalOnlyTP" ),
				param.getParameter<bool>( "chargedOnlyTP" ),
				param.getParameter<bool>( "stableOnlyTP" ),
				param.getParameter<std::vector<int> >("pdgIdTP") );
		selectorFlag_=true;

		// Also set these two variables, which are used to drop out early if the SimTrack doesn't conform.
		// The selector_ requires a full TrackingParticle object, but these two variables can veto things early.
		chargedOnly_=param.getParameter<bool>( "chargedOnlyTP" );
		signalOnly_=param.getParameter<bool>( "signalOnlyTP" );
	}
	else
	{
		selectorFlag_=false;
		chargedOnly_=false;
		signalOnly_=false;
	}

	//
	// Need to state what collections are going to be added to the event. This
	// depends on which of the merged and unmerged collections have been configured
	// to be created.
	//
	if( createUnmergedCollection_ )
	{
		mixMod.produces<TrackingVertexCollection>();
		mixMod.produces<TrackingParticleCollection>();
	}

	if( createMergedCollection_ )
	{
		mixMod.produces<TrackingParticleCollection>("MergedTrackTruth");
		mixMod.produces<TrackingVertexCollection>("MergedTrackTruth");
	}

	iC.consumes<std::vector<SimTrack> >(simTrackLabel_);
	iC.consumes<std::vector<SimVertex> >(simVertexLabel_);
	iC.consumes<std::vector<reco::GenParticle> >(genParticleLabel_);
	iC.consumes<std::vector<int> >(genParticleLabel_);
	iC.consumes<std::vector<int> >(hepMCproductLabel_);

	// Fill the collection tags
        const edm::ParameterSet& simHitCollectionConfig=config.getParameterSet("simHitCollections");
	std::vector<std::string> parameterNames=simHitCollectionConfig.getParameterNames();

	for( const auto& parameterName : parameterNames )
	{
           std::vector<edm::InputTag> tags=simHitCollectionConfig.getParameter<std::vector<edm::InputTag> >(parameterName);
           collectionTags_.insert(collectionTags_.end(), tags.begin(), tags.end());
        }

	for( const auto& collectionTag : collectionTags_ ) {
	  iC.consumes<std::vector<PSimHit> >(collectionTag);
        }

}

void TrackingTruthAccumulator::initializeEvent( edm::Event const& event, edm::EventSetup const& setup )
{
	if( createUnmergedCollection_ )
	{
		unmergedOutput_.pTrackingParticles.reset( new TrackingParticleCollection );
		unmergedOutput_.pTrackingVertices.reset( new TrackingVertexCollection );
		unmergedOutput_.refTrackingParticles=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingParticleCollection>();
		unmergedOutput_.refTrackingVertexes=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingVertexCollection>();
	}

	if( createMergedCollection_ )
	{
		mergedOutput_.pTrackingParticles.reset( new TrackingParticleCollection );
		mergedOutput_.pTrackingVertices.reset( new TrackingVertexCollection );
		mergedOutput_.refTrackingParticles=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingParticleCollection>("MergedTrackTruth");
		mergedOutput_.refTrackingVertexes=const_cast<edm::Event&>( event ).getRefBeforePut<TrackingVertexCollection>("MergedTrackTruth");
	}
}

/// create handle to edm::HepMCProduct here because event.getByLabel with edm::HepMCProduct only works for edm::Event
/// but not for PileUpEventPrincipal; PileUpEventPrincipal::getByLabel tries to call T::value_type and T::iterator
/// (where T is the type of the object one wants to get a handle to) which is only implemented for container-like objects
/// like std::vector but not for edm::HepMCProduct!

void TrackingTruthAccumulator::accumulate( edm::Event const& event, edm::EventSetup const& setup )
{
	// Call the templated version that does the same for both signal and pileup events
	
	edm::Handle< edm::HepMCProduct > hepmc;
	event.getByLabel(hepMCproductLabel_, hepmc);
	
	accumulateEvent( event, setup, hepmc );
}

void TrackingTruthAccumulator::accumulate( PileUpEventPrincipal const& event, edm::EventSetup const& setup, edm::StreamID const& )
{
	// If this bunch crossing is outside the user configured limit, don't do anything.
	if( event.bunchCrossing()>=-static_cast<int>(maximumPreviousBunchCrossing_) && event.bunchCrossing()<=static_cast<int>(maximumSubsequentBunchCrossing_) )
	{
		//edm::LogInfo(messageCategory_) << "Analysing pileup event for bunch crossing " << event.bunchCrossing();
		
		//simply create empty handle as we do not have a HepMCProduct in PU anyway
		edm::Handle< edm::HepMCProduct > hepmc;
		accumulateEvent( event, setup, hepmc );
	}
	else edm::LogInfo(messageCategory_) << "Skipping pileup event for bunch crossing " << event.bunchCrossing();
}

void TrackingTruthAccumulator::finalizeEvent( edm::Event& event, edm::EventSetup const& setup )
{

	if( createUnmergedCollection_ )
	{
		edm::LogInfo("TrackingTruthAccumulator") << "Adding " << unmergedOutput_.pTrackingParticles->size() << " TrackingParticles and " << unmergedOutput_.pTrackingVertices->size()
				<< " TrackingVertexs to the event.";

		event.put( unmergedOutput_.pTrackingParticles );
		event.put( unmergedOutput_.pTrackingVertices );
	}

	if( createMergedCollection_ )
	{
		edm::LogInfo("TrackingTruthAccumulator") << "Adding " << mergedOutput_.pTrackingParticles->size() << " merged TrackingParticles and " << mergedOutput_.pTrackingVertices->size()
				<< " merged TrackingVertexs to the event.";

		event.put( mergedOutput_.pTrackingParticles, "MergedTrackTruth" );
		event.put( mergedOutput_.pTrackingVertices, "MergedTrackTruth" );
	}

}

template<class T> void TrackingTruthAccumulator::accumulateEvent( const T& event, const edm::EventSetup& setup, const edm::Handle< edm::HepMCProduct >& hepMCproduct)
{
	//
	// Get the collections
	//
	edm::Handle<std::vector<SimTrack> > hSimTracks;
	edm::Handle<std::vector<SimVertex> > hSimVertices;
	edm::Handle< std::vector<reco::GenParticle> > hGenParticles;
	edm::Handle< std::vector<int> > hGenParticleIndices;

	event.getByLabel( simTrackLabel_, hSimTracks );
	event.getByLabel( simVertexLabel_, hSimVertices );

	try
	{
		event.getByLabel( genParticleLabel_, hGenParticles );
		event.getByLabel( genParticleLabel_, hGenParticleIndices );
	}
	catch( cms::Exception& exception )
	{
		//
		// The Monte Carlo is not always available, e.g. for pileup events. The information
		// is only used if it's available, but for some reason the PileUpEventPrincipal
		// wrapper throws an exception here rather than waiting to see if the handle is
		// used (as is the case for edm::Event). So I just want to catch this exception
		// and use the normal handle checking later on.
		//
	}

	//Retrieve tracker topology from geometry
	edm::ESHandle<TrackerTopology> tTopoHandle;
	setup.get<IdealGeometryRecord>().get(tTopoHandle);
	const TrackerTopology* const tTopo = tTopoHandle.product();


	// Run through the collections and work out the decay chain of each track/vertex. The
	// information in SimTrack and SimVertex only allows traversing upwards, but this will
	// allow traversal in both directions. This is required for things like grouping electrons
	// that bremsstrahlung as one TrackingParticle if "mergedBremsstrahlung" is set in the
	// config file.
	DecayChain decayChain( *hSimTracks, *hSimVertices );

	// I only want to create these collections if they're actually required
	std::auto_ptr< ::OutputCollectionWrapper> pUnmergedCollectionWrapper;
	std::auto_ptr< ::OutputCollectionWrapper> pMergedCollectionWrapper;
	if( createUnmergedCollection_ ) pUnmergedCollectionWrapper.reset( new ::OutputCollectionWrapper( decayChain, unmergedOutput_ ) );
	if( createMergedCollection_ ) pMergedCollectionWrapper.reset( new ::OutputCollectionWrapper( decayChain, mergedOutput_ ) );

	std::vector<const PSimHit*> simHitPointers;
	fillSimHits( simHitPointers, event, setup );
	TrackingParticleFactory objectFactory( decayChain, hGenParticles, hepMCproduct, hGenParticleIndices, simHitPointers, volumeRadius_, volumeZ_, vertexDistanceCut_, allowDifferentProcessTypeForDifferentDetectors_ );

	// While I'm testing, perform some checks.
	// TODO - drop this call once I'm happy it works in all situations.
	//decayChain.integrityCheck();

	TrackingParticleSelector* pSelector=NULL;
	if( selectorFlag_ ) pSelector=&selector_;

	// Run over all of the SimTracks, but because I'm interested in the decay hierarchy
	// do it through the DecayChainTrack objects. These are looped over in sequence here
	// but they have the hierarchy information for the functions called to traverse the
	// decay chain.

	for( size_t index=0; index<decayChain.decayTracksSize; ++index )
	{
		::DecayChainTrack* pDecayTrack=&decayChain.decayTracks[index];
		const SimTrack& simTrack=hSimTracks->at(pDecayTrack->simTrackIndex);


		// Perform some quick checks to see if we can drop out early. Note that these are
		// a subset of the cuts in the selector_ so the created TrackingParticle could still
		// fail. The selector_ requires the full TrackingParticle to be made however, which
		// can be computationally expensive.
		if( chargedOnly_ && simTrack.charge()==0 ) continue;
		if( signalOnly_ && (simTrack.eventId().bunchCrossing()!=0 || simTrack.eventId().event()!=0) ) continue;

		// Also perform a check to see if the production vertex is inside the tracker volume (if required).
		if( ignoreTracksOutsideVolume_ )
		{
			const SimVertex& simVertex=hSimVertices->at( pDecayTrack->pParentVertex->simVertexIndex );
			if( !objectFactory.vectorIsInsideVolume( simVertex.position() ) ) continue;
		}


		// This function creates the TrackinParticle and adds it to the collection if it
		// passes the selection criteria specified in the configuration. If the config
		// specifies adding ancestors, the function is called recursively to do that.
		::addTrack( pDecayTrack, pSelector, pUnmergedCollectionWrapper.get(), pMergedCollectionWrapper.get(), objectFactory, addAncestors_, tTopo );
	}
}

template<class T> void TrackingTruthAccumulator::fillSimHits( std::vector<const PSimHit*>& returnValue, const T& event, const edm::EventSetup& setup )
{
	// loop over the collections
	for( const auto& collectionTag : collectionTags_ )
	{
		edm::Handle< std::vector<PSimHit> > hSimHits;
		event.getByLabel( collectionTag, hSimHits );

		// TODO - implement removing the dead modules
		for( const auto& simHit : *hSimHits )
		{
			returnValue.push_back( &simHit );
		}

	} // end of loop over InputTags
}


//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//------                                                                     ------
//------  End of the definitions for the TrackingTruthAccumulator methods.   ------
//------  Definitions for the functions and classes declared in the unnamed  ------
//------  namespace are below. Documentation for those is above, where       ------
//------  they're declared.                                                  ------
//------                                                                     ------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------
//---------------------------------------------------------------------------------


namespace // Unnamed namespace for things only used in this file
{
	//---------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------
	//----   TrackingParticleFactory methods   ----------------------------------------
	//---------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------

	::TrackingParticleFactory::TrackingParticleFactory( const ::DecayChain& decayChain, const edm::Handle< std::vector<reco::GenParticle> >& hGenParticles,
			const edm::Handle< edm::HepMCProduct >& hepMCproduct, const edm::Handle< std::vector<int> >& hHepMCGenParticleIndices, const std::vector<const PSimHit*>& simHits,
			double volumeRadius, double volumeZ, double vertexDistanceCut, bool allowDifferentProcessTypes )
		: decayChain_(decayChain), hGenParticles_(hGenParticles), hepMCproduct_(hepMCproduct), simHits_(simHits), volumeRadius_(volumeRadius),
		  volumeZ_(volumeZ), vertexDistanceCut2_(vertexDistanceCut*vertexDistanceCut), allowDifferentProcessTypeForDifferentDetectors_(allowDifferentProcessTypes)
	{
		// Need to create a multimap to get from a SimTrackId to all of the hits in it. The SimTrackId
		// is an unsigned int.
		for( size_t index=0; index<simHits_.size(); ++index )
		{
			trackIdToHitIndex_.insert( std::make_pair( simHits_[index]->trackId(), index ) );
		}

		if( hHepMCGenParticleIndices.isValid() ) // Monte Carlo might not be available for the pileup events
		{
			genParticleIndices_.resize( hHepMCGenParticleIndices->size()+1 );

			// What I need is the reverse mapping of this vector. The sizes are already equivalent because I set
			// the size in the initialiser list.
			for( size_t recoGenParticleIndex=0; recoGenParticleIndex<hHepMCGenParticleIndices->size(); ++recoGenParticleIndex )
			{
				size_t hepMCGenParticleIndex=(*hHepMCGenParticleIndices)[recoGenParticleIndex];

				// They should be the same size, give or take a fencepost error, so this should never happen - but just in case
				if( genParticleIndices_.size()<=hepMCGenParticleIndex ) genParticleIndices_.resize(hepMCGenParticleIndex+1);

				genParticleIndices_[ hepMCGenParticleIndex ]=recoGenParticleIndex;
			}
		}
	}

	TrackingParticle TrackingParticleFactory::createTrackingParticle( const ::DecayChainTrack* pChainTrack, const TrackerTopology *tTopo ) const
	{
		typedef math::XYZTLorentzVectorD LorentzVector;
		typedef math::XYZPoint Vector;

		const SimTrack& simTrack=decayChain_.getSimTrack( pChainTrack );
		const SimVertex& parentSimVertex=decayChain_.getSimVertex( pChainTrack->pParentVertex );

		LorentzVector position( 0, 0, 0, 0 );
		if( !simTrack.noVertex() ) position=parentSimVertex.position();

		int pdgId=simTrack.type();

		TrackingParticle returnValue;
		// N.B. The sim track is added a few lines below, the parent and decay vertices are added in
		// another function later on.

		//
		// If there is some valid Monte Carlo for this track, take some information from that.
		// Only do so if it is from the signal event however. Not sure why but that's what the
		// old code did.
		//
		if( simTrack.eventId().event()==0 && simTrack.eventId().bunchCrossing()==0 ) // if this is a track in the signal event
		{
			int hepMCGenParticleIndex=simTrack.genpartIndex();
			if( hepMCGenParticleIndex>=0 && hepMCGenParticleIndex<static_cast<int>(genParticleIndices_.size()) && hGenParticles_.isValid() )
			{
				int recoGenParticleIndex=genParticleIndices_[hepMCGenParticleIndex];
				reco::GenParticleRef generatorParticleRef( hGenParticles_, recoGenParticleIndex );
				pdgId=generatorParticleRef->pdgId();
				returnValue.addGenParticle( generatorParticleRef );
			}
		}

		returnValue.addG4Track( simTrack );

		// I need to run over the sim hits and count up the different types of hits. To be honest
		// I don't fully understand this next section of code, I copied it from the old TrackingTruthProducer
		// to provide the same results. The different types of hits that I need to count are:
		size_t matchedHits=0; // As far as I can tell, this is the number of tracker layers with hits on them, i.e. taking account of overlaps.
		size_t numberOfHits=0; // The total number of hits not taking account of overlaps.
		size_t numberOfTrackerHits=0; // The number of tracker hits not taking account of overlaps.

		bool init=true;
		int processType=0;
		int particleType=0;
		int oldLayer = 0;
		int newLayer = 0;
		DetId oldDetector;
		DetId newDetector;

		for( std::multimap<unsigned int,size_t>::const_iterator iHitIndex=trackIdToHitIndex_.lower_bound( simTrack.trackId() );
				iHitIndex!=trackIdToHitIndex_.upper_bound( simTrack.trackId() );
				++iHitIndex )
		{
			const auto& pSimHit=simHits_[ iHitIndex->second ];

			// Initial condition for consistent simhit selection
			if( init )
			{
				processType=pSimHit->processType();
				particleType=pSimHit->particleType();
				newDetector=DetId( pSimHit->detUnitId() );
				init=false;
			}

			oldDetector=newDetector;
			newDetector=DetId( pSimHit->detUnitId() );

			// Fast sim seems to have different processType between tracker and muons, so if this flag
			// has been set allow the processType to change if the detector changes.
			if( allowDifferentProcessTypeForDifferentDetectors_ && newDetector.det()!=oldDetector.det() ) processType=pSimHit->processType();

			// Check for delta and interaction products discards
			if( processType==pSimHit->processType() && particleType==pSimHit->particleType() && pdgId==pSimHit->particleType() )
			{
				++numberOfHits;
				oldLayer=newLayer;
                                newLayer=0;
				if( newDetector.det() == DetId::Tracker ) {
                                  ++numberOfTrackerHits;

                                  newLayer=tTopo->layer( newDetector );

                                  // Count hits using layers for glued detectors
                                  if( (oldLayer!=newLayer || (oldLayer==newLayer && oldDetector.subdetId()!=newDetector.subdetId())) ) ++matchedHits;
                                }
			}
		} // end of loop over the sim hits for this sim track

		returnValue.setNumberOfTrackerLayers( matchedHits );
		returnValue.setNumberOfHits( numberOfHits );
		returnValue.setNumberOfTrackerHits( numberOfTrackerHits );

		return returnValue;
	}

	TrackingVertex TrackingParticleFactory::createTrackingVertex( const ::DecayChainVertex* pChainVertex) const
	{
                
		typedef math::XYZTLorentzVectorD LorentzVector;
		typedef math::XYZPoint Vector;
		typedef edm::Ref<edm::HepMCProduct, HepMC::GenVertex >   GenVertexRef;
		
		const SimVertex& simVertex=decayChain_.getSimVertex( pChainVertex );

		bool isInVolume=this->vectorIsInsideVolume( simVertex.position() );

		TrackingVertex returnValue( simVertex.position(), isInVolume, simVertex.eventId() );
		
		// add the SimVertex to the TrackingVertex
		returnValue.addG4Vertex(simVertex);
                
		// also add refs to nearby HepMC::GenVertexs; the way this is done (i.e. based on the position) is transcribed over from the old TrackingTruthProducer
		if( simVertex.eventId().event()==0 && simVertex.eventId().bunchCrossing()==0 && hepMCproduct_.isValid()) // if this is a track in the signal event
		{
			const HepMC::GenEvent* genEvent = hepMCproduct_->GetEvent();
			
			if (genEvent != NULL)
			{
				Vector tvPosition(returnValue.position().x(), returnValue.position().y(), returnValue.position().z());
                
				for (HepMC::GenEvent::vertex_const_iterator iGenVertex = genEvent->vertices_begin(); iGenVertex != genEvent->vertices_end(); ++iGenVertex)
				{
					HepMC::ThreeVector rawPosition = (*iGenVertex)->position();
                    
					Vector genPosition(rawPosition.x()*0.1, rawPosition.y()*0.1, rawPosition.z()*0.1);
                    
					auto distance2 = (tvPosition - genPosition).mag2();
                    
					if (distance2 < vertexDistanceCut2_)
						returnValue.addGenVertex( GenVertexRef(hepMCproduct_, (*iGenVertex)->barcode()) );
				}
			}
		}
                
		return returnValue;
	}

	bool ::TrackingParticleFactory::vectorIsInsideVolume( const math::XYZTLorentzVectorD& vector ) const
	{
		return ( vector.Pt()<volumeRadius_ && vector.z()<volumeZ_ );
	}

	//---------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------
	//----   DecayChain methods   -----------------------------------------------------
	//---------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------

	::DecayChain::DecayChain( const std::vector<SimTrack>& trackCollection, const std::vector<SimVertex>& vertexCollection )
		: decayTracksSize( trackCollection.size() ),
		  decayVerticesSize( vertexCollection.size() ),
		  decayTracks_( new DecayChainTrack[decayTracksSize] ),
		  decayVertices_( new DecayChainVertex[decayVerticesSize] ),
		  simTrackCollection_( trackCollection ),
		  simVertexCollection_( vertexCollection ),
		  decayTracks( decayTracks_ ), // These const references map onto the actual members for public const access
		  decayVertices( decayVertices_ ),
		  rootVertices( rootVertices_ )
	{
		// I need some maps to be able to get object pointers from the track/vertex ID
		std::map<int,::DecayChainTrack*> trackIdToDecayTrack;
		std::map<int,::DecayChainVertex*> vertexIdToDecayVertex;

		// First create a DecayChainTrack for every SimTrack and make a note of the
		// trackIds in the map. Also add a pointer to the daughter list of the parent
		// DecayChainVertex, which might include creating the vertex object if it
		// doesn't already exist.
		size_t decayVertexIndex=0; // The index of the next free slot in the DecayChainVertex array.
		for( size_t index=0; index<trackCollection.size(); ++index )
		{
			::DecayChainTrack* pDecayTrack=&decayTracks_[index]; // Get a pointer for ease of use

			// This is the array index of the SimTrack that this DecayChainTrack corresponds to. At
			// the moment this is a 1 to 1 relationship with the DecayChainTrack index, but they won't
			// necessarily be accessed through the array later so it's still required to store it.
			pDecayTrack->simTrackIndex=index;

			trackIdToDecayTrack[ trackCollection[index].trackId() ]=pDecayTrack;

			int parentVertexIndex=trackCollection[index].vertIndex();
			if( parentVertexIndex>=0 )
			{
				// Get the DecayChainVertex corresponding to this SimVertex, or initialise it if it hasn't been done already.
				::DecayChainVertex*& pParentVertex=vertexIdToDecayVertex[parentVertexIndex];
				if( pParentVertex==NULL )
				{
					// Note that I'm using a reference, so changing pParentVertex will change the entry in the map too.
					pParentVertex=&decayVertices_[decayVertexIndex];
					++decayVertexIndex;
					pParentVertex->simVertexIndex=parentVertexIndex;
				}
				pParentVertex->daughterTracks.push_back(pDecayTrack);
				pDecayTrack->pParentVertex=pParentVertex;
			}
			else throw std::runtime_error( "TrackingTruthAccumulator: Found a track with an invalid parent vertex index." );
		}

		// This assert was originally in to check the internal consistency of the decay chain. Fast sim
		// pileup seems to have a load of vertices with no tracks pointing to them though, so fast sim fails
		// this assert if pileup is added. I don't think the problem is vital however, so if the assert is
		// taken out these extra vertices are ignored.
		//assert( vertexIdToDecayVertex.size()==vertexCollection.size() && vertexCollection.size()==decayVertexIndex );

		// I still need to set DecayChainTrack::daughterVertices and DecayChainVertex::pParentTrack.
		// The information to do this comes from SimVertex::parentIndex. I couldn't do this before
		// because I need all of the DecayChainTracks initialised.
		for( auto& decayVertexMapPair : vertexIdToDecayVertex )
		{
			::DecayChainVertex* pDecayVertex=decayVertexMapPair.second;
			int parentTrackIndex=vertexCollection[pDecayVertex->simVertexIndex].parentIndex();
			if( parentTrackIndex!=-1 )
			{
				std::map<int,::DecayChainTrack*>::iterator iParentTrackMapPair=trackIdToDecayTrack.find(parentTrackIndex);
				if( iParentTrackMapPair==trackIdToDecayTrack.end() )
				{
					std::stringstream errorStream;
					errorStream << "TrackingTruthAccumulator: Something has gone wrong with the indexing. Parent track index is " << parentTrackIndex << ".";
					throw std::runtime_error( errorStream.str() );
				}

				::DecayChainTrack* pParentTrackHierarchy=iParentTrackMapPair->second;

				pParentTrackHierarchy->daughterVertices.push_back( pDecayVertex );
				pDecayVertex->pParentTrack=pParentTrackHierarchy;
			}
			else rootVertices_.push_back(pDecayVertex); // Has no parent so is at the top of the decay chain.
		} // end of loop over the vertexIdToDecayVertex map

		findBrem( trackCollection, vertexCollection );

	} // end of ::DecayChain constructor

	// Function documentation is with the declaration above. This function is only used for testing.
	void ::DecayChain::integrityCheck()
	{
		//
		// Start off checking the tracks
		//
		for( size_t index=0; index<decayTracksSize; ++index )
		{
			const auto& decayTrack=decayTracks[index];
			//
			// Tracks should always have a production vertex
			//
			if( decayTrack.pParentVertex==NULL ) throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainTrack with no parent vertex." );

			//
			// Make sure the parent has this as a child. Also make sure it's only listed once.
			//
			size_t numberOfTimesListed=0;
			for( const auto pSiblingTrack : decayTrack.pParentVertex->daughterTracks )
			{
				if( pSiblingTrack==&decayTrack ) ++numberOfTimesListed;
			}
			if( numberOfTimesListed!=1 ) throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainTrack whose parent does not have it listed once and only once as a child." );

			//
			// Make sure all of the children have this listed as a parent.
			//
			for( const auto pDaughterVertex : decayTrack.daughterVertices )
			{
				if( pDaughterVertex->pParentTrack!=&decayTrack ) throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainTrack whose child does not have it listed as the parent." );
			}

			//
			// Follow the chain up to the root vertex, and make sure that is listed in rootVertices
			//
			const DecayChainVertex* pAncestorVertex=decayTrack.pParentVertex;
			while( pAncestorVertex->pParentTrack!=NULL )
			{
				if( pAncestorVertex->pParentTrack->pParentVertex==NULL ) throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainTrack with no parent vertex higher in the decay chain." );
				pAncestorVertex=pAncestorVertex->pParentTrack->pParentVertex;
			}
			if( std::find( rootVertices.begin(), rootVertices.end(), pAncestorVertex )==rootVertices.end() )
			{
				throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainTrack whose root vertex is not recorded anywhere." );
			}
		} // end of loop over decayTracks

		//
		// Now check the vertices
		//
		for( size_t index=0; index<decayVerticesSize; ++index )
		{
			const auto& decayVertex=decayVertices[index];

			//
			// Make sure this, or a vertex higher in the chain, is in the list of root vertices.
			//
			const DecayChainVertex* pAncestorVertex=&decayVertex;
			while( pAncestorVertex->pParentTrack!=NULL )
			{
				if( pAncestorVertex->pParentTrack->pParentVertex==NULL ) throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainTrack with no parent vertex higher in the vertex decay chain." );
				pAncestorVertex=pAncestorVertex->pParentTrack->pParentVertex;
			}
			if( std::find( rootVertices.begin(), rootVertices.end(), pAncestorVertex )==rootVertices.end() )
			{
				throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainTrack whose root vertex is not recorded anywhere." );
			}

			//
			// Make sure the parent has this listed in its daughters once and only once.
			//
			if( decayVertex.pParentTrack!=NULL )
			{
				size_t numberOfTimesListed=0;
				for( const auto pSibling : decayVertex.pParentTrack->daughterVertices )
				{
					if( pSibling==&decayVertex ) ++numberOfTimesListed;
				}
				if( numberOfTimesListed!=1 ) throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainVertex whose parent does not have it listed once and only once as a child." );
			}

			//
			// Make sure all of the children have this listed as the parent
			//
			for( const auto pDaughter : decayVertex.daughterTracks )
			{
				if( pDaughter->pParentVertex!=&decayVertex ) throw std::runtime_error( "TrackingTruthAccumulator.cc integrityCheck(): Found DecayChainVertex whose child does not have it listed as the parent." );
			}
		} // end of loop over decay vertices

		std::cout << "TrackingTruthAccumulator.cc integrityCheck() completed successfully" << std::endl;
	} // end of ::DecayChain::integrityCheck()

	void ::DecayChain::findBrem( const std::vector<SimTrack>& trackCollection, const std::vector<SimVertex>& vertexCollection )
	{
		for( size_t index=0; index<decayVerticesSize; ++index )
		{
			auto& vertex=decayVertices_[index];

			// Make sure parent is an electron
			if( vertex.pParentTrack==NULL ) continue;
			int parentTrackPDG=trackCollection[vertex.pParentTrack->simTrackIndex].type();
			if( std::abs( parentTrackPDG )!=11 ) continue;

			size_t numberOfElectrons=0;
			size_t numberOfNonElectronsOrPhotons=0;
			for( auto& pDaughterTrack : vertex.daughterTracks )
			{
				const auto& simTrack=trackCollection[pDaughterTrack->simTrackIndex];
				if( simTrack.type()==11 || simTrack.type()==-11 ) ++numberOfElectrons;
				else if( simTrack.type()!=22 ) ++numberOfNonElectronsOrPhotons;
			}
			if( numberOfElectrons==1 && numberOfNonElectronsOrPhotons==0 )
			{
				// This is a valid brem, run through and tell all daughters to use the parent
				// as the brem
				for( auto& pDaughterTrack : vertex.daughterTracks ) pDaughterTrack->pMergedBremSource=vertex.pParentTrack;
				vertex.pMergedBremSource=vertex.pParentTrack->pParentVertex;
			}
		}
	} // end of ::DecayChain::findBrem()


	//---------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------
	//----   OutputCollectionWrapper methods   ----------------------------------------
	//---------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------

	::OutputCollectionWrapper::OutputCollectionWrapper( const ::DecayChain& decayChain, TrackingTruthAccumulator::OutputCollections& outputCollections )
		: output_(outputCollections),
		  trackingParticleIndices_(decayChain.decayTracksSize,-1),
		  trackingVertexIndices_(decayChain.decayVerticesSize,-1)
	{
		// No operation besides the initialiser list
	}

	TrackingParticle* ::OutputCollectionWrapper::addTrackingParticle( const ::DecayChainTrack* pDecayTrack, const TrackingParticle& trackingParticle )
	{
		if( trackingParticleIndices_[pDecayTrack->simTrackIndex]!=-1 ) throw std::runtime_error( "OutputCollectionWrapper::addTrackingParticle - trying to add a particle twice" );

		trackingParticleIndices_[pDecayTrack->simTrackIndex]=output_.pTrackingParticles->size();
		output_.pTrackingParticles->push_back( trackingParticle );

		// Clear any associations in case there were any beforehand
		output_.pTrackingParticles->back().clearDecayVertices();
		output_.pTrackingParticles->back().clearParentVertex();

		// Associate to the objects that are already in the output collections
		associateToExistingObjects( pDecayTrack );

		return &output_.pTrackingParticles->back();
	}

	TrackingVertex* ::OutputCollectionWrapper::addTrackingVertex( const ::DecayChainVertex* pDecayVertex, const TrackingVertex& trackingVertex )
	{
		if( trackingVertexIndices_[pDecayVertex->simVertexIndex]!=-1 ) throw std::runtime_error( "OutputCollectionWrapper::addTrackingVertex - trying to add a vertex twice" );

		trackingVertexIndices_[pDecayVertex->simVertexIndex]=output_.pTrackingVertices->size();
		output_.pTrackingVertices->push_back( trackingVertex );

		// Associate the new vertex to any parents or children that are already in the output collections
		associateToExistingObjects( pDecayVertex );

		return &output_.pTrackingVertices->back();
	}

	TrackingParticle* ::OutputCollectionWrapper::getTrackingParticle( const ::DecayChainTrack* pDecayTrack )
	{
		const int index=trackingParticleIndices_[pDecayTrack->simTrackIndex];
		if( index==-1 ) return NULL;
		else return &(*output_.pTrackingParticles)[index];
	}

	TrackingVertex* ::OutputCollectionWrapper::getTrackingVertex( const ::DecayChainVertex* pDecayVertex )
	{
		const int index=trackingVertexIndices_[pDecayVertex->simVertexIndex];
		if( index==-1 ) return NULL;
		else return &(*output_.pTrackingVertices)[index];
	}

	TrackingParticleRef OutputCollectionWrapper::getRef( const ::DecayChainTrack* pDecayTrack )
	{
		const int index=trackingParticleIndices_[pDecayTrack->simTrackIndex];
		if( index==-1 ) throw std::runtime_error( "OutputCollectionWrapper::getRefTrackingParticle - ref requested for a non existent TrackingParticle" );
		else return TrackingParticleRef( output_.refTrackingParticles, index );
	}

	TrackingVertexRef OutputCollectionWrapper::getRef( const ::DecayChainVertex* pDecayVertex )
	{
		const int index=trackingVertexIndices_[pDecayVertex->simVertexIndex];
		if( index==-1 ) throw std::runtime_error( "OutputCollectionWrapper::getRefTrackingParticle - ref requested for a non existent TrackingVertex" );
		else return TrackingVertexRef( output_.refTrackingVertexes, index );
	}

	void ::OutputCollectionWrapper::setProxy( const ::DecayChainTrack* pOriginalTrack, const ::DecayChainTrack* pProxyTrack )
	{
		int& index=trackingParticleIndices_[pOriginalTrack->simTrackIndex];
		if( index!=-1 ) throw std::runtime_error( "OutputCollectionWrapper::setProxy() was called for a TrackingParticle that has already been created" );
		// Note that index is a reference so this call changes the value in the vector too
		index=trackingParticleIndices_[pProxyTrack->simTrackIndex];
	}

	void ::OutputCollectionWrapper::setProxy( const ::DecayChainVertex* pOriginalVertex, const ::DecayChainVertex* pProxyVertex )
	{
		int& index=trackingVertexIndices_[pOriginalVertex->simVertexIndex];
		const int newIndex=trackingVertexIndices_[pProxyVertex->simVertexIndex];

		if( index!=-1 && index!=newIndex ) throw std::runtime_error( "OutputCollectionWrapper::setProxy() was called for a TrackingVertex that has already been created" );

		// Note that index is a reference so this call changes the value in the vector too
		index=newIndex;
	}

	void ::OutputCollectionWrapper::associateToExistingObjects( const ::DecayChainVertex* pChainVertex )
	{
		// First make sure the DecayChainVertex supplied has been turned into a TrackingVertex
		TrackingVertex* pTrackingVertex=getTrackingVertex( pChainVertex );
		if( pTrackingVertex==NULL ) throw std::runtime_error( "associateToExistingObjects was passed a non existent TrackingVertex" );

		//
		// Associate to the parent track (if there is one)
		//
		::DecayChainTrack* pParentChainTrack=pChainVertex->pParentTrack;
		if( pParentChainTrack!=NULL ) // Make sure there is a parent track first
		{
			// There is a parent track, but it might not have been turned into a TrackingParticle yet
			TrackingParticle* pParentTrackingParticle=getTrackingParticle(pParentChainTrack);
			if( pParentTrackingParticle!=NULL )
			{
				pParentTrackingParticle->addDecayVertex( getRef(pChainVertex) );
				pTrackingVertex->addParentTrack( getRef(pParentChainTrack) );
			}
			// Don't worry about the "else" case - the parent track might not necessarily get added
			// to the output collection at all. A check is done on daughter vertices when tracks
			// are added, so the association will be picked up then.
		}

		//
		// A parent TrackingVertex is always associated to a daughter TrackingParticle when
		// the TrackingParticle is created. Hence there is no need to check the list of
		// daughter tracks.
		//
	}

	void ::OutputCollectionWrapper::associateToExistingObjects( const ::DecayChainTrack* pChainTrack )
	{
		//
		// First make sure this DecayChainTrack has been turned into a TrackingParticle
		//
		TrackingParticle* pTrackingParticle=getTrackingParticle( pChainTrack );
		if( pTrackingParticle==NULL ) throw std::runtime_error( "associateToExistingObjects was passed a non existent TrackingParticle" );

		// Get the parent vertex. This should always already have been turned into a TrackingVertex, and
		// there should always be a parent DecayChainVertex.
		::DecayChainVertex* pParentChainVertex=pChainTrack->pParentVertex;
		TrackingVertex* pParentTrackingVertex=getTrackingVertex( pParentChainVertex );

		//
		// First associate to the parent vertex.
		//
		pTrackingParticle->setParentVertex( getRef(pParentChainVertex) );
		pParentTrackingVertex->addDaughterTrack( getRef(pChainTrack) );

		// If there are any daughter vertices that have already been made into a TrackingVertex
		// make sure they know about each other. If the daughter vertices haven't been made into
		// TrackingVertexs yet, I won't do anything and create the association when the vertex is
		// made.
		for( auto pDaughterChainVertex : pChainTrack->daughterVertices )
		{
			TrackingVertex* pDaughterTrackingVertex=getTrackingVertex( pDaughterChainVertex );
			if( pDaughterTrackingVertex!=NULL )
			{
				pTrackingParticle->addDecayVertex( getRef(pDaughterChainVertex) );
				pDaughterTrackingVertex->addParentTrack( getRef(pChainTrack) );
			}
		}
	}


	TrackingParticle* addTrackAndParentVertex( ::DecayChainTrack* pDecayTrack, const TrackingParticle& trackingParticle, ::OutputCollectionWrapper* pOutput )
	{
		// See if this TrackingParticle has already been created (could be if the DecayChainTracks are
		// looped over in a funny order). If it has then there's no need to do anything.
		TrackingParticle* pTrackingParticle=pOutput->getTrackingParticle( pDecayTrack );
		if( pTrackingParticle==NULL )
		{
			// Need to make sure the production vertex has been created first
			TrackingVertex* pProductionVertex=pOutput->getTrackingVertex( pDecayTrack->pParentVertex );
			if( pProductionVertex==NULL )
			{
				// TrackingVertex doesn't exist in the output collection yet. However, it's already been
				// created in the addTrack() function and a temporary reference to it set in the TrackingParticle.
				// I'll use that reference to create a copy in the output collection. When the TrackingParticle
				// is added to the output collection a few lines below the temporary reference to the parent
				// vertex will be cleared, and the correct one referring to the output collection will be set.
				pProductionVertex=pOutput->addTrackingVertex( pDecayTrack->pParentVertex, *trackingParticle.parentVertex() );
			}


			pTrackingParticle=pOutput->addTrackingParticle( pDecayTrack, trackingParticle );
		}

		return pTrackingParticle;
	}

	void addTrack( ::DecayChainTrack* pDecayChainTrack, const TrackingParticleSelector* pSelector, ::OutputCollectionWrapper* pUnmergedOutput, ::OutputCollectionWrapper* pMergedOutput, const ::TrackingParticleFactory& objectFactory, bool addAncestors, const TrackerTopology *tTopo )
	{
		if( pDecayChainTrack==NULL ) return; // This is required for when the addAncestors_ recursive call reaches the top of the chain

		// Check to see if this particle has already been processed while traversing up the parents
		// of another split in the decay chain. The check in the line above only stops when the top
		// of the chain is reached, whereas this will stop when a previously traversed split is reached.
		{ // block to limit the scope of temporary variables
			bool alreadyProcessed=true;
			if( pUnmergedOutput!=NULL )
			{
				if( pUnmergedOutput->getTrackingParticle( pDecayChainTrack )==NULL ) alreadyProcessed=false;
			}
			if( pMergedOutput!=NULL )
			{
				if( pMergedOutput->getTrackingParticle( pDecayChainTrack )==NULL ) alreadyProcessed=false;
			}
			if( alreadyProcessed ) return;
		}

		// Create a TrackingParticle.
		TrackingParticle newTrackingParticle=objectFactory.createTrackingParticle( pDecayChainTrack, tTopo );

		// The selector checks the impact parameters from the vertex, so I need to have a valid reference
		// to the parent vertex in the TrackingParticle before that can be called. TrackingParticle needs
		// an edm::Ref for the parent TrackingVertex though. I still don't know if this is going to be
		// added to the collection so I can't take it from there, so I need to create a temporary one.
		// When the addTrackAndParentVertex() is called (assuming it passes selection) it will use the
		// temporary reference to create a copy of the parent vertex, put that in the output collection,
		// and then set the reference in the TrackingParticle properly.
		TrackingVertexCollection dummyCollection; // Only needed to create an edm::Ref
		dummyCollection.push_back( objectFactory.createTrackingVertex( pDecayChainTrack->pParentVertex) );
		TrackingVertexRef temporaryRef( &dummyCollection, 0 );
		newTrackingParticle.setParentVertex( temporaryRef );

		// If a selector has been supplied apply it on the new TrackingParticle and return if it fails.
		if( pSelector )
		{
			if( !(*pSelector)( newTrackingParticle ) ) return; // Return if the TrackingParticle fails selection
		}

		// Add the ancestors first (if required) so that the collection has some kind of chronological
		// order. I don't know how important that is but other code might assume chronological order.
		// If adding ancestors, no selection is applied. Note that I've already checked that all
		// DecayChainTracks have a pParentVertex.
		if( addAncestors ) addTrack( pDecayChainTrack->pParentVertex->pParentTrack, NULL, pUnmergedOutput, pMergedOutput, objectFactory, addAncestors, tTopo );

		// If creation of the unmerged collection has been turned off in the config this pointer
		// will be null.
		if( pUnmergedOutput!=NULL ) addTrackAndParentVertex( pDecayChainTrack, newTrackingParticle, pUnmergedOutput );

		// If creation of the merged collection has been turned off in the config this pointer
		// will be null.
		if( pMergedOutput!=NULL )
		{
			::DecayChainTrack* pBremParentChainTrack=pDecayChainTrack;
			while( pBremParentChainTrack->pMergedBremSource!=NULL ) pBremParentChainTrack=pBremParentChainTrack->pMergedBremSource;

			if( pBremParentChainTrack!=pDecayChainTrack )
			{
				TrackingParticle* pBremParentTrackingParticle=addTrackAndParentVertex( pBremParentChainTrack, newTrackingParticle, pMergedOutput );
				// The original particle in the bremsstrahlung decay chain has been
				// created (or retrieved if it already existed), now copy in the
				// extra information.
				// TODO - copy extra information.

				if( std::abs( newTrackingParticle.pdgId() ) == 22 )
				{
					// Photons are added as separate TrackingParticles, but with the production
					// vertex changed to be the production vertex of the original electron.

					// Set up a proxy, so that all requests for the parent TrackingVertex get
					// redirected to the brem parent's TrackingVertex.
					pMergedOutput->setProxy( pDecayChainTrack->pParentVertex, pBremParentChainTrack->pParentVertex );

					// Now that pMergedOutput thinks the parent vertex is the brem parent's vertex I can
					// call this and it will set the TrackingParticle parent vertex correctly to the brem
					// parent vertex.
					addTrackAndParentVertex( pDecayChainTrack, newTrackingParticle, pMergedOutput );
				}
				else if( std::abs( newTrackingParticle.pdgId() ) == 11 )
				{
					// Electrons have their G4 tracks and SimHits copied to the parent TrackingParticle.
					for( const auto& trackSegment : newTrackingParticle.g4Tracks() )
					{
						pBremParentTrackingParticle->addG4Track( trackSegment );
					}

					// Also copy the generator particle references
					for( const auto& genParticleRef : newTrackingParticle.genParticles() )
					{
						pBremParentTrackingParticle->addGenParticle( genParticleRef );
					}

					pBremParentTrackingParticle->setNumberOfHits(pBremParentTrackingParticle->numberOfHits()+newTrackingParticle.numberOfHits());
					pBremParentTrackingParticle->setNumberOfTrackerHits(pBremParentTrackingParticle->numberOfTrackerHits()+newTrackingParticle.numberOfTrackerHits());
					pBremParentTrackingParticle->setNumberOfTrackerLayers(pBremParentTrackingParticle->numberOfTrackerLayers()+newTrackingParticle.numberOfTrackerLayers());

					// Set a proxy in the output collection wrapper so that any attempt to get objects for
					// this DecayChainTrack again get redirected to the brem parent.
					pMergedOutput->setProxy( pDecayChainTrack, pBremParentChainTrack );
				}
			}
			else
			{
				// This is not the result of bremsstrahlung so add to the collection as normal.
				addTrackAndParentVertex( pDecayChainTrack, newTrackingParticle, pMergedOutput );
			}
		} // end of "if( pMergedOutput!=NULL )", i.e. end of "if bremsstrahlung merging is turned on"

	} // end of addTrack function

} // end of the unnamed namespace

// Register with the framework
DEFINE_DIGI_ACCUMULATOR (TrackingTruthAccumulator);
