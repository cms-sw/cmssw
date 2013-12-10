#include <iostream>
#include <fstream>

#include "SimTracker/TrackAssociation/interface/QuickTrackAssociatorByHits.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

QuickTrackAssociatorByHits::QuickTrackAssociatorByHits( const edm::ParameterSet& config )
	: pHitAssociator_(NULL), pEventForWhichAssociatorIsValid_(NULL),
	  absoluteNumberOfHits_( config.getParameter<bool>( "AbsoluteNumberOfHits" ) ),
	  qualitySimToReco_( config.getParameter<double>( "Quality_SimToReco" ) ),
	  puritySimToReco_( config.getParameter<double>( "Purity_SimToReco" ) ),
	  cutRecoToSim_( config.getParameter<double>( "Cut_RecoToSim" ) ),
	  threeHitTracksAreSpecial_( config.getParameter<bool> ( "ThreeHitTracksAreSpecial" ) ),
          useClusterTPAssociation_(config.getParameter<bool>("useClusterTPAssociation")),
          cluster2TPSrc_(config.getParameter<edm::InputTag>("cluster2TPSrc"))
{
	//
	// Check whether the denominator when working out the percentage of shared hits should
	// be the number of simulated hits or the number of reconstructed hits.
	//
	std::string denominatorString=config.getParameter<std::string>("SimToRecoDenominator");
	if( denominatorString=="sim" ) simToRecoDenominator_=denomsim;
	else if( denominatorString=="reco" ) simToRecoDenominator_=denomreco;
	else throw cms::Exception( "QuickTrackAssociatorByHits" ) << "SimToRecoDenominator not specified as sim or reco";

	//
	// Set up the parameter set for the hit associator
	//
	hitAssociatorParameters_.addParameter<bool>( "associatePixel", config.getParameter<bool>("associatePixel") );
	hitAssociatorParameters_.addParameter<bool>( "associateStrip", config.getParameter<bool>("associateStrip") );
	// This is the important one, it stops the hit associator searching through the list of sim hits.
	// I only want to use the hit associator methods that work on the hit IDs (i.e. the uint32_t trackId
	// and the EncodedEventId eventId) so I'm not interested in matching that to the PSimHit objects.
	hitAssociatorParameters_.addParameter<bool>("associateRecoTracks",true);

	//
	// Do some checks on whether UseGrouped or UseSplitting have been set. They're not used
	// unlike the standard TrackAssociatorByHits so show a warning.
	//
	bool useGrouped, useSplitting;
	if( config.exists("UseGrouped") ) useGrouped=config.getParameter<bool>("UseGrouped");
	else useGrouped=true;

	if( config.exists("UseSplitting") ) useSplitting=config.getParameter<bool>("UseSplitting");
	else useSplitting=true;

	// This associator works as though both UseGrouped and UseSplitting were set to true, so show a
	// warning if this isn't the case.
	if( !(useGrouped && useSplitting) )
	{
		edm::LogWarning("QuickTrackAssociatorByHits") << "UseGrouped and/or UseSplitting has been set to false, but this associator ignores that setting.";
	}
}

QuickTrackAssociatorByHits::~QuickTrackAssociatorByHits()
{
	delete pHitAssociator_;
}

QuickTrackAssociatorByHits::QuickTrackAssociatorByHits( const QuickTrackAssociatorByHits& otherAssociator )
	: pEventForWhichAssociatorIsValid_(otherAssociator.pEventForWhichAssociatorIsValid_),
	  hitAssociatorParameters_(otherAssociator.hitAssociatorParameters_),
	  absoluteNumberOfHits_(otherAssociator.absoluteNumberOfHits_),
	  qualitySimToReco_(otherAssociator.qualitySimToReco_),
	  puritySimToReco_(otherAssociator.puritySimToReco_),
	  cutRecoToSim_(otherAssociator.cutRecoToSim_),
	  threeHitTracksAreSpecial_(otherAssociator.threeHitTracksAreSpecial_),
	  simToRecoDenominator_(otherAssociator.simToRecoDenominator_),
	  pTrackCollectionHandle_(otherAssociator.pTrackCollectionHandle_),
	  pTrackCollection_(otherAssociator.pTrackCollection_),
	  pTrackingParticleCollectionHandle_(otherAssociator.pTrackingParticleCollectionHandle_),
	  pTrackingParticleCollection_(otherAssociator.pTrackingParticleCollection_),
          useClusterTPAssociation_(otherAssociator.useClusterTPAssociation_),
          cluster2TPSrc_(otherAssociator.cluster2TPSrc_)
{
	// No operation other than the initialiser list. That copies everything straight from the other
	// associator, except for pHitAssociator_ which needs a deep copy or both instances will try
	// and free it on deletion.  If it wasn't for pHitAssociator_ the default copy constructor and
	// assignment operator would be sufficient.

	// Actually, need to check the other hit associator isn't null or the pointer dereference would
	// probably cause a segmentation fault.
	if( otherAssociator.pHitAssociator_ ) pHitAssociator_=new TrackerHitAssociator(*otherAssociator.pHitAssociator_);
	else pHitAssociator_=NULL;
}

QuickTrackAssociatorByHits& QuickTrackAssociatorByHits::operator=( const QuickTrackAssociatorByHits& otherAssociator )
{
	// Free up the old pHitAssociator_
	delete pHitAssociator_;

	//
	// pHitAssociator_ needs to be given a deep copy of the object, but everything else can
	// can be shallow copied from the other associator.
	//
	if( otherAssociator.pHitAssociator_ ) pHitAssociator_=new TrackerHitAssociator(*otherAssociator.pHitAssociator_);
	else pHitAssociator_=NULL;
	pEventForWhichAssociatorIsValid_=otherAssociator.pEventForWhichAssociatorIsValid_;
	hitAssociatorParameters_=otherAssociator.hitAssociatorParameters_;
	absoluteNumberOfHits_=otherAssociator.absoluteNumberOfHits_;
	qualitySimToReco_=otherAssociator.qualitySimToReco_;
	puritySimToReco_=otherAssociator.puritySimToReco_;
	cutRecoToSim_=otherAssociator.cutRecoToSim_;
	threeHitTracksAreSpecial_=otherAssociator.threeHitTracksAreSpecial_;
        useClusterTPAssociation_ = otherAssociator.useClusterTPAssociation_,
        cluster2TPSrc_ = otherAssociator.cluster2TPSrc_;
	simToRecoDenominator_=otherAssociator.simToRecoDenominator_;
	pTrackCollectionHandle_=otherAssociator.pTrackCollectionHandle_;
	pTrackCollection_=otherAssociator.pTrackCollection_;
	pTrackingParticleCollectionHandle_=otherAssociator.pTrackingParticleCollectionHandle_;
	pTrackingParticleCollection_=otherAssociator.pTrackingParticleCollection_;

	return *this;
}

reco::RecoToSimCollection QuickTrackAssociatorByHits::associateRecoToSim( edm::Handle<edm::View<reco::Track> >& trackCollectionHandle,
                                                                              edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle,
                                                                              const edm::Event* pEvent,
                                                                              const edm::EventSetup* pSetup ) const
{

        // get the Cluster2TPMap or initialize hit associator
        if (useClusterTPAssociation_) prepareCluster2TPMap(pEvent);
        else initialiseHitAssociator( pEvent );

	pTrackCollectionHandle_=&trackCollectionHandle;
	pTrackingParticleCollectionHandle_=&trackingParticleCollectionHandle;
	pTrackCollection_=NULL;
	pTrackingParticleCollection_=NULL;

	// This method checks which collection type is set to NULL, and uses the other one.
	return associateRecoToSimImplementation();
}

reco::SimToRecoCollection QuickTrackAssociatorByHits::associateSimToReco( edm::Handle<edm::View<reco::Track> >& trackCollectionHandle,
                                                                              edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle,
                                                                              const edm::Event * pEvent,
                                                                              const edm::EventSetup * pSetup ) const
{

        // get the Cluster2TPMap or initialize hit associator
        if (useClusterTPAssociation_) prepareCluster2TPMap(pEvent);
        else initialiseHitAssociator( pEvent );

	pTrackCollectionHandle_=&trackCollectionHandle;
	pTrackingParticleCollectionHandle_=&trackingParticleCollectionHandle;
	pTrackCollection_=NULL;
	pTrackingParticleCollection_=NULL;

	// This method checks which collection type is set to NULL, and uses the other one.
	return associateSimToRecoImplementation();
}


reco::RecoToSimCollection QuickTrackAssociatorByHits::associateRecoToSim(const edm::RefToBaseVector<reco::Track>& trackCollection,
									 const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection,
									 const edm::Event* pEvent,
									 const edm::EventSetup* pSetup ) const
{

  // get the Cluster2TPMap or initialize hit associator
  if (useClusterTPAssociation_) prepareCluster2TPMap(pEvent);
  else initialiseHitAssociator( pEvent );

  pTrackCollectionHandle_=NULL;
  pTrackingParticleCollectionHandle_=NULL;
  pTrackCollection_=&trackCollection;
  pTrackingParticleCollection_=&trackingParticleCollection;

  // This method checks which collection type is set to NULL, and uses the other one.
  return associateRecoToSimImplementation();
}

reco::SimToRecoCollection QuickTrackAssociatorByHits::associateSimToReco(const edm::RefToBaseVector<reco::Track>& trackCollection,
									 const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection,
									 const edm::Event* pEvent,
									 const edm::EventSetup* pSetup ) const
{

  // get the Cluster2TPMap or initialize hit associator
  if (useClusterTPAssociation_) prepareCluster2TPMap(pEvent);
  else initialiseHitAssociator( pEvent );

  pTrackCollectionHandle_=NULL;
  pTrackingParticleCollectionHandle_=NULL;
  pTrackCollection_=&trackCollection;
  pTrackingParticleCollection_=&trackingParticleCollection;

  // This method checks which collection type is set to NULL, and uses the other one.
  return associateSimToRecoImplementation();
}

reco::RecoToSimCollection QuickTrackAssociatorByHits::associateRecoToSimImplementation() const
{
  reco::RecoToSimCollection returnValue;

  size_t collectionSize;
  // Need to check which pointer is valid to get the collection size
  if ( pTrackCollection_ ) collectionSize=pTrackCollection_->size();
  else collectionSize=(*pTrackCollectionHandle_)->size();

  //std::cout << "#reco Tracks = " << collectionSize << std::endl;
  for( size_t i=0; i<collectionSize; ++i )
  {
    const reco::Track* pTrack; // Get a normal pointer for ease of use.
    if( pTrackCollection_ ) pTrack=&*(*pTrackCollection_)[i]; // Possibly the most obscure dereference I've ever had to write
    else pTrack=&(*pTrackCollectionHandle_->product())[i];
    //    std::cout << ">>> recoTrack #index = " << i << " pt = " << pTrack->pt() << std::endl;

    // The return of this function has first as the index and second as the number of associated hits
    std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs = (useClusterTPAssociation_) 
      ? associateTrackByCluster( pTrack->recHitsBegin(),pTrack->recHitsEnd() )
      : associateTrack( pTrack->recHitsBegin(), pTrack->recHitsEnd() );

    // int nt = 0;
    for (std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=trackingParticleQualityPairs.begin(); iTrackingParticleQualityPair!=trackingParticleQualityPairs.end(); ++iTrackingParticleQualityPair )
    {
      const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
      size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
      size_t numberOfValidTrackHits=pTrack->found();
	  
      //std::cout << ">>> reco2sim. numberOfSharedHits = " << nt++ << ", " << numberOfSharedHits << std::endl;
      if (numberOfSharedHits==0) continue; // No point in continuing if there was no association

      //if electron subtract double counting
      if (abs(trackingParticleRef->pdgId())==11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
      {
	numberOfSharedHits -= getDoubleCount( pTrack->recHitsBegin(), pTrack->recHitsEnd(), trackingParticleRef );
      }

      double quality;
      if (absoluteNumberOfHits_ ) 
        quality = static_cast<double>(numberOfSharedHits);
      else if (numberOfValidTrackHits != 0) 
        quality = (static_cast<double>(numberOfSharedHits) / static_cast<double>(numberOfValidTrackHits));
      else 
        quality = 0;
      if (quality > cutRecoToSim_ && !( threeHitTracksAreSpecial_ && numberOfValidTrackHits==3 && numberOfSharedHits<3 ) )
      {
	if (pTrackCollection_) returnValue.insert( (*pTrackCollection_)[i], std::make_pair( trackingParticleRef, quality ));
	else returnValue.insert( edm::RefToBase<reco::Track>(*pTrackCollectionHandle_,i), std::make_pair( trackingParticleRef, quality ));
      }
    }
  }
  return returnValue;
}

reco::SimToRecoCollection QuickTrackAssociatorByHits::associateSimToRecoImplementation() const
{
  reco::SimToRecoCollection returnValue;

	size_t collectionSize;
	// Need to check which pointer is valid to get the collection size
	if( pTrackCollection_ ) collectionSize=pTrackCollection_->size();
	else collectionSize=(*pTrackCollectionHandle_)->size();

	for( size_t i=0; i<collectionSize; ++i )
	{
		const reco::Track* pTrack; // Get a normal pointer for ease of use.
		if( pTrackCollection_ ) pTrack=&*(*pTrackCollection_)[i]; // Possibly the most obscure dereference I've ever had to write
		else pTrack=&(*pTrackCollectionHandle_->product())[i];

		// The return of this function has first as an edm:Ref to the associated TrackingParticle, and second as the number of associated hits
		std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs = (useClusterTPAssociation_) 
		? associateTrackByCluster( pTrack->recHitsBegin(),pTrack->recHitsEnd() )
		: associateTrack( pTrack->recHitsBegin(),pTrack->recHitsEnd() );

		// int nt = 0;
		for( std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=trackingParticleQualityPairs.begin();
				iTrackingParticleQualityPair!=trackingParticleQualityPairs.end(); ++iTrackingParticleQualityPair )
		{
			const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
			size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
			size_t numberOfValidTrackHits=pTrack->found();
			size_t numberOfSimulatedHits=0; // Set a few lines below, but only if required.

			//std::cout << ">>> sim2reco. numberOfSharedHits = " << nt++ << ", " << numberOfSharedHits << std::endl;
			if( numberOfSharedHits==0 ) continue; // No point in continuing if there was no association

			if( simToRecoDenominator_==denomsim || (numberOfSharedHits<3 && threeHitTracksAreSpecial_) ) // the numberOfSimulatedHits is not always required, so can skip counting in some circumstances
			{
				// Note that in the standard TrackAssociatorByHits, all of the hits in associatedTrackingParticleHits are checked for
				// various things.  I'm not sure what these checks are for but they depend on the UseGrouping and UseSplitting settings.
				// This associator works as though both UseGrouping and UseSplitting were set to true, i.e. just counts the number of
				// hits in the tracker.
				numberOfSimulatedHits=trackingParticleRef->numberOfTrackerHits();
			}


			//if electron subtract double counting
			if (abs(trackingParticleRef->pdgId())==11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
			{
			  numberOfSharedHits -= getDoubleCount( pTrack->recHitsBegin(), pTrack->recHitsEnd(), trackingParticleRef );
			}


			double purity=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfValidTrackHits);
			double quality;
			if( absoluteNumberOfHits_ ) quality=static_cast<double>(numberOfSharedHits);
			else if( simToRecoDenominator_==denomsim && numberOfSimulatedHits != 0 ) quality=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfSimulatedHits);
			else if( simToRecoDenominator_==denomreco && numberOfValidTrackHits != 0 ) quality=purity;
			else quality=0;

			if( quality>qualitySimToReco_ && !( threeHitTracksAreSpecial_ && numberOfSimulatedHits==3 && numberOfSharedHits<3 ) && ( absoluteNumberOfHits_ || (purity>puritySimToReco_) ) )
			{
				if( pTrackCollection_ ) returnValue.insert( trackingParticleRef, std::make_pair( (*pTrackCollection_)[i], quality ) );
				else returnValue.insert( trackingParticleRef, std::make_pair( edm::RefToBase<reco::Track>(*pTrackCollectionHandle_,i) , quality ) );
			}
		}
	}
	return returnValue;

}

template<typename iter> std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > QuickTrackAssociatorByHits::associateTrack( iter begin, iter end ) const
{
	// The pairs in this vector have a Ref to the associated TrackingParticle as "first" and the number of associated hits as "second"
	std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > returnValue;

	// The pairs in this vector have first as the sim track identifiers, and second the number of reco hits associated to that sim track.
	// Most reco hits will probably have come from the same sim track, so the number of entries in this vector should be fewer than the
	// number of reco hits.  The pair::second entries should add up to the total number of reco hits though.
	std::vector< std::pair<SimTrackIdentifiers,size_t> > hitIdentifiers=getAllSimTrackIdentifiers(begin, end);

	// Loop over the TrackingParticles
	size_t collectionSize;
	if( pTrackingParticleCollection_ ) collectionSize=pTrackingParticleCollection_->size();
	else collectionSize=(*pTrackingParticleCollectionHandle_)->size();

	for( size_t i=0; i<collectionSize; ++i )
	{
		const TrackingParticle* pTrackingParticle; // Convert to raw pointer for ease of use
		if( pTrackingParticleCollection_ ) pTrackingParticle=&*(*pTrackingParticleCollection_)[i];
		else pTrackingParticle=&(*pTrackingParticleCollectionHandle_->product())[i];

		// Ignore TrackingParticles with no hits
		if( pTrackingParticle->numberOfHits()==0 ) continue;

		size_t numberOfAssociatedHits=0;
		// Loop over all of the sim track identifiers and see if any of them are part of this TrackingParticle. If they are, add
		// the number of reco hits associated to that sim track to the total number of associated hits.
		for( std::vector< std::pair<SimTrackIdentifiers,size_t> >::const_iterator iIdentifierCountPair=hitIdentifiers.begin(); iIdentifierCountPair!=hitIdentifiers.end(); ++iIdentifierCountPair )
		{
			if( trackingParticleContainsIdentifier( pTrackingParticle, iIdentifierCountPair->first ) ) numberOfAssociatedHits+=iIdentifierCountPair->second;
		}

		if( numberOfAssociatedHits>0 )
		{
			if( pTrackingParticleCollection_ ) returnValue.push_back( std::make_pair( (*pTrackingParticleCollection_)[i], numberOfAssociatedHits ) );
			else returnValue.push_back( std::make_pair( edm::Ref<TrackingParticleCollection>( *pTrackingParticleCollectionHandle_, i ), numberOfAssociatedHits ) );
		}
	}

	return returnValue;
}

void QuickTrackAssociatorByHits::prepareCluster2TPMap(const edm::Event* pEvent) const
{
  //get the Cluster2TPList
  edm::Handle<ClusterTPAssociationList> pCluster2TPListH;
  pEvent->getByLabel(cluster2TPSrc_, pCluster2TPListH);
  if (pCluster2TPListH.isValid()) {
    pCluster2TPList_ = *(pCluster2TPListH.product());
    //make sure it is properly sorted
    std::sort(pCluster2TPList_.begin(), pCluster2TPList_.end(), clusterTPAssociationListGreater);
  } else {
    edm::LogInfo("TrackAssociator") << "ClusterTPAssociationList with label "<<cluster2TPSrc_<<" not found. Using DigiSimLink based associator";
    initialiseHitAssociator( pEvent );
    useClusterTPAssociation_ = false;
  }
}

template<typename iter> std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > QuickTrackAssociatorByHits::associateTrackByCluster(iter begin, iter end) const
{
  // The pairs in this vector have a Ref to the associated TrackingParticle as "first" and the number of associated clusters as "second"
  // Note: typedef edm::Ref<TrackingParticleCollection> TrackingParticleRef;
  std::vector<std::pair<edm::Ref<TrackingParticleCollection>, size_t> > returnValue;
  if (!pCluster2TPList_.size()) return returnValue;
  
  // The pairs in this vector have first as the TP, and second the number of reco clusters associated to that TP.
  // Most reco clusters will probably have come from the same sim track (i.e TP), so the number of entries in this 
  // vector should be fewer than the number of clusters. The pair::second entries should add up to the total 
  // number of reco clusters though.
  std::vector<OmniClusterRef> oClusters = getMatchedClusters(begin, end);

  std::map<TrackingParticleRef, size_t> lmap; 
  for (std::vector<OmniClusterRef>::const_iterator it = oClusters.begin(); it != oClusters.end(); ++it) {

    std::pair<OmniClusterRef, TrackingParticleRef> clusterTPpairWithDummyTP(*it,TrackingParticleRef());//TP is dummy: for clusterTPAssociationListGreater sorting only the cluster is needed
    auto range = std::equal_range(pCluster2TPList_.begin(), pCluster2TPList_.end(), clusterTPpairWithDummyTP, clusterTPAssociationListGreater);
    if(range.first != range.second) {
      for(auto ip = range.first; ip != range.second; ++ip) {

        const TrackingParticleRef trackingParticle = (ip->second);

        // Ignore TrackingParticles with no hits
        if (trackingParticle->numberOfHits()==0) continue;

	/* Alternative implementation to avoid the use of lmap... memory slightly improved but slightly slower...
	std::pair<edm::Ref<TrackingParticleCollection>,size_t> tpIntPair(trackingParticle, 1);
	auto tp_range = std::equal_range(returnValue.begin(), returnValue.end(), tpIntPair, tpIntPairGreater);
	if ((tp_range.second-tp_range.first)>1) {
	  edm::LogError("TrackAssociator") << ">>> Error in counting TPs!" << " file: " << __FILE__ << " line: " << __LINE__;
	}
	if(tp_range.first != tp_range.second) {
	  tp_range.first->second++;
	} else {
	  returnValue.push_back(tpIntPair);
	  std::sort(returnValue.begin(), returnValue.end(), tpIntPairGreater);
	}
	*/
        auto jpos = lmap.find(trackingParticle);     
	if (jpos != lmap.end())
	  ++jpos->second; 
        else 
	  lmap.insert(std::make_pair(trackingParticle, 1));
      }
    }
  }
  // now copy the map to returnValue
  for (auto ip = lmap.begin(); ip != lmap.end(); ++ip) {
    returnValue.push_back(std::make_pair(ip->first, ip->second));
  }
  return returnValue;
}

template<typename iter> std::vector<OmniClusterRef> QuickTrackAssociatorByHits::getMatchedClusters(iter begin, iter end) const
{
  std::vector<OmniClusterRef> returnValue;
  for (iter iRecHit = begin; iRecHit != end; ++iRecHit) {
    const TrackingRecHit* rhit = getHitFromIter(iRecHit);
    if (rhit->isValid()) {
      int subdetid = rhit->geographicalId().subdetId();
      if (subdetid==PixelSubdetector::PixelBarrel||subdetid==PixelSubdetector::PixelEndcap) {
	const SiPixelRecHit* pRHit = dynamic_cast<const SiPixelRecHit*>(rhit);
        if (!pRHit->cluster().isNonnull()) 
	  edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
        returnValue.push_back(pRHit->omniClusterRef());
      }
      else if (subdetid==SiStripDetId::TIB||subdetid==SiStripDetId::TOB||subdetid==SiStripDetId::TID||subdetid==SiStripDetId::TEC) {
	const std::type_info &tid = typeid(*rhit);
	if (tid == typeid(SiStripMatchedRecHit2D)) {
	  const SiStripMatchedRecHit2D* sMatchedRHit = dynamic_cast<const SiStripMatchedRecHit2D*>(rhit);
          if (!sMatchedRHit->monoHit().cluster().isNonnull() || !sMatchedRHit->stereoHit().cluster().isNonnull())  
	    edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
	  returnValue.push_back(sMatchedRHit->monoClusterRef()); 
	  returnValue.push_back(sMatchedRHit->stereoClusterRef()); 
        }
	else if (tid == typeid(SiStripRecHit2D)) {
	  const SiStripRecHit2D* sRHit = dynamic_cast<const SiStripRecHit2D*>(rhit);
          if (!sRHit->cluster().isNonnull()) 
	    edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
	  returnValue.push_back(sRHit->omniClusterRef());
	}
	else if (tid == typeid(SiStripRecHit1D)) {
	  const SiStripRecHit1D* sRHit = dynamic_cast<const SiStripRecHit1D*>(rhit);
          if (!sRHit->cluster().isNonnull()) 
	    edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
	  returnValue.push_back(sRHit->omniClusterRef());
	}
        else {
  	  edm::LogError("TrackAssociator") << ">>> getMatchedClusters: TrackingRecHit not associated to any SiStripCluster! subdetid = " << subdetid;
        }
      }
      else {
	edm::LogError("TrackAssociator") << ">>> getMatchedClusters: TrackingRecHit not associated to any cluster! subdetid = " << subdetid;
      }
    }
  }
  return returnValue;
}

template<typename iter> std::vector< std::pair<QuickTrackAssociatorByHits::SimTrackIdentifiers,size_t> > QuickTrackAssociatorByHits::getAllSimTrackIdentifiers( iter begin, iter end ) const
{
  // The pairs in this vector have first as the sim track identifiers, and second the number of reco hits associated to that sim track.
  std::vector< std::pair<SimTrackIdentifiers,size_t> > returnValue;

  std::vector<SimTrackIdentifiers> simTrackIdentifiers;
  // Loop over all of the rec hits in the track
  //iter tRHIterBeginEnd = getTRHIterBeginEnd( pTrack );
  for( iter iRecHit=begin; iRecHit!=end; ++iRecHit )
  {
    if( getHitFromIter(iRecHit)->isValid() )
    {
      simTrackIdentifiers.clear();
	  
      // Get the identifiers for the sim track that this hit came from. There should only be one entry unless clusters
      // have merged (as far as I know).
      pHitAssociator_->associateHitId( *(getHitFromIter(iRecHit)), simTrackIdentifiers ); // This call fills simTrackIdentifiers
      // Loop over each identifier, and add it to the return value only if it's not already in there
      for ( std::vector<SimTrackIdentifiers>::const_iterator iIdentifier  = simTrackIdentifiers.begin(); 
	                                                     iIdentifier != simTrackIdentifiers.end(); 
                                                           ++iIdentifier )
      {
	std::vector< std::pair<SimTrackIdentifiers,size_t> >::iterator iIdentifierCountPair;
	for (iIdentifierCountPair=returnValue.begin(); iIdentifierCountPair!=returnValue.end(); ++iIdentifierCountPair)
	{
	  if (iIdentifierCountPair->first.first==iIdentifier->first && iIdentifierCountPair->first.second==iIdentifier->second )
	  {
	    // This sim track identifier is already in the list, so increment the count of how many hits it relates to.
	    ++iIdentifierCountPair->second;
	    break;
	  }
	}
	if( iIdentifierCountPair==returnValue.end() ) returnValue.push_back( std::make_pair(*iIdentifier,1) ); 
	  // This identifier wasn't found, so add it
      }
    }
  }
  return returnValue;
}

bool QuickTrackAssociatorByHits::trackingParticleContainsIdentifier( const TrackingParticle* pTrackingParticle, const SimTrackIdentifiers& identifier ) const
{
  // Loop over all of the g4 tracks in the tracking particle
  for( std::vector<SimTrack>::const_iterator iSimTrack=pTrackingParticle->g4Track_begin(); iSimTrack!=pTrackingParticle->g4Track_end(); ++iSimTrack )
    {
      // And see if the sim track identifiers match
      if( iSimTrack->eventId()==identifier.second && iSimTrack->trackId()==identifier.first )
	{
	  return true;
	}
    }

  // If control has made it this far then none of the identifiers were found in
  // any of the g4 tracks, so return false.
  return false;
}

template<typename iter> int QuickTrackAssociatorByHits::getDoubleCount( iter startIterator, iter endIterator, TrackingParticleRef associatedTrackingParticle ) const
{
  // This method is largely copied from the standard TrackAssociatorByHits. Once I've tested how much difference
  // it makes I'll go through and comment it properly.

  int doubleCount=0;
  std::vector<SimHitIdpr> SimTrackIdsDC;

  for( iter iHit=startIterator; iHit != endIterator; iHit++ )
    {
      int idcount=0;

      if (useClusterTPAssociation_) {
	std::vector<OmniClusterRef> oClusters = getMatchedClusters(iHit, iHit+1);//only for the cluster being checked
	for (std::vector<OmniClusterRef>::const_iterator it = oClusters.begin(); it != oClusters.end(); ++it) {	  
	  std::pair<OmniClusterRef, TrackingParticleRef> clusterTPpairWithDummyTP(*it,TrackingParticleRef());//TP is dummy: for clusterTPAssociationListGreater sorting only the cluster is needed
	  auto range = std::equal_range(pCluster2TPList_.begin(), pCluster2TPList_.end(), clusterTPpairWithDummyTP, clusterTPAssociationListGreater);
	  if(range.first != range.second) {
	    for(auto ip = range.first; ip != range.second; ++ip) {
	      const TrackingParticleRef trackingParticle = (ip->second);
	      if (associatedTrackingParticle==trackingParticle) {
		idcount++;
	      }
	    }
	  }
	}
      } else {
	SimTrackIdsDC.clear();
	pHitAssociator_->associateHitId( *(getHitFromIter(iHit)), SimTrackIdsDC );
	if( SimTrackIdsDC.size() > 1 )
	  {
	    for( TrackingParticle::g4t_iterator g4T=associatedTrackingParticle->g4Track_begin(); g4T != associatedTrackingParticle->g4Track_end(); ++g4T )
	      {
		if( find( SimTrackIdsDC.begin(), SimTrackIdsDC.end(), SimHitIdpr( ( *g4T).trackId(), SimTrackIdsDC.begin()->second ) )
		    != SimTrackIdsDC.end() )
		  {
		    idcount++;
		  }
	      }
	  }
      }
      if( idcount > 1 ) doubleCount+=(idcount - 1);
    }
  
  return doubleCount;
}


void QuickTrackAssociatorByHits::initialiseHitAssociator( const edm::Event* pEvent ) const
{
	// The intention of this function was to check whether the hit associator is still valid
	// (since in general associateSimToReco and associateRecoToSim are called for the same
	// event). I was doing this by recording the event pointer and checking it hasn't changed
	// but this doesn't appear to work. Until I find a way of uniquely identifying an event
	// I'll just create it anew each time.
//	if( pEventForWhichAssociatorIsValid_==pEvent && pEventForWhichAssociatorIsValid_!=NULL ) return; // Already set up so no need to do anything

	// Free up the previous instantiation
	delete pHitAssociator_;

	// Create a new instantiation using the new event
	pHitAssociator_=new TrackerHitAssociator( *pEvent, hitAssociatorParameters_ );
	pEventForWhichAssociatorIsValid_=pEvent;
}



reco::RecoToSimCollectionSeed  
QuickTrackAssociatorByHits::associateRecoToSim(edm::Handle<edm::View<TrajectorySeed> >& pSeedCollectionHandle_,
					       edm::Handle<TrackingParticleCollection>&  trackingParticleCollectionHandle,     
					       const edm::Event * pEvent,
					       const edm::EventSetup *setup ) const{

  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHits::associateRecoToSim - #seeds="
                                      << pSeedCollectionHandle_->size()<<" #TPs="<<trackingParticleCollectionHandle->size();

  initialiseHitAssociator( pEvent );
  pTrackCollectionHandle_=NULL;
  pTrackingParticleCollectionHandle_=&trackingParticleCollectionHandle;
  pTrackCollection_=NULL;
  pTrackingParticleCollection_=NULL;

  reco::RecoToSimCollectionSeed  returnValue;

  size_t collectionSize=pSeedCollectionHandle_->size();
  
  for( size_t i=0; i<collectionSize; ++i )
    {
      const TrajectorySeed* pSeed = &(*pSeedCollectionHandle_)[i];
      
      // The return of this function has first as the index and second as the number of associated hits
      std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs =  (useClusterTPAssociation_) 
        ? associateTrackByCluster( pSeed->recHits().first,pSeed->recHits().second )
        : associateTrack( pSeed->recHits().first,pSeed->recHits().second );
      for( std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=trackingParticleQualityPairs.begin();
	   iTrackingParticleQualityPair!=trackingParticleQualityPairs.end(); ++iTrackingParticleQualityPair )
	{
	  const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
	  size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
	  size_t numberOfValidTrackHits=pSeed->recHits().second-pSeed->recHits().first;
	  
	  if( numberOfSharedHits==0 ) continue; // No point in continuing if there was no association
	  
	  //if electron subtract double counting
	  if( abs(trackingParticleRef->pdgId())==11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
	    {
	      numberOfSharedHits-=getDoubleCount( pSeed->recHits().first, pSeed->recHits().second, trackingParticleRef );
	    }
	  
	  double quality;
	  if( absoluteNumberOfHits_ ) quality=static_cast<double>( numberOfSharedHits );
	  else if( numberOfValidTrackHits != 0 ) quality=(static_cast<double>(numberOfSharedHits) / static_cast<double>(numberOfValidTrackHits) );
	  else quality=0;
	  
	  if( quality > cutRecoToSim_ && !( threeHitTracksAreSpecial_ && numberOfValidTrackHits==3 && numberOfSharedHits<3 ) )
	    {
	      returnValue.insert( edm::RefToBase<TrajectorySeed>(pSeedCollectionHandle_,i), std::make_pair( trackingParticleRef, quality ));
	    }
	}
    }
  
  LogTrace("TrackAssociator") << "% of Assoc Seeds=" << ((double)returnValue.size())/((double)pSeedCollectionHandle_->size());
  returnValue.post_insert();
  return returnValue;
  
}


reco::SimToRecoCollectionSeed
QuickTrackAssociatorByHits::associateSimToReco(edm::Handle<edm::View<TrajectorySeed> >& pSeedCollectionHandle_,
					       edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle, 
					       const edm::Event * pEvent,
					       const edm::EventSetup *setup ) const{

  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHits::associateSimToReco - #seeds="
                                      <<pSeedCollectionHandle_->size()<<" #TPs="<<trackingParticleCollectionHandle->size();

  initialiseHitAssociator( pEvent );
  pTrackCollectionHandle_=NULL;
  pTrackingParticleCollectionHandle_=&trackingParticleCollectionHandle;
  pTrackCollection_=NULL;
  pTrackingParticleCollection_=NULL;

  reco::SimToRecoCollectionSeed  returnValue;

  size_t collectionSize=pSeedCollectionHandle_->size();
  
  for( size_t i=0; i<collectionSize; ++i )
    {
      const TrajectorySeed* pSeed=&(*pSeedCollectionHandle_)[i];
      
      // The return of this function has first as an edm:Ref to the associated TrackingParticle, and second as the number of associated hits
      std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs =  (useClusterTPAssociation_) 
        ? associateTrackByCluster( pSeed->recHits().first,pSeed->recHits().second )
        : associateTrack( pSeed->recHits().first,pSeed->recHits().second );
      for( std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=trackingParticleQualityPairs.begin();
	   iTrackingParticleQualityPair!=trackingParticleQualityPairs.end(); ++iTrackingParticleQualityPair )
	{
	  const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
	  size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
	  size_t numberOfValidTrackHits=pSeed->recHits().second-pSeed->recHits().first;
	  size_t numberOfSimulatedHits=0; // Set a few lines below, but only if required.
	  
	  if( numberOfSharedHits==0 ) continue; // No point in continuing if there was no association

	  //if electron subtract double counting
	  if( abs(trackingParticleRef->pdgId())==11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
	    {
	      numberOfSharedHits-=getDoubleCount( pSeed->recHits().first, pSeed->recHits().second, trackingParticleRef );
	    }
	  
	  if( simToRecoDenominator_==denomsim || (numberOfSharedHits<3 && threeHitTracksAreSpecial_) ) // the numberOfSimulatedHits is not always required, so can skip counting in some circumstances
	    {
	      // Note that in the standard TrackAssociatorByHits, all of the hits in associatedTrackingParticleHits are checked for
	      // various things.  I'm not sure what these checks are for but they depend on the UseGrouping and UseSplitting settings.
	      // This associator works as though both UseGrouping and UseSplitting were set to true, i.e. just counts the number of
	      // hits in the tracker.
	      numberOfSimulatedHits=trackingParticleRef->numberOfTrackerHits();
	    }
	  
	  double purity=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfValidTrackHits);
	  double quality;
	  if( absoluteNumberOfHits_ ) quality=static_cast<double>(numberOfSharedHits);
	  else if( simToRecoDenominator_==denomsim && numberOfSimulatedHits != 0 ) quality=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfSimulatedHits);
	  else if( simToRecoDenominator_==denomreco && numberOfValidTrackHits != 0 ) quality=purity;
	  else quality=0;
	  
	  if( quality>qualitySimToReco_ && !( threeHitTracksAreSpecial_ && numberOfSimulatedHits==3 && numberOfSharedHits<3 ) && ( absoluteNumberOfHits_ || (purity>puritySimToReco_) ) )
	    {
	      returnValue.insert( trackingParticleRef, std::make_pair( edm::RefToBase<TrajectorySeed>(pSeedCollectionHandle_,i) , quality ) );
	    }
	}
    }
  return returnValue;
  
  LogTrace("TrackAssociator") << "% of Assoc TPs=" << ((double)returnValue.size())/((double)trackingParticleCollectionHandle->size());
  returnValue.post_insert();
  return returnValue;
}
