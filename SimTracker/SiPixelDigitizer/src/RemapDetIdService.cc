#include <SimTracker/SiPixelDigitizer/interface/RemapDetIdService.h>

#include <iostream>
#include <fstream>

#include <FWCore/ServiceRegistry/interface/ServiceMaker.h> // Required for DEFINE_FWK_SERVICE
#include <FWCore/Framework/interface/Event.h>
#include <SimGeneral/MixingModule/interface/PileUpEventPrincipal.h>
#include <SimDataFormats/TrackingHit/interface/PSimHit.h>
#include <DataFormats/Provenance/interface/ProcessHistoryRegistry.h>
#include <DataFormats/Provenance/interface/Provenance.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>


namespace simtracker { namespace services { DEFINE_FWK_SERVICE( RemapDetIdService ); } }


simtracker::services::RemapDetIdService::RemapDetIdService( const edm::ParameterSet& parameterSet, edm::ActivityRegistry& activityRegister )
{
	std::vector<edm::InputTag> inputCollectionNames=parameterSet.getParameter<std::vector<edm::InputTag> >( "inputCollections" );
	// Put in a null pointer to a collection for each of the InputTags
	for( const auto& tag : inputCollectionNames ) remappedCollections_.push_back( std::make_pair(tag,std::unique_ptr<std::vector<PSimHit> >()) );

	cmsswVersionsToRemap_=parameterSet.getParameter<std::vector<std::string> >( "versionsToRemap" );

	edm::FileInPath remapFilename=parameterSet.getParameter<edm::FileInPath>( "mapFilename" );

	std::ifstream inputFile( remapFilename.fullPath() );
	if( !inputFile.is_open() ) throw std::runtime_error( "ModifyTrackerDetIds - unable to load the input file "+remapFilename.relativePath() );

	uint32_t oldDetId;
	uint32_t newDetId;
	while( inputFile.good() )
	{
		inputFile >> oldDetId >> newDetId;
		detIdMap_[oldDetId]=newDetId;
	}
}

simtracker::services::RemapDetIdService::~RemapDetIdService()
{
}

template<class T> bool simtracker::services::RemapDetIdService::getByLabel_( T& event, const edm::InputTag& inputTag, edm::Handle<std::vector<PSimHit> >& handle )
{
	edm::Handle<std::vector<PSimHit> > hInputCollection;

	if( !event.getByLabel( inputTag, hInputCollection ) )
	{
		handle=hInputCollection;
		return false;
	}

	// See if I should remap this collection
	std::vector<PSimHit>* pRemappedCollection=nullptr;
	for( auto& tagCollectionPointerPair : remappedCollections_ )
	{
		const auto& configuredTag=tagCollectionPointerPair.first;
		if( inputTag==configuredTag )
		{
			// The service has been configured to remap this collection. Also need to check
			// which version of CMSSW was used to create it.
			std::string cmsswVersion=cmsswVersionForProduct( hInputCollection );
			// Use "find" rather than absolute equality in case the string contains
			// e.g. "_patch3". I only want to remap the collection if it was made with
			// a version of CMSSW before SLHC13.
			for( const auto& cmsswVersionToCheck : cmsswVersionsToRemap_ )
			{
				if( cmsswVersion.find(cmsswVersionToCheck)!=std::string::npos )
				{
					// This collection should be remapped. I'll reset the collection and
					// set pRemappedCollection to the new one so that I know later to remap
					auto& collectionPointer=tagCollectionPointerPair.second;
					collectionPointer.reset( new std::vector<PSimHit>() );
					pRemappedCollection=collectionPointer.get();
					break;
				}
			}
		} // end of "if inputTag matches one set in the configuration"
	} // end of loop over InputTags

	// If I don't need to perform the remapping then pRemappedCollection will be null
	if( pRemappedCollection==nullptr )
	{
		handle=hInputCollection;
		return true;
	}

	edm::LogInfo("RemapDetIdService") << "Remapping the DetIds for collection " << inputTag;

	for( const auto& simHit : *hInputCollection )
	{
		uint32_t oldDetId=simHit.detUnitId();

		std::map<uint32_t,uint32_t>::const_iterator iFindResult=detIdMap_.find( oldDetId );
		if( iFindResult!=detIdMap_.end() ) // If the DetId is in the last of mappings
		{
			uint32_t newDetId=iFindResult->second;

			LogDebug("RemapDetIdService") << "    Modifying " << oldDetId << " to " << newDetId;
			PSimHit modifiedSimHit( simHit.entryPoint(), simHit.exitPoint(),
					simHit.pabs(), simHit.tof(), simHit.energyLoss(),
					simHit.particleType(), newDetId, simHit.trackId(),
					simHit.thetaAtEntry(), simHit.phiAtEntry(), simHit.processType() );

			modifiedSimHit.setEventId( simHit.eventId() );

			pRemappedCollection->push_back( std::move(modifiedSimHit) );
		}
		else // the DetId does not need to be remapped
		{
			LogDebug("RemapDetIdService") << "    Leaving " << oldDetId << " alone";
			pRemappedCollection->push_back( simHit );
		}

	} // end of loop over simhits

	handle=edm::Handle<std::vector<PSimHit> >( pRemappedCollection, hInputCollection.provenance() );
	return true;

}

bool simtracker::services::RemapDetIdService::getByLabel( const edm::Event& event, const edm::InputTag& inputTag, edm::Handle<std::vector<PSimHit> >& handle )
{
	return getByLabel_( event, inputTag, handle );
}

bool simtracker::services::RemapDetIdService::getByLabel( const PileUpEventPrincipal& event, const edm::InputTag& inputTag, edm::Handle<std::vector<PSimHit> >& handle )
{
	return getByLabel_( event, inputTag, handle );
}

template<class T> std::string simtracker::services::RemapDetIdService::cmsswVersionForProduct( const edm::Handle<T>& handle )
{
	if( !handle.isValid() ) return ""; // Should maybe throw an exception instead?

	const std::string processForHandle=handle.provenance()->processName();

	for( const auto& entry : *edm::ProcessHistoryRegistry::instance() )
	{
		for( const auto& processConfiguration : entry.second )
		{
			if( processConfiguration.processName()==processForHandle )
			{
				std::string cmsswVersion=processConfiguration.releaseVersion();
				// For some reason the release versions are given with quote marks around
				// them. I want to strip this off.
				if( cmsswVersion.front()=='"' && cmsswVersion.back()=='"' ) cmsswVersion=cmsswVersion.substr(1,cmsswVersion.size()-2);
				return cmsswVersion;
			} // End of "if process name matches the one for the handle"
		} // End of loop over edm::ProcessConfigurations
	} // End of loop over ProcessHistoryKeys

	// If control gets this far then the process name for the handle couldn't
	// be found in the registry. This should never happen.
	throw std::runtime_error( "cmsswVersionForProduct was queried for a handle who's process is not in the registry" );
}

