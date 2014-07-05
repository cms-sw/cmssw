#include "RemapDetIdService.h"

#include <iostream>
#include <fstream>

#include <FWCore/ServiceRegistry/interface/ServiceMaker.h> // Required for DEFINE_FWK_SERVICE
#include <SimDataFormats/TrackingHit/interface/PSimHit.h>
#include <DataFormats/Provenance/interface/ProcessHistoryRegistry.h>
#include <DataFormats/Provenance/interface/Provenance.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>


namespace simtracker { namespace services { DEFINE_FWK_SERVICE( RemapDetIdService ); } }


simtracker::services::RemapDetIdService::RemapDetIdService( const edm::ParameterSet& parameterSet, edm::ActivityRegistry& activityRegister )
{
	inputCollectionNames_=parameterSet.getParameter<std::vector<edm::InputTag> >( "inputCollections" );
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

bool simtracker::services::RemapDetIdService::collectionShouldBeRemapped( const edm::Handle<std::vector<PSimHit> >& handle )
{
	if( !handle.isValid() ) return false;

	// Run through all of the tags and see if this service has been configured to remap
	// the collection in the handle supplied.
	for( const auto& configuredTag : inputCollectionNames_ )
	{
		if( handle.provenance()->moduleLabel()==configuredTag.label()
				&& handle.provenance()->productInstanceName()==configuredTag.instance()
				&& (configuredTag.process()=="" || handle.provenance()->processName()==configuredTag.process()) )
		{
			// The service has been configured to remap this collection. Need to check
			// which version of CMSSW was used to create it.
			std::string cmsswVersion=cmsswVersionForProduct( handle );
			// Use "find" rather than absolute equality in case the string contains
			// e.g. "_patch3". I only want to remap the collection if it was made with
			// a version of CMSSW before SLHC13.
			if( cmsswVersion.find("CMSSW_6_2_0_SLHC11")!=std::string::npos ) return true;
			else if( cmsswVersion.find("CMSSW_6_2_0_SLHC12")!=std::string::npos ) return true;
			else return false;
		}
	}

	// If flow got this far then the collection has not been configured for remapping.
	return false;
}

void simtracker::services::RemapDetIdService::remapCollection( const edm::Handle<std::vector<PSimHit> >& handle, std::vector<PSimHit>& returnValue )
{
	if( collectionShouldBeRemapped(handle) )
	{
		edm::LogInfo("RemapDetIdService") << "Remapping the DetIds for collection " << handle.provenance()->moduleLabel() << " " << handle.provenance()->productInstanceName() << " " << handle.provenance()->processName();
		for( const auto& simHit : *handle )
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

				returnValue.push_back( std::move(modifiedSimHit) );
			}
			else // the DetId does not need to be remapped
			{
				LogDebug("RemapDetIdService") << "    Leaving " << oldDetId << " alone";
				returnValue.push_back( simHit );
			}

		} // end of loop over simhits
	}
	else returnValue=*handle; // If the collection isn't being remapped, just copy over the collection unchanged
}

