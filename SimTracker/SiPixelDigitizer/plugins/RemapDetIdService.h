#ifndef SimTracker_SiPixelDigitizer_RemapDetIdService_h
#define SimTracker_SiPixelDigitizer_RemapDetIdService_h

#include <string>
#include <map>
#include <vector>

#include <DataFormats/Common/interface/Handle.h> // Template so can't forward declare

//
// Forward declarations
//
class PSimHit;
namespace edm
{
	class ActivityRegistry;
	class ParameterSet;
	class Event;
	class InputTag;
}

namespace simtracker
{
	namespace services
	{
		/** @brief Service that provides a function to remap the DetIds in a SimHit collection.
		 *
		 * The DetId numbering in the forward pixel discs changed when going from 6_2_0_SLHC12 to
		 * 6_2_0_SLHC13. A huge set of production GEN-SIM samples was created in 6_2_0_SLHC11 however,
		 * and there isn't sufficient time to redo them. As a **temporary** hack the DetIds of the
		 * affected detectors can be remapped using the methods in this service.
		 *
		 * This code should be removed as soon as the ability to read the SLHC11 files is no longer
		 * required.
		 *
		 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
		 * @date 05/Jul/2014
		 */
		class RemapDetIdService
		{
		public:
			RemapDetIdService( const edm::ParameterSet& parameterSet, edm::ActivityRegistry& activityRegister );
			~RemapDetIdService();

			/** @brief Checks the provinence and returns the version of CMSSW that the given collection was created in */
			template<class T> std::string cmsswVersionForProduct( const edm::Handle<T>& handle );

			/** @brief Checks the collection given to see if it has been configured for remapping */
			bool collectionShouldBeRemapped( const edm::Handle<std::vector<PSimHit> >& handle );

			/** @brief Takes all of the SimHits in the handle supplied, and copies them to the vector changing DetIds as required.
			 *
			 * It is safe to call this on a collection that doesn't need remapping. A check is made first, if the collection
			 * doesn't need to be remapped then it is just copied straight to the vector.
			 */
			void remapCollection( const edm::Handle<std::vector<PSimHit> >& handle, std::vector<PSimHit>& returnValue );
		private:
			std::map<uint32_t,uint32_t> detIdMap_;
			std::vector<edm::InputTag> inputCollectionNames_;
			std::vector<std::string> cmsswVersionsToRemap_;
		};

	} // end of namespace services
} // end of namespace simtracker

#endif // end of ifndef SimTracker_SiPixelDigitizer_RemapDetIdService_h
