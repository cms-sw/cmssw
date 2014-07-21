#ifndef SimTracker_SiPixelDigitizer_RemapDetIdService_h
#define SimTracker_SiPixelDigitizer_RemapDetIdService_h

#include <string>
#include <map>
#include <vector>
#include <memory>

#include <DataFormats/Common/interface/Handle.h> // Template so can't forward declare
//#include <SimDataFormats/CrossingFrame/interface/CrossingFrame.h>

#include <DataFormats/Provenance/interface/EventID.h>

//
// Forward declarations
//
class PSimHit;
class PileUpEventPrincipal;
template<class T> class CrossingFrame;

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
		 * Consumers that want a PSimHit collection with the DetIds remapped call the getByLabel
		 * proxy methods. This gets the collection from the event and checks whether it should be
		 * remapped. Things that are checked are whether this service was configured to remap a
		 * collection with that InputTag, and what version of CMSSW the requested collection was
		 * made with. If the collection was made with one of the ones listed in the config
		 * parameters (i.e. SLHC11 or SLHC12) then a new collection is created copying the SimHits
		 * but changing the DetId according to the mapping in the file specified in the config.
		 * Hence it is safe to call the method for collections that don't need to be remapped.
		 *
		 * Some consumers require the remapped collection to be persistent for the duration of
		 * its processing. E.g. TrackingTruthAccumulator takes pointers to the SimHits in the
		 * collection. The remapped collection will remain valid until the next call for the
		 * same InputTag. **This is in no way thread safe**. 6_2_0_SLHCX is not threaded anyway,
		 * and this code should be taken out long before we move to the threaded framework.
		 *
		 * N.B. The returned handle always has the provinence information of the original handle,
		 * regardless of whether it was remapped or not. This is probably not a good idea but I
		 * don't see much of an alternative.
		 *
		 * The configuration parameters are:
		 * <table>
		 * <tr><th>Parameter name    </th><th> Type                       </th><th> Description </th></tr>
		 * <tr><td> mapFilename      </td><td> edm::FileInPath            </td><td> The filename of the mapping between old DetId and new DetId </td></tr>
		 * <tr><td> inputCollections </td><td> std::vector<edm::InputTag> </td><td> The list of input collections that should be remapped </td></tr>
		 * <tr><td> versionsToRemap  </td><td> std::vector<std::string>   </td><td> The CMSSW versions that should be remapped. Note that the version the
		 *                                                                          collection was made with only needs to contain this string, so e.g.
		 *                                                                          setting this to "CMSSW_6_2_0_SLHC12" would also remap collections made
		 *                                                                          with CMSSW_6_2_0_SLHC12_patch1 </td></tr>
		 * </table>
		 *
 		 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
		 * @date 05/Jul/2014
		 */
		class RemapDetIdService
		{
		public:
			RemapDetIdService( const edm::ParameterSet& parameterSet, edm::ActivityRegistry& activityRegister );
			~RemapDetIdService();

			/** @brief Gets the collection and remaps the DetIds if required. Safe to call on all collections. */
			bool getByLabel( const edm::Event& event, const edm::InputTag& inputTag, edm::Handle<std::vector<PSimHit> >& handle );
			/** @brief Gets the collection and remaps the DetIds if required. Safe to call on all collections. */
			bool getByLabel( const PileUpEventPrincipal& event, const edm::InputTag& inputTag, edm::Handle<std::vector<PSimHit> >& handle );

			/** @brief Basically just returns the collection from the event, but prints a warning if the collection should have been remapped.
			 *
			 * TrackerHitAssociator needs a version to get the SimHit collection from the CrossingFrame. That would be quite a lot
			 * of code that's going to be taken out soon (as soon as we no longer need to reed the TP production samples). */
			bool getByLabel( const edm::Event& event, const edm::InputTag& inputTag, edm::Handle<CrossingFrame<PSimHit> >& handle );

			/** @brief Checks the provinence and returns the version of CMSSW that the given collection was created in */
			template<class T> std::string cmsswVersionForProduct( const edm::Handle<T>& handle );
		private:
			/** @brief Templated version that both of the public methods delegate to. */
			template<class T> bool getByLabel_( T& event, const edm::InputTag& inputTag, edm::Handle<std::vector<PSimHit> >& handle );
			std::map<uint32_t,uint32_t> detIdMap_;
			std::vector<std::string> cmsswVersionsToRemap_;

			/** @brief The remapped collections for each InputTag.
			 *
			 * As soon as another collection with the same InputTag is requested the stored remapped collection
			 * is freed and the remapping performed again. The assumption is that this will allow the remapped
			 * collection to persist for the duration that that the crossing is being processed.
			 */
			std::vector< std::pair<edm::InputTag, std::unique_ptr<std::vector<PSimHit> > > > remappedCollections_;
		};

	} // end of namespace services
} // end of namespace simtracker

#endif // end of ifndef SimTracker_SiPixelDigitizer_RemapDetIdService_h
