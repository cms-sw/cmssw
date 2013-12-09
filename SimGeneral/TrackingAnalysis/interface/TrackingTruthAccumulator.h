#ifndef TrackingAnalysis_TrackingTruthAccumulator_h
#define TrackingAnalysis_TrackingTruthAccumulator_h

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "SimGeneral/TrackingAnalysis/interface/TrackingParticleSelector.h"
#include <memory> // required for std::auto_ptr
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"


// Forward declarations
namespace edm
{
	class ParameterSet;
	class EDProducer;
	class Event;
	class EventSetup;
}
class PileUpEventPrincipal;
class PSimHit;



/** @brief Replacement for TrackingTruthProducer in the new pileup mixing setup.
 *
 * The configuration parameters are:
 *
 * <table>
 * <tr>
 *     <th>Parameter name                  </th><th> Type              </th><th> Description </th>
 * </tr>
 * <tr><td> volumeRadius                   </td><td> double            </td><td> The volume radius in cm used if ignoreTracksOutsideVolume is true. </td></tr>
 * <tr><td> volumeZ                        </td><td> double            </td><td> The volume z in cm used if ignoreTracksOutsideVolume is true. </td></tr>
 * <tr><td> ignoreTracksOutsideVolume      </td><td> bool              </td><td> If true, sim tracks that have a production vertex outside the volume specified by
 *                                                                               volumeRadius and volumeZ won't be turned into TrackingParticles. Doesn't make much
 *                                                                               difference to be honest, over a huge range of volume sizes so there must be a cut
 *                                                                               earlier in the simulation. </td></tr>
 * <tr><td> maximumPreviousBunchCrossing   </td><td> unsigned int      </td><td> Bunch crossings before this number (inclusive; use positive integer) won't be included.
 *                                                                               Setting to zero means only in-time. </td></tr>
 * <tr><td> maximumSubsequentBunchCrossing </td><td> unsigned int      </td><td> Bunch crossings after this won't create any TrackingParticles. </td></tr>
 * <tr><td> createUnmergedCollection       </td><td> bool              </td><td> Whether to create the TrackingParticle collection without bremsstrahlung merged. </td></tr>
 * <tr><td> createMergedBremsstrahlung     </td><td> bool              </td><td> Whether to create the TrackingParticle collection with bremsstrahlung merged. At
 *                                                                               least one of createUnmergedCollection or createMergedBremsstrahlung should be true
 *                                                                               otherwise nothing will be produced. </td></tr>
 * <tr><td> alwaysAddAncestors             </td><td> bool              </td><td> If a sim track passes selection and is turned into a TrackingParticle, all of it's
 *                                                                               parents will also be created even if they fail the selection. This was the default
 *                                                                               behaviour for the old TrackingParticleProducer. </td></tr>
 * <tr><td> removeDeadModules              </td><td> bool              </td><td> Hasn't been implemented yet (as of 22/May/2013). </td></tr>
 * <tr><td> simTrackCollection             </td><td> edm::InputTag     </td><td> The input SimTrack collection </td></tr>
 * <tr><td> simVertexCollection            </td><td> edm::InputTag     </td><td> The input SimVerted collection </td></tr>
 * <tr><td> simHitCollections              </td><td> edm::ParameterSet </td><td> A ParameterSet of vectors of InputTags that are the input PSimHits </td></tr>
 * <tr><td> genParticleCollection          </td><td> edm::InputTag     </td><td> The input reco::GenParticle collection. Note that there's a difference between
 *                                                                               reco::GenParticle and HepMC::GenParticle; the old TrackingTruthProducer used to
 *                                                                               use HepMC::GenParticle. </td></tr>
 * <tr><td> allowDifferentSimHitProcesses  </td><td> bool              </td><td> Should be false for FullSim and true for FastSim. There's more documentation in
 *                                                                               the code if you're really interested. </td></tr>
 * <tr><td> select                         </td><td> edm::ParameterSet </td><td> A ParameterSet used to configure a TrackingParticleSelector. If the TrackingParticle
 *                                                                               doesn't pass this selector then it's not added to the output. </td></tr>
 * </table>
 *
 * @author Mark Grimes (mark.grimes@bristol.ac.uk)
 * @date 11/Oct/2012
 */
class TrackingTruthAccumulator : public DigiAccumulatorMixMod
{
public:
	explicit TrackingTruthAccumulator( const edm::ParameterSet& config, edm::EDProducer& mixMod );
private:
	virtual void initializeEvent( const edm::Event& event, const edm::EventSetup& setup );
	virtual void accumulate( const edm::Event& event, const edm::EventSetup& setup );
	virtual void accumulate( const PileUpEventPrincipal& event, const edm::EventSetup& setup );
	virtual void finalizeEvent( edm::Event& event, const edm::EventSetup& setup );

	/** @brief Both forms of accumulate() delegate to this templated method. */
	template<class T> void accumulateEvent( const T& event, const edm::EventSetup& setup );

	/** @brief Fills the supplied vector with pointers to the SimHits, checking for bad modules if required */
	template<class T> void fillSimHits( std::vector<const PSimHit*>& returnValue, const T& event, const edm::EventSetup& setup );

	const std::string messageCategory_; ///< The message category used to send messages to MessageLogger

	const double volumeRadius_;
	const double volumeZ_;
	const bool ignoreTracksOutsideVolume_;

	/** The maximum bunch crossing BEFORE the signal crossing to create TrackinParticles for. Use positive values. If set to zero no
	 * previous bunches are added and only in-time, signal and after bunches (defined by maximumSubsequentBunchCrossing_) are used.*/
	const unsigned int maximumPreviousBunchCrossing_;
	/** The maximum bunch crossing AFTER the signal crossing to create TrackinParticles for. E.g. if set to zero only
	 * uses the signal and in time pileup (and previous bunches defined by the maximumPreviousBunchCrossing_ parameter). */
	const unsigned int maximumSubsequentBunchCrossing_;
	/// If bremsstrahlung merging, whether to also add the unmerged collection to the event or not.
	const bool createUnmergedCollection_;
	const bool createMergedCollection_;
	/// Whether or not to add the full parentage of any TrackingParticle that is inserted in the collection.
	const bool addAncestors_;

	/// As of 11/Feb/2013 this option hasn't been implemented yet.
	const bool removeDeadModules_;
	const edm::InputTag simTrackLabel_;
	const edm::InputTag simVertexLabel_;
	edm::ParameterSet simHitCollectionConfig_;
	edm::InputTag genParticleLabel_;

	bool selectorFlag_;
	TrackingParticleSelector selector_;
	/// Uses the same config as selector_, but can be used to drop out early since selector_ requires the TrackingParticle to be created first.
	bool chargedOnly_;
	/// Uses the same config as selector_, but can be used to drop out early since selector_ requires the TrackingParticle to be created first.
	bool signalOnly_;

	/** @brief When counting hits, allows hits in different detectors to have a different process type.
	 *
	 * Fast sim PSimHits seem to have a peculiarity where the process type (as reported by PSimHit::processType()) is
	 * different for the tracker than the muons. When counting how many hits there are, the code usually only counts
	 * the number of hits that have the same process type as the first hit. Setting this to true will also count hits
	 * that have the same process type as the first hit in the second detector.<br/>
	 */
	bool allowDifferentProcessTypeForDifferentDetectors_;
public:
	// These always go hand in hand, and I need to pass them around in the internal
	// functions, so I might as well package them up in a struct.
	struct OutputCollections
	{
		std::auto_ptr<TrackingParticleCollection> pTrackingParticles;
		std::auto_ptr<TrackingVertexCollection> pTrackingVertices;
		TrackingParticleRefProd refTrackingParticles;
		TrackingVertexRefProd refTrackingVertexes;
	};
private:
	OutputCollections unmergedOutput_;
	OutputCollections mergedOutput_;
};

#endif // end of "#ifndef TrackingAnalysis_TrackingTruthAccumulator_h"
