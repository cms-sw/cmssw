#ifndef SimDataMixingGeneralTrackWorker_h
#define SimDataMixingGeneralTrackWorker_h

/** \class DataMixingGeneralTrackWorker
 *
 * DataMixingModule is the EDProducer subclass
 * that overlays rawdata events on top of MC,
 * using real data for pileup simulation
 * This class takes care of the GeneralTrack information
 *
 * \author Mike Hildreth, University of Notre Dame
 *
 * \version   1st Version October 2007
 *
 ************************************************************/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
// Data Formats
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <map>
#include <string>
#include <vector>

namespace edm {
  class ModuleCallingContext;

  class DataMixingGeneralTrackWorker {
  public:
    DataMixingGeneralTrackWorker();

    /** standard constructor*/
    explicit DataMixingGeneralTrackWorker(const edm::ParameterSet &ps, edm::ConsumesCollector &&iC);

    /**Default destructor*/
    virtual ~DataMixingGeneralTrackWorker();

    void putGeneralTrack(edm::Event &e);
    void addGeneralTrackSignals(const edm::Event &e);
    void addGeneralTrackPileups(const int bcr,
                                const edm::EventPrincipal *,
                                unsigned int EventId,
                                ModuleCallingContext const *);

  private:
    // data specifiers

    edm::InputTag GeneralTrackcollectionSig_;  // primary name given to collection
                                               // of GeneralTracks
    edm::InputTag GeneralTrackLabelSig_;       // secondary name given to collection of
                                               // GeneralTracks
    edm::InputTag GeneralTrackPileInputTag_;   // InputTag for pileup tracks
    std::string GeneralTrackCollectionDM_;     // secondary name to be given to new
                                               // GeneralTrack

    edm::EDGetTokenT<reco::TrackCollection> GTrackSigToken_;   // Token to retrieve information
    edm::EDGetTokenT<reco::TrackCollection> GTrackPileToken_;  // Token to retrieve information

    //

    std::unique_ptr<reco::TrackCollection> NewTrackList_;
  };
}  // namespace edm

#endif  // SimDataMixingGeneralTrackWorker_h
