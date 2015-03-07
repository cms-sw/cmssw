#ifndef MixingModule_h
#define MixingModule_h

/** \class MixingModule
 *
 * MixingModule is the EDProducer subclass 
 * which fills the CrossingFrame object to allow to add
 * pileup events in digitisations
 *
 * \author Ursula Berthon, LLR Palaiseau
 *
 * \version   1st Version June 2005
 * \version   2nd Version Sep 2005

 *
 ************************************************************/
#include "Mixing/Base/interface/BMixingModule.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"
#include <vector>
#include <string>

class CrossingFramePlaybackInfoNew;
class DigiAccumulatorMixMod;
class PileUpEventPrincipal;

namespace edm {
  class AdjusterBase;
  class ConsumesCollector;
  class MixingWorkerBase;
  class ModuleCallingContext;
  class StreamID;

  class MixingModule : public BMixingModule {
    public:
      typedef std::vector<DigiAccumulatorMixMod*> Accumulators;

      /** standard constructor*/
      explicit MixingModule(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~MixingModule();

      virtual void beginJob() {}

      virtual void beginRun(Run const& r1, EventSetup const& c) override;

      virtual void endRun(Run const& r1, EventSetup const& c) override;

      virtual void beginLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) override;

      virtual void endLuminosityBlock(LuminosityBlock const& l1, EventSetup const& c) override;

      void initializeEvent(Event const& event, EventSetup const& setup);

      void accumulateEvent(Event const& event, EventSetup const& setup);

      void accumulateEvent(PileUpEventPrincipal const& event, EventSetup const& setup, edm::StreamID const&);

      void finalizeEvent(Event& event, EventSetup const& setup);

      virtual void reload(const edm::EventSetup &);
 
    private:
      virtual void branchesActivate(const std::string &friendlyName, const std::string &subdet, InputTag &tag, std::string &label);
      virtual void put(edm::Event &e,const edm::EventSetup& es);
      virtual void createnewEDProduct();
      virtual void checkSignal(const edm::Event &e);
      virtual void addSignals(const edm::Event &e, const edm::EventSetup& es); 
      virtual void doPileUp(edm::Event &e, const edm::EventSetup& es) override;
      void pileAllWorkers(EventPrincipal const& ep, ModuleCallingContext const*, int bcr, int id, int& offset,
			  const edm::EventSetup& setup, edm::StreamID const&);
      void createDigiAccumulators(const edm::ParameterSet& mixingPSet, edm::ConsumesCollector& iC);

      InputTag inputTagPlayback_;
      bool mixProdStep2_;
      bool mixProdStep1_;
      CrossingFramePlaybackInfoNew *playbackInfo_;

      std::vector<AdjusterBase *> adjusters_;
      std::vector<AdjusterBase *> adjustersObjects_;
      std::vector<MixingWorkerBase *> workers_;
      std::vector<MixingWorkerBase *> workersObjects_;
      std::vector<std::string> wantedBranches_;
      bool useCurrentProcessOnly_;

      // Digi-producing algorithms
      Accumulators digiAccumulators_ ;

  };
}//edm

#endif
