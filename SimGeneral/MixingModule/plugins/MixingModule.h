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
#include "FWCore/Framework/interface/Selector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/Handle.h"

#include "MixingWorkerBase.h"

#include <vector>
#include <string>

class CrossingFramePlaybackInfoExtended;
class MixingWorkerBase;
namespace edm
{
  class MixingModule : public BMixingModule
    {
    public:

      /** standard constructor*/
      explicit MixingModule(const edm::ParameterSet& ps);

      /**Default destructor*/
      virtual ~MixingModule();

      virtual void beginJob() {}
 
    private:
      virtual void branchesActivate(const std::string &friendlyName, const std::string &subdet, InputTag &tag, std::string &label);
      virtual void put(edm::Event &e,const edm::EventSetup& es);
      virtual void createnewEDProduct();
      virtual void checkSignal(const edm::Event &e);
      virtual void addSignals(const edm::Event &e, const edm::EventSetup& es); 
      virtual void doPileUp(edm::Event &e, const edm::EventSetup& es);
      virtual void addPileups(const int bcr, EventPrincipal*,unsigned int EventId,unsigned int worker, const edm::EventSetup& e);
      virtual void setEventStartInfo(const unsigned int s); // set in CF-s
      virtual void getEventStartInfo(edm::Event & e, const unsigned int s); // fill in in base class

      std::string labelPlayback_;
      bool mixProdStep2_;
      bool mixProdStep1_;
      CrossingFramePlaybackInfoExtended *playbackInfo_;

      Selector * sel_;

      std::vector<MixingWorkerBase *> workers_;
      std::vector<MixingWorkerBase *> workersObjects_;
      std::vector<std::string> wantedBranches_;
      bool useCurrentProcessOnly_;
    };
}//edm

#endif
