#include "SimGeneral/TrackingAnalysis/test/TrackingTruthOutputTest.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;
typedef std::vector<TrackingParticle>                   TrackingParticleCollection;

typedef TrackingParticleRefVector::iterator               tp_iterator;
typedef TrackingVertex::genv_iterator                   genv_iterator;
typedef TrackingVertex::g4v_iterator                     g4v_iterator;

TrackingTruthOutputTest::TrackingTruthOutputTest(const edm::ParameterSet& conf){
  conf_ = conf;
}

void TrackingTruthOutputTest::analyze(const edm::Event& event, const edm::EventSetup& c){
  using namespace std;

  edm::Handle<TrackingParticleCollection> mergedPH;
  edm::Handle<TrackingVertexCollection>   mergedVH;

  edm::InputTag trackingTruth = conf_.getUntrackedParameter<edm::InputTag>("trackingTruth");

  event.getByLabel(trackingTruth, mergedPH);
  event.getByLabel(trackingTruth, mergedVH);

  if ( conf_.getUntrackedParameter<bool>("dumpVertexes") )
  {
    std::cout << std::endl << "Dumping merged vertices: " << std::endl;
    for (TrackingVertexCollection::const_iterator iVertex = mergedVH->begin(); iVertex != mergedVH->end(); ++iVertex) 
    {
      std::cout << std::endl << *iVertex;
      std::cout << "Daughters of this vertex:" << std::endl;
      for (tp_iterator iTrack = iVertex->daughterTracks_begin(); iTrack != iVertex->daughterTracks_end(); ++iTrack) 
        std::cout << **iTrack;
    }
    std::cout << std::endl;
  }

  if ( conf_.getUntrackedParameter<bool>("dumpOnlyBremsstrahlung") )
  {
     std::cout << std::endl << "Dumping only merged tracks: " << std::endl;
     for (TrackingParticleCollection::const_iterator iTrack = mergedPH->begin(); iTrack != mergedPH->end(); ++iTrack)
        if (iTrack->g4Tracks().size() > 1)
            std::cout << *iTrack << std::endl;
  }
  else
  {
    std::cout << std::endl << "Dump of merged tracks: " << std::endl;
    for (TrackingParticleCollection::const_iterator iTrack = mergedPH->begin(); iTrack != mergedPH->end(); ++iTrack)
        std::cout << *iTrack << std::endl;
  }
}


DEFINE_FWK_MODULE(TrackingTruthOutputTest);


