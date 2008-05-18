#include "SimGeneral/TrackingAnalysis/test/TrackingTruthOutputTest.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

typedef edm::RefVector< std::vector<TrackingParticle> > TrackingParticleContainer;
typedef std::vector<TrackingParticle>                   TrackingParticleCollection;

typedef TrackingParticleRefVector::iterator               tp_iterator;
typedef TrackingParticle::g4t_iterator                   g4t_iterator;
typedef TrackingParticle::genp_iterator                 genp_iterator;
typedef TrackingVertex::genv_iterator                   genv_iterator;
typedef TrackingVertex::g4v_iterator                     g4v_iterator;

TrackingTruthOutputTest::TrackingTruthOutputTest(const edm::ParameterSet& conf){
  conf_ = conf;
}

void TrackingTruthOutputTest::analyze(const edm::Event& event, const edm::EventSetup& c){
  using namespace std;

  edm::Handle<TrackingParticleCollection> mergedPH;
  edm::Handle<TrackingVertexCollection>   mergedVH;

  std::string trackingTruthModule = conf_.getUntrackedParameter<std::string>("trackingTruthModule");
  std::string trackingTruthProduct = conf_.getUntrackedParameter<std::string>("trackingTruthProduct"); 

  event.getByLabel(trackingTruthModule, trackingTruthProduct, mergedPH);
  event.getByLabel(trackingTruthModule, trackingTruthProduct, mergedVH);

  if ( conf_.getUntrackedParameter<bool>("dumpVertexes") )
  {
    cout << endl << "Dumping merged vertices: " << endl;
    for (TrackingVertexCollection::const_iterator iVertex = mergedVH->begin(); iVertex != mergedVH->end(); ++iVertex) 
    {
      cout << endl << *iVertex;
      cout << "Daughters of this vertex:" << endl;
      for (tp_iterator iTrack = iVertex->daughterTracks_begin(); iTrack != iVertex->daughterTracks_end(); ++iTrack) 
        cout << **iTrack;
    }
    cout << endl;
  }

  if ( conf_.getUntrackedParameter<bool>("dumpOnlyBremsstrahlung") )
  {
     cout << endl << "Dumping only merged tracks: " << endl;
     for (TrackingParticleCollection::const_iterator iTrack = mergedPH->begin(); iTrack != mergedPH->end(); ++iTrack)
        if (iTrack->g4Tracks().size() > 1)
          cout << *iTrack << endl;
  }
  else
  {
    cout << endl << "Dump of merged tracks: " << endl;
    for (TrackingParticleCollection::const_iterator iTrack = mergedPH->begin(); iTrack != mergedPH->end(); ++iTrack)
      cout << *iTrack << endl;
  }
}

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TrackingTruthOutputTest);


