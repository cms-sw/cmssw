#ifndef TrackingAnalysis_TrackingTruthProducer_h
#define TrackingAnalysis_TrackingTruthProducer_h

#include <map>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimGeneral/TrackingAnalysis/interface/EncodedTruthId.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "Utilities/Timing/interface/TimerStack.h"

using namespace edm;
using namespace std;

class TrackingTruthProducer : public edm::EDProducer {

public:
  explicit TrackingTruthProducer( const edm::ParameterSet & );
//  ~TrackingTruthProducer() { TimingReport::current()->dump(std::cout); }

private:

  void produce( edm::Event &, const edm::EventSetup & );

  int LayerFromDetid(const unsigned int&);

  edm::ParameterSet conf_;

  double                   distanceCut_;
  std::vector<std::string> dataLabels_;
  std::vector<std::string> hitLabelsVector_;
  double                   volumeRadius_;
  double                   volumeZ_;
  bool                     discardOutVolume_;
  bool					   mergedBremsstrahlung_;  
  bool                     discardHitsFromDeltas_;
  std::string              simHitLabel_;

  std::string MessageCategory_;

  // Related to production

  // Encoded SimTrack to encoded source vertex
  map<EncodedTruthId,EncodedTruthId> simTrack_sourceV; 

  // Encoded SimTrack to PSimHit
  multimap<EncodedTruthId,PSimHit> simTrack_hit;

  // Encoded SimTrack to TrackingParticle index
  map<EncodedTruthId,int> simTrack_tP;

  //! Create a one to many association between simtracks and hits
  void simTrackHitsAssociator(
    std::auto_ptr<MixCollection<PSimHit> > &
  );

  //! Assamble the tracking particles in function of the simtrack collection    
  void trackingParticleAssambler(
    auto_ptr<TrackingParticleCollection> &,
    auto_ptr<MixCollection<SimTrack> > &,
    Handle<edm::HepMCProduct> const &
  );

  //! Assamble the tracking vertexes including parents-daughters relations
  void trackingVertexAssambler(
    auto_ptr<TrackingParticleCollection> &,
    auto_ptr<TrackingVertexCollection> &,
    auto_ptr<MixCollection<SimTrack> > &,  
    auto_ptr<MixCollection<SimVertex> > &,
    TrackingParticleRefProd &,
    TrackingVertexRefProd &,
    Handle<edm::HepMCProduct> const &
  );

  //! Merged Bremsstrahlung and copy the new collection into mergedTPC and mergedTVC
  void mergeBremsstrahlung(
    auto_ptr<TrackingParticleCollection> &,
    auto_ptr<TrackingVertexCollection>   &,
    auto_ptr<TrackingParticleCollection> &,
    auto_ptr<TrackingVertexCollection>   &,
    TrackingParticleRefProd &,
    TrackingVertexRefProd &
  );



   //! Verify is a given vertex is a Bremsstrahlung vertex
  bool isBremsstrahlungVertex(
    TrackingVertex const &,
    auto_ptr<TrackingParticleCollection> &
  );

};

#endif
