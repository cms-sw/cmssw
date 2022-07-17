// -*- C++ -*-
//
// Package:    KineExample
// Class:      KineExample
//
/**\class KineExample KineExample.cc RecoVertex/KineExample/src/KineExample.cc

 Description: steers tracker primary vertex reconstruction and storage

 Implementation:
     <Notes on implementation>
*/

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include <RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h>
//#include "MagneticField/Engine/interface/MagneticField.h"
//#include "MagneticField/Records/interface/IdealMagneticField.h"

#include <TFile.h>

/**
   * This is a very simple test analyzer mean to test the KalmanVertexFitter
   */

class KineExample : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit KineExample(const edm::ParameterSet&);
  ~KineExample() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override{};

private:
  void printout(const RefCountedKinematicVertex& myVertex) const;
  void printout(const RefCountedKinematicParticle& myParticle) const;
  void printout(const RefCountedKinematicTree& myTree) const;

  TrackingVertex getSimVertex(const edm::Event& iEvent) const;

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> estoken_ttk;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> estoken_mf;

  edm::ParameterSet theConfig;
  edm::ParameterSet kvfPSet;
  //   TFile*  rootFile_;

  std::string outputFile_;  // output file
  edm::EDGetTokenT<reco::TrackCollection> token_tracks;
  //   edm::EDGetTokenT<TrackingParticleCollection> token_TrackTruth;
  edm::EDGetTokenT<TrackingVertexCollection> token_VertexTruth;
};

using namespace reco;
using namespace edm;
using namespace std;

KineExample::KineExample(const edm::ParameterSet& iConfig)
    : estoken_ttk(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      estoken_mf(esConsumes<edm::Transition::BeginRun>()) {
  token_tracks = consumes<TrackCollection>(iConfig.getParameter<string>("TrackLabel"));
  outputFile_ = iConfig.getUntrackedParameter<std::string>("outputFile");
  kvfPSet = iConfig.getParameter<edm::ParameterSet>("KVFParameters");
  //   rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE");
  edm::LogInfo("RecoVertex/KineExample") << "Initializing KVF TEST analyser  - Output file: " << outputFile_ << "\n";
  //   token_TrackTruth = consumes<TrackingParticleCollection>(edm::InputTag("trackingtruth", "TrackTruth"));
  token_VertexTruth = consumes<TrackingVertexCollection>(edm::InputTag("trackingtruth", "VertexTruth"));
}

KineExample::~KineExample() = default;

void KineExample::beginRun(Run const& run, EventSetup const& setup) {
  //const MagneticField* magField = &setup.getData(estoken_mf);
  //tree.reset(new SimpleVertexTree("VertexFitter", magField.product()));
}

//
// member functions
//

void KineExample::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  try {
    edm::LogPrint("KineExample") << "Reconstructing event number: " << iEvent.id() << "\n";

    // get RECO tracks from the event
    // `tks` can be used as a ptr to a reco::TrackCollection
    edm::Handle<reco::TrackCollection> tks;
    iEvent.getByToken(token_tracks, tks);
    if (!tks.isValid()) {
      edm::LogPrint("KineExample") << "Couln't find track collection: " << iEvent.id() << "\n";
    } else {
      edm::LogInfo("RecoVertex/KineExample") << "Found: " << (*tks).size() << " reconstructed tracks"
                                             << "\n";
      edm::LogPrint("KineExample") << "got " << (*tks).size() << " tracks " << endl;

      // Transform Track to TransientTrack
      //get the builder:
      edm::ESHandle<TransientTrackBuilder> theB = iSetup.getHandle(estoken_ttk);
      //do the conversion:
      vector<TransientTrack> t_tks = (*theB).build(tks);

      edm::LogPrint("KineExample") << "Found: " << t_tks.size() << " reconstructed tracks"
                                   << "\n";

      // Do a KindFit, if >= 4 tracks.
      if (t_tks.size() > 3) {
        // For a first test, suppose that the first four tracks are the 2 muons,
        // then the 2 kaons. Since this will not be true, the result of the fit
        // will not be meaningfull, but at least you will get the idea of how to
        // do such a fit.

        //First, to get started, a simple vertex fit:

        vector<TransientTrack> ttv;
        ttv.push_back(t_tks[0]);
        ttv.push_back(t_tks[1]);
        ttv.push_back(t_tks[2]);
        ttv.push_back(t_tks[3]);
        KalmanVertexFitter kvf(false);
        TransientVertex tv = kvf.vertex(ttv);
        if (!tv.isValid())
          edm::LogPrint("KineExample") << "KVF failed\n";
        else
          edm::LogPrint("KineExample") << "KVF fit Position: " << Vertex::Point(tv.position()) << "\n";

        TransientTrack ttMuPlus = t_tks[0];
        TransientTrack ttMuMinus = t_tks[1];
        TransientTrack ttKPlus = t_tks[2];
        TransientTrack ttKMinus = t_tks[3];

        //the final state muons and kaons from the Bs->J/PsiPhi->mumuKK decay
        //Creating a KinematicParticleFactory
        KinematicParticleFactoryFromTransientTrack pFactory;

        //The mass of a muon and the insignificant mass sigma to avoid singularities in the covariance matrix.
        ParticleMass muon_mass = 0.1056583;
        ParticleMass kaon_mass = 0.493677;
        float muon_sigma = 0.0000001;
        float kaon_sigma = 0.000016;

        //initial chi2 and ndf before kinematic fits. The chi2 of the reconstruction is not considered
        float chi = 0.;
        float ndf = 0.;

        //making particles
        vector<RefCountedKinematicParticle> muonParticles;
        vector<RefCountedKinematicParticle> phiParticles;
        vector<RefCountedKinematicParticle> allParticles;
        muonParticles.push_back(pFactory.particle(ttMuPlus, muon_mass, chi, ndf, muon_sigma));
        muonParticles.push_back(pFactory.particle(ttMuMinus, muon_mass, chi, ndf, muon_sigma));
        allParticles.push_back(pFactory.particle(ttMuPlus, muon_mass, chi, ndf, muon_sigma));
        allParticles.push_back(pFactory.particle(ttMuMinus, muon_mass, chi, ndf, muon_sigma));

        phiParticles.push_back(pFactory.particle(ttKPlus, kaon_mass, chi, ndf, kaon_sigma));
        phiParticles.push_back(pFactory.particle(ttKMinus, kaon_mass, chi, ndf, kaon_sigma));
        allParticles.push_back(pFactory.particle(ttKPlus, kaon_mass, chi, ndf, kaon_sigma));
        allParticles.push_back(pFactory.particle(ttKMinus, kaon_mass, chi, ndf, kaon_sigma));

        /* Example of a simple vertex fit, without other constraints
       * The reconstructed decay tree is a result of the kinematic fit
       * The KinematicParticleVertexFitter fits the final state particles to their vertex and
       * reconstructs the decayed state
       */
        KinematicParticleVertexFitter fitter;
        edm::LogPrint("KineExample") << "Simple vertex fit with KinematicParticleVertexFitter:\n";
        RefCountedKinematicTree vertexFitTree = fitter.fit(allParticles);

        printout(vertexFitTree);

        /////Example of global fit:

        //creating the constraint for the J/Psi mass
        ParticleMass jpsi = 3.09687;

        //creating the two track mass constraint
        MultiTrackKinematicConstraint* j_psi_c = new TwoTrackMassKinematicConstraint(jpsi);

        //creating the fitter
        KinematicConstrainedVertexFitter kcvFitter;

        //obtaining the resulting tree
        RefCountedKinematicTree myTree = kcvFitter.fit(allParticles, j_psi_c);

        edm::LogPrint("KineExample") << "\nGlobal fit done:\n";
        printout(myTree);

        //creating the vertex fitter
        KinematicParticleVertexFitter kpvFitter;

        //reconstructing a J/Psi decay
        RefCountedKinematicTree jpTree = kpvFitter.fit(muonParticles);

        //creating the particle fitter
        KinematicParticleFitter csFitter;

        // creating the constraint
        float jp_m_sigma = 0.00004;
        KinematicConstraint* jpsi_c2 = new MassKinematicConstraint(jpsi, jp_m_sigma);

        //the constrained fit:
        jpTree = csFitter.fit(jpsi_c2, jpTree);

        //getting the J/Psi KinematicParticle and putting it together with the kaons.
        //The J/Psi KinematicParticle has a pointer to the tree it belongs to
        jpTree->movePointerToTheTop();
        RefCountedKinematicParticle jpsi_part = jpTree->currentParticle();
        phiParticles.push_back(jpsi_part);

        //making a vertex fit and thus reconstructing the Bs parameters
        // the resulting tree includes all the final state tracks, the J/Psi meson,
        // its decay vertex, the Bs meson and its decay vertex.
        RefCountedKinematicTree bsTree = kpvFitter.fit(phiParticles);
        edm::LogPrint("KineExample") << "Sequential fit done:\n";
        printout(bsTree);

        //       // For the analysis: compare to your SimVertex
        //       TrackingVertex sv = getSimVertex(iEvent);
        //   edm::Handle<TrackingParticleCollection>  TPCollectionH ;
        //   iEvent.getByToken(token_TrackTruth, TPCollectionH);
        //   const TrackingParticleCollection tPC = *(TPCollectionH.product());
        //       reco::RecoToSimCollection recSimColl=associatorForParamAtPca->associateRecoToSim(tks,
        // 									      TPCollectionH,
        // 									      &iEvent,
        //                                                                            &iSetup);
        //
        //       tree->fill(tv, &sv, &recSimColl);
        //     }
      }
    }
  } catch (cms::Exception& err) {
    edm::LogError("KineExample") << "Exception during event number: " << iEvent.id() << "\n" << err.what() << "\n";
  }
}

void KineExample::printout(const RefCountedKinematicVertex& myVertex) const {
  if (myVertex->vertexIsValid()) {
    edm::LogPrint("KineExample") << "Decay vertex: " << myVertex->position() << myVertex->chiSquared() << " "
                                 << myVertex->degreesOfFreedom() << endl;
  } else
    edm::LogPrint("KineExample") << "Decay vertex Not valid\n";
}

void KineExample::printout(const RefCountedKinematicParticle& myParticle) const {
  edm::LogPrint("KineExample") << "Particle: \n";
  //accessing the reconstructed Bs meson parameters:
  //SK: uncomment if needed  AlgebraicVector7 bs_par = myParticle->currentState().kinematicParameters().vector();

  //and their joint covariance matrix:
  //SK:uncomment if needed  AlgebraicSymMatrix77 bs_er = myParticle->currentState().kinematicParametersError().matrix();
  edm::LogPrint("KineExample") << "Momentum at vertex: " << myParticle->currentState().globalMomentum() << endl;
  edm::LogPrint("KineExample") << "Parameters at vertex: " << myParticle->currentState().kinematicParameters().vector()
                               << endl;
}

void KineExample::printout(const RefCountedKinematicTree& myTree) const {
  if (!myTree->isValid()) {
    edm::LogPrint("KineExample") << "Tree is invalid. Fit failed.\n";
    return;
  }

  //accessing the tree components, move pointer to top
  myTree->movePointerToTheTop();

  //We are now at the top of the decay tree getting the B_s reconstructed KinematicPartlcle
  RefCountedKinematicParticle b_s = myTree->currentParticle();
  printout(b_s);

  // The B_s decay vertex
  RefCountedKinematicVertex b_dec_vertex = myTree->currentDecayVertex();
  printout(b_dec_vertex);

  // Get all the children of Bs:
  //In this way, the pointer is not moved
  vector<RefCountedKinematicParticle> bs_children = myTree->finalStateParticles();

  for (unsigned int i = 0; i < bs_children.size(); ++i) {
    printout(bs_children[i]);
  }

  //Now navigating down the tree , pointer is moved:
  bool child = myTree->movePointerToTheFirstChild();

  if (child)
    while (myTree->movePointerToTheNextChild()) {
      RefCountedKinematicParticle aChild = myTree->currentParticle();
      printout(aChild);
    }
}

//Returns the first vertex in the list.

TrackingVertex KineExample::getSimVertex(const edm::Event& iEvent) const {
  // get the simulated vertices
  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(token_VertexTruth, TVCollectionH);
  const TrackingVertexCollection tPC = *(TVCollectionH.product());

  //    Handle<edm::SimVertexContainer> simVtcs;
  //    iEvent.getByLabel("g4SimHits", simVtcs);
  //    edm::LogPrint("KineExample") << "SimVertex " << simVtcs->size() << std::endl;
  //    for(edm::SimVertexContainer::const_iterator v=simVtcs->begin();
  //        v!=simVtcs->end(); ++v){
  //      edm::LogPrint("KineExample") << "simvtx "
  // 	       << v->position().x() << " "
  // 	       << v->position().y() << " "
  // 	       << v->position().z() << " "
  // 	       << v->parentIndex() << " "
  // 	       << v->noParent() << " "
  //               << std::endl;
  //    }
  return *(tPC.begin());
}
DEFINE_FWK_MODULE(KineExample);
