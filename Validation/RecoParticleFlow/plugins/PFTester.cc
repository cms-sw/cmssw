// author: Mike Schmitt, University of Florida
// first version 11/7/2007

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

class PFTester : public edm::one::EDAnalyzer<> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit PFTester(const edm::ParameterSet &);
  ~PFTester() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override;
  void endJob() override;

private:
  // DAQ Tools
  DQMStore *dbe_;
  std::map<std::string, MonitorElement *> me;

  // Inputs from Configuration File
  std::string outputFile_;
  edm::EDGetTokenT<reco::PFCandidateCollection> inputPFlowLabel_tok_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFTester);

using namespace edm;
using namespace std;
using namespace reco;

PFTester::PFTester(const edm::ParameterSet &iConfig) {
  inputPFlowLabel_tok_ = consumes<reco::PFCandidateCollection>(iConfig.getParameter<std::string>("InputPFlowLabel"));
  outputFile_ = iConfig.getUntrackedParameter<std::string>("OutputFile");

  if (!outputFile_.empty())
    edm::LogInfo("OutputInfo") << " ParticleFLow Task histograms will be saved to '" << outputFile_.c_str() << "'";
  else
    edm::LogInfo("OutputInfo") << " ParticleFlow Task histograms will NOT be saved";
}

PFTester::~PFTester() {}

void PFTester::beginJob() {
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();

  if (dbe_) {
    dbe_->setCurrentFolder("PFTask/PFCandidates");

    me["CandidateEt"] = dbe_->book1D("CandidateEt", "CandidateEt", 1000, 0, 1000);
    me["CandidateEta"] = dbe_->book1D("CandidateEta", "CandidateEta", 200, -5, 5);
    me["CandidatePhi"] = dbe_->book1D("CandidatePhi", "CandidatePhi", 200, -M_PI, M_PI);
    me["CandidateCharge"] = dbe_->book1D("CandidateCharge", "CandidateCharge", 5, -2, 2);
    me["PFCandidateType"] = dbe_->book1D("PFCandidateType", "PFCandidateType", 10, 0, 10);

    dbe_->setCurrentFolder("PFTask/PFBlocks");

    me["NumElements"] = dbe_->book1D("NumElements", "NumElements", 25, 0, 25);
    me["NumTrackElements"] = dbe_->book1D("NumTrackElements", "NumTrackElements", 5, 0, 5);
    me["NumPS1Elements"] = dbe_->book1D("NumPS1Elements", "NumPS1Elements", 5, 0, 5);
    me["NumPS2Elements"] = dbe_->book1D("NumPS2Elements", "NumPS2Elements", 5, 0, 5);
    me["NumECALElements"] = dbe_->book1D("NumECALElements", "NumECALElements", 5, 0, 5);
    me["NumHCALElements"] = dbe_->book1D("NumHCALElements", "NumHCALElements", 5, 0, 5);
    me["NumMuonElements"] = dbe_->book1D("NumMuonElements", "NumMuonElements", 5, 0, 5);

    dbe_->setCurrentFolder("PFTask/PFTracks");

    me["TrackCharge"] = dbe_->book1D("TrackCharge", "TrackCharge", 5, -2, 2);
    me["TrackNumPoints"] = dbe_->book1D("TrackNumPoints", "TrackNumPoints", 100, 0, 100);
    me["TrackNumMeasurements"] = dbe_->book1D("TrackNumMeasurements", "TrackNumMeasurements", 100, 0, 100);
    me["TrackImpactParameter"] = dbe_->book1D("TrackImpactParameter", "TrackImpactParameter", 1000, 0, 1);

    dbe_->setCurrentFolder("PFTask/PFClusters");
  }
}

void PFTester::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // Data to Retrieve from the Event
  const PFCandidateCollection *pflow_candidates;

  // ==========================================================
  // Retrieve!
  // ==========================================================

  {
    // Get Particle Flow Candidates
    Handle<PFCandidateCollection> pflow_hnd;
    iEvent.getByToken(inputPFlowLabel_tok_, pflow_hnd);
    pflow_candidates = pflow_hnd.product();
  }

  if (!pflow_candidates) {
    edm::LogInfo("OutputInfo") << " failed to retrieve data required by ParticleFlow Task";
    edm::LogInfo("OutputInfo") << " ParticleFlow Task cannot continue...!";
    return;
  }

  // ==========================================================
  // Analyze!
  // ==========================================================

  // Loop Over Particle Flow Candidates
  PFCandidateCollection::const_iterator pf;
  for (pf = pflow_candidates->begin(); pf != pflow_candidates->end(); pf++) {
    const PFCandidate *particle = &(*pf);

    // Fill Histograms for Candidate Methods
    me["CandidateEt"]->Fill(particle->et());
    me["CandidateEta"]->Fill(particle->eta());
    me["CandidatePhi"]->Fill(particle->phi());
    me["CandidateCharge"]->Fill(particle->charge());
    me["CandidatePdgId"]->Fill(particle->pdgId());

    // Fill Histograms for PFCandidate Specific Methods
    me["PFCandidateType"]->Fill(particle->particleId());
    // particle->elementsInBlocks();

    // Get the PFBlock and Elements
    // JW: Returns vector of blocks now ,TO BE FIXED ----
    /*PFBlock block = *(particle->block());
    OwnVector<PFBlockElement> elements = block.elements();
    int numElements = elements.size();
    int numTrackElements = 0;
    int numPS1Elements = 0;
    int numPS2Elements = 0;
    int numECALElements = 0;
    int numHCALElements = 0;
    int numMuonElements = 0;

    // Loop over Elements in Block
    OwnVector<PFBlockElement>::const_iterator element;
    for (element = elements.begin(); element != elements.end(); element++) {

      int element_type = element->type();
      // Element is a Tracker Track
      if (element_type == PFBlockElement::TRACK) {

        // Get General Information about the Track
        PFRecTrack track = *(element->trackRefPF());
        me["TrackCharge"]->Fill(track.charge());
        me["TrackNumPoints"]->Fill(track.nTrajectoryPoints());
        me["TrackNumMeasurements"]->Fill(track.nTrajectoryMeasurements());

        // Loop Over Points in the Track
        vector<PFTrajectoryPoint> points = track.trajectoryPoints();
        vector<PFTrajectoryPoint>::iterator point;
        for (point = points.begin(); point != points.end(); point++) {
          int point_layer = point->layer();
          double x = point->positionXYZ().x();
          double y = point->positionXYZ().y();
          double z = point->positionXYZ().z();
          //switch (point_layer) {
          //case PFTrajectoryPoint::ClosestApproach:
          // Fill the Track's D0
          if (point_layer == PFTrajectoryPoint::ClosestApproach) {
            me["TrackImpactParameter"]->Fill(sqrt(x*x + y*y + z*z));
          }
        }
        numTrackElements++;
      }

      // Element is an ECAL Cluster
      else if (element_type == PFBlockElement::ECAL) {
        numECALElements++;
      }
      // Element is a HCAL Cluster
      else if (element_type == PFBlockElement::HCAL) {
        numHCALElements++;
      }
      // Element is a Muon Track
      else if (element_type == PFBlockElement::MUON) {
        numMuonElements++;
      }
      // Fill the Respective Elements Sizes
      me["NumElements"]->Fill(numElements);
      me["NumTrackElements"]->Fill(numTrackElements);
      me["NumPS1Elements"]->Fill(numPS1Elements);
      me["NumPS2Elements"]->Fill(numPS2Elements);
      me["NumECALElements"]->Fill(numECALElements);
      me["NumHCALElements"]->Fill(numHCALElements);
      me["NumMuonElements"]->Fill(numMuonElements);
    } ----------------------------------------------  */
  }
}

void PFTester::endJob() {
  // Store the DAQ Histograms
  if (!outputFile_.empty() && dbe_)
    dbe_->save(outputFile_);
}
