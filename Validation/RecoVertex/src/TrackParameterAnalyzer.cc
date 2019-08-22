#include "Validation/RecoVertex/interface/TrackParameterAnalyzer.h"

//system includes
#include <memory>
#include <vector>

// core framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

// Hep MC stuff from CLHEP
#include "CLHEP/Vector/LorentzVector.h"

// track
#include "DataFormats/TrackReco/interface/Track.h"

// Root
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>

//
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackParameterAnalyzer::TrackParameterAnalyzer(const edm::ParameterSet& iConfig)
    : edmSimVertexContainerToken_(consumes<edm::SimVertexContainer>(iConfig.getParameter<edm::InputTag>("simG4"))),
      edmSimTrackContainerToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("simG4"))),
      recoTrackCollectionToken_(consumes<reco::TrackCollection>(
          edm::InputTag(iConfig.getUntrackedParameter<std::string>("recoTrackProducer")))),
      outputFile_(iConfig.getUntrackedParameter<std::string>("outputFile")),
      simUnit_(1.0)  //  starting from  CMSSW_1_2_x, I think
      ,
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)) {
  //now do whatever initialization is needed
  // open output file to store histograms}
  auto tversion = edm::getReleaseVersion();
  tversion = tversion.erase(tversion.size() - 1, 1).erase(0, 1);
  outputFile_ = tversion + "_" + outputFile_;
  rootFile_ = TFile::Open(outputFile_.c_str(), "RECREATE");
  if ((edm::getReleaseVersion()).find("CMSSW_1_1_", 0) != std::string::npos) {
    simUnit_ = 0.1;  // for use in  CMSSW_1_1_1 tutorial
  }
}

TrackParameterAnalyzer::~TrackParameterAnalyzer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  delete rootFile_;
}

//
// member functions
//
void TrackParameterAnalyzer::beginJob() {
  std::cout << " TrackParameterAnalyzer::beginJob  conversion from sim units to rec units is " << simUnit_ << std::endl;

  rootFile_->cd();
  h1_pull0_ = new TH1F("pull0", "pull q/p", 100, -25., 25.);
  h1_pull1_ = new TH1F("pull1", "pull lambda", 100, -25., 25.);
  h1_pull2_ = new TH1F("pull2", "pull phi  ", 100, -25., 25.);
  h1_pull3_ = new TH1F("pull3", "pull dca  ", 100, -25., 25.);
  h1_pull4_ = new TH1F("pull4", "pull zdca ", 100, -25., 25.);

  h1_res0_ = new TH1F("res0", "res q/p", 100, -0.1, 0.1);
  h1_res1_ = new TH1F("res1", "res lambda", 100, -0.1, 0.1);
  h1_res2_ = new TH1F("res2", "res phi  ", 100, -0.1, 0.1);
  h1_res3_ = new TH1F("res3", "res dca  ", 100, -0.1, 0.1);
  h1_res4_ = new TH1F("res4", "res zdca ", 100, -0.1, 0.1);

  h1_Beff_ = new TH1F("Beff", "Beff", 2000, -10., 10.);
  h2_dvsphi_ = new TH2F("dvsphi", "dvsphi", 360, -M_PI, M_PI, 100, -0.1, 0.1);
  h1_par0_ = new TH1F("par0", "q/p", 100, -0.1, 0.1);
  h1_par1_ = new TH1F("par1", "lambda", 100, -M_PI / 2., M_PI / 2.);
  h1_par2_ = new TH1F("par2", "phi  ", 100, -M_PI, M_PI);
  h1_par3_ = new TH1F("par3", "dca  ", 100, -0.1, 0.1);
  h1_par4_ = new TH1F("par4", "zdca ", 1000, -10., 10.);
}

void TrackParameterAnalyzer::endJob() {
  rootFile_->cd();
  h1_pull0_->Write();
  h1_pull1_->Write();
  h1_pull2_->Write();
  h1_pull3_->Write();
  h1_pull4_->Write();

  h1_res0_->Write();
  h1_res1_->Write();
  h1_res2_->Write();
  h1_res3_->Write();
  h1_res4_->Write();

  h1_Beff_->Write();
  h2_dvsphi_->Write();
  h1_par0_->Write();
  h1_par1_->Write();
  h1_par2_->Write();
  h1_par3_->Write();
  h1_par4_->Write();
}

// helper function
bool TrackParameterAnalyzer::match(const ParameterVector& a, const ParameterVector& b) {
  double dtheta = a(1) - b(1);
  double dphi = a(2) - b(2);
  if (dphi > M_PI) {
    dphi -= M_2_PI;
  } else if (dphi < -M_PI) {
    dphi += M_2_PI;
  }
  return ((fabs(dtheta) < 0.02) && (fabs(dphi) < 0.04));
}

// ------------ method called to produce the data  ------------
void TrackParameterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using CLHEP::HepLorentzVector;

  const double fBfield = 3.8;

  edm::Handle<edm::SimVertexContainer> simVtcs;
  iEvent.getByToken(edmSimVertexContainerToken_, simVtcs);
  if (verbose_) {
    std::cout << "SimVertex " << simVtcs->size() << std::endl;
    for (edm::SimVertexContainer::const_iterator v = simVtcs->begin(); v != simVtcs->end(); ++v) {
      std::cout << "simvtx " << std::setw(10) << std::setprecision(4) << v->position().x() << " " << v->position().y()
                << " " << v->position().z() << " " << v->parentIndex() << " " << v->noParent() << " " << std::endl;
    }
  }

  // get the simulated tracks, extract perigee parameters
  edm::Handle<edm::SimTrackContainer> simTrks;
  iEvent.getByToken(edmSimTrackContainerToken_, simTrks);

  if (verbose_) {
    std::cout << "simtrks " << simTrks->size() << std::endl;
  }
  std::vector<ParameterVector> tsim;
  for (edm::SimTrackContainer::const_iterator t = simTrks->begin(); t != simTrks->end(); ++t) {
    if (t->noVertex()) {
      std::cout << "simtrk  has no vertex" << std::endl;
      return;
    } else {
      // get the vertex position
      HepLorentzVector v((*simVtcs)[t->vertIndex()].position().x(),
                         (*simVtcs)[t->vertIndex()].position().y(),
                         (*simVtcs)[t->vertIndex()].position().z(),
                         (*simVtcs)[t->vertIndex()].position().e());
      int pdgCode = t->type();

      if (pdgCode == -99) {
        // such entries cause crashes, no idea what they are
        std::cout << "funny particle skipped  , code=" << pdgCode << std::endl;
      } else {
        double Q = 0;
        if ((pdgCode == 11) || (pdgCode == 13) || (pdgCode == 15) || (pdgCode == -211) || (pdgCode == -2212) ||
            (pdgCode == 321)) {
          Q = -1;
        } else if ((pdgCode == -11) || (pdgCode == -13) || (pdgCode == -15) || (pdgCode == 211) || (pdgCode == 2212) ||
                   (pdgCode == 321)) {
          Q = 1;
        } else {
          std::cout << pdgCode << " " << std::endl;
        }
        HepLorentzVector p(t->momentum().x(), t->momentum().y(), t->momentum().z(), t->momentum().e());
        if (verbose_) {
          std::cout << "simtrk "
                    << " gen=" << std::setw(4) << t->genpartIndex() << " vtx=" << std::setw(4) << t->vertIndex()
                    << " pdg=" << std::setw(5) << t->type() << " Q=" << std::setw(3) << Q << " pt=" << std::setw(6)
                    << p.perp() << " vx=" << std::setw(6) << v.x() << " vy=" << std::setw(6) << v.y()
                    << " vz=" << std::setw(6) << v.z() << std::endl;
        }
        if ((Q != 0) && (p.perp() > 0.1)) {
          double x0 = v.x() * simUnit_;
          double y0 = v.y() * simUnit_;
          double z0 = v.z() * simUnit_;
          double kappa = -Q * 0.002998 * fBfield / p.perp();
          double D0 = x0 * sin(p.phi()) - y0 * cos(p.phi()) - 0.5 * kappa * (x0 * x0 + y0 * y0);
          double q = sqrt(1. - 2. * kappa * D0);
          double s0 = (x0 * cos(p.phi()) + y0 * sin(p.phi())) / q;
          double s1;
          if (fabs(kappa * s0) > 0.001) {
            s1 = asin(kappa * s0) / kappa;
          } else {
            double ks02 = (kappa * s0) * (kappa * s0);
            s1 = s0 * (1. + ks02 / 6. + 3. / 40. * ks02 * ks02 + 5. / 112. * pow(ks02, 3));
          }
          ParameterVector par;
          par[reco::TrackBase::i_qoverp] = Q / sqrt(p.perp2() + p.pz() * p.pz());
          par[reco::TrackBase::i_lambda] = M_PI / 2. - p.theta();
          par[reco::TrackBase::i_phi] = p.phi() - asin(kappa * s0);
          par[reco::TrackBase::i_dxy] = 2. * D0 / (1. + q);
          par[reco::TrackBase::i_dsz] = z0 * sin(p.theta()) - s1 * cos(p.theta());
          tsim.push_back(par);
        }
      }
    }  // has vertex
  }    //for loop

  // simtrack parameters are in now tsim
  // loop over tracks and try to match them to simulated tracks

  edm::Handle<reco::TrackCollection> recTracks;
  iEvent.getByToken(recoTrackCollectionToken_, recTracks);

  for (reco::TrackCollection::const_iterator t = recTracks->begin(); t != recTracks->end(); ++t) {
    reco::TrackBase::ParameterVector p = t->parameters();
    reco::TrackBase::CovarianceMatrix c = t->covariance();
    if (verbose_) {
      std::cout << "reco pars= " << p << std::endl;
    }
    for (std::vector<ParameterVector>::const_iterator s = tsim.begin(); s != tsim.end(); ++s) {
      if (match(*s, p)) {
        h1_pull0_->Fill((p(0) - (*s)(0)) / sqrt(c(0, 0)));
        h1_pull1_->Fill((p(1) - (*s)(1)) / sqrt(c(1, 1)));
        h1_pull2_->Fill((p(2) - (*s)(2)) / sqrt(c(2, 2)));
        h1_pull3_->Fill((p(3) - (*s)(3)) / sqrt(c(3, 3)));
        h1_pull4_->Fill((p(4) - (*s)(4)) / sqrt(c(4, 4)));

        h1_res0_->Fill(p(0) - (*s)(0));
        h1_res1_->Fill(p(1) - (*s)(1));
        h1_res2_->Fill(p(2) - (*s)(2));
        h1_res3_->Fill(p(3) - (*s)(3));
        h1_res4_->Fill(p(4) - (*s)(4));

        h1_Beff_->Fill(p(0) / (*s)(0) * fBfield);
        h2_dvsphi_->Fill(p(2), p(3));
        h1_par0_->Fill(p(0));
        h1_par1_->Fill(p(1));
        h1_par2_->Fill(p(2));
        h1_par3_->Fill(p(3));
        h1_par4_->Fill(p(4));
      }
    }
  }
}
