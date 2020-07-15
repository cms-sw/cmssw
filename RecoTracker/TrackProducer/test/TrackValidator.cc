#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <iostream>
#include <string>

#include <TH1.h>
#include <TH2.h>
#include <TROOT.h>
#include <TFile.h>
#include <TCanvas.h>

using namespace edm;
using namespace std;
using namespace reco;

class TrackValidator : public edm::EDAnalyzer {
public:
  TrackValidator(const edm::ParameterSet& pset)
      : sim(pset.getParameter<string>("sim")),
        label(pset.getParameter<vector<string> >("label")),
        out(pset.getParameter<string>("out")),
        open(pset.getParameter<string>("open")),
        min(pset.getParameter<double>("min")),
        max(pset.getParameter<double>("max")),
        nint(pset.getParameter<int>("nint")),
        partId(pset.getParameter<int>("partId")) {
    hFile = new TFile(out.c_str(), open.c_str());
  }

  ~TrackValidator() override {
    if (hFile != nullptr) {
      hFile->Close();
      delete hFile;
    }
  }

  void beginRun(edm::Run const& run, const edm::EventSetup& setup) override {
    for (unsigned int j = 0; j < label.size(); j++) {
      vector<double> etaintervalsv;
      vector<int> totSIMv, totRECv;
      vector<TH1F*> ptdistribv;
      vector<TH1F*> etadistribv;

      double step = (max - min) / nint;
      ostringstream title, name;
      etaintervalsv.push_back(0);
      for (double d = min; d < max; d = d + step) {
        etaintervalsv.push_back(d + step);
        totSIMv.push_back(0);
        totRECv.push_back(0);
        name.str("");
        title.str("");
        name << "pt[" << d << "," << d + step << "]";
        title << "p_{t} residue " << d << "<#eta<" << d + step;
        ptdistribv.push_back(new TH1F(name.str().c_str(), title.str().c_str(), 200, -2, 2));
        name.str("");
        title.str("");
        name << "eta[" << d << "," << d + step << "]";
        title << "eta residue " << d << "<#eta<" << d + step;
        etadistribv.push_back(new TH1F(name.str().c_str(), title.str().c_str(), 200, -0.2, 0.2));
      }
      etaintervals.push_back(etaintervalsv);
      totSIM.push_back(totSIMv);
      totREC.push_back(totRECv);
      ptdistrib.push_back(ptdistribv);
      etadistrib.push_back(etadistribv);

      h_ptSIM.push_back(new TH1F("ptSIM", "generated p_{t}", 5500, 0, 110));
      h_etaSIM.push_back(new TH1F("etaSIM", "generated pseudorapidity", 500, 0, 5));
      h_tracksSIM.push_back(new TH1F("tracksSIM", "number of simluated tracks", 100, -0.5, 99.5));
      h_vertposSIM.push_back(new TH1F("vertposSIM", "Transverse position of sim vertices", 1000, -0.5, 10000.5));

      //     h_pt     = new TH1F("pt", "p_{t} residue", 2000, -500, 500 );
      h_pt.push_back(new TH1F("pullPt", "pull of p_{t}", 100, -10, 10));
      h_pt2.push_back(new TH1F("pt2", "p_{t} residue (#tracks>1)", 300, -15, 15));
      h_eta.push_back(new TH1F("eta", "pseudorapidity residue", 1000, -0.1, 0.1));
      h_tracks.push_back(new TH1F("tracks", "number of reconstructed tracks", 10, -0.5, 9.5));
      h_nchi2.push_back(new TH1F("nchi2", "normalized chi2", 200, 0, 20));
      h_hits.push_back(new TH1F("hits", "number of hits per track", 30, -0.5, 29.5));
      h_effic.push_back(new TH1F("effic", "efficiency vs #eta", nint, &etaintervals[j][0]));
      h_ptrmsh.push_back(new TH1F("PtRMS", "PtRMS vs #eta", nint, &etaintervals[j][0]));
      h_deltaeta.push_back(new TH1F("etaRMS", "etaRMS vs #eta", nint, &etaintervals[j][0]));
      h_charge.push_back(new TH1F("charge", "charge", 3, -1.5, 1.5));

      h_pullTheta.push_back(new TH1F("pullTheta", "pull of theta parameter", 100, -10, 10));
      h_pullPhi0.push_back(new TH1F("pullPhi0", "pull of phi0 parameter", 100, -10, 10));
      h_pullD0.push_back(new TH1F("pullD0", "pull of d0 parameter", 100, -10, 10));
      h_pullDz.push_back(new TH1F("pullDz", "pull of dz parameter", 100, -10, 10));
      h_pullK.push_back(new TH1F("pullK", "pull of k parameter", 100, -10, 10));

      chi2_vs_nhits.push_back(new TH2F("chi2_vs_nhits", "chi2 vs nhits", 25, 0, 25, 100, 0, 10));
      chi2_vs_eta.push_back(new TH2F("chi2_vs_eta", "chi2 vs eta", nint, min, max, 100, 0, 10));
      nhits_vs_eta.push_back(new TH2F("nhits_vs_eta", "nhits vs eta", nint, min, max, 25, 0, 25));
      ptres_vs_eta.push_back(new TH2F("ptres_vs_eta", "ptresidue vs eta", nint, min, max, 200, -2, 2));
      etares_vs_eta.push_back(new TH2F("etares_vs_eta", "etaresidue vs eta", nint, min, max, 200, -0.1, 0.1));
    }

    setup.get<IdealMagneticFieldRecord>().get(theMF);
  }

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {
    std::cout << "In TrackValidator\n";
    edm::ESHandle<TransientTrackBuilder> theB;
    setup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);
    for (unsigned int w = 0; w < label.size(); w++) {
      //
      //get collections from the event
      //
      edm::Handle<SimTrackContainer> simTrackCollection;
      event.getByLabel(sim, simTrackCollection);
      const SimTrackContainer simTC = *(simTrackCollection.product());

      edm::Handle<SimVertexContainer> simVertexCollection;
      event.getByLabel(sim, simVertexCollection);
      const SimVertexContainer simVC = *(simVertexCollection.product());

      edm::Handle<reco::TrackCollection> trackCollection;
      event.getByLabel(label[w], trackCollection);
      const reco::TrackCollection tC = *(trackCollection.product());

      vector<TransientTrack> t_tks = (*theB).build(trackCollection);
      cout << "Found: " << t_tks.size() << " reconstructed tracks"
           << "\n";

      //
      //fill simulation histograms
      //
      int st = 0;
      for (SimTrackContainer::const_iterator simTrack = simTC.begin(); simTrack != simTC.end(); simTrack++) {
        if (abs(simTrack->momentum().eta()) > max || abs(simTrack->momentum().eta()) < min)
          continue;
        st++;
        h_ptSIM[w]->Fill(simTrack->momentum().pt());
        h_etaSIM[w]->Fill(simTrack->momentum().eta());

        h_vertposSIM[w]->Fill(simVC[simTrack->vertIndex()].position().pt());

        if (simTrack->type() != partId)
          continue;
        //compute number of tracks per eta interval
        int i = 0;
        for (vector<double>::iterator h = etaintervals[w].begin(); h != etaintervals[w].end() - 1; h++) {
          if (abs(simTrack->momentum().eta()) > etaintervals[w][i] &&
              abs(simTrack->momentum().eta()) < etaintervals[w][i + 1]) {
            totSIM[w][i]++;
            bool doit = false;
            for (reco::TrackCollection::const_iterator track = tC.begin(); track != tC.end(); track++) {
              if (abs(track->pt() - simTrack->momentum().pt()) < (simTrack->momentum().pt() * 0.1))
                doit = true;
            }
            if (doit)
              totREC[w][i]++;
          }
          i++;
        }
      }
      if (st != 0)
        h_tracksSIM[w]->Fill(st);

      //       for (SimVertexContainer::const_iterator simVertex=simVC.begin();simVertex!=simVC.end();simVertex++){
      // 	h_vertposSIM[w]->Fill(simVertex->position().perp());
      //       }

      //
      //fill reconstructed track histograms
      //
      int rt = 0;
      for (vector<TransientTrack>::const_iterator track = t_tks.begin(); track != t_tks.end(); track++) {
        TrajectoryStateClosestToPoint tscp = track->impactPointTSCP();
        cout << tscp.perigeeParameters().vector() << tscp.perigeeError().covarianceMatrix() << endl;

        if (abs(track->track().eta()) > max || abs(track->track().eta()) < min)
          continue;

        rt++;

        //nchi2 and hits global distributions
        h_nchi2[w]->Fill(track->normalizedChi2());
        h_hits[w]->Fill(track->numberOfValidHits());
        chi2_vs_nhits[w]->Fill(track->numberOfValidHits(), track->normalizedChi2());
        chi2_vs_eta[w]->Fill(track->track().eta(), track->normalizedChi2());
        nhits_vs_eta[w]->Fill(track->track().eta(), track->numberOfValidHits());
        h_charge[w]->Fill(track->charge());

        //pt, eta residue, theta, phi0, d0, dz pull
        double ptres = 1000;
        double etares = 1000;
        double thetares = 1000;
        double phi0res = 1000;
        double d0res = 1000;
        double dzres = 1000;
        double kres = 1000;
        for (SimTrackContainer::const_iterator simTrack = simTC.begin(); simTrack != simTC.end(); simTrack++) {
          if (simTrack->type() != partId)
            continue;
          double tmp = track->track().pt() - simTrack->momentum().pt();
          if (tC.size() > 1)
            h_pt2[w]->Fill(tmp);
          if (abs(tmp) < abs(ptres)) {
            ptres = tmp;

            etares = track->initialFreeState().momentum().eta() - simTrack->momentum().eta();
            thetares =
                (tscp.perigeeParameters().theta() - simTrack->momentum().theta()) / tscp.perigeeError().thetaError();
            phi0res = (tscp.perigeeParameters().phi() - simTrack->momentum().phi()) / tscp.perigeeError().phiError();
            d0res =
                (tscp.perigeeParameters().transverseImpactParameter() - simVC[simTrack->vertIndex()].position().pt()) /
                tscp.perigeeError().transverseImpactParameterError();
            dzres =
                (tscp.perigeeParameters().longitudinalImpactParameter() - simVC[simTrack->vertIndex()].position().z()) /
                tscp.perigeeError().longitudinalImpactParameterError();

            const math::XYZTLorentzVectorD& vertexPosition = simVC[simTrack->vertIndex()].position();
            GlobalVector magField =
                theMF->inTesla(GlobalPoint(vertexPosition.x(), vertexPosition.y(), vertexPosition.z()));
            kres = (tscp.perigeeParameters().transverseCurvature() -
                    (-track->charge() * 2.99792458e-3 * magField.z() / simTrack->momentum().pt())) /
                   tscp.perigeeError().transverseCurvatureError();

            // 	    cout << "track->d0(): " << track->d0() << endl;
            // 	    cout << "simVC[simTrack->vertIndex()].position().perp(): " << simVC[simTrack->vertIndex()].position().perp() << endl;
            // 	    cout << "track->dz(): " << track->dz() << endl;
            // 	    cout << "simVC[simTrack->vertIndex()].position().z(): " << simVC[simTrack->vertIndex()].position().z() << endl;
            // 	    cout << "track->transverseCurvature(): " << track->transverseCurvature() << endl;

            // 	    cout << "-track->charge()*2.99792458e-3 * 4./simTrack->momentum().perp(): " << -track->charge()*2.99792458e-3 * 4./simTrack->momentum().perp() << endl;
            // 	    cout << "-track->charge()*2.99792458e-3 * 4./simTrack->momentum().perp(): " << -track->charge()*2.99792458e-3 * magField.z()/simTrack->momentum().perp() << endl;
          }
        }
        cout << etares << endl;
        h_pt[w]->Fill(
            ptres / (tscp.perigeeError().transverseCurvatureError() / tscp.perigeeParameters().transverseCurvature()));
        h_eta[w]->Fill(etares);
        ptres_vs_eta[w]->Fill(track->track().eta(), ptres);
        etares_vs_eta[w]->Fill(track->track().eta(), etares);
        h_pullTheta[w]->Fill(thetares);
        h_pullPhi0[w]->Fill(phi0res);
        h_pullD0[w]->Fill(d0res);
        h_pullDz[w]->Fill(dzres);
        h_pullK[w]->Fill(kres);

        //pt residue distribution per eta interval
        int i = 0;
        for (vector<TH1F*>::iterator h = ptdistrib[w].begin(); h != ptdistrib[w].end(); h++) {
          for (SimTrackContainer::const_iterator simTrack = simTC.begin(); simTrack != simTC.end(); simTrack++) {
            if (simTrack->type() != partId)
              continue;
            ptres = 1000;
            if (abs(simTrack->momentum().eta()) > etaintervals[w][i] &&
                abs(simTrack->momentum().eta()) < etaintervals[w][i + 1]) {
              double tmp = track->track().pt() - simTrack->momentum().pt();
              if (abs(tmp) < abs(ptres))
                ptres = tmp;
            }
          }
          (*h)->Fill(ptres);
          i++;
        }
        //eta residue distribution per eta interval
        i = 0;
        for (vector<TH1F*>::iterator h = etadistrib[w].begin(); h != etadistrib[w].end(); h++) {
          for (SimTrackContainer::const_iterator simTrack = simTC.begin(); simTrack != simTC.end(); simTrack++) {
            if (simTrack->type() != partId)
              continue;
            etares = 1000;
            ptres = 1000;
            if (abs(simTrack->momentum().eta()) > etaintervals[w][i] &&
                abs(simTrack->momentum().eta()) < etaintervals[w][i + 1]) {
              double tmp = track->track().pt() - simTrack->momentum().pt();
              if (abs(tmp) < abs(ptres))
                etares = track->track().eta() - simTrack->momentum().eta();
            }
          }
          (*h)->Fill(etares);
          i++;
        }
      }
      if (rt != 0)
        h_tracks[w]->Fill(rt);
    }
  }

  void endJob() override {
    for (unsigned int w = 0; w < label.size(); w++) {
      TDirectory* p = hFile->mkdir(label[w].c_str());

      //write simulation histos
      TDirectory* simD = p->mkdir("simulation");
      simD->cd();
      h_ptSIM[w]->Write();
      h_etaSIM[w]->Write();
      h_tracksSIM[w]->Write();
      h_vertposSIM[w]->Write();

      //fill pt rms plot versus eta and write pt residue distribution per eta interval histo
      TDirectory* ptD = p->mkdir("ptdistribution");
      ptD->cd();
      int i = 0;
      for (vector<TH1F*>::iterator h = ptdistrib[w].begin(); h != ptdistrib[w].end(); h++) {
        (*h)->Write();
        h_ptrmsh[w]->Fill(etaintervals[w][i + 1] - 0.00001, (*h)->GetRMS());
        i++;
      }

      //fill eta rms plot versus eta and write eta residue distribution per eta interval histo
      TDirectory* etaD = p->mkdir("etadistribution");
      etaD->cd();
      i = 0;
      for (vector<TH1F*>::iterator h = etadistrib[w].begin(); h != etadistrib[w].end(); h++) {
        (*h)->Write();
        h_deltaeta[w]->Fill(etaintervals[w][i + 1] - 0.00001, (*h)->GetRMS());
        i++;
      }

      //write the other histos
      p->cd();
      int j = 0;
      for (vector<int>::iterator h = totSIM[w].begin(); h != totSIM[w].end(); h++) {
        if (totSIM[w][j])
          h_effic[w]->Fill(etaintervals[w][j + 1] - 0.00001, ((double)totREC[w][j]) / ((double)totSIM[w][j]));
        else
          h_effic[w]->Fill(etaintervals[w][j + 1] - 0.00001, 0);
        j++;
      }

      h_pt[w]->Write();
      h_pt2[w]->Write();
      h_eta[w]->Write();
      h_tracks[w]->Write();
      h_nchi2[w]->Write();
      h_hits[w]->Write();
      h_effic[w]->Write();
      h_ptrmsh[w]->Write();
      h_deltaeta[w]->Write();
      chi2_vs_nhits[w]->Write();
      chi2_vs_eta[w]->Write();
      nhits_vs_eta[w]->Write();
      ptres_vs_eta[w]->Write();
      etares_vs_eta[w]->Write();
      h_charge[w]->Write();

      h_pullTheta[w]->Write();
      h_pullPhi0[w]->Write();
      h_pullD0[w]->Write();
      h_pullDz[w]->Write();
      h_pullK[w]->Write();
    }

    hFile->Close();
  }

private:
  string sim;
  vector<string> label;
  string out, open;
  double min, max;
  int nint, partId;

  vector<TH1F*> h_ptSIM, h_etaSIM, h_tracksSIM, h_vertposSIM;
  vector<TH1F*> h_tracks, h_nchi2, h_hits, h_effic, h_ptrmsh, h_deltaeta, h_charge;
  vector<TH1F*> h_pt, h_eta, h_pullTheta, h_pullPhi0, h_pullD0, h_pullDz, h_pullK, h_pt2;
  vector<TH2F*> chi2_vs_nhits, chi2_vs_eta, nhits_vs_eta, ptres_vs_eta, etares_vs_eta;

  vector<vector<double> > etaintervals;
  vector<vector<int> > totSIM, totREC;

  vector<vector<TH1F*> > ptdistrib;
  vector<vector<TH1F*> > etadistrib;
  TFile* hFile;

  edm::ESHandle<MagneticField> theMF;
};

DEFINE_FWK_MODULE(TrackValidator);
