#ifndef Validation_RecoMuon_Histograms_H
#define Validation_RecoMuon_Histograms_H

/** \class Histograms
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"
#include "TFile.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include <iostream>
#include <vector>
#include <cmath>

class HTrackVariables {
public:
  HTrackVariables(DQMStore::IBooker &ibooker, std::string dirName_, std::string name, std::string whereIs = "")
      : theName(name), where(whereIs) {
    ibooker.cd();
    const std::string &dirName = dirName_;
    //dirName+="/";
    //dirName+=name.c_str();

    ibooker.setCurrentFolder(dirName);

    hEta = ibooker.book1D(theName + "_Eta_" + where, "#eta at " + where, 120, -3., 3.);
    hPhi = ibooker.book1D(theName + "_Phi_" + where, "#phi at " + where, 100, -Geom::pi(), Geom::pi());
    hP = ibooker.book1D(theName + "_P_" + where, "p_{t} at " + where, 1000, 0, 2000);
    hPt = ibooker.book1D(theName + "_Pt_" + where, "p_{t} at " + where, 1000, 0, 2000);
    hCharge = ibooker.book1D(theName + "_charge_" + where, "Charge at " + where, 4, -2., 2.);

    hEtaVsGen = ibooker.book1D(theName + "_EtaVsGen_" + where, "#eta at " + where, 120, -3., 3.);
    hPhiVsGen = ibooker.book1D(theName + "_PhiVsGen_" + where, "#phi at " + where, 100, -Geom::pi(), Geom::pi());
    hPtVsGen = ibooker.book1D(theName + "_PtVsGen_" + where, "p_{t} at " + where, 1000, 0, 2000);

    hDeltaR = ibooker.book1D(theName + "_DeltaR_" + where, "Delta R w.r.t. sim track for " + where, 1000, 0, 20);

    theEntries = 0;
  }

  /*
  HTrackVariables(std::string name, TFile* file):theName(name.c_str()){ 
    
    hEta = dynamic_cast<TH1F*>( file->Get(theName+"_Eta_"+where));
    hPhi = dynamic_cast<TH1F*>( file->Get(theName+"_Phi_"+where));
    hP   = dynamic_cast<TH1F*>( file->Get(theName+"_P_"+where));
    hPt  = dynamic_cast<TH1F*>( file->Get(theName+"_Pt_"+where));
    hCharge = dynamic_cast<TH1F*>( file->Get(theName+"_charge_"+where)); 
  }
  */

  ~HTrackVariables() {}

  MonitorElement *eta() { return hEta; }
  MonitorElement *phi() { return hPhi; }
  MonitorElement *p() { return hP; }
  MonitorElement *pt() { return hPt; }
  MonitorElement *charge() { return hCharge; }
  int entries() { return theEntries; }

  void Fill(double p, double pt, double eta, double phi, double charge) {
    hEta->Fill(eta);
    hPhi->Fill(phi);
    hP->Fill(p);
    hPt->Fill(pt);
    hCharge->Fill(charge);
    ++theEntries;
  }

  void Fill(double pt, double eta, double phi) {
    hEtaVsGen->Fill(eta);
    hPhiVsGen->Fill(phi);
    hPtVsGen->Fill(pt);
  }

  void FillDeltaR(double deltaR) { hDeltaR->Fill(deltaR); }

  double computeEfficiency(HTrackVariables *sim, DQMStore::IBooker &ibooker) {
    efficiencyHistos.push_back(computeEfficiency(hEtaVsGen, sim->eta(), ibooker));
    efficiencyHistos.push_back(computeEfficiency(hPhiVsGen, sim->phi(), ibooker));
    efficiencyHistos.push_back(computeEfficiency(hPtVsGen, sim->pt(), ibooker));

    double efficiency = 100 * entries() / sim->entries();
    return efficiency;
  }

  MonitorElement *computeEfficiency(MonitorElement *reco, MonitorElement *sim, DQMStore::IBooker &ibooker) {
    TH1F *hReco = reco->getTH1F();
    TH1F *hSim = sim->getTH1F();

    std::string name = hReco->GetName();
    std::string title = hReco->GetTitle();

    MonitorElement *me = ibooker.book1D("Eff_" + name,
                                        "Efficiecny as a function of " + title,
                                        hSim->GetNbinsX(),
                                        hSim->GetXaxis()->GetXmin(),
                                        hSim->GetXaxis()->GetXmax());

    me->getTH1F()->Divide(hReco, hSim, 1., 1., "b");

    // Set the error accordingly to binomial statistics
    int nBinsEta = me->getTH1F()->GetNbinsX();
    for (int bin = 1; bin <= nBinsEta; bin++) {
      float nSimHit = hSim->GetBinContent(bin);
      float eff = me->getTH1F()->GetBinContent(bin);
      float error = 0;
      if (nSimHit != 0 && eff <= 1) {
        error = sqrt(eff * (1 - eff) / nSimHit);
      }
      me->getTH1F()->SetBinError(bin, error);
    }

    return me;
  }

private:
  std::string theName;
  std::string where;

  int theEntries;

  MonitorElement *hEta;
  MonitorElement *hPhi;
  MonitorElement *hP;
  MonitorElement *hPt;
  MonitorElement *hCharge;

  MonitorElement *hEtaVsGen;
  MonitorElement *hPhiVsGen;
  MonitorElement *hPtVsGen;

  MonitorElement *hDeltaR;

  std::vector<MonitorElement *> efficiencyHistos;
};

class HResolution {
public:
  HResolution(DQMStore::IBooker &ibooker, std::string dirName_, std::string name, std::string whereIs)
      : theName(name), where(whereIs) {
    ibooker.cd();
    const std::string &dirName = dirName_;
    //dirName+="/";
    //dirName+=name.c_str();

    ibooker.setCurrentFolder(dirName);

    double eta = 15.;
    int neta = 800;
    double phi = 12.;
    int nphi = 400;
    double pt = 60.;
    int npt = 2000;

    hEta = ibooker.book1D(theName + "_Eta_" + where, "#eta " + theName, neta, -eta, eta);  // 400
    hPhi = ibooker.book1D(theName + "_Phi_" + where, "#phi " + theName, nphi, -phi, phi);  // 100

    hP = ibooker.book1D(theName + "_P_" + where, "P " + theName, 400, -4, 4);          // 200
    hPt = ibooker.book1D(theName + "_Pt_" + where, "P_{t} " + theName, npt, -pt, pt);  // 200

    hCharge = ibooker.book1D(theName + "_charge_" + where, "Charge " + theName, 4, -2., 2.);

    h2Eta = ibooker.book2D(
        theName + "_Eta_vs_Eta" + where, "#eta " + theName + " as a function of #eta", 200, -2.5, 2.5, neta, -eta, eta);
    h2Phi = ibooker.book2D(theName + "_Phi_vs_Phi" + where,
                           "#phi " + theName + " as a function of #phi",
                           100,
                           -Geom::pi(),
                           Geom::pi(),
                           nphi,
                           -phi,
                           phi);

    h2P =
        ibooker.book2D(theName + "_P_vs_P" + where, "P " + theName + " as a function of P", 1000, 0, 2000, 400, -4, 4);
    h2Pt = ibooker.book2D(
        theName + "_Pt_vs_Pt" + where, "P_{t} " + theName + " as a function of P_{t}", 1000, 0, 2000, npt, -pt, pt);

    h2PtVsEta = ibooker.book2D(
        theName + "_Pt_vs_Eta" + where, "P_{t} " + theName + " as a function of #eta", 200, -2.5, 2.5, npt, -pt, pt);
    h2PtVsPhi = ibooker.book2D(theName + "_Pt_vs_Phi" + where,
                               "P_{t} " + theName + " as a function of #phi",
                               100,
                               -Geom::pi(),
                               Geom::pi(),
                               npt,
                               -pt,
                               pt);

    h2EtaVsPt = ibooker.book2D(
        theName + "_Eta_vs_Pt" + where, "#eta " + theName + " as a function of P_{t}", 1000, 0, 2000, neta, -eta, eta);
    h2EtaVsPhi = ibooker.book2D(theName + "_Eta_vs_Phi" + where,
                                "#eta " + theName + " as a function of #phi",
                                100,
                                -Geom::pi(),
                                Geom::pi(),
                                neta,
                                -eta,
                                eta);

    h2PhiVsPt = ibooker.book2D(
        theName + "_Phi_vs_Pt" + where, "#phi " + theName + " as a function of P_{t}", 1000, 0, 2000, nphi, -phi, phi);
    h2PhiVsEta = ibooker.book2D(
        theName + "_Phi_vs_Eta" + where, "#phi " + theName + " as a function of #eta", 200, -2.5, 2.5, nphi, -phi, phi);
  }

  HResolution(DQMStore::IBooker &ibooker, std::string name, TFile *file) : theName(name) {
    //    dynamic_cast<TH1F*>( file->Get(theName+"") );
  }

  ~HResolution() {}

  void Fill(
      double p, double pt, double eta, double phi, double rp, double rpt, double reta, double rphi, double rcharge) {
    Fill(rp, rpt, reta, rphi, rcharge);

    h2Eta->Fill(eta, reta);
    h2Phi->Fill(phi, rphi);

    h2P->Fill(p, rp);
    h2Pt->Fill(pt, rpt);

    h2PtVsEta->Fill(eta, rpt);
    h2PtVsPhi->Fill(phi, rpt);

    h2EtaVsPt->Fill(pt, reta);
    h2EtaVsPhi->Fill(phi, reta);

    h2PhiVsPt->Fill(pt, rphi);
    h2PhiVsEta->Fill(eta, rphi);
  }

  void Fill(double p, double pt, double eta, double phi, double rp, double rpt) {
    hP->Fill(rp);
    hPt->Fill(rpt);

    h2P->Fill(p, rp);
    // h2PVsEta->Fill(eta,rp);
    // h2PVsPhi->Fill(phi,rp);

    h2Pt->Fill(pt, rpt);
    h2PtVsEta->Fill(eta, rpt);
    h2PtVsPhi->Fill(phi, rpt);
  }

  void Fill(double rp, double rpt, double reta, double rphi, double rcharge) {
    hEta->Fill(reta);
    hPhi->Fill(rphi);

    hP->Fill(rp);
    hPt->Fill(rpt);

    hCharge->Fill(rcharge);
  }

private:
  std::string theName;
  std::string where;

  MonitorElement *hEta;
  MonitorElement *hPhi;

  MonitorElement *hP;
  MonitorElement *hPt;

  MonitorElement *hCharge;

  MonitorElement *h2Eta;
  MonitorElement *h2Phi;

  MonitorElement *h2P;
  MonitorElement *h2Pt;

  MonitorElement *h2PtVsEta;
  MonitorElement *h2PtVsPhi;

  MonitorElement *h2EtaVsPt;
  MonitorElement *h2EtaVsPhi;

  MonitorElement *h2PhiVsPt;
  MonitorElement *h2PhiVsEta;
};

class HResolution1DRecHit {
public:
  HResolution1DRecHit(DQMStore::IBooker &ibooker, std::string name) : theName(name) {
    // Position, sigma, residual, pull
    hResX = ibooker.book1D(theName + "_X_Res", "X residual", 5000, -0.5, 0.5);
    hResY = ibooker.book1D(theName + "_Y_Res", "Y residual", 5000, -1., 1.);
    hResZ = ibooker.book1D(theName + "_Z_Res", "Z residual", 5000, -1.5, 1.5);

    hResXVsEta = ibooker.book2D(theName + "_XResVsEta", "X residual vs eta", 200, -2.5, 2.5, 5000, -1.5, 1.5);
    hResYVsEta = ibooker.book2D(theName + "_YResVsEta", "Y residual vs eta", 200, -2.5, 2.5, 5000, -1., 1.);
    hResZVsEta = ibooker.book2D(theName + "_ZResVsEta", "Z residual vs eta", 200, -2.5, 2.5, 5000, -1.5, 1.5);

    hXResVsPhi =
        ibooker.book2D(theName + "_XResVsPhi", "X residual vs phi", 100, -Geom::pi(), Geom::pi(), 5000, -0.5, 0.5);
    hYResVsPhi =
        ibooker.book2D(theName + "_YResVsPhi", "Y residual vs phi", 100, -Geom::pi(), Geom::pi(), 5000, -1., 1.);
    hZResVsPhi =
        ibooker.book2D(theName + "_ZResVsPhi", "Z residual vs phi", 100, -Geom::pi(), Geom::pi(), 5000, -1.5, 1.5);

    hXResVsPos = ibooker.book2D(theName + "_XResVsPos", "X residual vs position", 10000, -750, 750, 5000, -0.5, 0.5);
    hYResVsPos = ibooker.book2D(theName + "_YResVsPos", "Y residual vs position", 10000, -740, 740, 5000, -1., 1.);
    hZResVsPos = ibooker.book2D(theName + "_ZResVsPos", "Z residual vs position", 10000, -1100, 1100, 5000, -1.5, 1.5);

    hXPull = ibooker.book1D(theName + "_XPull", "X pull", 600, -2, 2);
    hYPull = ibooker.book1D(theName + "_YPull", "Y pull", 600, -2, 2);
    hZPull = ibooker.book1D(theName + "_ZPull", "Z pull", 600, -2, 2);

    hXPullVsPos = ibooker.book2D(theName + "_XPullVsPos", "X pull vs position", 10000, -750, 750, 600, -2, 2);
    hYPullVsPos = ibooker.book2D(theName + "_YPullVsPos", "Y pull vs position", 10000, -740, 740, 600, -2, 2);
    hZPullVsPos = ibooker.book2D(theName + "_ZPullVsPos", "Z pull vs position", 10000, -1100, 1100, 600, -2, 2);
  }

  HResolution1DRecHit(DQMStore::IBooker &ibooker, const TString &name_, TFile *file) {}

  ~HResolution1DRecHit() {}

  void Fill(double x,
            double y,
            double z,
            double dx,
            double dy,
            double dz,
            double errx,
            double erry,
            double errz,
            double eta,
            double phi) {
    double rx = dx / x, ry = dy / y, rz = dz / z;

    hResX->Fill(rx);
    hResY->Fill(ry);
    hResZ->Fill(rz);

    hResXVsEta->Fill(eta, rx);
    hResYVsEta->Fill(eta, ry);
    hResZVsEta->Fill(eta, rz);

    hXResVsPhi->Fill(phi, rx);
    hYResVsPhi->Fill(phi, ry);
    hZResVsPhi->Fill(phi, rz);

    hXResVsPos->Fill(x, rx);
    hYResVsPos->Fill(y, ry);
    hZResVsPos->Fill(z, rz);

    if (errx < 1e-6)
      std::cout << "NO proper error set for X: " << errx << std::endl;
    else {
      hXPull->Fill(dx / errx);
      hXPullVsPos->Fill(x, dx / errx);
    }

    if (erry < 1e-6)
      std::cout << "NO proper error set for Y: " << erry << std::endl;
    else {
      hYPull->Fill(dy / erry);
      hYPullVsPos->Fill(y, dy / erry);
    }
    if (errz < 1e-6)
      std::cout << "NO proper error set for Z: " << errz << std::endl;
    else {
      hZPull->Fill(dz / errz);
      hZPullVsPos->Fill(z, dz / errz);
    }
  }

public:
  std::string theName;

  MonitorElement *hResX;
  MonitorElement *hResY;
  MonitorElement *hResZ;

  MonitorElement *hResXVsEta;
  MonitorElement *hResYVsEta;
  MonitorElement *hResZVsEta;

  MonitorElement *hXResVsPhi;
  MonitorElement *hYResVsPhi;
  MonitorElement *hZResVsPhi;

  MonitorElement *hXResVsPos;
  MonitorElement *hYResVsPos;
  MonitorElement *hZResVsPos;

  MonitorElement *hXPull;
  MonitorElement *hYPull;
  MonitorElement *hZPull;

  MonitorElement *hXPullVsPos;
  MonitorElement *hYPullVsPos;
  MonitorElement *hZPullVsPos;
};
#endif
