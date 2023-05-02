#include "SimG4Core/Generators/interface/Generator.h"
#include "SimG4Core/Generators/interface/HepMCParticle.h"
#include "SimG4Core/Generators/interface/LumiMonitorFilter.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HepPDT/ParticleID.hh"

#include "G4Event.hh"

#include "G4HEPEvtParticle.hh"
#include "G4Log.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

#include <sstream>

using namespace edm;

Generator::Generator(const ParameterSet &p)
    : fPCuts(p.getParameter<bool>("ApplyPCuts")),
      fPtransCut(p.getParameter<bool>("ApplyPtransCut")),
      fEtaCuts(p.getParameter<bool>("ApplyEtaCuts")),
      fPhiCuts(p.getParameter<bool>("ApplyPhiCuts")),
      fFinalState(p.getParameter<bool>("FixOfFinalStateRadiation")),
      theMinPhiCut(p.getParameter<double>("MinPhiCut")),  // in radians (CMS standard)
      theMaxPhiCut(p.getParameter<double>("MaxPhiCut")),
      theMinEtaCut(p.getParameter<double>("MinEtaCut")),
      theMaxEtaCut(p.getParameter<double>("MaxEtaCut")),
      theMinPCut(p.getParameter<double>("MinPCut")),  // in GeV (CMS standard)
      theMaxPCut(p.getParameter<double>("MaxPCut")),
      theEtaCutForHector(p.getParameter<double>("EtaCutForHector")),
      verbose(p.getUntrackedParameter<int>("Verbosity", 0)),
      fLumiFilter(nullptr),
      evt_(nullptr),
      vtx_(nullptr),
      weight_(0),
      Z_lmin(0),
      Z_lmax(0),
      Z_hector(0),
      pdgFilterSel(false),
      fPDGFilter(false) {
  bool lumi = p.getParameter<bool>("ApplyLumiMonitorCuts");
  if (lumi) {
    fLumiFilter = new LumiMonitorFilter();
  }

  double theRDecLenCut = p.getParameter<double>("RDecLenCut") * cm;
  theDecRCut2 = theRDecLenCut * theRDecLenCut;

  theMinPtCut2 = theMinPCut * theMinPCut;

  double theDecLenCut = p.getParameter<double>("LDecLenCut") * cm;

  fFiductialCuts = (fPCuts || fPtransCut || fEtaCuts || fPhiCuts);

  pdgFilter.resize(0);
  if (p.exists("PDGselection")) {
    pdgFilterSel = (p.getParameter<edm::ParameterSet>("PDGselection")).getParameter<bool>("PDGfilterSel");
    pdgFilter = (p.getParameter<edm::ParameterSet>("PDGselection")).getParameter<std::vector<int>>("PDGfilter");
    if (!pdgFilter.empty()) {
      fPDGFilter = true;
      std::stringstream ss;
      ss << "SimG4Core/Generator: ";
      if (pdgFilterSel) {
        ss << " Selecting only PDG ID = ";
      } else {
        ss << " Filtering out PDG ID = ";
      }
      for (unsigned int ii = 0; ii < pdgFilter.size(); ++ii) {
        ss << pdgFilter[ii] << "  ";
      }
      edm::LogVerbatim("SimG4CoreGenerator") << ss.str();
    }
  }

  if (fEtaCuts) {
    Z_lmax = theRDecLenCut * ((1 - exp(-2 * theMaxEtaCut)) / (2 * exp(-theMaxEtaCut)));
    Z_lmin = theRDecLenCut * ((1 - exp(-2 * theMinEtaCut)) / (2 * exp(-theMinEtaCut)));
  }

  Z_hector = theRDecLenCut * ((1 - exp(-2 * theEtaCutForHector)) / (2 * exp(-theEtaCutForHector)));

  edm::LogVerbatim("SimG4CoreGenerator") << "SimG4Core/Generator: Rdecaycut= " << theRDecLenCut / cm
                                         << " cm;  Zdecaycut= " << theDecLenCut / cm << "Z_min= " << Z_lmin / cm
                                         << " cm; Z_max= " << Z_lmax << " cm;  Z_hector = " << Z_hector << " cm\n"
                                         << "                     ApplyCuts: " << fFiductialCuts
                                         << "  PCuts: " << fPCuts << "  PtransCut: " << fPtransCut
                                         << "  EtaCut: " << fEtaCuts << "  PhiCut: " << fPhiCuts
                                         << "  LumiMonitorCut: " << lumi << " FinalState: " << fFinalState;
  if (lumi) {
    fLumiFilter->Describe();
  }
}

Generator::~Generator() { delete fLumiFilter; }

void Generator::HepMC2G4(const HepMC::GenEvent *evt_orig, G4Event *g4evt) {
  HepMC::GenEvent *evt = new HepMC::GenEvent(*evt_orig);

  if (*(evt->vertices_begin()) == nullptr) {
    std::stringstream ss;
    ss << "SimG4Core/Generator: in event " << g4evt->GetEventID() << " Corrupted Event - GenEvent with no vertex \n";
    throw SimG4Exception(ss.str());
  }

  if (!evt->weights().empty()) {
    weight_ = evt->weights()[0];
    for (unsigned int iw = 1; iw < evt->weights().size(); ++iw) {
      // terminate if the vector of weights contains a zero-weight
      if (evt->weights()[iw] <= 0)
        break;
      weight_ *= evt->weights()[iw];
    }
  }

  if (vtx_ != nullptr) {
    delete vtx_;
  }
  vtx_ = new math::XYZTLorentzVector((*(evt->vertices_begin()))->position().x(),
                                     (*(evt->vertices_begin()))->position().y(),
                                     (*(evt->vertices_begin()))->position().z(),
                                     (*(evt->vertices_begin()))->position().t());

  edm::LogVerbatim("SimG4CoreGenerator") << "Generator: primary Vertex = (" << vtx_->x() << ", " << vtx_->y() << ", "
                                         << vtx_->z() << ")";

  unsigned int ng4vtx = 0;

  for (HepMC::GenEvent::vertex_const_iterator vitr = evt->vertices_begin(); vitr != evt->vertices_end(); ++vitr) {
    // loop for vertex, is it a real vertex?
    // Set qvtx to true for any particles that should be propagated by GEANT,
    // i.e., status 1 particles or status 2 particles that decay outside the
    // beampipe.
    G4bool qvtx = false;
    HepMC::GenVertex::particle_iterator pitr;
    for (pitr = (*vitr)->particles_begin(HepMC::children); pitr != (*vitr)->particles_end(HepMC::children); ++pitr) {
      // For purposes of this function, the status is defined as follows:
      // 1:  particles are not decayed by generator
      // 2:  particles are decayed by generator but need to be propagated by
      // GEANT 3:  particles are decayed by generator but do not need to be
      // propagated by GEANT
      int status = (*pitr)->status();
      int pdg = (*pitr)->pdg_id();
      if (status > 3 && isExotic(pdg) && (!(isExoticNonDetectable(pdg)))) {
        // In Pythia 8, there are many status codes besides 1, 2, 3.
        // By setting the status to 2 for exotic particles, they will be
        // checked: if its decay vertex is outside the beampipe, it will be
        // propagated by GEANT. Some Standard Model particles, e.g., K0, cannot
        // be propagated by GEANT, so do not change their status code.
        status = 2;
      }
      if(std::abs(pdg) == 9900015) status = 3;

      // Particles which are not decayed by generator
      if (status == 1) {
        // filter out unwanted particles and vertices
        if (fPDGFilter && !pdgFilterSel && IsInTheFilterList(pdg)) {
          continue;
        }

        qvtx = true;
        if (verbose > 2)
          LogDebug("SimG4CoreGenerator") << "GenVertex barcode = " << (*vitr)->barcode()
                                         << " selected for GenParticle barcode = " << (*pitr)->barcode();
        break;
      }
      // The selection is made considering if the partcile with status = 2
      // have the end_vertex with a radius greater than the radius of beampipe
      // cylinder (no requirement on the Z of the vertex is applyed).
      else if (status == 2) {
        if ((*pitr)->end_vertex() != nullptr) {
          double xx = (*pitr)->end_vertex()->position().x();
          double yy = (*pitr)->end_vertex()->position().y();
          double r_dd = xx * xx + yy * yy;
          if (r_dd > theDecRCut2) {
            qvtx = true;
            if (verbose > 2)
              LogDebug("SimG4CoreGenerator")
                  << "GenVertex barcode = " << (*vitr)->barcode()
                  << " selected for GenParticle barcode = " << (*pitr)->barcode() << " radius = " << std::sqrt(r_dd);
            break;
          }
        } else {
          // particles with status 2 without end_vertex are
          // equivalent to stable
          qvtx = true;
          break;
        }
      }
    }

    // if this vertex is inside fiductial volume inside the beam pipe
    // and has no long-lived secondary the vertex is not saved
    if (!qvtx) {
      continue;
    }

    double x1 = (*vitr)->position().x() * mm;
    double y1 = (*vitr)->position().y() * mm;
    double z1 = (*vitr)->position().z() * mm;
    double t1 = (*vitr)->position().t() * mm / c_light;

    G4PrimaryVertex *g4vtx = new G4PrimaryVertex(x1, y1, z1, t1);

    for (pitr = (*vitr)->particles_begin(HepMC::children); pitr != (*vitr)->particles_end(HepMC::children); ++pitr) {
      int status = (*pitr)->status();
      int pdg = (*pitr)->pdg_id();
      bool hasDecayVertex = (nullptr != (*pitr)->end_vertex()) ? true : false;

      // Filter on allowed particle species if required
      if (fPDGFilter) {
        bool isInTheList = IsInTheFilterList(pdg);
        if ((!pdgFilterSel && isInTheList) || (pdgFilterSel && !isInTheList)) {
          edm::LogVerbatim("SimG4CoreGenerator")
              << " Skiped GenParticle barcode= " << (*pitr)->barcode() << " PDGid= " << pdg << " status= " << status
              << " isExotic: " << isExotic(pdg) << " isExoticNotDet: " << isExoticNonDetectable(pdg)
              << " isInTheList: " << isInTheList << " hasDecayVertex: " << hasDecayVertex;
          continue;
        }
      }

      edm::LogVerbatim("SimG4CoreGenerator")
          << "Generator: pdg= " << pdg << " status= " << status << " hasPreDefinedDecay: " << hasDecayVertex
          << " isExotic: " << isExotic(pdg) << " isExoticNotDet: " << isExoticNonDetectable(pdg)
          << " isInTheList: " << IsInTheFilterList(pdg) << "\n         (x,y,z,t): (" << x1 << "," << y1 << "," << z1
          << "," << t1 << ")";

      if (status > 3 && isExotic(pdg) && (!(isExoticNonDetectable(pdg)))) {
        status = hasDecayVertex ? 2 : 1;
      }

      // this particle has predefined decay but has no vertex
      if (2 == status && !hasDecayVertex) {
        edm::LogWarning("SimG4CoreGenerator: in event ") << g4evt->GetEventID() << " a particle "
                                                         << " pdgid= " << pdg
                                                         << " has status=2 but has no decay vertex, so will be fully "
                                                            "tracked by Geant4";
        status = 1;
      }
      // heavy neutrino should not be propagated by Geant4
      if(std::abs(pdg) == 9900015) status = 3;

      double x2 = x1;
      double y2 = y1;
      double z2 = z1;
      double decay_length = 0.0;
      if (2 == status) {
        x2 = (*pitr)->end_vertex()->position().x();
        y2 = (*pitr)->end_vertex()->position().y();
        z2 = (*pitr)->end_vertex()->position().z();
        decay_length = std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
      }

      bool toBeAdded = (fFiductialCuts) ? false : true;

      double px = (*pitr)->momentum().px();
      double py = (*pitr)->momentum().py();
      double pz = (*pitr)->momentum().pz();
      double ptot = std::sqrt(px * px + py * py + pz * pz);
      math::XYZTLorentzVector p(px, py, pz, (*pitr)->momentum().e());

      double ximpact = x1;
      double yimpact = y1;
      double zimpact = z1;

      // protection against numerical problems for extremely low momenta
      // compute impact point at transition to Hector
      const double minTan = 1.e-20;
      if (std::abs(z1) < Z_hector && std::abs(pz) >= minTan * ptot) {
        if (pz > 0.0) {
          zimpact = Z_hector;
        } else {
          zimpact = -Z_hector;
        }
        double del = (zimpact - z1) / pz;
        ximpact += del * px;
        yimpact += del * py;
      }
      double rimpact2 = ximpact * ximpact + yimpact * yimpact;

      if (verbose > 2)
        LogDebug("SimG4CoreGenerator") << "Processing GenParticle barcode= " << (*pitr)->barcode() << " pdg= " << pdg
                                       << " status= " << (*pitr)->status() << " st= " << status
                                       << " rimpact(cm)= " << std::sqrt(rimpact2) / cm
                                       << " zimpact(cm)= " << zimpact / cm << " ptot(GeV)= " << ptot
                                       << " pz(GeV)= " << pz;

      // Particles of status 1 trasnported along the beam pipe for forward
      // detectors (HECTOR) always pass to Geant4 without cuts
      if (1 == status && std::abs(zimpact) >= Z_hector && rimpact2 <= theDecRCut2) {
        toBeAdded = true;
        if (verbose > 2)
          LogDebug("SimG4CoreGenerator") << "GenParticle barcode = " << (*pitr)->barcode() << " passed case 3";
      } else {
        // Standard case: particles not decayed by the generator
        if (1 == status && (std::abs(zimpact) < Z_hector || rimpact2 > theDecRCut2)) {
          // Ptot cut for all particles
          if (fPCuts && (ptot < theMinPCut || ptot > theMaxPCut)) {
            continue;
          }
          // phi cut is applied mainly for particle gun
          if (fPhiCuts) {
            double phi = p.phi();
            if (phi < theMinPhiCut || phi > theMaxPhiCut) {
              continue;
            }
          }
          // eta cut is applied if position of the decay
          // is within vacuum chamber and limited in Z
          if (fEtaCuts) {
            // eta cut
            double xi = x1;
            double yi = y1;
            double zi = z1;

            // can be propagated along Z
            if (std::abs(pz) >= minTan * ptot) {
              if ((zi >= Z_lmax) & (pz < 0.0)) {
                zi = Z_lmax;
              } else if ((zi <= Z_lmin) & (pz > 0.0)) {
                zi = Z_lmin;
              } else {
                if (pz > 0) {
                  zi = Z_lmax;
                } else {
                  zi = Z_lmin;
                }
              }
              double del = (zi - z1) / pz;
              xi += del * px;
              yi += del * py;
            }
            // check eta cut
            if ((zi >= Z_lmin) & (zi <= Z_lmax) & (xi * xi + yi * yi < theDecRCut2)) {
              continue;
            }
          }
          if (fLumiFilter && !fLumiFilter->isGoodForLumiMonitor(*pitr)) {
            continue;
          }
          toBeAdded = true;
          if (verbose > 2)
            LogDebug("SimG4CoreGenerator") << "GenParticle barcode = " << (*pitr)->barcode() << " passed case 1";

          // Decay chain outside the fiducial cylinder defined by theRDecLenCut
          // are used for Geant4 tracking with predefined decay channel
          // In the case of decay in vacuum particle is not tracked by Geant4
        } else if (2 == status && x2 * x2 + y2 * y2 >= theDecRCut2 && std::abs(z2) < Z_hector) {
          toBeAdded = true;
          if (verbose > 2)
            LogDebug("SimG4CoreGenerator") << "GenParticle barcode = " << (*pitr)->barcode() << " passed case 2"
                                           << " decay_length(cm)= " << decay_length / cm;
        }
      }
      if (toBeAdded) {
        G4PrimaryParticle *g4prim = new G4PrimaryParticle(pdg, px * GeV, py * GeV, pz * GeV);

        if (g4prim->GetG4code() != nullptr) {
          g4prim->SetMass(g4prim->GetG4code()->GetPDGMass());
          double charge = g4prim->GetG4code()->GetPDGCharge();

          // apply Pt cut
          if (fPtransCut && 0.0 != charge && px * px + py * py < theMinPtCut2) {
            delete g4prim;
            continue;
          }
          g4prim->SetCharge(charge);
        }

        // V.I. do not use SetWeight but the same code
        // value of the code compute inside TrackWithHistory
        // g4prim->SetWeight( 10000*(*vpitr)->barcode() ) ;
        setGenId(g4prim, (*pitr)->barcode());

        if (2 == status) {
          particleAssignDaughters(g4prim, (HepMC::GenParticle *)*pitr, decay_length);
        }
        if (verbose > 1)
          g4prim->Print();
        g4vtx->SetPrimary(g4prim);
        edm::LogVerbatim("SimG4CoreGenerator") << "Generator: new particle pdg= " << pdg << " Ptot(GeV/c)= " << ptot
                                               << " Pt= " << std::sqrt(px * px + py * py);
      }
    }

    if (verbose > 1)
      g4vtx->Print();
    g4evt->AddPrimaryVertex(g4vtx);
    ++ng4vtx;
  }

  // Add a protection for completely empty events (produced by LHCTransport):
  // add a dummy vertex with no particle attached to it
  if (ng4vtx == 0) {
    G4PrimaryVertex *g4vtx = new G4PrimaryVertex(0.0, 0.0, 0.0, 0.0);
    if (verbose > 1)
      g4vtx->Print();
    g4evt->AddPrimaryVertex(g4vtx);
  }

  delete evt;
}

void Generator::particleAssignDaughters(G4PrimaryParticle *g4p, HepMC::GenParticle *vp, double decaylength) {
  if (verbose > 1) {
    LogDebug("SimG4CoreGenerator") << "Special case of long decay length \n"
                                   << "Assign daughters with to mother with decaylength=" << decaylength / cm << " cm";
  }
  math::XYZTLorentzVector p(vp->momentum().px(), vp->momentum().py(), vp->momentum().pz(), vp->momentum().e());

  // defined preassigned decay time
  double proper_time = decaylength / (p.Beta() * p.Gamma() * c_light);
  g4p->SetProperTime(proper_time);

  if (verbose > 2) {
    LogDebug("SimG4CoreGenerator") << " px= " << p.px() << " py= " << p.py() << " pz= " << p.pz() << " e= " << p.e()
                                   << " beta= " << p.Beta() << " gamma= " << p.Gamma()
                                   << " Proper time= " << proper_time / ns << " ns";
  }

  // the particle will decay after the same length if it
  // has not interacted before
  double x1 = vp->end_vertex()->position().x();
  double y1 = vp->end_vertex()->position().y();
  double z1 = vp->end_vertex()->position().z();

  for (HepMC::GenVertex::particle_iterator vpdec = vp->end_vertex()->particles_begin(HepMC::children);
       vpdec != vp->end_vertex()->particles_end(HepMC::children);
       ++vpdec) {
    // transform decay products such that in the rest frame of mother
    math::XYZTLorentzVector pdec(
        (*vpdec)->momentum().px(), (*vpdec)->momentum().py(), (*vpdec)->momentum().pz(), (*vpdec)->momentum().e());

    // children should only be taken into account once
    G4PrimaryParticle *g4daught =
        new G4PrimaryParticle((*vpdec)->pdg_id(), pdec.x() * GeV, pdec.y() * GeV, pdec.z() * GeV);

    if (g4daught->GetG4code() != nullptr) {
      g4daught->SetMass(g4daught->GetG4code()->GetPDGMass());
      g4daught->SetCharge(g4daught->GetG4code()->GetPDGCharge());
    }

    // V.I. do not use SetWeight but the same code
    // value of the code compute inside TrackWithHistory
    setGenId(g4daught, (*vpdec)->barcode());

    if (verbose > 2)
      LogDebug("SimG4CoreGenerator") << "Assigning a " << (*vpdec)->pdg_id() << " as daughter of a " << vp->pdg_id();

    bool checkStatus = fFinalState ? (*vpdec)->status() > 3
      : ((*vpdec)->status() == 23 && std::abs(vp->pdg_id()) == 1000015);   
    if (((*vpdec)->status() == 2 || checkStatus) &&
        (*vpdec)->end_vertex() != nullptr) {
      double x2 = (*vpdec)->end_vertex()->position().x();
      double y2 = (*vpdec)->end_vertex()->position().y();
      double z2 = (*vpdec)->end_vertex()->position().z();
      double dd = std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
      particleAssignDaughters(g4daught, *vpdec, dd);
    }
    (*vpdec)->set_status(1000 + (*vpdec)->status());
    g4p->SetDaughter(g4daught);

    if (verbose > 1)
      g4daught->Print();
  }
}

// Used for non-beam particles
bool Generator::particlePassesPrimaryCuts(const G4ThreeVector &p) const {
  bool flag = true;
  double ptot = p.mag();
  if (fPCuts && (ptot < theMinPCut * GeV || ptot > theMaxPCut * GeV)) {
    flag = false;
  }
  if (fEtaCuts && flag) {
    double pz = p.z();
    if (ptot < pz + 1.e-10) {
      flag = false;

    } else {
      double eta = 0.5 * G4Log((ptot + pz) / (ptot - pz));
      if (eta < theMinEtaCut || eta > theMaxEtaCut) {
        flag = false;
      }
    }
  }
  if (fPhiCuts && flag) {
    double phi = p.phi();
    if (phi < theMinPhiCut || phi > theMaxPhiCut) {
      flag = false;
    }
  }

  if (verbose > 2)
    LogDebug("SimG4CoreGenerator") << "Generator ptot(GeV)= " << ptot / GeV << " eta= " << p.eta()
                                   << "  phi= " << p.phi() << " Flag= " << flag;

  return flag;
}

bool Generator::isExotic(int pdgcode) const {
  int pdgid = std::abs(pdgcode);
  return ((pdgid >= 1000000 && pdgid < 4000000 && pdgid != 3000022) ||  // SUSY, R-hadron, and technicolor particles
          pdgid == 17 ||                                                // 4th generation lepton
          pdgid == 34 ||                                                // W-prime
          pdgid == 37)                                                  // charged Higgs
             ? true
             : false;
}

bool Generator::isExoticNonDetectable(int pdgcode) const {
  int pdgid = std::abs(pdgcode);
  HepPDT::ParticleID pid(pdgcode);
  int charge = pid.threeCharge();
  return ((charge == 0) && (pdgid >= 1000000 && pdgid < 1000040))  // SUSY
             ? true
             : false;
}

bool Generator::IsInTheFilterList(int pdgcode) const {
  int pdgid = std::abs(pdgcode);
  for (auto &pdg : pdgFilter) {
    if (std::abs(pdg) == pdgid) {
      return true;
    }
  }
  return false;
}

void Generator::nonBeamEvent2G4(const HepMC::GenEvent *evt, G4Event *g4evt) {
  int i = 0;
  for (HepMC::GenEvent::particle_const_iterator it = evt->particles_begin(); it != evt->particles_end(); ++it) {
    ++i;
    HepMC::GenParticle *gp = (*it);
    int g_status = gp->status();
    // storing only particle with status == 1
    if (g_status == 1) {
      int g_id = gp->pdg_id();
      G4PrimaryParticle *g4p =
          new G4PrimaryParticle(g_id, gp->momentum().px() * GeV, gp->momentum().py() * GeV, gp->momentum().pz() * GeV);
      if (g4p->GetG4code() != nullptr) {
        g4p->SetMass(g4p->GetG4code()->GetPDGMass());
        g4p->SetCharge(g4p->GetG4code()->GetPDGCharge());
      }
      setGenId(g4p, i);
      if (particlePassesPrimaryCuts(g4p->GetMomentum())) {
        G4PrimaryVertex *v = new G4PrimaryVertex(gp->production_vertex()->position().x() * mm,
                                                 gp->production_vertex()->position().y() * mm,
                                                 gp->production_vertex()->position().z() * mm,
                                                 gp->production_vertex()->position().t() * mm / c_light);
        v->SetPrimary(g4p);
        g4evt->AddPrimaryVertex(v);
        if (verbose > 0) {
          v->Print();
        }

      } else {
        delete g4p;
      }
    }
  }  // end loop on HepMC particles
}
