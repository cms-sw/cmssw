// Debug watcher for quirk pairs, port of the Athena DebugSteppingAction:
// every DebugStep mm of flight prints the total 4-momentum of quirks plus
// string, the angular momentum and the center of mass; per event it prints
// the step count, the number of suspensions and the largest pair separation.
// With DumpEvery > 0 every N-th quirk step is written to DumpFile (one file per thread).

#include "SimG4Core/Watcher/interface/SimWatcher.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/CustomPhysics/interface/Quirk.h"
#include "SimG4Core/CustomPhysics/interface/QuirkStringStore.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4Event.hh"
#include "G4Threading.hh"
#include <CLHEP/Units/SystemOfUnits.h>
#include <CLHEP/Units/PhysicalConstants.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

class QuirkDebugWatcher : public SimWatcher,
                          public Observer<const BeginOfEvent*>,
                          public Observer<const EndOfEvent*>,
                          public Observer<const G4Step*> {
public:
  explicit QuirkDebugWatcher(edm::ParameterSet const& p);
  ~QuirkDebugWatcher() override = default;

  void update(const BeginOfEvent*) override;
  void update(const EndOfEvent*) override;
  void update(const G4Step*) override;

private:
  void printConservation(int i, const G4Track* track, const G4StepPoint* ps);

  double debugStep_;
  int verbose_;
  long nSteps_[2]{0, 0};
  long nSuspend_[2]{0, 0};
  int iStep_[2]{0, 0};
  double maxDist_{0};
  double edep_[2]{0, 0};
  G4LorentzVector x_[2], p_[2], p0_[2];
  std::chrono::steady_clock::time_point start_;
  int dumpEvery_;
  std::unique_ptr<std::ofstream> dump_;
  int eventID_{0};
};

QuirkDebugWatcher::QuirkDebugWatcher(edm::ParameterSet const& p) {
  // one instance per worker thread, string state is thread local
  setMT(true);
  edm::ParameterSet ps = p.getParameter<edm::ParameterSet>("QuirkDebugWatcher");
  debugStep_ = ps.getUntrackedParameter<double>("DebugStep", 0.) * CLHEP::mm;
  verbose_ = ps.getUntrackedParameter<int>("Verbose", 0);
  dumpEvery_ = ps.getUntrackedParameter<int>("DumpEvery", 0);
  if (dumpEvery_ > 0) {
    std::string name = ps.getUntrackedParameter<std::string>("DumpFile", "quirkSteps") + "_" +
                       std::to_string(G4Threading::G4GetThreadId()) + ".txt";
    dump_ = std::make_unique<std::ofstream>(name);
    *dump_ << std::setprecision(9);
  }
  edm::LogVerbatim("QuirkDebugWatcher") << "QuirkDebugWatcher: conservation printout every " << debugStep_ << " mm";
}

void QuirkDebugWatcher::update(const BeginOfEvent* evt) {
  eventID_ = (*evt)()->GetEventID();
  for (int i = 0; i < 2; ++i) {
    nSteps_[i] = nSuspend_[i] = iStep_[i] = 0;
    edep_[i] = 0;
    x_[i] = p_[i] = p0_[i] = G4LorentzVector();
  }
  maxDist_ = 0;
  start_ = std::chrono::steady_clock::now();
}

void QuirkDebugWatcher::update(const EndOfEvent* evt) {
  double cpu = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
  std::ostringstream os;
  os << std::setprecision(6) << "QuirkDebugWatcher: event " << (*evt)()->GetEventID() << " steps " << nSteps_[0] << " "
     << nSteps_[1] << " suspensions " << nSuspend_[0] << " " << nSuspend_[1] << " max separation(mm) " << maxDist_
     << " edep(MeV) " << edep_[0] << " " << edep_[1] << " time(s) " << cpu;
  for (int i = 0; i < 2; ++i) {
    os << "\n  quirk " << i << " start p " << p0_[i] << " end x " << x_[i] << " end p " << p_[i];
  }
  edm::LogVerbatim("QuirkDebugWatcher") << os.str();
  if (dump_) {
    dump_->flush();
  }
}

void QuirkDebugWatcher::update(const G4Step* step) {
  const G4Track* track = step->GetTrack();
  if (!Quirk::isQuirk(track->GetParticleDefinition())) {
    return;
  }
  const int i = (track->GetParticleDefinition()->GetPDGEncoding() > 0) ? 0 : 1;
  const G4StepPoint* ps = step->GetPostStepPoint();
  ++nSteps_[i];
  if (track->GetTrackStatus() == fSuspend) {
    ++nSuspend_[i];
  }
  edep_[i] += step->GetTotalEnergyDeposit();
  if (track->GetCurrentStepNumber() == 1) {
    iStep_[i] = 0;
    p0_[i] = G4LorentzVector(step->GetPreStepPoint()->GetMomentum(), step->GetPreStepPoint()->GetTotalEnergy());
  }
  x_[i] = G4LorentzVector(ps->GetPosition(), CLHEP::c_light * ps->GetGlobalTime());
  p_[i] = track->GetDynamicParticle()->Get4Momentum();
  if (nSteps_[0] > 0 && nSteps_[1] > 0) {
    // last positions of the two quirks, not at equal times: an upper estimate of the separation
    maxDist_ = std::max(maxDist_, (x_[1].vect() - x_[0].vect()).mag());
  }
  if (verbose_ > 1) {
    edm::LogVerbatim("QuirkDebugWatcher")
        << std::setprecision(10) << "quirk " << i << " step " << track->GetCurrentStepNumber() << " x " << x_[i]
        << " p " << p_[i] << " status " << track->GetTrackStatus();
  }
  if (dump_ && (nSteps_[i] % dumpEvery_ == 0 || track->GetTrackStatus() == fStopAndKill)) {
    // event quirk step x y z [mm] t [ns] px py pz [MeV] status
    *dump_ << eventID_ << ' ' << i << ' ' << track->GetCurrentStepNumber() << ' ' << ps->GetPosition().x() << ' '
           << ps->GetPosition().y() << ' ' << ps->GetPosition().z() << ' ' << ps->GetGlobalTime() << ' ' << p_[i].x()
           << ' ' << p_[i].y() << ' ' << p_[i].z() << ' ' << track->GetTrackStatus() << '\n';
  }
  if (debugStep_ > 0 && x_[i].vect().mag() >= iStep_[i] * debugStep_) {
    ++iStep_[i];
    printConservation(i, track, ps);
  }
}

void QuirkDebugWatcher::printConservation(int i, const G4Track* track, const G4StepPoint* ps) {
  const QuirkInfracolorForce* s[2];
  s[i] = &QuirkStringStore::instance().stringFor(track->GetParticleDefinition()->GetPDGEncoding());
  s[1 - i] = s[i]->GetReactionForce();
  const double force = s[0]->GetStringForce();
  G4LorentzVector ss[2] = {s[0]->GetSumStrings(), s[1]->GetSumStrings()};
  G4LorentzVector dx = ss[0] - ss[1];
  G4LorentzVector ptot = force * (ss[0] + ss[1]) + p_[0] + p_[1];
  G4LorentzVector p1s = p_[1] + force * ss[1];
  G4ThreeVector L = dx.vect().cross(p1s);
  G4ThreeVector Excm = p1s.t() * dx.vect() - p1s.vect() * dx.t();
  for (int k = 0; k < 2; ++k) {
    L += s[k]->GetAngMomentum();
    Excm += s[k]->GetMomentOfE();
  }
  L -= Excm.cross(ptot.vect() / ptot.t());
  Excm += ptot.t() * x_[0].vect() - ptot.vect() * x_[0].t();
  edm::LogVerbatim("QuirkDebugWatcher") << std::setprecision(10) << "QuirkCons quirk " << i << " step "
                                        << track->GetCurrentStepNumber() << " t(ns) " << ps->GetGlobalTime() << " ptot "
                                        << ptot << " L " << L << " cm " << Excm / ptot.t() << " |dx| "
                                        << (x_[1] - x_[0]).vect().mag() << " nstr " << s[0]->GetNStrings() << " "
                                        << s[1]->GetNStrings();
}

DEFINE_SIMWATCHER(QuirkDebugWatcher);
