#include "SimG4Core/MagneticField/test/FieldStepWatcher.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"

#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4TransportationManager.hh"
#include "G4VPhysicalVolume.hh"

FieldStepWatcher::FieldStepWatcher(const edm::ParameterSet &p) {
  level = p.getUntrackedParameter<int>("DepthLevel", 2);
  outFile = p.getUntrackedParameter<std::string>("OutputFile", "field.root");

  edm::LogInfo("FieldStepWatcher") << "FieldStepWatcher initialised to monitor"
                                   << " level " << level << " with o/p on " << outFile;
  dbe_ = edm::Service<DQMStore>().operator->();
}

FieldStepWatcher::~FieldStepWatcher() {
  if (dbe_ && !outFile.empty())
    dbe_->save(outFile);
}

void FieldStepWatcher::update(const BeginOfRun *) {
  G4VPhysicalVolume *pv =
      G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume();
  findTouch(pv, 0);
  lvnames.push_back("Not Found");

  edm::LogInfo("FieldStepWatcher") << "FieldStepWatcher: Finds " << lvnames.size() << " different volumes"
                                   << " at level " << level;
  unsigned num = lvnames.size();
  steps.push_back(0);
  for (unsigned int i = 0; i < num; i++) {
    edm::LogInfo("FieldStepWatcher") << "FieldStepWatcher: lvnames[" << i << "] = " << lvnames[i];
    steps.push_back(0);
  }

  if (dbe_) {
    char titl[60], name[20];
    for (unsigned int i = 0; i <= lvnames.size(); i++) {
      std::string lvname = "All";
      if (i != 0)
        lvname = lvnames[i - 1];
      sprintf(name, "Step%d", i);
      sprintf(titl, "Step Length in Volume %s", lvname.c_str());
      MonitorElement *m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStep.push_back(m1);
      sprintf(name, "Call%d", i);
      sprintf(titl, "Number of steps in Volume %s", lvname.c_str());
      MonitorElement *m2 = dbe_->book1D(name, titl, 50000, 0., 5000000.);
      meCall.push_back(m2);
      sprintf(name, "StepE%d", i);
      sprintf(titl, "Step Length for Electrons in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepE.push_back(m1);
      sprintf(name, "StepG%d", i);
      sprintf(titl, "Step Length for Photons in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepG.push_back(m1);
      sprintf(name, "StepMu%d", i);
      sprintf(titl, "Step Length for Muons in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepMu.push_back(m1);
      sprintf(name, "StepNu%d", i);
      sprintf(titl, "Step Length for Neutrinos in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepNu.push_back(m1);
      sprintf(name, "StepCH%d", i);
      sprintf(titl, "Step Length for Charged Hadrons in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepCH.push_back(m1);
      sprintf(name, "StepNH%d", i);
      sprintf(titl, "Step Length for Neutral Hadrons in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepNH.push_back(m1);
      sprintf(name, "StepC%d", i);
      sprintf(titl, "Step Length for Charged Particles in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepC.push_back(m1);
      sprintf(name, "StepN%d", i);
      sprintf(titl, "Step Length for Neutral Particles in Volume %s", lvname.c_str());
      m1 = dbe_->book1D(name, titl, 5000, 0., 10000.);
      meStepN.push_back(m1);
    }
  }
}

void FieldStepWatcher::update(const BeginOfEvent *) {
  for (unsigned int i = 0; i < steps.size(); i++)
    steps[i] = 0;
}

void FieldStepWatcher::update(const EndOfEvent *) {
  if (dbe_) {
    for (unsigned int i = 0; i < steps.size(); i++)
      meCall[i]->Fill(steps[i]);
  }
}

void FieldStepWatcher::update(const G4Step *aStep) {
  if (aStep) {
    G4StepPoint *preStepPoint = aStep->GetPreStepPoint();
    const G4VTouchable *pre_touch = preStepPoint->GetTouchable();
    int pre_level = ((pre_touch->GetHistoryDepth()) + 1);
    std::string name = "Not Found";
    if (pre_level > 0 && pre_level >= level) {
      int ii = pre_level - level;
      name = pre_touch->GetVolume(ii)->GetName();
    }
    int indx = findName(name);
    double charge = aStep->GetTrack()->GetDefinition()->GetPDGCharge();
    int code = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
    LogDebug("FieldStepWatcher") << "FieldStepWatcher:: Step at Level " << pre_level << " with " << name << " at"
                                 << " level " << level << " corresponding"
                                 << " to index " << indx << " due to "
                                 << "particle " << code << " of charge " << charge;
    steps[0]++;
    if (indx >= 0) {
      int i = indx + 1;
      steps[indx + 1]++;
      if (dbe_) {
        meStep[0]->Fill(aStep->GetStepLength());
        meStep[i]->Fill(aStep->GetStepLength());
        if (code == 11 || code == -11) {
          meStepE[0]->Fill(aStep->GetStepLength());
          meStepE[i]->Fill(aStep->GetStepLength());
        } else if (code == 22) {
          meStepG[0]->Fill(aStep->GetStepLength());
          meStepG[i]->Fill(aStep->GetStepLength());
        } else if (code == 13 || code == -13) {
          meStepMu[0]->Fill(aStep->GetStepLength());
          meStepMu[i]->Fill(aStep->GetStepLength());
        } else if (code == 12 || code == -12 || code == 14 || code == -14 || code == 16 || code == -16) {
          meStepNu[0]->Fill(aStep->GetStepLength());
          meStepNu[i]->Fill(aStep->GetStepLength());
        } else if (charge == 0) {
          meStepNH[0]->Fill(aStep->GetStepLength());
          meStepNH[i]->Fill(aStep->GetStepLength());
        } else {
          meStepCH[0]->Fill(aStep->GetStepLength());
          meStepCH[i]->Fill(aStep->GetStepLength());
        }
        if (charge == 0) {
          meStepN[0]->Fill(aStep->GetStepLength());
          meStepN[i]->Fill(aStep->GetStepLength());
        } else {
          meStepC[0]->Fill(aStep->GetStepLength());
          meStepC[i]->Fill(aStep->GetStepLength());
        }
      }
    }
  }
}

void FieldStepWatcher::findTouch(G4VPhysicalVolume *pv, int leafDepth) {
  if (leafDepth == 0)
    fHistory.SetFirstEntry(pv);
  else
    fHistory.NewLevel(pv, kNormal, pv->GetCopyNo());

  G4LogicalVolume *lv = pv->GetLogicalVolume();
  LogDebug("FieldStepWatcher") << "FieldStepWatcher::find Touch " << lv->GetName() << " at level " << leafDepth;
  if (leafDepth == level - 1) {
    const std::string &lvname = lv->GetName();
    if (findName(lvname) < 0)
      lvnames.push_back(lvname);
  } else if (leafDepth < level - 1) {
    int noDaughters = lv->GetNoDaughters();
    while ((noDaughters--) > 0) {
      G4VPhysicalVolume *pvD = lv->GetDaughter(noDaughters);
      if (!pvD->IsReplicated())
        findTouch(pvD, leafDepth + 1);
    }
  }
  if (leafDepth > 0)
    fHistory.BackLevel();
}

int FieldStepWatcher::findName(std::string name) {
  for (unsigned int i = 0; i < lvnames.size(); i++)
    if (name == lvnames[i])
      return i;
  return -1;
}

// define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

DEFINE_SIMWATCHER(FieldStepWatcher);
