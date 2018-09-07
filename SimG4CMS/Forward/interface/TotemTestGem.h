#ifndef Forward_TotemTestGem_h
#define Forward_TotemTestGem_h 1
// -*- C++ -*-
//
// Package:     Forward
// Class  :     TotemTestGem
//
/**\class TotemTestGem TotemTestGem.h SimG4CMS/Forward/interface/TotemTestGem.h
 
 Description: Manages Root file creation for Totem Tests
 
 Usage:
    Used in testing Totem simulation
 
*/
//
// Original Author: 
//         Created:  Tue May 16 10:14:34 CEST 2006
//
 
// system include files
#include <iostream>
#include <memory>
#include <vector>
#include <string>

// user include files
#include "SimG4Core/Notification/interface/Observer.h"
#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfEvent.h"
#include "SimG4Core/Notification/interface/EndOfEvent.h"
#include "SimG4Core/Watcher/interface/SimProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Forward/interface/TotemTestHistoClass.h"
#include "SimG4CMS/Forward/interface/TotemG4Hit.h"

class G4Step;

class TotemTestGem : public SimProducer,
		     public Observer<const BeginOfEvent *>,
		     public Observer<const EndOfEvent *> {

public: 

  TotemTestGem(const edm::ParameterSet &p);
  ~TotemTestGem() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

protected:
  // observer classes
  void update(const BeginOfEvent * evt) override;
  void update(const EndOfEvent * evt) override;

private:
  void clear();
  void fillEvent(TotemTestHistoClass&);

private:

  //Keep parameters and internal memory
  std::vector<std::string>                names;
  int                                     evtnum;
  std::vector<TotemG4Hit*>                hits;
 
};

#endif
