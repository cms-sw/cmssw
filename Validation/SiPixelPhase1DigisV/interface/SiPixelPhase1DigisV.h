#ifndef SiPixelPhase1DigisV_h // Can we use #pragma once?
#define SiPixelPhase1DigisV_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1DigisV
// Class  :     SiPixelPhase1DigisV
// 

// Original Author: Marcel Schneider
// Additional Authors: Alexander Morton - modifying code for validation use

// Input data stuff
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

// PixelDQM Framework
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"

class SiPixelPhase1DigisV : public SiPixelPhase1Base {
  // List of quantities to be plotted. 
  enum {
    ADC, // digi ADC readouts
    NDIGIS, // number of digis per event and module
    ROW, // number of digis per row
    COLUMN, // number of digis per column

    MAX_HIST // a sentinel that gives the number of quantities (not a plot).
  };
  public:
  explicit SiPixelPhase1DigisV(const edm::ParameterSet& conf);

  void analyze(const edm::Event&, const edm::EventSetup&) ;

  private:
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> srcToken_;

};

class SiPixelPhase1DigisHarvesterV : public SiPixelPhase1Harvester {
  enum {
    ADC, // digi ADC readouts
    NDIGIS, // number of digis per event and module
    ROW, // number of digis per row
    COLUMN, // number of digis per column

    MAX_HIST
  };
  public:
  explicit SiPixelPhase1DigisHarvesterV(const edm::ParameterSet& conf);

};

#endif

