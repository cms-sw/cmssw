#ifndef GEMBaseValidation_H
#define GEMBaseValidation_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMDigi.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigi.h"
#include "DataFormats/GEMDigi/interface/GEMCSCPadDigiCollection.h"


class GEMBaseValidation
{
public:
  GEMBaseValidation(DQMStore* dbe,
                         const edm::InputTag & inputTag);
  virtual ~GEMBaseValidation();
  //virtual void analyze(const edm::Event& e, const edm::EventSetup&) = 0;

  void setGeometry(const GEMGeometry* geom) { theGEMGeometry = geom; }
  // void set SimHitMap(const PSimHitMap* simHitMap) { theSimHitMap = simHitMap; } 


 protected:

  DQMStore* dbe_;
  edm::InputTag theInputTag;
  const GEMGeometry* theGEMGeometry;

};

#endif
