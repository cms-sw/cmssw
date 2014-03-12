#ifndef MuonGEMHits_H
#define MuonGEMHits_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Validation/MuonGEMHits/interface/GEMHitsValidation.h"
#include "Validation/MuonGEMHits/interface/GEMSimTrackMatch.h"
class MuonGEMHits : public edm::EDAnalyzer
{
public:
  explicit MuonGEMHits(const edm::ParameterSet&);
  ~MuonGEMHits();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;

  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  DQMStore* dbe_;
  std::string outputFile_;
  std::string simInputLabel_;

  GEMHitsValidation* theGEMHitsValidation;
  GEMSimTrackMatch* theGEMSimTrackMatch;
  

  edm::ESHandle<GEMGeometry> gem_geom;

  const GEMGeometry* gem_geometry_;
  bool hasGEMGeometry_;

  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;
};
#endif
