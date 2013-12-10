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
  /// constructor
  explicit MuonGEMHits(const edm::ParameterSet&);
  /// destructor
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

  GEMHitsValidation* theGEMHitsValidation;
  GEMSimTrackMatch* theGEMSimTrackMatch;
  

  void buildLUT();
  std::pair<int,int> getClosestChambers(int region, float phi);


  edm::ESHandle<GEMGeometry> gem_geom;

  const GEMGeometry* gem_geometry_;



  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;
};
#endif
