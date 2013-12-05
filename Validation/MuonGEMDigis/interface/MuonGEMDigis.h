#ifndef MuonGEMDigis_H
#define MuonGEMDigis_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "SimMuon/MCTruth/interface/PSimHitMap.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
#include "Validation/MuonGEMDigis/interface/GEMCSCPadDigiValidation.h"
#include "Validation/MuonGEMDigis/interface/GEMCSCCoPadDigiValidation.h"
#include "Validation/MuonGEMDigis/interface/GEMTrackMatch.h"

class GEMStripDigiValidation;

class MuonGEMDigis : public edm::EDAnalyzer
{
public:
  /// constructor
  explicit MuonGEMDigis(const edm::ParameterSet&);
  /// destructor
  ~MuonGEMDigis();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);

  virtual void beginJob() ;

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual void endJob() ;

  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  DQMStore* dbe_;
  std::string outputFile_;

  GEMStripDigiValidation* theGEMStripDigiValidation;
  GEMCSCPadDigiValidation* theGEMCSCPadDigiValidation;
  GEMCSCCoPadDigiValidation* theGEMCSCCoPadDigiValidation;
  GEMTrackMatch* theGEMTrackMatch;
  

  void buildLUT();
  std::pair<int,int> getClosestChambers(int region, float phi);


  edm::Handle<GEMDigiCollection> gem_digis;
  edm::Handle<GEMCSCPadDigiCollection> gemcscpad_digis;
  edm::Handle<GEMCSCPadDigiCollection> gemcsccopad_digis;
  edm::ESHandle<GEMGeometry> gem_geo_;

  const GEMGeometry* gem_geometry_;

  edm::InputTag input_tag_gem_;
  edm::InputTag input_tag_gemcscpad_;
  edm::InputTag input_tag_gemcsccopad_;


  //edm::ParameterSet cfg_;
  //float minPt_;


  std::pair<std::vector<float>,std::vector<int> > positiveLUT_;
  std::pair<std::vector<float>,std::vector<int> > negativeLUT_;
};
#endif
