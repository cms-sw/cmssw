#ifndef GEMRecHitsValidation_H
#define GEMRecHitsValidation_H

#include "Validation/MuonGEMHits/interface/GEMBaseValidation.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/GEMRecHit/interface/GEMRecHit.h>
#include <DataFormats/GEMRecHit/interface/GEMRecHitCollection.h>

class GEMRecHitsValidation : public GEMBaseValidation
{
public:
  explicit GEMRecHitsValidation( const edm::ParameterSet& );
  ~GEMRecHitsValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
 private:

  //bool isGEMRecHitMatched(const MyGEMRecHit& gem_recHit_, const MyGEMSimHit& gem_sh);

  MonitorElement* gem_rh_xy[2][3][2];
  MonitorElement* gem_rh_zr[2][3][2];
 
  
  edm::EDGetToken InputTagToken_RH;
  Int_t npart;
  

};

#endif
