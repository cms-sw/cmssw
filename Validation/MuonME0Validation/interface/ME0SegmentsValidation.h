#ifndef ME0SegmentsValidation_H
#define ME0SegmentsValidation_H

#include "Validation/MuonME0Validation/interface/ME0BaseValidation.h"

#include <DataFormats/GEMRecHit/interface/ME0Segment.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"


class ME0SegmentsValidation : public ME0BaseValidation
{
public:
  explicit ME0SegmentsValidation( const edm::ParameterSet& );
  ~ME0SegmentsValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
  std::pair<int,int> isMatched(auto, auto, auto );
 private:

  MonitorElement* me0_specRH_xy[2][6];
  MonitorElement* me0_rh_xy_Muon[2][6];
  MonitorElement* me0_specRH_zr[2];
  
  MonitorElement *me0_segment_chi2, *me0_segment_redchi2, *me0_segment_ndof;
  MonitorElement *me0_segment_time, *me0_segment_timeErr;
  MonitorElement *me0_segment_numRH, *me0_segment_numRHSig, *me0_segment_numRHBkg;
  MonitorElement *me0_segment_EtaRH, *me0_segment_PhiRH, *me0_segment_size;
  
  MonitorElement* me0_specRH_DeltaX[2][6];
  MonitorElement* me0_specRH_DeltaY[2][6];
  MonitorElement* me0_specRH_PullX[2][6];
  MonitorElement* me0_specRH_PullY[2][6];
  
  edm::EDGetToken InputTagToken_Segments;
  edm::EDGetToken InputTagToken_Digis;
  
  Int_t npart;
  
};
  
#endif
