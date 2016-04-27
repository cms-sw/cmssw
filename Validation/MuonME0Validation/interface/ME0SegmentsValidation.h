#ifndef ME0SegmentsValidation_H
#define ME0SegmentsValidation_H

#include "Validation/MuonME0Validation/interface/ME0BaseValidation.h"

#include <DataFormats/GEMRecHit/interface/ME0Segment.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>




class ME0SegmentsValidation : public ME0BaseValidation
{
public:
  explicit ME0SegmentsValidation( const edm::ParameterSet& );
  ~ME0SegmentsValidation();
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup&) override;
 private:

  MonitorElement* me0_specRH_xy[2][6];
  MonitorElement* me0_rh_xy_Muon[2][6];
  MonitorElement* me0_specRH_zr[2];
  
  MonitorElement* me0_segment_chi2;
  MonitorElement *me0_segment_time, *me0_segment_timeErr;
  MonitorElement* me0_segment_numRH;
  MonitorElement *me0_segment_EtaRH,*me0_segment_PhiRH;
  
  MonitorElement* me0_specRH_DeltaX[2][6];
  MonitorElement* me0_specRH_DeltaY[2][6];
  MonitorElement* me0_specRH_PullX[2][6];
  MonitorElement* me0_specRH_PullY[2][6];
  
  edm::EDGetToken InputTagToken_Segments;
  
  Int_t npart;
  
  
};
  
#endif
