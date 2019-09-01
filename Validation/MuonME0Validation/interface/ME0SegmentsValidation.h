#ifndef ME0SegmentsValidation_H
#define ME0SegmentsValidation_H

#include "Validation/MuonME0Validation/interface/ME0BaseValidation.h"

#include <DataFormats/GEMRecHit/interface/ME0Segment.h>
#include <DataFormats/GEMRecHit/interface/ME0SegmentCollection.h>

#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include <DataFormats/GEMRecHit/interface/ME0RecHit.h>

class ME0SegmentsValidation : public ME0BaseValidation {
public:
  explicit ME0SegmentsValidation(const edm::ParameterSet &);
  ~ME0SegmentsValidation() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &) override;
  std::pair<int, int> isMatched(ME0DetId, LocalPoint, edm::Handle<ME0DigiPreRecoCollection>);
  bool isSimTrackGood(edm::SimTrackContainer::const_iterator simTrack);
  bool isSimMatched(edm::SimTrackContainer::const_iterator, edm::PSimHitContainer::const_iterator);

private:
  MonitorElement *me0_specRH_xy[2][6];
  MonitorElement *me0_rh_xy_Muon[2][6];
  MonitorElement *me0_specRH_zr[2];

  MonitorElement *me0_segment_chi2, *me0_segment_redchi2, *me0_segment_ndof;
  MonitorElement *me0_segment_time, *me0_segment_timeErr;
  MonitorElement *me0_segment_numRH, *me0_segment_numRHSig, *me0_segment_numRHBkg;
  MonitorElement *me0_segment_EtaRH, *me0_segment_PhiRH, *me0_segment_size;

  MonitorElement *me0_simsegment_eta, *me0_simsegment_pt, *me0_simsegment_phi;
  MonitorElement *me0_matchedsimsegment_eta, *me0_matchedsimsegment_pt, *me0_matchedsimsegment_phi;

  MonitorElement *me0_specRH_DeltaX[2][6];
  MonitorElement *me0_specRH_DeltaY[2][6];
  MonitorElement *me0_specRH_PullX[2][6];
  MonitorElement *me0_specRH_PullY[2][6];

  edm::EDGetToken InputTagToken_Segments;
  edm::EDGetToken InputTagToken_Digis;
  edm::EDGetToken InputTagToken_;
  edm::EDGetToken InputTagTokenST_;

  int npart;
  double sigma_x_, sigma_y_;
  double eta_max_, eta_min_;
  double pt_min_;
  bool isMuonGun_;

  typedef std::map<edm::SimTrackContainer::const_iterator, edm::PSimHitContainer> MapTypeSim;
  typedef std::map<ME0SegmentCollection::const_iterator, std::vector<ME0RecHit>> MapTypeSeg;
};

#endif
