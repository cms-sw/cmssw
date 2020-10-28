#ifndef __SimTracker_SiPhase2Digitizer_PixelTestBeamValidation_h
#define __SimTracker_SiPhase2Digitizer_PixelTestBeamValidation_h

// -*- C++ -*-
//
// Package:    SimTracker/SiPhase2Digitizer
// Class:      PixelTestBeamValidation
//
/**\class PixelTestBeamValidation PixelTestBeamValidation.cc
 
  Description: Access Digi collection and creates histograms accumulating
               data from the module to different pixel cells, 1x1 cell, 
               2x2 cell, etc..

*/
// Author:  J.Duarte-Campderros (IFCA)
// Created: 2019-10-02

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// CMSSW framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// CMSSW Data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/DetId/interface/DetId.h"

// system
#include <string>
#include <set>
#include <vector>
#include <map>
#include <functional>
#include <cstdint>

//class MonitorElement;
class GeomDet;
class PSimHit;
class PixelGeomDetUnit;

class PixelTestBeamValidation : public DQMEDAnalyzer {
public:
  explicit PixelTestBeamValidation(const edm::ParameterSet &);
  ~PixelTestBeamValidation() override;

  void dqmBeginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) override;
  void bookHistograms(DQMStore::IBooker &ibooker, edm::Run const &iRun, edm::EventSetup const &iSetup) override;
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  // GeomDet units belonging to the tracker pixel system (Barrel and endcap)
  bool isPixelSystem_(const GeomDet *detunit) const;

  // The detector unit associated to a list of histograms
  // Barrel_Layer / Endcap_Layer_side: Need 3 numbers
  int meUnit_(bool isBarrel, int layer, int side) const;

  // Get the pixel digi found at that channel for a given subdetector unit
  const PixelDigi &get_digi_from_channel_(int ch, const edm::DetSetVector<PixelDigi>::const_iterator &itdigis);

  // Get the sim track with trackId for given subdetector unit
  const SimTrack *get_simtrack_from_id_(unsigned int tid, const edm::SimTrackContainer *simtracks);

  // Get the set of simhits created by a track
  const std::vector<const PSimHit *> get_simhits_from_trackid_(
      unsigned int tid, unsigned int detid_raw, const std::vector<const edm::PSimHitContainer *> &psimhits);

  // Helper container to memoize the SimHits created by a track (Id) per event
  // This should be cleared at the begining of the event
  // Note that the PSimHit needs to store the pointer, as the
  // memoizer functions present ar
  std::map<unsigned int, std::map<unsigned int, std::vector<const PSimHit *>>> m_tId_det_simhits_;

  // Transform a given measurement points into the i-cell pixel frame
  const std::pair<double, double> pixel_cell_transformation_(const std::pair<double, double> &pos,
                                                             unsigned int icell,
                                                             const std::pair<double, double> &pitch);
  const std::pair<double, double> pixel_cell_transformation_(const MeasurementPoint &pos,
                                                             unsigned int icell,
                                                             const std::pair<double, double> &pitch);

  // Check if the given local position is close enough to the given channel,
  // the "tolerance" argument is quantifying 'close enough' in a square
  //bool channel_iluminated_by_(const MeasurementPoint & localpos,int channel, double tolerance) const;
  bool channel_iluminated_by_(const PSimHit &localpos, int channel, const PixelGeomDetUnit *tkDet);

  // The list of channels illuminated by the PSimHit
  std::set<int> get_illuminated_channels_(const PSimHit &ps,
                                          const DetId &detid,
                                          const edm::DetSetVector<PixelDigiSimLink> *simdigis);

  // The list of pixels illuminated by the PSimHit
  std::set<std::pair<int, int>> get_illuminated_pixels_(const PSimHit &ps, const PixelGeomDetUnit *tkDetUnit);

  // General function container to decide if a hit has to be processed or not
  // depending the user request (check constructor and _check_input_angles method
  // for implementation details, as the real algorithm is created in there)
  std::function<bool(const PSimHit *)> use_this_track_;

  // The actual algorithm to check if a track entered a detector
  // within a given angles (to be used by use_this_track_ if needed, not
  // intended to be directly called)
  bool _check_input_angles_(const PSimHit *ps);

  // Helper member ot track the list of pixels illuminated by
  // the PSimHit (see definition of illuminate in get_illuminated_pixels_)
  std::map<std::uintptr_t, std::set<std::pair<int, int>>> m_illuminated_pixels_;

  // Histograms:
  // HElper setup functions
  MonitorElement *setupH1D_(DQMStore::IBooker &ibooker, const std::string &histoname, const std::string &title);
  MonitorElement *setupH2D_(DQMStore::IBooker &ibooker, const std::string &histoname, const std::string &title);
  MonitorElement *setupH2D_(DQMStore::IBooker &ibooker,
                            const std::string &histoname,
                            const std::string &title,
                            const std::pair<double, double> &xranges,
                            const std::pair<double, double> &yranges);
  MonitorElement *setupProf2D_(DQMStore::IBooker &ibooker,
                               const std::string &histoname,
                               const std::string &title,
                               const std::pair<double, double> &xranges,
                               const std::pair<double, double> &yranges);
  // Whole CMS histos
  MonitorElement *vME_track_XYMap_;
  MonitorElement *vME_track_RZMap_;
  MonitorElement *vME_digi_XYMap_;
  MonitorElement *vME_digi_RZMap_;
  // Per detector unit plots
  std::map<int, MonitorElement *> vME_clsize1D_;
  std::map<int, MonitorElement *> vME_clsize1Dx_;
  std::map<int, MonitorElement *> vME_clsize1Dy_;
  std::map<int, MonitorElement *> vME_charge1D_;
  std::map<int, MonitorElement *> vME_charge_elec1D_;
  std::map<int, MonitorElement *> vME_track_dxdz_;
  std::map<int, MonitorElement *> vME_track_dydz_;
  std::map<int, MonitorElement *> vME_track_dxdzAngle_;
  std::map<int, MonitorElement *> vME_track_dydzAngle_;
  std::map<int, MonitorElement *> vME_dx1D_;
  std::map<int, MonitorElement *> vME_dy1D_;
  std::map<int, MonitorElement *> vME_digi_charge1D_;
  std::map<int, MonitorElement *> vME_digi_chargeElec1D_;
  std::map<int, MonitorElement *> vME_sim_cluster_charge_;
  // --- cell histograms per subdector , each element on the
  //     vector 0: total, 1: 1x1, 2: 2x2, (3: 3x3, 4: 4x4)?
  std::map<int, std::vector<MonitorElement *>> vME_pshpos_cell_;
  std::map<int, std::vector<MonitorElement *>> vME_position_cell_;
  std::map<int, std::vector<MonitorElement *>> vME_eff_cell_;
  std::map<int, std::vector<MonitorElement *>> vME_clsize_cell_;
  std::map<int, std::vector<MonitorElement *>> vME_charge_cell_;
  std::map<int, std::vector<MonitorElement *>> vME_charge_elec_cell_;
  std::map<int, std::vector<MonitorElement *>> vME_dx_cell_;
  std::map<int, std::vector<MonitorElement *>> vME_dy_cell_;

  // Map to take care of common and tedious
  //const std::vector<std::map<int,std::vector<MonitorElement *>>*> helperMap_;

  // Configuration
  edm::ParameterSet config_;
  // Geometry to use
  std::string geomType_;

  // The conversion between ToT to electrons (Be carefull, this should
  // be using the same value used in the digitization module)
  double electronsPerADC_;
  // The tracks entry angle to accept (if any)
  std::vector<double> tracksEntryAngleX_;
  std::vector<double> tracksEntryAngleY_;
  // The actual angles already parsed (0- x aix, 1- y axis)
  std::map<unsigned int, std::pair<double, double>> active_entry_angles_;

  //std::vector<double> phiValues;
  // EDM token for the input collections
  std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> digiToken_;
  const edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> digiSimLinkToken_;
  const edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
};
#endif
