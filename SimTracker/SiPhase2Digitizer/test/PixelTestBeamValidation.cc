//
// Access Digi collection and creates histograms accumulating
// data from the module to different pixel cells, 1x1 cell,
// 2x2 cell, etc..
//
// Author:  J.Duarte-Campderros (IFCA)
// Created: 2019-10-02
//

// CMSSW Framework
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

// CMSSW Data formats
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"

// system
#include <algorithm>

// XXX - Be careful the relative position
#include "PixelTestBeamValidation.h"

// Some needed units (um are more suited to the pixel size of the sensors)
using cms_units::operators::operator""_deg;
// Analogously to CMSUnits (no um defined, using inverse)
constexpr double operator""_inv_um(long double length) { return length * 1e4; }
// Energy (keV) -- to be used with the PSimHit::energyLoss with energy in GeV
constexpr double operator""_inv_keV(long double energy_in_GeV) { return energy_in_GeV * 1e6; }

using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;

PixelTestBeamValidation::PixelTestBeamValidation(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      geomType_(iConfig.getParameter<std::string>("GeometryType")),
      //phiValues(iConfig.getParameter<std::vector<double> >("PhiAngles")),
      electronsPerADC_(iConfig.getParameter<double>("ElectronsPerADC")),
      tracksEntryAngleX_(
          iConfig.getUntrackedParameter<std::vector<double>>("TracksEntryAngleX", std::vector<double>())),
      tracksEntryAngleY_(
          iConfig.getUntrackedParameter<std::vector<double>>("TracksEntryAngleY", std::vector<double>())),
      digiToken_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("PixelDigiSource"))),
      digiSimLinkToken_(
          consumes<edm::DetSetVector<PixelDigiSimLink>>(iConfig.getParameter<edm::InputTag>("PixelDigiSimSource"))),
      simTrackToken_(consumes<edm::SimTrackContainer>(iConfig.getParameter<edm::InputTag>("SimTrackSource"))) {
  LogDebug("PixelTestBeamValidation") << ">>> Construct PixelTestBeamValidation ";

  const std::vector<edm::InputTag> psimhit_v(config_.getParameter<std::vector<edm::InputTag>>("PSimHitSource"));

  for (const auto& itag : psimhit_v) {
    simHitTokens_.push_back(consumes<edm::PSimHitContainer>(itag));
  }

  // Parse the entry angles parameter. Remember the angles are defined in
  // radians and defined as 0 when perpendicular to the detector plane
  // ---------------------------------------------------------------------
  // Helper map to build up the active entry angles
  std::map<std::string, std::vector<double>*> prov_ref_m;
  // Get the range of entry angles for the tracks on the detector surfaces, if any
  if (tracksEntryAngleX_.size() != 0) {
    prov_ref_m["X"] = &tracksEntryAngleX_;
  }
  if (tracksEntryAngleY_.size() != 0) {
    prov_ref_m["Y"] = &tracksEntryAngleY_;
  }

  // translation from string to int
  std::map<std::string, unsigned int> conversor({{"X", 0}, {"Y", 1}});

  // For each range vector do some consistency checks
  for (const auto& label_v : prov_ref_m) {
    // If provided 2
    if (label_v.second->size() == 2) {
      // Just order the ranges, lower index the minimum
      std::sort(label_v.second->begin(), label_v.second->end());
    } else if (label_v.second->size() == 1) {
      // Create the range with  a +- 1 deg
      label_v.second->push_back((*label_v.second)[0] + 1.0_deg);
      (*label_v.second)[0] -= 1.0_deg;
    } else {
      // Not valid,
      throw cms::Exception("Configuration") << "Setup TrackEntryAngle parameters"
                                            << "Invalid number of elements in 'TracksEntryAngle" << label_v.first
                                            << ".size()' = " << label_v.second->size() << ". Valid sizes are 1 or 2.";
    }
    // convert the angles into its tangent (the check is going to be done
    // against the tangent dx/dz and dy/dz)
    (*label_v.second)[0] = std::tan((*label_v.second)[0]);
    (*label_v.second)[1] = std::tan((*label_v.second)[1]);
    active_entry_angles_[conversor[label_v.first]] =
        std::pair<double, double>({(*label_v.second)[0], (*label_v.second)[1]});
  }

  if (prov_ref_m.size() != 0) {
    // The algorithm is defined in the implementation of _check_input_angles_
    use_this_track_ = std::bind(&PixelTestBeamValidation::_check_input_angles_, this, std::placeholders::_1);
    edm::LogInfo("Configuration") << "Considering hits from tracks entering the detectors between\n "
                                  << "Considering hits from tracks entering the detectors between\n "
                                  << "\tX-plane: (" << std::atan(active_entry_angles_[0].first) << ","
                                  << std::atan(active_entry_angles_[0].second) << ") rad. "
                                  << "\tY-plane: (" << std::atan(active_entry_angles_[1].first) << ","
                                  << std::atan(active_entry_angles_[1].second) << ") rad. ";
  } else {
    // There is no requiremnt, so always process it
    use_this_track_ = [](const PSimHit*) -> bool { return true; };
  }
}

//
// destructor
//
PixelTestBeamValidation::~PixelTestBeamValidation() {
  LogDebug("PixelTestBeamValidation") << ">>> Destroy PixelTestBeamValidation ";
}

// -- DQM Begin Run
//
void PixelTestBeamValidation::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::LogInfo("PixelTestBeamValidation") << "Initialize PixelTestBeamValidation ";
}
//
// -- Analyze
//
void PixelTestBeamValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // First clear the memoizers
  m_tId_det_simhits_.clear();
  m_illuminated_pixels_.clear();

  // Get digis and the simhit links and the simhits
  edm::Handle<edm::DetSetVector<PixelDigiSimLink>> digiSimLinkHandle;
  iEvent.getByToken(digiSimLinkToken_, digiSimLinkHandle);
  const edm::DetSetVector<PixelDigiSimLink>* simdigis = digiSimLinkHandle.product();

  edm::Handle<edm::DetSetVector<PixelDigi>> digiHandle;
  iEvent.getByToken(digiToken_, digiHandle);
  const edm::DetSetVector<PixelDigi>* digis = digiHandle.product();

  // Vector of simHits
  // XXX : NOt like that, just an example
  std::vector<const edm::PSimHitContainer*> simhits;
  simhits.reserve(simHitTokens_.size());
  for (const auto& sh_token : simHitTokens_) {
    edm::Handle<edm::PSimHitContainer> simHitHandle;
    iEvent.getByToken(sh_token, simHitHandle);
    if (!simHitHandle.isValid()) {
      continue;
    }
    simhits.push_back(simHitHandle.product());
  }

  // Get SimTrack
  /*edm::Handle<edm::SimTrackContainer> simTrackHandle;
    iEvent.getByToken(simTrackToken_, simTrackHandle);
    const edm::SimTrackContainer * simtracks = simTrackHandle.product();*/

  // Geometry description
  edm::ESWatcher<TrackerDigiGeometryRecord> tkDigiGeomWatcher;
  if (!tkDigiGeomWatcher.check(iSetup)) {
    // XXX -- Should raise a Warning??
    return;
  }

  // Tracker geometry and topology
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* topo = tTopoHandle.product();

  edm::ESHandle<TrackerGeometry> geomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geomHandle);
  const TrackerGeometry* tkGeom = geomHandle.product();

  // Let's loop over the detectors
  for (auto const& dunit : tkGeom->detUnits()) {
    if (!isPixelSystem_(dunit)) {
      continue;
    }

    // --------------------------------------------
    // Get more info about the detector unit
    // -- layer, topology (nrows,ncolumns,pitch...)
    const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(dunit);
    //const PixelTopology * topo = tkDetUnit->specificTopology();

    const auto detId = dunit->geographicalId();
    const int layer = topo->layer(detId);
    // Get the relevant histo key
    const auto& me_unit = meUnit_(tkDetUnit->type().isBarrel(), layer, topo->side(detId));

    // Get the id of the detector unit
    const unsigned int detId_raw = detId.rawId();

    // Loop over the simhits to obtain the list of PSimHits created
    // in this detector unit
    std::vector<const PSimHit*> it_simhits;

    for (const auto* sh_c : simhits) {
      for (const auto& sh : *sh_c) {
        if (sh.detUnitId() == detId_raw) {
          it_simhits.push_back(&sh);  //insert and/or reserve (?)
        }
      }
    }

    if (it_simhits.size() == 0) {
      continue;
    }

    // Find RAW digis (digis) created in this det unit
    const auto& it_digis = digis->find(detId);

    //std::cout << "DETECTOR: " << tkDetUnit->type().name() << " ME UNIT: " << me_unit << std::endl;

    // Loop over the list of PSimHits (i.e. the deposits created
    // by the primary+secundaries) to check if they are associated with
    // some digi, that is, if the simdigi link exists and to obtain the list
    // of channels illuminated
    for (const auto* psh : it_simhits) {
      // Check user conditions to accept the hits
      if (!use_this_track_(psh)) {
        continue;
      }
      // Fill some sim histograms
      const GlobalPoint tk_ep_gbl(dunit->surface().toGlobal(psh->entryPoint()));
      vME_track_XYMap_->Fill(tk_ep_gbl.x(), tk_ep_gbl.y());
      vME_track_RZMap_->Fill(tk_ep_gbl.z(), std::hypot(tk_ep_gbl.x(), tk_ep_gbl.y()));
      vME_track_dxdzAngle_[me_unit]->Fill(std::atan2(psh->momentumAtEntry().x(), psh->momentumAtEntry().z()));
      vME_track_dydzAngle_[me_unit]->Fill(std::atan2(psh->momentumAtEntry().y(), psh->momentumAtEntry().z()));

      // Obtain the detected position of the sim particle:
      // the middle point between the entry and the exit
      const auto psh_pos = tkDetUnit->specificTopology().measurementPosition(psh->localPosition());

      // MC Cluster finding: get the channels illuminated during digitization by this PSimHit
      // based on MC truth info (simdigi links)
      const auto psh_channels = get_illuminated_channels_(*psh, detId, simdigis);

      // Get the total charge for this cluster size
      // and obtain the center of the cluster using a charge-weighted mean
      int cluster_tot = 0;
      double cluster_tot_elec = 0.0;
      int cluster_size = 0;
      std::pair<std::set<int>, std::set<int>> cluster_size_xy;
      std::pair<double, double> cluster_position({0.0, 0.0});
      std::set<int> used_channel;
      for (const auto& ch : psh_channels) {
        // Not re-using the digi XXX
        if (used_channel.find(ch) != used_channel.end()) {
          continue;
        }
        const PixelDigi& current_digi = get_digi_from_channel_(ch, it_digis);
        used_channel.insert(ch);
        // Fill the digi histograms
        vME_digi_charge1D_[me_unit]->Fill(current_digi.adc());
        // Fill maps: get the position in the sensor local frame to convert into global
        const LocalPoint digi_local_pos(
            tkDetUnit->specificTopology().localPosition(MeasurementPoint(current_digi.row(), current_digi.column())));
        const GlobalPoint digi_global_pos(dunit->surface().toGlobal(digi_local_pos));
        vME_digi_XYMap_->Fill(digi_global_pos.x(), digi_global_pos.y());
        vME_digi_RZMap_->Fill(digi_global_pos.z(), std::hypot(digi_global_pos.x(), digi_global_pos.y()));
        // Create the MC-cluster
        cluster_tot += current_digi.adc();
        // Add 0.5 to allow ToT = 0 (valid value)
        cluster_tot_elec += (current_digi.adc() + 0.5) * electronsPerADC_;
        // Use the center of the pixel
        cluster_position.first += current_digi.adc() * (current_digi.row() + 0.5);
        cluster_position.second += current_digi.adc() * (current_digi.column() + 0.5);
        // Size
        cluster_size_xy.first.insert(current_digi.row());
        cluster_size_xy.second.insert(current_digi.column());
        ++cluster_size;
      }

      // Be careful here, there is 1 entry per each simhit
      vME_clsize1D_[me_unit]->Fill(cluster_size);
      vME_clsize1Dx_[me_unit]->Fill(cluster_size_xy.first.size());
      vME_clsize1Dy_[me_unit]->Fill(cluster_size_xy.second.size());

      // mean weighted
      cluster_position.first /= double(cluster_tot);
      cluster_position.second /= double(cluster_tot);

      // -- XXX Be careful, secondaries with already used the digis
      //        are going the be lost (then lost on efficiency)
      // Efficiency --> It was found a cluster of digis?
      const bool is_cluster_present = (cluster_size > 0);

      // Get topology info of the module sensor
      //-const int n_rows = tkDetUnit->specificTopology().nrows();
      //-const int n_cols = tkDetUnit->specificTopology().ncolumns();
      const auto pitch = tkDetUnit->specificTopology().pitch();
      // Residuals, convert them to longitud units (so far, in units of row, col)
      const double dx_um = (psh_pos.x() - cluster_position.first) * pitch.first * 1.0_inv_um;
      const double dy_um = (psh_pos.y() - cluster_position.second) * pitch.second * 1.0_inv_um;
      if (is_cluster_present) {
        vME_charge1D_[me_unit]->Fill(cluster_tot);
        vME_charge_elec1D_[me_unit]->Fill(cluster_tot_elec);
        vME_dx1D_[me_unit]->Fill(dx_um);
        vME_dy1D_[me_unit]->Fill(dy_um);
        // The track energy loss corresponding to that cluster
        vME_sim_cluster_charge_[me_unit]->Fill(psh->energyLoss() * 1.0_inv_keV, cluster_tot_elec);
      }
      // Histograms per cell
      for (unsigned int i = 0; i < vME_position_cell_[me_unit].size(); ++i) {
        // Convert the PSimHit center position to the IxI-cell
        const std::pair<double, double> icell_psh = pixel_cell_transformation_(psh_pos, i, pitch);
        // Efficiency: (PSimHit matched to a digi-cluster)/PSimHit
        vME_eff_cell_[me_unit][i]->Fill(
            icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um, is_cluster_present);
        vME_pshpos_cell_[me_unit][i]->Fill(icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um);
        // Digi clusters related histoos
        if (is_cluster_present) {
          // Convert to the i-cell
          //const std::pair<double,double> icell_digi_cluster   = pixel_cell_transformation_(cluster_position,i,pitch);
          // Position
          vME_position_cell_[me_unit][i]->Fill(icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um);
          // Residuals
          vME_dx_cell_[me_unit][i]->Fill(icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um, dx_um);
          vME_dy_cell_[me_unit][i]->Fill(icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um, dy_um);
          // Charge
          vME_charge_cell_[me_unit][i]->Fill(icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um, cluster_tot);
          vME_charge_elec_cell_[me_unit][i]->Fill(
              icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um, cluster_tot_elec);
          // Cluster size
          vME_clsize_cell_[me_unit][i]->Fill(icell_psh.first * 1.0_inv_um, icell_psh.second * 1.0_inv_um, cluster_size);
        }
      }
    }
  }
}

//
// -- Book Histograms
//
void PixelTestBeamValidation::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const& iRun,
                                             edm::EventSetup const& iSetup) {
  // Get Geometry to associate a folder to each Pixel subdetector
  edm::ESHandle<TrackerGeometry> geomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geomHandle);
  const TrackerGeometry* tkGeom = geomHandle.product();

  // Tracker Topology (layers, modules, side, etc..)
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* topo = tTopoHandle.product();

  const std::string top_folder = config_.getParameter<std::string>("TopFolderName");

  // Histograms independent of the subdetector units
  ibooker.cd();
  ibooker.setCurrentFolder(top_folder);
  vME_track_XYMap_ =
      setupH2D_(ibooker, "TrackXY", "Track entering position in the tracker system;x [cm];y [cm];N_{tracksId}");
  vME_track_RZMap_ =
      setupH2D_(ibooker, "TrackRZ", "Track entering position in the tracker system;z [cm];r [cm];N_{tracksId}");
  vME_digi_XYMap_ = setupH2D_(ibooker, "DigiXY", "Digi position;x [cm];y [cm];N_{digi}");
  vME_digi_RZMap_ = setupH2D_(ibooker, "DigiRZ", "Digi position;z [cm];r [cm];N_{digi}");

  // Get all pixel subdetector, create histogram by layers
  // -- More granularity can be accomplished by modules (1 central + 5 modules: in z,
  //    each one composed by 12 pixel modules (1 long pixel + 2 ROCs) in Phi = 108 pixel-modules
  for (auto const& dunit : tkGeom->detUnits()) {
    if (!isPixelSystem_(dunit)) {
      continue;
    }
    //if( ! dtype.isInnerTracker() || ! dtype.isTrackerPixel() )
    //{
    //    continue;
    //}
    const auto& dtype = dunit->type();

    const auto detId = dunit->geographicalId();
    const int layer = topo->layer(detId);
    // Create the sub-detector histogram if needed
    // Barrel_Layer/Endcap_Layer_Side
    const int me_unit = meUnit_(dtype.isBarrel(), layer, topo->side(detId));
    if (vME_position_cell_.find(me_unit) == vME_position_cell_.end()) {
      std::string folder_name(top_folder + "/");
      if (dtype.isBarrel()) {
        folder_name += "Barrel";
      } else if (dtype.isEndcap()) {
        folder_name += "Endcap";
      } else {
        cms::Exception("Geometry") << "Tracker subdetector '" << dtype.subDetector()
                                   << "' malformed: does not belong to the Endcap neither the Barrel";
      }
      folder_name += "/Layer" + std::to_string(layer);

      if (topo->side(detId)) {
        folder_name += "/Side" + std::to_string(topo->side(detId));
      }
      // Go the proper folder
      ibooker.cd();
      ibooker.setCurrentFolder(folder_name);

      const TrackerGeomDet* geomDetUnit = tkGeom->idToDetUnit(detId);
      const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(geomDetUnit);
      const int nrows = tkDetUnit->specificTopology().nrows();
      const int ncols = tkDetUnit->specificTopology().ncolumns();
      const auto pitch = tkDetUnit->specificTopology().pitch();
      //const double x_size = nrows*pitch.first;
      //const double y_size = ncols*pitch.second;

      // And create the histos
      // Per detector unit histos
      vME_clsize1D_[me_unit] =
          setupH1D_(ibooker, "ClusterSize1D", "MC-truth DIGI cluster size;Cluster size;N_{clusters}");
      vME_clsize1Dx_[me_unit] =
          setupH1D_(ibooker, "ClusterSize1Dx", "MC-truth DIGI cluster size in X;Cluster size;N_{clusters}");
      vME_clsize1Dy_[me_unit] =
          setupH1D_(ibooker, "ClusterSize1Dy", "MC-truth DIGI cluster size in Y;Cluster size;N_{clusters}");
      vME_charge1D_[me_unit] =
          setupH1D_(ibooker, "Charge1D", "MC-truth DIGI cluster charge;Cluster charge [ToT];N_{clusters}");
      vME_charge_elec1D_[me_unit] =
          setupH1D_(ibooker, "ChargeElec1D", "MC-truth DIGI cluster charge;Cluster charge [Electrons];N_{clusters}");
      vME_track_dxdzAngle_[me_unit] = setupH1D_(
          ibooker,
          "TrackAngleDxdz",
          "Angle between the track-momentum and detector surface (X-plane);#pi/2-#theta_{x} [rad];N_{tracks}");
      vME_track_dydzAngle_[me_unit] = setupH1D_(
          ibooker,
          "TrackAngleDydz",
          "Angle between the track-momentum and detector surface (Y-plane);#pi/2-#theta_{y} [rad];N_{tracks}");
      vME_dx1D_[me_unit] = setupH1D_(
          ibooker, "Dx1D", "MC-truth DIGI cluster residuals X;x_{PSimHit}-x^{cluster}_{digi} [#mum];N_{digi clusters}");
      vME_dy1D_[me_unit] = setupH1D_(
          ibooker, "Dy1D", "MC-truth DIGI cluster residual Ys;y_{PSimHit}-y^{cluster}_{digi} [#mum];N_{digi clusters}");
      vME_digi_charge1D_[me_unit] = setupH1D_(ibooker, "DigiCharge1D", "Digi charge;digi charge [ToT];N_{digi}");
      vME_sim_cluster_charge_[me_unit] =
          setupH2D_(ibooker,
                    "SimClusterCharge",
                    "PSimHit Energy deposit vs. Cluster Charge;deposited E_{sim} [keV];cluster_{charge} [Electrons];");

      // The histos per cell
      // Prepare the ranges: 0- whole sensor, 1- cell 1x1, 2-cell 2x2,
      const std::vector<std::pair<double, double>> xranges = {
          std::make_pair<double, double>(0, (nrows - 1) * pitch.first * 1.0_inv_um),
          std::make_pair<double, double>(0, pitch.first * 1.0_inv_um),
          std::make_pair<double, double>(0, 2.0 * pitch.first * 1.0_inv_um)};
      const std::vector<std::pair<double, double>> yranges = {
          std::make_pair<double, double>(0, (ncols - 1) * pitch.second * 1.0_inv_um),
          std::make_pair<double, double>(0, pitch.second * 1.0_inv_um),
          std::make_pair<double, double>(0, 2.0 * pitch.second * 1.0_inv_um)};
      for (unsigned int i = 0; i < xranges.size(); ++i) {
        const std::string cell("Cell " + std::to_string(i) + "x" + std::to_string(i) + ": ");
        vME_pshpos_cell_[me_unit].push_back(
            setupH2D_(ibooker,
                      "Position_" + std::to_string(i),
                      cell + "PSimHit middle point position;x [#mum];y [#mum];N_{clusters}",
                      xranges[i],
                      yranges[i]));
        vME_position_cell_[me_unit].push_back(
            setupH2D_(ibooker,
                      "MatchedPosition_" + std::to_string(i),
                      cell + "PSimHit matched to a DIGI cluster position ;x [#mum];y [#mum];N_{clusters}",
                      xranges[i],
                      yranges[i]));
        vME_eff_cell_[me_unit].push_back(setupProf2D_(ibooker,
                                                      "Efficiency_" + std::to_string(i),
                                                      cell + "MC-truth efficiency;x [#mum];y [#mum];<#varepsilon>",
                                                      xranges[i],
                                                      yranges[i]));
        vME_clsize_cell_[me_unit].push_back(
            setupProf2D_(ibooker,
                         "ClusterSize_" + std::to_string(i),
                         cell + "MC-truth cluster size;x [#mum];y [#mum];<Cluster size>",
                         xranges[i],
                         yranges[i]));
        vME_charge_cell_[me_unit].push_back(setupProf2D_(ibooker,
                                                         "Charge_" + std::to_string(i),
                                                         cell + "MC-truth charge;x [#mum];y [#mum];<ToT>",
                                                         xranges[i],
                                                         yranges[i]));
        vME_charge_elec_cell_[me_unit].push_back(setupProf2D_(ibooker,
                                                              "Charge_elec_" + std::to_string(i),
                                                              cell + "MC-truth charge;x [#mum];y [#mum];<Electrons>",
                                                              xranges[i],
                                                              yranges[i]));
        vME_dx_cell_[me_unit].push_back(setupProf2D_(ibooker,
                                                     "Dx_" + std::to_string(i),
                                                     cell + "MC-truth residuals;x [#mum];y [#mum];<#Deltax [#mum]>",
                                                     xranges[i],
                                                     yranges[i]));
        vME_dy_cell_[me_unit].push_back(setupProf2D_(ibooker,
                                                     "Dy_" + std::to_string(i),
                                                     cell + "MC-truth residuals;x [#mum];y [#mum];<#Deltay [#mum]>",
                                                     xranges[i],
                                                     yranges[i]));
      }

      edm::LogInfo("PixelTestBeamValidation") << "Booking Histograms in: " << folder_name << " ME UNIT:" << me_unit;
      edm::LogInfo("PixelTestBeamValidation") << "Booking Histograms in: " << folder_name;
    }
  }
}

bool PixelTestBeamValidation::isPixelSystem_(const GeomDetUnit* dunit) const {
  const auto& dtype = dunit->type();
  if (!dtype.isInnerTracker() || !dtype.isTrackerPixel()) {
    return false;
  }
  return true;
}

// Helper functions to setup
int PixelTestBeamValidation::meUnit_(bool isBarrel, int layer, int side) const {
  // [isBarrel][layer][side]
  // [X][XXX][XX]
  return (static_cast<int>(isBarrel) << 6) | (layer << 3) | side;
}

PixelTestBeamValidation::MonitorElement* PixelTestBeamValidation::setupH1D_(DQMStore::IBooker& ibooker,
                                                                            const std::string& histoname,
                                                                            const std::string& title) {
  // Config need to have exactly the same histo name
  edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
  return ibooker.book1D(histoname,
                        title,
                        params.getParameter<int32_t>("Nxbins"),
                        params.getParameter<double>("xmin"),
                        params.getParameter<double>("xmax"));
}

PixelTestBeamValidation::MonitorElement* PixelTestBeamValidation::setupH2D_(DQMStore::IBooker& ibooker,
                                                                            const std::string& histoname,
                                                                            const std::string& title) {
  // Config need to have exactly the same histo name
  edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
  return ibooker.book2D(histoname,
                        title,
                        params.getParameter<int32_t>("Nxbins"),
                        params.getParameter<double>("xmin"),
                        params.getParameter<double>("xmax"),
                        params.getParameter<int32_t>("Nybins"),
                        params.getParameter<double>("ymin"),
                        params.getParameter<double>("ymax"));
}

PixelTestBeamValidation::MonitorElement* PixelTestBeamValidation::setupH2D_(DQMStore::IBooker& ibooker,
                                                                            const std::string& histoname,
                                                                            const std::string& title,
                                                                            const std::pair<double, double>& xranges,
                                                                            const std::pair<double, double>& yranges) {
  // Config need to have exactly the same histo name
  edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
  return ibooker.book2D(histoname,
                        title,
                        params.getParameter<int32_t>("Nxbins"),
                        xranges.first,
                        xranges.second,
                        params.getParameter<int32_t>("Nybins"),
                        yranges.first,
                        yranges.second);
}

PixelTestBeamValidation::MonitorElement* PixelTestBeamValidation::setupProf2D_(
    DQMStore::IBooker& ibooker,
    const std::string& histoname,
    const std::string& title,
    const std::pair<double, double>& xranges,
    const std::pair<double, double>& yranges) {
  // Config need to have exactly the same histo name
  edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
  return ibooker.bookProfile2D(histoname,
                               title,
                               params.getParameter<int32_t>("Nxbins"),
                               xranges.first,
                               xranges.second,
                               params.getParameter<int32_t>("Nybins"),
                               yranges.first,
                               yranges.second,
                               params.getParameter<double>("zmin"),
                               params.getParameter<double>("zmax"));
}

const PixelDigi& PixelTestBeamValidation::get_digi_from_channel_(
    int ch, const edm::DetSetVector<PixelDigi>::const_iterator& itdigis) {
  for (const auto& dh : *itdigis) {
    if (dh.channel() == ch) {
      return dh;
    }
  }
  // MAke sense not to find a digi?
  throw cms::Exception("DIGI Pixel Validation") << "Not found a PixelDig for the given channel: " << ch;
}

const SimTrack* PixelTestBeamValidation::get_simtrack_from_id_(unsigned int idx, const edm::SimTrackContainer* stc) {
  for (const auto& st : *stc) {
    if (st.trackId() == idx) {
      return &st;
    }
  }
  // Any simtrack correspond to this trackid index
  //edm::LogWarning("PixelTestBeamValidation::get_simtrack_from_id_")
  edm::LogInfo("PixelTestBeamValidation::get_simtrack_from_id_") << "Not found any SimTrack with trackId: " << idx;
  return nullptr;
}

const std::vector<const PSimHit*> PixelTestBeamValidation::get_simhits_from_trackid_(
    unsigned int tid, unsigned int detid_raw, const std::vector<const edm::PSimHitContainer*>& psimhits) {
  // It was already found?
  if (m_tId_det_simhits_.find(tid) != m_tId_det_simhits_.end()) {
    // Note that if there were no simhits in detid_raw, now it
    // is creating an empty std::vector<const edm::PSimHitContainer*>
    return m_tId_det_simhits_[tid][detid_raw];
  }

  // Otherwise,
  // Create the new map for the track
  m_tId_det_simhits_[tid] = std::map<unsigned int, std::vector<const PSimHit*>>();
  // and search for it, all the PsimHit found in all detectors
  // are going to be seeked once, therefore memoizing already
  for (const auto* sh_c : psimhits) {
    for (const auto& sh : *sh_c) {
      if (sh.trackId() == tid) {
        m_tId_det_simhits_[tid][sh.detUnitId()].push_back(&sh);
      }
    }
  }
  // And returning what was requested
  return m_tId_det_simhits_[tid][detid_raw];
}

const std::pair<double, double> PixelTestBeamValidation::pixel_cell_transformation_(
    const MeasurementPoint& pos, unsigned int icell, const std::pair<double, double>& pitch) {
  return pixel_cell_transformation_(std::pair<double, double>({pos.x(), pos.y()}), icell, pitch);
}

const std::pair<double, double> PixelTestBeamValidation::pixel_cell_transformation_(
    const std::pair<double, double>& pos, unsigned int icell, const std::pair<double, double>& pitch) {
  // Get the position modulo icell
  // Case the whole detector (icell==0)
  double xmod = pos.first;
  double ymod = pos.second;

  // Actual pixel cells
  if (icell != 0) {
    xmod = std::fmod(pos.first, icell);
    ymod = std::fmod(pos.second, icell);
  }

  const double xcell = xmod * pitch.first;
  const double ycell = ymod * pitch.second;

  return std::pair<double, double>({xcell, ycell});
}

//bool PixelTestBeamValidation::channel_iluminated_by_(const MeasurementPoint & localpos,int channel, double tolerance) const
//{
//    const auto pos_channel(PixelDigi::channelToPixel(channel));
//    if( std::fabs(localpos.x()-pos_channel.first) <= tolerance
//            && std::fabs(localpos.y()-pos_channel.second) <= tolerance )
//    {
//        return true;
//    }
//    return false;
//}

bool PixelTestBeamValidation::channel_iluminated_by_(const PSimHit& ps,
                                                     int channel,
                                                     const PixelGeomDetUnit* tkDetUnit) {
  // Get the list of pixels illuminated by the PSimHit
  const auto pixel_list = get_illuminated_pixels_(ps, tkDetUnit);
  // Get the digi position
  const auto pos_channel(PixelDigi::channelToPixel(channel));

  for (const auto& px_py : pixel_list) {
    if (px_py.first == pos_channel.first && px_py.second == pos_channel.second) {
      return true;
    }
  }
  return false;
}

std::set<int> PixelTestBeamValidation::get_illuminated_channels_(const PSimHit& ps,
                                                                 const DetId& detid,
                                                                 const edm::DetSetVector<PixelDigiSimLink>* simdigis) {
  // Find simulated digi links (simdigis) created in this det unit
  const auto& it_simdigilink = simdigis->find(detid);

  if (it_simdigilink == simdigis->end()) {
    return std::set<int>();
  }

  std::set<int> channels;
  for (const auto& hdsim : *it_simdigilink) {
    if (ps.trackId() == hdsim.SimTrackId()) {
      channels.insert(hdsim.channel());
    }
  }
  return channels;
}

std::set<std::pair<int, int>> PixelTestBeamValidation::get_illuminated_pixels_(const PSimHit& ps,
                                                                               const PixelGeomDetUnit* tkDetUnit) {
  auto ps_key = reinterpret_cast<std::uintptr_t>(&ps);

  //  -- Check if was already memoized
  if (m_illuminated_pixels_.find(ps_key) != m_illuminated_pixels_.end()) {
    return m_illuminated_pixels_[ps_key];
  }

  // Get the entry point - exit point position
  const double min_x = std::min(ps.entryPoint().x(), ps.exitPoint().x());
  const double min_y = std::min(ps.entryPoint().y(), ps.exitPoint().y());
  const double max_x = std::max(ps.entryPoint().x(), ps.exitPoint().x());
  const double max_y = std::max(ps.entryPoint().y(), ps.exitPoint().y());
  // Get the position in readout units for each point
  const auto min_pos = tkDetUnit->specificTopology().measurementPosition(LocalPoint(min_x, min_y));
  const auto max_pos = tkDetUnit->specificTopology().measurementPosition(LocalPoint(max_x, max_y));
  // Count how many cells has passed. Use the most conservative rounding:
  // round for maximums and int (floor) for minimums
  //const int ncells_x = std::round(max_pos.x())-std::floor(min_pos.x());
  //const int ncells_y = std::round(max_pos.y())-std::floor(min_pos.y());

  //std::cout << "ENTRANDO --- PSimHit (" << ps_key << ") entry point: " << ps.entryPoint() << " exitPoint: " << ps.exitPoint()
  //  << " Entry (pixel units): " << min_pos
  //  << " Exit  (pixel units): " << max_pos << ")"
  //  << " ILLUMINATED PIXELS: ";

  std::set<std::pair<int, int>> illuminated_pixels;
  for (unsigned int px = std::floor(min_pos.x()); px <= std::round(max_pos.x()); ++px) {
    for (unsigned int py = std::floor(min_pos.y()); py <= std::round(max_pos.y()); ++py) {
      //std::cout << " [" << px << "," << py << "]";
      illuminated_pixels.insert(std::pair<int, int>(px, py));
    }
  }
  //std::cout << " TOTAL PIXELS: " << illuminated_pixels.size() << std::endl;
  // Memoize, and return what expected.
  m_illuminated_pixels_[ps_key] = illuminated_pixels;
  return m_illuminated_pixels_[ps_key];
}

bool PixelTestBeamValidation::_check_input_angles_(const PSimHit* psimhit) {
  // Create a vector to check against the range map where
  // X axis is in the key 0, Y axis in the key 1
  const std::vector<double> entry_tan({psimhit->momentumAtEntry().x() / psimhit->momentumAtEntry().z(),
                                       psimhit->momentumAtEntry().y() / psimhit->momentumAtEntry().z()});

  for (const auto& axis_ranges : active_entry_angles_) {
    if (axis_ranges.second.first > entry_tan[axis_ranges.first] ||
        axis_ranges.second.second < entry_tan[axis_ranges.first]) {
      return false;
    }
  }
  return true;
}

//define this as a plug-in
DEFINE_FWK_MODULE(PixelTestBeamValidation);
