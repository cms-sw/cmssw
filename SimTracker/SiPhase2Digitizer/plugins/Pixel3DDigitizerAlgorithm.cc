#include "SimTracker/SiPhase2Digitizer/plugins/Pixel3DDigitizerAlgorithm.h"

// Framework infrastructure
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Calibration & Conditions
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineSimService.h"
#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleSimRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

//#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace sipixelobjects;

// Analogously to CMSUnits (no um defined)
constexpr double operator""_um(long double length) { return length * 1e-4; }
constexpr double operator""_um_inv(long double length) { return length * 1e4; }

void Pixel3DDigitizerAlgorithm::init(const edm::EventSetup& es) {
  // XXX: Just copied from PixelDigitizer Algorithm
  //      CHECK if all these is needed

  if (use_ineff_from_db_) {
    // load gain calibration service fromdb...
    theSiPixelGainCalibrationService_->setESObjects(es);
  }

  if (use_deadmodule_DB_) {
    es.get<SiPixelQualityRcd>().get(siPixelBadModule_);
  }

  if (use_LorentzAngle_DB_) {
    // Get Lorentz angle from DB record
    es.get<SiPixelLorentzAngleSimRcd>().get(siPixelLorentzAngle_);
  }

  // gets the map and geometry from the DB (to kill ROCs)
  es.get<SiPixelFedCablingMapRcd>().get(fedCablingMap_);
  es.get<TrackerDigiGeometryRecord>().get(geom_);
}

Pixel3DDigitizerAlgorithm::Pixel3DDigitizerAlgorithm(const edm::ParameterSet& conf)
    : Phase2TrackerDigitizerAlgorithm(conf.getParameter<edm::ParameterSet>("AlgorithmCommon"),
                                      conf.getParameter<edm::ParameterSet>("Pixel3DDigitizerAlgorithm")),
      np_column_radius_(
          (conf.getParameter<edm::ParameterSet>("Pixel3DDigitizerAlgorithm").getParameter<double>("NPColumnRadius")) *
          1.0_um),
      ohm_column_radius_(
          (conf.getParameter<edm::ParameterSet>("Pixel3DDigitizerAlgorithm").getParameter<double>("OhmicColumnRadius")) *
          1.0_um),
      np_column_gap_(
          (conf.getParameter<edm::ParameterSet>("Pixel3DDigitizerAlgorithm").getParameter<double>("NPColumnGap")) *
          1.0_um) {
  // XXX - NEEDED?
  pixelFlag_ = true;

  edm::LogInfo("Pixel3DDigitizerAlgorithm")
      << "Algorithm constructed \n"
      << "Configuration parameters:\n"
      << "\n*** Threshold"
      << "\n    Endcap = " << theThresholdInE_Endcap_ << " electrons"
      << "\n    Barrel = " << theThresholdInE_Barrel_ << " electrons"
      << "\n*** Gain"
      << "\n    Electrons per ADC:" << theElectronPerADC_ << "\n    ADC Full Scale: " << theAdcFullScale_
      << "\n*** The delta cut-off is set to " << tMax_ << "\n*** Pixel-inefficiency: " << addPixelInefficiency_;
}

Pixel3DDigitizerAlgorithm::~Pixel3DDigitizerAlgorithm() {}

//
// -- Select the Hit for Digitization
//
bool Pixel3DDigitizerAlgorithm::select_hit(const PSimHit& hit, double tCorr, double& sigScale) const {
  double time = hit.tof() - tCorr;
  return (time >= theTofLowerCut_ && time < theTofUpperCut_);
}

const bool Pixel3DDigitizerAlgorithm::is_inside_n_column_(const LocalPoint& p, const float& sensor_thickness) const {
  // The insensitive volume of the column: sensor thickness - column gap distance
  return (p.perp() <= np_column_radius_ && p.z() <= (sensor_thickness - np_column_gap_));
}

const bool Pixel3DDigitizerAlgorithm::is_inside_ohmic_column_(const LocalPoint& p,
                                                              const std::pair<float, float>& half_pitch) const {
  // The four corners of the cell
  return ((p - LocalVector(half_pitch.first, half_pitch.second, 0)).perp() <= ohm_column_radius_) ||
         ((p - LocalVector(-half_pitch.first, half_pitch.second, 0)).perp() <= ohm_column_radius_) ||
         ((p - LocalVector(half_pitch.first, -half_pitch.second, 0)).perp() <= ohm_column_radius_) ||
         ((p - LocalVector(-half_pitch.first, -half_pitch.second, 0)).perp() <= ohm_column_radius_);
}

// Diffusion algorithm: Probably not needed,
// Assuming the position point is given in the reference system of the proxy
// cell, centered at the n-column.
// The algorithm assumes only 1-axis could produce the charge migration, this assumption
// could be enough given that the p-columns (5 um radius) are in the corners of the cell
// (no producing charge in there)
// The output is vector of newly created charge in the neighbour pixel i+1 or i-1,
// defined by its position higher than abs(half_pitch) and the the sign providing
// the addition or subtraction in the pixel  (i+-1)
std::vector<DigitizerUtility::EnergyDepositUnit> Pixel3DDigitizerAlgorithm::diffusion(
    const LocalPoint& pos,
    const float& ncarriers,
    const std::function<LocalVector(float, float)>& u_drift,
    const std::pair<float, float> hpitches,
    const float& thickness) const {
  // FIXME -- DM : Note that with a 0.3 will be enough (if using current sigma formulae)
  //          With the current sigma, this value is dependent of the thickness,
  //          Note that this formulae is coming from planar sensors, a similar
  //          study with data will be needed to extract the sigma for 3D
  const float max_migration_radius = 0.4_um;
  // Need to know which axis is the relevant one
  int displ_ind = -1;
  float pitch = 0.0;

  // Check the group is near the edge of the pixel, so diffusion will
  // be relevant in order to migrate between pixel cells
  if (std::abs(pos.x() - hpitches.first) < max_migration_radius) {
    displ_ind = 0;
    pitch = hpitches.first;
  } else if (std::abs(pos.y() - hpitches.second) < max_migration_radius) {
    displ_ind = 1;
    pitch = hpitches.second;
  } else {
    // Nothing to do, too far away
    return std::vector<DigitizerUtility::EnergyDepositUnit>();
  }

  // The new EnergyDeposits in the neighbour pixels
  // (defined by +1 to the right (first axis) and +1 to the up (second axis)) <-- XXX
  std::vector<DigitizerUtility::EnergyDepositUnit> migrated_charge;

  // FIXME -- DM
  const float diffusion_step = 0.1_um;

  // The position while drifting
  std::vector<float> pos_moving({pos.x(), pos.y(), pos.z()});
  // The drifting: drift field and steps
  std::function<std::vector<float>(int)> do_step =
      [&pos_moving, &u_drift, diffusion_step](int i) -> std::vector<float> {
    auto dd = u_drift(pos_moving[0], pos_moving[1]);
    return std::vector<float>({i * diffusion_step * dd.x(), i * diffusion_step * dd.y(), i * diffusion_step * dd.z()});
  };

  LogDebug("Pixel3DDigitizerAlgorithm::diffusion")
      << "\nMax. radius from the pixel edge to migrate charge: " << max_migration_radius * 1.0_um_inv << " [um]"
      << "\nMigration axis: " << displ_ind
      << "\n(super-)Charge distance to the pixel edge: " << (pitch - pos_moving[displ_ind]) * 1.0_um_inv << " [um]";

  // FIXME -- Sigma reference, DM?
  const float distance0 = 300.0_um;
  const float sigma0 = 3.4_um;
  // FIXME -- Tolerance, DM?
  const float TOL = 1e-6;
  // How many sigmas (probably a configurable, to be decided not now)
  const float N_SIGMA = 3.0;

  // Start the drift and check every step
  // initial position
  int i = 0;
  // Some variables needed
  float current_carriers = ncarriers;
  std::vector<float> newpos({pos_moving[0], pos_moving[1], pos_moving[2]});
  float distance_edge = 0.0_um;
  do {
    std::transform(pos_moving.begin(), pos_moving.end(), do_step(i).begin(), pos_moving.begin(), std::plus<float>());
    distance_edge = std::abs(pos_moving[displ_ind] - pitch);
    // current diffusion value
    double sigma = std::sqrt(i * diffusion_step / distance0) * (distance0 / thickness) * sigma0;
    // Get the amount of charge on the neighbor pixel: note the
    // transformation to a Normal
    float migrated_e = current_carriers * (1.0 - std::erf(distance_edge / sigma));

    LogDebug("(super-)charge diffusion") << "step-" << i << ", Initial Ne= " << ncarriers << ", "
                                         << "r=(" << pos_moving[0] * 1.0_um_inv << ", " << pos_moving[1] * 1.0_um_inv
                                         << ", " << pos_moving[2] * 1.0_um_inv << ") [um], "
                                         << "Migrated charge: " << migrated_e;

    // No charge was migrated (ignore creation time)
    if (i != 0) {
      // At least 1 electron migrated
      if ((migrated_e - TOL) < 1.0) {
        break;
      }
      // Move the migrated charge
      current_carriers -= migrated_e;
      // Create the ionization point:
      // First update the newpos vector: the new charge positions at the neighbourg pixels
      // are created in the same position that its "parent carriers"
      // except the direction of migration
      std::vector<float> newpos(pos_moving);
      // Lest create the new charges around 3 sigmas away
      newpos[displ_ind] += std::copysign(N_SIGMA * sigma, newpos[displ_ind]);
      migrated_charge.push_back(DigitizerUtility::EnergyDepositUnit(migrated_e, newpos[0], newpos[1], newpos[2]));
    }
    // Next step
    ++i;
  } while (std::abs(distance_edge) < max_migration_radius);

  return migrated_charge;
}

// ======================================================================
//
// Drift the charge segments to the column (collection surface)
// Include the effect of E-field and B-field
//
// =====================================================================
std::vector<DigitizerUtility::SignalPoint> Pixel3DDigitizerAlgorithm::drift(
    const PSimHit& hit,
    const Phase2TrackerGeomDetUnit* pixdet,
    const GlobalVector& bfield,
    const std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points) const {
  return drift(hit, pixdet, bfield, ionization_points, true);
}
std::vector<DigitizerUtility::SignalPoint> Pixel3DDigitizerAlgorithm::drift(
    const PSimHit& hit,
    const Phase2TrackerGeomDetUnit* pixdet,
    const GlobalVector& bfield,
    const std::vector<DigitizerUtility::EnergyDepositUnit>& ionization_points,
    bool diffusion_activated) const {
  // -- Current reference system is placed in the center on the module
  // -- The natural reference frame should be discribed taking advantatge of
  // -- the cylindrical nature of the pixel geometry -->
  // -- the new reference frame should be place in the center of the columncy, and in the
  // -- surface of the ROC using cylindrical coordinates

  // Get ROC pitch, half_pitch and sensor thickness to be used to create the
  // proxy pixel cell reference frame
  const auto pitch = pixdet->specificTopology().pitch();
  const auto half_pitch = std::make_pair<float, float>(pitch.first * 0.5, pitch.second * 0.5);
  const float thickness = pixdet->specificSurface().bounds().thickness();
  const int nrows = pixdet->specificTopology().nrows();
  const int ncolumns = pixdet->specificTopology().ncolumns();
  const float pix_rounding = 0.99;

  // the maximum radial distance is going to be use to evaluate radiation damage XXX?
  const float max_radial_distance =
      std::sqrt(half_pitch.first * half_pitch.first + half_pitch.second * half_pitch.second);

  // All pixels are going to be translated to a proxy pixel cell (all pixels should behave
  // equally no matter their position w.r.t. the module) and describe the movements there
  // Define the center of the pixel local reference frame: the current cartesian local reference
  // frame is centered at half_width_x,half_width_y,half_thickness
  // XXX -- This info could be obtained at init/construction time?
  LocalPoint center_proxy_cell(half_pitch.first, half_pitch.second, -0.5 * thickness);

  LogDebug("Pixel3DDigitizerAlgorithm::drift")
      << "Pixel pitch:" << pitch.first * 1.0_um_inv << ", " << pitch.second * 1.0_um_inv << " [um]";

  // And the drift direction (assumed same for all the sensor)
  // XXX call the function which will return a functional
  std::function<LocalVector(float, float)> drift_direction = [](float x, float y) -> LocalVector {
    const float theta = std::atan2(y, x);
    return LocalVector(-std::cos(theta), -std::sin(theta), 0.0);
  };
  // The output
  std::vector<DigitizerUtility::SignalPoint> collection_points;
  //collection_points.resize(ionization_points.size());
  collection_points.reserve(ionization_points.size());

  // Radiation damage limit of application
  // (XXX: No sense for 3D, let this be until decided what algorithm to use)
  const float RAD_DAMAGE = 0.001;

  for (const auto& super_charge : ionization_points) {
    // Extract the pixel cell
    auto current_pixel = pixdet->specificTopology().pixel(LocalPoint(super_charge.x(), super_charge.y()));
    // `pixel` function does not check to be in the ROC bounds,
    // so check it here and fix potential rounding problems.
    // Careful, this is assuming a rounding problem (1 unit), more than 1 pixel
    // away is probably showing some backward problem worth it to track.
    // This is also correcting out of bounds migrated charge from diffusion.
    // The charge will be moved to the edge of the row/column.
    current_pixel.first = std::clamp(current_pixel.first, float(0.0), (nrows - 1) + pix_rounding);
    current_pixel.second = std::clamp(current_pixel.second, float(0.0), (ncolumns - 1) + pix_rounding);

    const auto current_pixel_int = std::make_pair(std::floor(current_pixel.first), std::floor(current_pixel.second));

    // Convert to the 1x1 proxy pixel cell (pc), where all calculations are going to be
    // performed. The pixel is scaled to the actual pitch
    const auto relative_position_at_pc =
        std::make_pair((current_pixel.first - current_pixel_int.first) * pitch.first,
                       (current_pixel.second - current_pixel_int.second) * pitch.second);

    // Changing the reference frame to the proxy pixel cell
    LocalPoint position_at_pc(relative_position_at_pc.first - center_proxy_cell.x(),
                              relative_position_at_pc.second - center_proxy_cell.y(),
                              super_charge.z() - center_proxy_cell.z());

    LogDebug("Pixel3DDigitizerAlgorithm::drift")
        << "(super-)Charge\nlocal position: (" << super_charge.x() * 1.0_um_inv << ", " << super_charge.y() * 1.0_um_inv
        << ", " << super_charge.z() * 1.0_um_inv << ") [um]"
        << "\nMeasurement Point (row,column) (" << current_pixel.first << ", " << current_pixel.second << ")"
        << "\nProxy pixel-cell frame (centered at  left-down corner): (" << relative_position_at_pc.first * 1.0_um_inv
        << ", " << relative_position_at_pc.second * 1.0_um_inv << ") [um]"
        << "\nProxy pixel-cell frame (centered at n-column): (" << position_at_pc.x() * 1.0_um_inv << ", "
        << position_at_pc.y() * 1.0_um_inv << ") [um] "
        << "\nNe=" << super_charge.energy() << " electrons";

    // Check if the point is inside any of the column --> no charge was actually created then
    if (is_inside_n_column_(position_at_pc, thickness) || is_inside_ohmic_column_(position_at_pc, half_pitch)) {
      LogDebug("Pixel3DDigitizerAlgorithm::drift") << "Remove charge,  inside the n-column or p-column!!";
      continue;
    }

    float nelectrons = super_charge.energy();
    // XXX -- Diffusion: using the center frame
    if (diffusion_activated) {
      auto migrated_charges = diffusion(position_at_pc, super_charge.energy(), drift_direction, half_pitch, thickness);
      // remove the migrated charges
      for (auto& mc : migrated_charges) {
        nelectrons -= mc.energy();
        // and convert back to the pixel ref. system
        // Low-left origin/pitch -> relative within the pixel (a)
        // Adding the pixel
        const float pixel_x = current_pixel_int.first + (mc.x() + center_proxy_cell.x()) / pitch.first;
        const float pixel_y = current_pixel_int.second + (mc.y() + center_proxy_cell.y()) / pitch.second;
        const auto lp = pixdet->specificTopology().localPosition(MeasurementPoint(pixel_x, pixel_y));
        //Remember: the drift function will move the reference system to the bottom. We need to add what we previously subtract
        //in order to avoid a double translation when calling the drift function once again below
        mc.migrate_position(LocalPoint(lp.x(), lp.y(), mc.z() + center_proxy_cell.z()));
      }
      if (!migrated_charges.empty()) {
        LogDebug("Pixel3DDigitizerAlgorithm::drift") << "****************"
                                                     << "MIGRATING (super-)charges"
                                                     << "****************";
        // Drift this charges on the other pixel
        auto mig_colpoints = drift(hit, pixdet, bfield, migrated_charges, false);
        LogDebug("Pixel3DDigitizerAlgorithm::drift") << "*****************"
                                                     << "DOME MIGRATION"
                                                     << "****************";
      }
    }

    // Perform the drift, and check a potential lost of carriers because
    // they reach the pasivation region (-z < thickness/2)
    // XXX: not doing nothing, the carriers reach the electrode surface,
    const float drift_distance = position_at_pc.perp() - np_column_radius_;

    // Insert a charge loss due to Rad Damage here
    // XXX: ??
    float energyOnCollector = nelectrons;
    // FIXME: is this correct?  Not for 3D...

    if (pseudoRadDamage_ >= RAD_DAMAGE) {
      const float module_radius = pixdet->surface().position().perp();
      if (module_radius <= pseudoRadDamageRadius_) {
        const float kValue = pseudoRadDamage_ / (module_radius * module_radius);
        energyOnCollector = energyOnCollector * std::exp(-1.0 * kValue * drift_distance / max_radial_distance);
      }
    }
    LogDebug("Pixel3DDigitizerAlgorithm::drift")
        << "Drift distance = " << drift_distance * 1.0_um_inv << " [um], "
        << "Initial electrons = " << super_charge.energy()
        << " [electrons], Electrons after loss/diff= " << energyOnCollector << " [electrons] ";
    // Load the Charge distribution parameters
    // XXX -- probably makes no sense the SignalPoint anymore...
    collection_points.push_back(DigitizerUtility::SignalPoint(
        current_pixel_int.first, current_pixel_int.second, 0.0, 0.0, hit.tof(), energyOnCollector));
  }

  return collection_points;
}

// ====================================================================
// Signal is already "induced" (actually electrons transported to the
// n-column) at the electrode. Just collecting and adding-up all pixel
// signal and linking it to the simulated energy deposit (hit)
void Pixel3DDigitizerAlgorithm::induce_signal(const PSimHit& hit,
                                              const size_t hitIndex,
                                              const uint32_t tofBin,
                                              const Phase2TrackerGeomDetUnit* pixdet,
                                              const std::vector<DigitizerUtility::SignalPoint>& collection_points) {
  // X  - Rows, Left-Right
  // Y  - Columns, Down-Up
  const uint32_t detId = pixdet->geographicalId().rawId();
  // Accumulated signal at each channel of this detector
  signal_map_type& the_signal = _signal[detId];

  // Choose the proper pixel-to-channel converter
  std::function<int(int, int)> pixelToChannel =
      pixelFlag_ ? PixelDigi::pixelToChannel
                 : static_cast<std::function<int(int, int)> >(Phase2TrackerDigi::pixelToChannel);

  // Iterate over collection points on the collection plane
  for (const auto& pt : collection_points) {
    // Extract corresponding channel (position is already given in pixel indices)
    const int channel = pixelToChannel(pt.position().x(), pt.position().y());

    float corr_time = hit.tof() - pixdet->surface().toGlobal(hit.localPosition()).mag() * c_inv;
    if (makeDigiSimLinks_) {
      the_signal[channel] +=
          DigitizerUtility::Amplitude(pt.amplitude(), &hit, pt.amplitude(), corr_time, hitIndex, tofBin);
    } else {
      the_signal[channel] += DigitizerUtility::Amplitude(pt.amplitude(), nullptr, pt.amplitude());
    }

    LogDebug("Pixel3DDigitizerAlgorithm::induce_signal")
        << " Induce charge at row,col:" << pt.position() << " N_electrons:" << pt.amplitude() << " [Channel:" << channel
        << "]\n   [Accumulated signal in this channel:" << the_signal[channel].ampl() << "] "
        << " Global index linked PSimHit:" << hitIndex;
    ;
  }
}
