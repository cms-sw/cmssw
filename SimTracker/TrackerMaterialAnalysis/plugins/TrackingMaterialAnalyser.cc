#include <iostream>     // FIXME: switch to MessagLogger & friends
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <boost/tuple/tuple.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"
#include "MaterialAccountingGroup.h"
#include "TrackingMaterialAnalyser.h"
#include "TrackingMaterialPlotter.h"

//-------------------------------------------------------------------------
TrackingMaterialAnalyser::TrackingMaterialAnalyser(const edm::ParameterSet& iPSet)
{
  m_material                = iPSet.getParameter<edm::InputTag>("MaterialAccounting");
  m_groupNames              = iPSet.getParameter<std::vector<std::string> >("Groups");
  const std::string & splitmode = iPSet.getParameter<std::string>("SplitMode");
  if (strcasecmp(splitmode.c_str(), "NearestLayer") == 0) {
    m_splitMode = NEAREST_LAYER;
  } else if (strcasecmp(splitmode.c_str(), "InnerLayer") == 0) {
    m_splitMode = INNER_LAYER;
  } else if (strcasecmp(splitmode.c_str(), "OuterLayer") == 0) {
    m_splitMode = OUTER_LAYER;
  } else {
    m_splitMode = UNDEFINED;
    throw edm::Exception(edm::errors::LogicError) << "Invalid SplitMode \"" << splitmode << "\". Acceptable values are \"NearestLayer\", \"InnerLayer\", \"OuterLayer\".";
  }
  m_skipAfterLastDetector   = iPSet.getParameter<bool>("SkipAfterLastDetector");
  m_skipBeforeFirstDetector = iPSet.getParameter<bool>("SkipBeforeFirstDetector");
  m_saveSummaryPlot         = iPSet.getParameter<bool>("SaveSummaryPlot");
  m_saveDetailedPlots       = iPSet.getParameter<bool>("SaveDetailedPlots");
  m_saveParameters          = iPSet.getParameter<bool>("SaveParameters");
  m_saveXml                 = iPSet.getParameter<bool>("SaveXML");
  if (m_saveSummaryPlot)
    m_plotter               = new TrackingMaterialPlotter( 300., 120., 10 );      // 10x10 points per cm2
  else
    m_plotter               = NULL;
}

//-------------------------------------------------------------------------
TrackingMaterialAnalyser::~TrackingMaterialAnalyser(void)
{
  if (m_plotter)
    delete m_plotter;
}

//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::saveParameters(const char* name)
{
  std::ofstream parameters(name);
  std::cout << std::endl;
  for (unsigned int i = 0; i < m_groups.size(); ++i) {
    MaterialAccountingGroup & layer = *(m_groups[i]);
    std::cout << layer.name() << std::endl;
    std::cout << boost::format("\tnumber of hits:               %9d") % layer.tracks() << std::endl;
    std::cout << boost::format("\tnormalized segment length:    %9.1f ± %9.1f cm")  % layer.averageLength()           % layer.sigmaLength()           << std::endl;
    std::cout << boost::format("\tnormalized radiation lengths: %9.3f ± %9.3f")     % layer.averageRadiationLengths() % layer.sigmaRadiationLengths() << std::endl;
    std::cout << boost::format("\tnormalized energy loss:       %9.3f ± %9.3f MeV") % layer.averageEnergyLoss()       % layer.sigmaEnergyLoss()       << std::endl;
    parameters << boost::format("%-20s\t%7d\t%5.1f ± %5.1f cm\t%6.4f ± %6.4f \t%6.4fe-03 ± %6.4fe-03 GeV")
                                % layer.name()
                                % layer.tracks()
                                % layer.averageLength()               % layer.sigmaLength()
                                % layer.averageRadiationLengths()     % layer.sigmaRadiationLengths()
                                % layer.averageEnergyLoss()           % layer.sigmaEnergyLoss()
               << std::endl;
  }
  std::cout << std::endl;

  parameters.close();
}

//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::saveXml(const char* name)
{
  std::ofstream xml(name);
  xml << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
  xml << "<Groups>" << std::endl;
  for (unsigned int i = 0; i < m_groups.size(); ++i) {
    MaterialAccountingGroup & layer = *(m_groups[i]);
    xml << "  <Group name=\"" << layer.name() << "\">\n"
        << "    <Parameter name=\"TrackerRadLength\" value=\"" << layer.averageRadiationLengths() << "\"/>\n"
        << "    <Parameter name=\"TrackerXi\" value=\"" << layer.averageEnergyLoss() << "\"/>\n"
        << "  </Group>\n" 
        << std::endl;
  }
  xml << "</Groups>" << std::endl;
}

//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::saveLayerPlots(const char * name)
{
  if (boost::filesystem::is_directory(name))
    boost::filesystem::remove_all(name);
  boost::filesystem::create_directory(name);
  for (unsigned int i = 0; i < m_groups.size(); ++i) {
    MaterialAccountingGroup & layer = *(m_groups[i]);
    layer.savePlots(name);
  }
}

//-------------------------------------------------------------------------
// FIXME we should save the parameters/xml/plots to a different file per IOV
void TrackingMaterialAnalyser::save()
{
  if (m_saveParameters)
    saveParameters("parameters");

  if (m_saveXml)
    saveXml("parameters.xml");

  if (m_saveDetailedPlots)
    saveLayerPlots("layers");

  if (m_saveSummaryPlot and m_plotter) {
    m_plotter->normalize();
    m_plotter->draw();
  }
}

//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::endJob(void)
{
  save();
}

//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::analyze(const edm::Event & event, const edm::EventSetup & setup)
{
  if (m_geometryWatcher.check(setup)) {
    edm::ESTransientHandle<DDCompactView> hDDD;
    setup.get<IdealGeometryRecord>().get( hDDD );

    if (m_groups.empty()) {
      // initialise the layers for the first time
      m_groups.resize( m_groupNames.size(), nullptr );
    } else {
      // the geometry has changed: save the current parameters/xml/plots and re-initialise the layers
      save();
      for (unsigned int i = 0; i < m_groups.size(); ++i)
        delete m_groups[i];
      if (m_plotter)
        m_plotter->reset();
    }

    for (unsigned int i = 0; i < m_groups.size(); ++i)
      m_groups[i] = new MaterialAccountingGroup( m_groupNames[i], * hDDD);

    // INFO
    std::cout << "TrackingMaterialAnalyser: List of the tracker groups: " << std::endl;
    for (unsigned int i = 0; i < m_groups.size(); ++i)
      std::cout << '\t' << m_groups[i]->info() << std::endl;
    std::cout << std::endl;
  }

  edm::Handle< std::vector<MaterialAccountingTrack> > h_tracks;
  event.getByLabel(m_material, h_tracks);

  for (std::vector<MaterialAccountingTrack>::const_iterator t = h_tracks->begin(), end = h_tracks->end(); t != end; ++t) {
    MaterialAccountingTrack track(*t);
    split( track );
  }
}

//-------------------------------------------------------------------------
// split a track in segments, each associated to a sensitive detector in a DetLayer;
// then, associate each step to one segment, splitting the steps across the segment boundaries
//
// Nota Bene: this implementation assumes that the steps stored along each track are consecutive and adjacent,
// and that no step can span across 3 layers, since all steps should split at layer boundaries

void TrackingMaterialAnalyser::split( MaterialAccountingTrack & track )
{
  // group sensitive detectors by their DetLayer
  std::vector<int> group( track.m_detectors.size() );
  for (unsigned int i = 0; i < track.m_detectors.size(); ++i)
    group[i] = findLayer( track.m_detectors[i] );

  unsigned int detectors = track.m_detectors.size();
  if (detectors == 0) {
    // the track doesn't cross any active detector:
    // keep al material as unassigned
    if (m_plotter)
      for (unsigned int i = 1; i < track.m_steps.size(); ++i)
        m_plotter->plotSegmentUnassigned( track.m_steps[i] );
  } else {
    const double TOLERANCE = 0.0001;    // 1 um tolerance
    std::vector<double> limits(detectors + 2);

    // define the trivial limits
    if (m_skipBeforeFirstDetector)
      limits[0] = track.m_detectors[0].m_curvilinearIn - TOLERANCE;
    else
      limits[0] = - TOLERANCE;
    if (m_skipAfterLastDetector)
      limits[detectors] = track.m_detectors[detectors-1].m_curvilinearOut + TOLERANCE;
    else
      limits[detectors] = track.m_total.length() + TOLERANCE;
    limits[detectors+1] = INFINITY;     // this is probably no more needed, but doesn't harm...

    // pick the algorithm to define the non-trivial limits
    switch (m_splitMode) {
      // assign each segment to the the nearest layer
      // e.g. the material between pixel barrel 3 and TIB 1 will be split among the two
      case NEAREST_LAYER:
        for (unsigned int i = 1; i < detectors; ++i)
          limits[i] = (track.m_detectors[i-1].m_curvilinearOut + track.m_detectors[i].m_curvilinearIn) / 2.;
        break;

      // assign each segment to the the inner layer
      // e.g. all material between pixel barrel 3 and TIB 1 will go into the pixel barrel
      case INNER_LAYER:
        for (unsigned int i = 1; i < detectors; ++i)
          limits[i] = track.m_detectors[i].m_curvilinearIn - TOLERANCE;
        break;

      // assign each segment to the the outer layer
      // e.g. all material between pixel barrel 3 and TIB 1 will go into the TIB
      case OUTER_LAYER:
        for (unsigned int i = 1; i < detectors; ++i)
          limits[i] = track.m_detectors[i-1].m_curvilinearOut + TOLERANCE;
        break;

      case UNDEFINED:
      default:
        // throw something
        throw edm::Exception(edm::errors::LogicError) << "Invalid SplitMode";
    }

    //for (unsigned int i = 0; i < detectors; ++i)
    //  std::cout << "MaterialAccountingTrack::split(): detector region boundaries: [" << limits[i] << ", " << limits[i+1] << "] along track" << std::endl;

    double begin = 0.;          // begginning of step, along the track
    double end   = 0.;          // end of step, along the track
    unsigned int i = 1;         // step conter

    // skip the material before the first layer
    //std::cout << "before first layer, skipping" << std::endl;
    while (end < limits[0]) {
      const MaterialAccountingStep & step = track.m_steps[i++];
      end = begin + step.length();

      // do not account material before the first layer
      if (m_plotter)
        m_plotter->plotSegmentUnassigned( step );

      begin = end;
      //std::cout << '.';
    }
    //std::cout << std::endl;

    // optionally split a step across the first layer boundary
    //std::cout << "first layer (0): " << limits[0] << ".." << limits[1] << std::endl;
    if (begin < limits[0] and end > limits[0]) {
      const MaterialAccountingStep & step = track.m_steps[i++];
      end = begin + step.length();

      double fraction = (limits[0] - begin) / (end - begin);
      std::pair<MaterialAccountingStep, MaterialAccountingStep> parts = step.split(fraction);

      //std::cout << '!' << std::endl;
      track.m_detectors[0].account( parts.second, limits[1], end );

      if (m_plotter) {
        // step partially before first layer, keep first part as unassocated
        m_plotter->plotSegmentUnassigned( parts.first );

        // associate second part to first layer
        m_plotter->plotSegmentInLayer( parts.second,  group[0] );
      }
      begin = end;
    }

    unsigned int index = 0;     // which detector
    while (i < track.m_steps.size()) {
      const MaterialAccountingStep & step = track.m_steps[i++];

      end = begin + step.length();

      if (begin > limits[detectors]) {
        // segment after last layer and skipping requested in configuation
        if (m_plotter)
          m_plotter->plotSegmentUnassigned( step );
        begin = end;
        continue;
      }

      // from here onwards we should be in the accountable region, either completely in a single layer:
      //   limits[index] <= begin < end <= limits[index+1]
      // or possibly split between 2 layers
      //   limits[index] < begin < limits[index+1] < end <  limits[index+2]
      if (begin < limits[index] or end > limits[index+2]) {
        // sanity check
        std::cerr << "MaterialAccountingTrack::split(): ERROR: internal logic error, expected " << limits[index] << " < " << begin << " < " << limits[index+1] << std::endl;
        break;
      }

      //std::cout << '.';
      if (limits[index] <= begin and end <= limits[index+1]) {
        // step completely inside current detector range
        track.m_detectors[index].account( step, begin, end );
        if (m_plotter)
          m_plotter->plotSegmentInLayer( step, group[index] );
      } else {
        // step shared beteewn two detectors, transition at limits[index+1]
        double fraction = (limits[index+1] - begin) / (end - begin);
        std::pair<MaterialAccountingStep, MaterialAccountingStep> parts = step.split(fraction);

        if (m_plotter) {
          if (index > 0)
            m_plotter->plotSegmentInLayer( parts.first, group[index] );
          else
            // track outside acceptance, keep as unassocated
            m_plotter->plotSegmentUnassigned( parts.first );

          if (index+1 < detectors)
            m_plotter->plotSegmentInLayer( parts.second,  group[index+1] );
          else
            // track outside acceptance, keep as unassocated
            m_plotter->plotSegmentUnassigned( parts.second );
        }

        track.m_detectors[index].account( parts.first, begin, limits[index+1] );
        ++index;          // next layer
        //std::cout << '!' << std::endl;
        //std::cout << "next layer (" << index << "): " << limits[index] << ".." << limits[index+1] << std::endl;
        if (index < detectors)
          track.m_detectors[index].account( parts.second, limits[index+1], end );
      }
      begin = end;
    }

  }
  //std::cout << std::endl;

  // add the material from each detector to its layer (if there is one and only one)
  for (unsigned int i = 0; i < track.m_detectors.size(); ++i)
    if (group[i] != 0)
      m_groups[group[i]-1]->addDetector( track.m_detectors[i] );

  // end of track: commit internal buffers and reset the m_groups internal state for a new track
  for (unsigned int i = 0; i < m_groups.size(); ++i)
    m_groups[i]->endOfTrack();
}

//-------------------------------------------------------------------------
// find the layer index (0: none, 1-3: PixelBarrel, 4-7: TID, 8-13: TOB, 14-15,28-29: PixelEndcap, 16-18,30-32: TID, 19-27,33-41: TEC)
int TrackingMaterialAnalyser::findLayer( const MaterialAccountingDetector & detector )
{
  int    index  = 0;
  size_t inside = 0;
  for (size_t i = 0; i < m_groups.size(); ++i)
    if (m_groups[i]->inside(detector)) {
      ++inside;
      index = i+1;
    }
  if (inside == 0) {
    index = 0;
    std::cerr << "TrackingMaterialAnalyser::findLayer(...): ERROR: detector does not belong to any DetLayer" << std::endl;
    std::cerr << "TrackingMaterialAnalyser::findLayer(...): detector position: " << std::fixed
              << " (r: " << std::setprecision(1) << std::setw(5) << detector.position().perp()
              << ", z: " << std::setprecision(1) << std::setw(6) << detector.position().z()
              << ", phi: " << std::setprecision(3) << std::setw(6) << detector.position().phi() << ")" 
              << std::endl;
  }
  if (inside > 1) {
    index = 0;
    std::cerr << "TrackingMaterialAnalyser::findLayer(...): ERROR: detector belongs to " << inside << "DetLayers" << std::endl;
    std::cerr << "TrackingMaterialAnalyser::findLayer(...): detector position: " << std::fixed
              << " (r: " << std::setprecision(1) << std::setw(5) << detector.position().perp()
              << ", z: " << std::setprecision(1) << std::setw(6) << detector.position().z()
              << ", phi: " << std::setprecision(3) << std::setw(6) << detector.position().phi() << ")" 
              << std::endl;
  }

  return index;
}

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingMaterialAnalyser);
