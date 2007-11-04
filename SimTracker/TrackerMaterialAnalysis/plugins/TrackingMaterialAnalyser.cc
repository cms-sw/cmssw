#include <iostream>     // FIXME: switch to MessagLogger & friends
#include <fstream>
#include <vector>
#include <string>
#include <exception>
#include <boost/tuple/tuple.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingStep.h"
#include "SimDataFormats/ValidationFormats/interface/MaterialAccountingTrack.h"
#include "MaterialAccountingLayer.h"
#include "TrackingMaterialAnalyser.h"
#include "TrackingMaterialPlotter.h"

//-------------------------------------------------------------------------
TrackingMaterialAnalyser::TrackingMaterialAnalyser(const edm::ParameterSet& iPSet)
{
  m_material            = iPSet.getParameter<edm::InputTag>("MaterialAccounting");
  m_stopAtLastDetector  = iPSet.getParameter<bool>("StopAtLastDetector");
  m_plotter             = new TrackingMaterialPlotter( 300., 120., 10 );      // 10x10 points per cm2
}    

//-------------------------------------------------------------------------
TrackingMaterialAnalyser::~TrackingMaterialAnalyser(void)
{
  if (m_plotter)
    delete m_plotter;
}    

//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::endJob(void)
{
  std::ofstream parameters("parameters");
  std::cout << std::endl << std::fixed;
  for (std::vector<MaterialAccountingLayer>::const_iterator layer = m_layers.begin(); layer != m_layers.end(); ++layer) {
    std::cout << std::setprecision(3);
    std::cout << std::setw(20) << std::left << layer->getName()
              << "z: [" << std::setw(8) << std::right << layer->getRangeZ().first << "," << std::setw(8) << std::right << layer->getRangeZ().second << "] "
              << "r: [" << std::setw(7) << std::right << layer->getRangeR().first << "," << std::setw(7) << std::right << layer->getRangeR().second << "] "
              << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "\tnumber of hits:        "        << std::setw(9) << layer->tracks()                  << std::endl;
    std::cout << "\tnormalized segment length:    " << std::setw(9) << layer->averageLength()           << " ± " << std::setw(9) << layer->sigmaLength()           << " cm" << std::endl;
    std::cout << "\tnormalized radiation lengths: " << std::setw(9) << layer->averageRadiationLengths() << " ± " << std::setw(9) << layer->sigmaRadiationLengths() << "    ";
    if (layer->material())
      std::cout << "\t(was " << layer->material()->radLen() << ")";
    std::cout << std::endl;
    std::cout << "\tnormalized energy loss:       " << std::setw(9) << layer->averageEnergyLoss()       << " ± " << std::setw(9) << layer->sigmaEnergyLoss()       << " MeV";
    if (layer->material())
      std::cout << "\t(was " << layer->material()->xi() * 1000. << " MeV)";     // GeV --> MeV
    std::cout << std::endl;

    parameters << std::setw(20) << std::left << layer->getName() << std::right << std::setprecision(6)
               << std::setw(6) << layer->tracks()
               << '\t' << std::setw(9) << layer->averageLength()             << " ± " << std::setw(9) << layer->sigmaLength()             << " cm"
               << '\t' << std::setw(9) << layer->averageRadiationLengths()   << " ± " << std::setw(9) << layer->sigmaRadiationLengths()
               << '\t' << std::setw(9) << layer->averageEnergyLoss() / 1000. << " ± " << std::setw(9) << layer->sigmaEnergyLoss() / 1000. << " Gev"
               << std::endl;
  }

  parameters.close();
  
  if (m_plotter) {
    m_plotter->normalize();
    m_plotter->draw();
  }
}


//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::beginJob(const edm::EventSetup & iSetup)
{
  edm::ESHandle<GeometricSearchTracker> hTracker;
  iSetup.get<TrackerRecoGeometryRecord>().get(hTracker);

  std::vector<DetLayer*> layers = hTracker->allLayers();
  m_layers.reserve( layers.size() );
  // INFO
  std::cout << "TrackingMaterialAnalyser: List of the tracker layers: " << std::endl;
  for (std::vector<DetLayer*>::const_iterator layer = layers.begin(); layer != layers.end(); ++layer) {
    std::cout << '\t' << (*layer)->subDetector() << std::endl;
    m_layers.push_back( MaterialAccountingLayer( **layer ) );
  }
  std::cout << std::endl;
}

//-------------------------------------------------------------------------
void TrackingMaterialAnalyser::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  edm::Handle< std::vector<MaterialAccountingTrack> > h_tracks;
  iEvent.getByLabel(m_material, h_tracks);

  for (std::vector<MaterialAccountingTrack>::const_iterator t = h_tracks->begin(), end = h_tracks->end(); t != end; ++t) {
    MaterialAccountingTrack track(*t);
    split( track );
  }
}

//-------------------------------------------------------------------------
// split a track in segments, each associated to a sensitive detector in a DetLayer;
// then, associate each step to one segment, splitting the steps across the segment boundaries
void TrackingMaterialAnalyser::split( MaterialAccountingTrack & track )
{
  // group sensitive detectors by their DetLayer
  std::vector<int> group( track.m_detectors.size() );
  for (unsigned int i = 0; i < track.m_detectors.size(); ++i)
    group[i] = findLayer( track.m_detectors[i] );

  unsigned int detectors = track.m_detectors.size();
  std::vector<double> limits(detectors + 2);
  limits[0] = -0.0001;                                                          // 1 um tolerance 
  for (unsigned int i = 1; i < detectors; ++i)
    limits[i] = (track.m_detectors[i-1].m_curvilinearOut + track.m_detectors[i].m_curvilinearIn) / 2.;
  if (detectors) {
    if (m_stopAtLastDetector)
      limits[detectors] = track.m_detectors[detectors-1].m_curvilinearOut + 0.001;    // 1 um tolerance
    else
      limits[detectors] = track.m_total.length() + 0.001;                             // 1 um tolerance
  }
  limits[detectors+1] = INFINITY;
  //for (unsigned int i = 0; i < detectors; ++i)
  //  std::cout << "MaterialAccountingTrack::split(): detector region boundaries: [" << limits[i] << ", " << limits[i+1] << "] along track" << std::endl;
 
  unsigned int index = 0;     // which detector 
  double begin = 0.;          // begginning of step, along the track
  double end   = 0.;          // end of step, along the track
  for (unsigned int i = 1; i < track.m_steps.size(); ++i) {
    if (index >= detectors) {
      // track outside acceptance, keep as unassocated
      if (m_plotter)
        m_plotter->plotSegmentUnassigned( track.m_steps[i] );
      continue;
    }
    if (begin <= limits[index]) {
      std::cerr << "MaterialAccountingTrack::split(): ERROR: internal logic error, expected " << limits[index] << " < " << begin << " < " << limits[index+1] << "]" << std::endl;
      break;
    }
    end = begin + track.m_steps[i].length();
    
    if (end <= limits[index+1]) {
      // step inside detector range
      //std::cout << "MaterialAccountingTrack::split(): step[" << i << "] at [" << begin << ", " << end << "] inside [" << limits[index] << ", " << limits[index+1] << "]" << std::endl;
      track.m_detectors[index].account( track.m_steps[i], begin, end );
      if (m_plotter)
        m_plotter->plotSegmentInLayer( track.m_steps[i], group[index] );
    } else {
      // step shared beteewn two detectors, transition at limits[index+1]
      //std::cout << "MaterialAccountingTrack::split(): step[" << i << "] at [" << begin << ", " << end << "] going from [" << limits[index] << ", " << limits[index+1] << "] into [" << limits[index+1] << ", " << limits[index+2] << "]" << std::endl;
      double fraction = (limits[index+1] - begin) / (end - begin);
      GlobalPoint limit(
          track.m_steps[i].in().x() + (track.m_steps[i].out().x() - track.m_steps[i].in().x()) * fraction,
          track.m_steps[i].in().y() + (track.m_steps[i].out().y() - track.m_steps[i].in().y()) * fraction,
          track.m_steps[i].in().z() + (track.m_steps[i].out().z() - track.m_steps[i].in().z()) * fraction
      );
     
      track.m_detectors[index].account( fraction * track.m_steps[i], begin, limits[index+1] );
      if (m_plotter) {
        MaterialAccountingStep step( fraction * track.m_steps[i].length(),
                                     fraction * track.m_steps[i].radiationLengths(),
                                     fraction * track.m_steps[i].energyLoss(),
                                     track.m_steps[i].in(), 
                                     limit );
        m_plotter->plotSegmentInLayer( step, group[index] );
      }
      ++index;          // next layer
      if (index < detectors)
        track.m_detectors[index].account( (1 - fraction) * track.m_steps[i], limits[index+1], end );

      if (m_plotter) {
        MaterialAccountingStep step( (1 - fraction) * track.m_steps[i].length(),
                                     (1 - fraction) * track.m_steps[i].radiationLengths(),
                                     (1 - fraction) * track.m_steps[i].energyLoss(),
                                     limit,
                                     track.m_steps[i].out() );
        if (index < detectors)
          m_plotter->plotSegmentInLayer( step,  group[index] );
        else
          // track outside acceptance, keep as unassocated
          m_plotter->plotSegmentUnassigned( step );
      }
    }
    begin = end;
  }
    
  // add the material from each detector to its layer (if there is one and only one)
  for (unsigned int i = 0; i < track.m_detectors.size(); ++i)
    if (group[i] != 0)
      m_layers[group[i]-1].addDetector( track.m_detectors[i] );

  // end of track: commit internal buffers and reset the m_layers internal state for a new track
  for (std::vector<MaterialAccountingLayer>::iterator layer = m_layers.begin(); layer != m_layers.end(); ++layer)
    layer->endOfTrack();
}

//-------------------------------------------------------------------------
// find the layer index (0: none, 1-3: PixelBarrel, 4-7: TID, 8-13: TOB, 14-15,28-29: PixelEndcap, 16-18,30-32: TID, 19-27,33-41: TEC)
int TrackingMaterialAnalyser::findLayer( const MaterialAccountingDetector & detector )
{
  int    index  = 0;
  size_t inside = 0;
  for (size_t i = 0; i < m_layers.size(); ++i)
    if (m_layers[i].inside(detector)) {
      ++inside;
      index = i+1;
    }
  if (inside == 0) {
    index = 0;
    std::cerr << "TrackingMaterialAnalyser::findLayer(...): ERROR: detector does not belong to any DetLayer" << std::endl;
  }
  if (inside > 1) {
    index = 0;
    std::cerr << "TrackingMaterialAnalyser::findLayer(...): ERROR: detector belongs to " << inside << "DetLayers" << std::endl;
  }

  return index;
} 

//-------------------------------------------------------------------------
// define as a plugin
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingMaterialAnalyser);
