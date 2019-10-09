#ifndef __SimTracker_SiPhase2Digitizer_DQMPixelCell_h
#define __SimTracker_SiPhase2Digitizer_DQMPixelCell_h

// -*- C++ -*-
//
// Package:    SimTracker/SiPhase2Digitizer
// Class:      DQMPixelCell
//
/**\class DQMPixelCell DQMPixelCell.cc
 
  Description: Access Digi collection and creates histograms accumulating
               data from the module to different pixel cells, 1x1 cell, 
               2x2 cell, etc..

*/
// Author:  J.Duarte-Campderros (IFCA)
// Created: 2019-10-02
//
//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "FWCore/Utilities/interface/InputTag.h"
//#include "DataFormats/Math/interface/deltaPhi.h"
//#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
//#include "SimTracker/SiPhase2Digitizer/plugins/Phase2TrackerDigitizerFwd.h"
//#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
//
//#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

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

// system
#include <vector>
#include <map>
//#include <memory>
//#include <cmath>
//

//class MonitorElement;
class GeomDet;

class DQMPixelCell : public DQMEDAnalyzer 
{
    public:
        explicit DQMPixelCell(const edm::ParameterSet&);
        ~DQMPixelCell() override;
        
        void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
        void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
        void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
        
    private:
        // enumerate defining the different subdetectors
        //int matchedSimTrackIndex(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& linkHandle,
        //                   edm::Handle<edm::SimTrackContainer>& simTkHandle,
        //                   DetId detId,
        //                   unsigned int& channel);
        //void fillClusterWidth(DigiMEs& mes, float dphi, float width)
        //
        // GeomDet units belonging to the tracker pixel system (Barrel and endcap)
        bool isPixelSystem_(const GeomDet * detunit) const;

        // The detector unit associated to a list of histograms
        // Barrel_Layer / Endcap_Layer_side: Need 3 numbers 
        int meUnit_(bool isBarrel, int layer, int side) const;

        // Get the pixel digi found at that channel for a given subdetector unit
        const PixelDigi & get_digi_from_channel_(int ch, const edm::DetSetVector<PixelDigi>::const_iterator & itdigis);
        
        // Get the sim track with trackId for given subdetector unit
        const SimTrack * get_simtrack_from_id_(unsigned int tid, const edm::SimTrackContainer * simtracks);

        // Get the set of simhits created by a track 
        const edm::PSimHitContainer get_simhits_from_trackid_(
                unsigned int tid, 
                unsigned int detid_raw, 
                const std::vector<const edm::PSimHitContainer*> & psimhits);

        // Helper container to memoize the SimHits created by a track (Id) per event
        // This should be cleared at the begining of the event
        std::map<unsigned int,std::map<unsigned int,edm::PSimHitContainer>> m_tId_det_simhits_;

        // Transform a given measurement points into the i-cell pixel frame
        const std::pair<double,double> pixel_cell_transformation_(
                const std::pair<double,double> & pos,
                unsigned int icell,
                const std::pair<double,double> & pitch);

        // Histograms:
        // HElper setup functions 
        MonitorElement * setupH1D_(DQMStore::IBooker& ibooker, const std::string & histoname, const std::string & title);
        MonitorElement * setupH2D_(DQMStore::IBooker& ibooker, const std::string & histoname, const std::string & title);
        MonitorElement * setupH2D_(DQMStore::IBooker& ibooker, 
                const std::string & histoname, 
                const std::string & title, 
                const std::pair<double,double> & xranges, 
                const std::pair<double,double> & yranges);
        MonitorElement * setupProf2D_(DQMStore::IBooker& ibooker, 
                const std::string & histoname, 
                const std::string & title, 
                const std::pair<double,double> & xranges, 
                const std::pair<double,double> & yranges);
        // Whole CMS histos
        MonitorElement * vME_track_XYMap_;
        MonitorElement * vME_track_RZMap_;
        MonitorElement * vME_digi_XYMap_;
        MonitorElement * vME_digi_RZMap_;
        // Per detector unit plots
        std::map<int,MonitorElement *> vME_clsize1D_;
        std::map<int,MonitorElement *> vME_clsize1Dx_;
        std::map<int,MonitorElement *> vME_clsize1Dy_;
        std::map<int,MonitorElement *> vME_charge1D_;
        std::map<int,MonitorElement *> vME_track_dxdz_;
        std::map<int,MonitorElement *> vME_track_dydz_;
        std::map<int,MonitorElement *> vME_track_dxdzAngle_;
        std::map<int,MonitorElement *> vME_track_dydzAngle_;
        std::map<int,MonitorElement *> vME_dx1D_;
        std::map<int,MonitorElement *> vME_dy1D_;        
        std::map<int,MonitorElement *> vME_digi_charge1D_;
        // --- cell efficiency per subdector , each element on the 
        //     vector 0: total, 1: 1x1, 2: 2x2, (3: 3x3, 4: 4x4)?
        std::map<int,std::vector<MonitorElement *>> vME_position_cell_;
        std::map<int,std::vector<MonitorElement *>> vME_eff_cell_;
        std::map<int,std::vector<MonitorElement *>> vME_clsize_cell_;
        std::map<int,std::vector<MonitorElement *>> vME_charge_cell_;
        std::map<int,std::vector<MonitorElement *>> vME_dx_cell_;
        std::map<int,std::vector<MonitorElement *>> vME_dy_cell_;

        // Map to take care of common and tedious
        //const std::vector<std::map<int,std::vector<MonitorElement *>>*> helperMap_;
        
        // Configuration 
        edm::ParameterSet config_;
        // Geometry to use
        std::string geomType_;
        
        //std::vector<double> phiValues;
        // EDM token for the input collections 
        std::vector<edm::EDGetTokenT<edm::PSimHitContainer>> simHitTokens_;
        const edm::EDGetTokenT<edm::DetSetVector<PixelDigi> > digiToken_;
        const edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > digiSimLinkToken_;
        const edm::EDGetTokenT<edm::SimTrackContainer> simTrackToken_;
};
#endif
