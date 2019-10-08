// 
// Access Digi collection and creates histograms accumulating
// data from the module to different pixel cells, 1x1 cell, 
// 2x2 cell, etc..
//
// Author:  J.Duarte-Campderros (IFCA)
// Created: 2019-10-02
//

// CMSSW Framework 
//#include "DQMServices/Core/interface/MonitorElement.h"
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

// CMSSW Data formats
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"

// system
#include <algorithm>

// XXX - Be careful the relative position
#include "DQMPixelCell.h"


double unit_um = 1e4; // [cm]

using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;


DQMPixelCell::DQMPixelCell(const edm::ParameterSet& iConfig) : 
    config_(iConfig),
    geomType_(iConfig.getParameter<std::string>("GeometryType")),
    //phiValues(iConfig.getParameter<std::vector<double> >("PhiAngles")),
    digiToken_(consumes<edm::DetSetVector<PixelDigi> >(
                iConfig.getParameter<edm::InputTag>("PixelDigiSource"))
            ),
    digiSimLinkToken_(consumes<edm::DetSetVector<PixelDigiSimLink> >(
                iConfig.getParameter<edm::InputTag>("PixelDigiSimSource"))),
    simTrackToken_(consumes<edm::SimTrackContainer>(
                iConfig.getParameter<edm::InputTag>("SimTrackSource"))) 
{
    LogDebug("DQMPixelCell") << ">>> Construct DQMPixelCell ";
    
    const std::vector<edm::InputTag> psimhit_v(config_.getParameter<std::vector<edm::InputTag>>("PSimHitSource"));

    for(const auto & itag: psimhit_v)
    {
        simHitTokens_.push_back(consumes<edm::PSimHitContainer>(itag));
    }
}

//
// destructor
//
DQMPixelCell::~DQMPixelCell() 
{
    LogDebug("DQMPixelCell") << ">>> Destroy DQMPixelCell ";
}
//
// -- DQM Begin Run
//
void DQMPixelCell::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::LogInfo("DQMPixelCell") << "Initialize DQMPixelCell ";
}
//
// -- Analyze
//
void DQMPixelCell::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // Get digis and the simhit links and the simhits
    edm::Handle<edm::DetSetVector<PixelDigiSimLink> > digiSimLinkHandle;
    iEvent.getByToken(digiSimLinkToken_, digiSimLinkHandle);
    const edm::DetSetVector<PixelDigiSimLink> * simdigis = digiSimLinkHandle.product();
    
    edm::Handle<edm::DetSetVector<PixelDigi> > digiHandle;
    iEvent.getByToken(digiToken_, digiHandle);
    const edm::DetSetVector<PixelDigi> * digis = digiHandle.product();
    
    // Vector of simHits
    // XXX : NOt like that, just an example
    /*std::vector<const edm::PSimHitContainer*> simhits;
    simhits.reserve(simHitTokens_.size());
    for(const auto & sh_token: simHitTokens_)
    {
        edm::Handle<edm::PSimHitContainer> simHitHandle;
        iEvent.getByToken(sh_token, simHitHandle);
        //const edm::PSimHitContainer * simhits = simHitHandle.product();
        simhits.push_back( simHitHandle.product() );
    }*/

    // Get SimTrack
    edm::Handle<edm::SimTrackContainer> simTrackHandle;
    iEvent.getByToken(simTrackToken_, simTrackHandle);
    const edm::SimTrackContainer * simtracks = simTrackHandle.product();
    
    // Geometry description
    edm::ESWatcher<TrackerDigiGeometryRecord> tkDigiGeomWatcher;
    if(! tkDigiGeomWatcher.check(iSetup) )
    {
        // XXX -- Should raise a Warning?? 
        return;
    }
    
    // Tracker geometry and topology
    edm::ESHandle<TrackerTopology> tTopoHandle;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology * topo = tTopoHandle.product();

    edm::ESHandle<TrackerGeometry> geomHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geomHandle);
    const TrackerGeometry* tkGeom = geomHandle.product();

    // Let's loop over the detectors
    for(auto const & dunit: tkGeom->detUnits())
    {
        if( ! isPixelSystem_(dunit) )
        {
            continue;
        }
        
        // --------------------------------------------
        // Get more info about the detector unit 
        // -- layer, topology (nrows,ncolumns,pitch...)
        const Phase2TrackerGeomDetUnit * tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(dunit);
        //const PixelTopology * topo = tkDetUnit->specificTopology();

        const auto detId = dunit->geographicalId();
        const int layer = topo->layer(detId);
        // Get the relevant histo key
        const auto & me_unit = meUnit_(tkDetUnit->type().isBarrel(),layer,topo->side(detId));

        // Find simulated digis links on this det unit
        const auto & it_simdigilink = simdigis->find(detId);
        if(it_simdigilink == simdigis->end())
        {
            // FIXME: CHeck if there is any digi ... Should not
            continue;
        }

        // Find created RAW digis on this det unit
        const auto & it_digis = digis->find(detId);
        /*if(it_digilink == digis->end())
        {
            // FIXME: CHeck if there is any digi ... Should not
            //continue;
        }*/
        //std::cout << "DETECTOR: " << tkDetUnit->type().name() << std::endl;

        // Loop over the simulated digi links, for each one get the track id, 
        // used to obtain the simhit, and the raw digi (via the channel)
        std::map<unsigned int,std::set<int> > stracks_channels;
        for(const auto & dhsim: *it_simdigilink)
        {
            const int current_channel = dhsim.channel();
            // Already processed (ignoring fractions in the same pixel channel)
            if(stracks_channels.find(dhsim.SimTrackId()) != stracks_channels.end())
            {
                if(stracks_channels[dhsim.SimTrackId()].find(current_channel) != stracks_channels[dhsim.SimTrackId()].end())
                {
                    continue;
                }
            }
            // Create/update the list of channels created by this simtrack
            stracks_channels[dhsim.SimTrackId()].insert(current_channel);
            
            // Get The Sim tracks (once per simtrack)
            // Obtain the dz/dx y dz/dy from the track?
            const SimTrack & current_simtrack = get_simtrack_from_index_(dhsim.SimTrackId(),simtracks);
            // Convert the momentum into the local frame in order to evaluate the incident
            // angle, but first it needs to be converted into a GlobalVector
            const GlobalVector cst_momentum(current_simtrack.momentum().x(),
                    current_simtrack.momentum().y(),
                    current_simtrack.momentum().z());
            const LocalVector cst_m_local(dunit->surface().toLocal(cst_momentum));
            vME_track_dxdz_[me_unit]->Fill(cst_m_local.x()/cst_m_local.z());
            vME_track_dydz_[me_unit]->Fill(cst_m_local.y()/cst_m_local.z());

            // Get the pixels col and row
            //const auto current_pixel(PixelDigi::channelToPixel(current_channel));

            // Get the SimTrack, i.e. the persistent version of a G4Track, corresponding
            // to a SimHit --> Therefore the actual info is in the PSimHit
            // XXX: Need to check the validity of the Id?
//            const SimTrack & current_simtrack = (*simtracks)[dhsim.SimTrackId()];
//            const GlobalPoint st_position(current_simtrack.trackerSurfacePosition().x(),
//                    current_simtrack.trackerSurfacePosition().y(),
//                    current_simtrack.trackerSurfacePosition().z());
//            const GlobalVector st_momentum(current_simtrack.momentum().x(),
//                    current_simtrack.momentum().y(),
//                    current_simtrack.momentum().z());

//std::cout << "---- # Found a sim link [channel:" << current_channel << "] track id:" 
//    << dhsim.SimTrackId() << " -- Fraction of the track Eloss: " << dhsim.fraction() 
//    << " At measurement position: " << current_pixel.first << " " << current_pixel.second 
//    << " Local Position: " << tkDetUnit->specificTopology().localPosition(MeasurementPoint(current_pixel.first,current_pixel.second)) 
//    << " Global Position: " << dunit->surface().toGlobal(
//            tkDetUnit->specificTopology().localPosition(MeasurementPoint(current_pixel.first,current_pixel.second)) ) << std::endl;
//std::cout << "------- Corresponding Simtrack id: " << dhsim.SimTrackId() << " which is a [PDG code]" 
//    << current_simtrack.type() << "  : Global TrackerSurfacePosition: " << st_position << "" 
//    << " momentum: " << st_momentum 
//    << " /// Local position: " << dunit->surface().toLocal(st_position) 
//    << "  (without using surface: " << dunit->toLocal(st_position) << " ) "
//    << " local momentum: " << dunit->surface().toLocal(st_momentum) << std::endl;


//            const PixelDigi * digi_linked = nullptr;
            // Search for the digilink of the current channel
//            for(const auto & dh: *it_digilink)
//            {
//
//                if(dh.channel() == current_channel)
//                {
//                    digi_linked = &dh;
//                    vME_eff_cell_[me_unit][0]->Fill(current_pixel.first,current_pixel.second,1.0);
//                    // Efficiency
//                    
////std::cout << "    ---->>>> Found a digi in that channel [channel:" << digi_linked->channel() << std::endl;
//                }
//            }
//            if(digi_linked == nullptr)
//            {
//                vME_eff_cell_[me_unit][0]->Fill(current_pixel.first,current_pixel.second,0.0);
////std::cout << "   ------- ---->>>> NOT Found a digi in that channel [channel:" << channel << "]" << std::endl;
//            } 
//            const edm::DetSet<PixelDigi>::iterator dh = std::find(it_digilink->begin(),it_digilink->end(), 
//                    [&channel] (const PixelDigi & dlink) -> bool
//                    {
//                        return true;
//                        //return (channel == static_cast<int>(dlink->channel()));
//                    });
            //for(const auto & dh: *it_digilink)
            //{
            //}
                //break;
            //if(dh.channel() == static_cast<int>(dhsim.channel()))
            //{
            //}
        }

        // Fill  per detector histograms
        for(const auto & st_ch: stracks_channels)
        {
            // Cluster size on the detector unit
            vME_clsize1D_[me_unit]->Fill(st_ch.second.size());

            // Get the total charge for this cluster size
            int cluster_tot = 0;
            for(const auto & ch: st_ch.second)
            {
                const PixelDigi & current_digi = get_digi_from_channel_(ch,it_digis);
                cluster_tot += current_digi.adc();
            }
            vME_charge1D_[me_unit]->Fill(cluster_tot);
            
        }

for(const auto & _kk: stracks_channels)
{
 std::cout << "    ---> Track id: [" << _kk.first << "]  At channels: -->  ";
     for(const auto & _ch: _kk.second)
     {
std::cout << " |row: "      << PixelDigi::channelToPixel(_ch).first << " col: "
     << PixelDigi::channelToPixel(_ch).second << "| ";
     }
std::cout <<  " TOTAL cluster size: " <<  _kk.second.size() << std::endl;

}
    }

    // Input digis loop
    /*for(const auto & digi: *digis)
    {
        const DetId detId(digi.id);
        // Just keep tracker digis
        if(detId.det() != DetId::Detector::Tracker)
        {
            continue;
        }
        // And only pixels (Forward and barrel, Endcap??)`
        if(detId.subdetId() != PixelSubdetector::PixelBarrel 
                && detId.subdetId() != PixelSubdetector::PixelEndcap )
        {
            continue;
        }

        // ----------------------------
        // Get the module where is the hit
        const GeomDetUnit * geomDetUnit = tkGeom->idToDetUnit(detId);
        const Phase2TrackerGeomDetUnit * tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(geomDetUnit);

        const int layer = topo->layer(detId);
        
        // Get the relevant histo
        const auto & me_unit = meUnit_(tkDetUnit->type().isBarrel(),layer,topo->side(detId));
        // FIXME? Check the existence of this?

        // Module topology
        //const int nRows = tkDetUnit->specificTopology().nrows();
        //const int nCols = tkDetUnit->specificTopology().ncolumns();
        //const auto pitch = tkDetUnit->specificTopology().pitch();
        
        // Get the simlinks
        const auto & it_digilink = simdigis->find(detId);        
    
        // Loop over the digis on the Detector unit.
        for(const auto & dh: digi)
        {
            // Get local position
            //const auto localpos = tkDetUnit->specificTopology().localPosition(MeasurementPoint(dh.row(),dh.column()));
            // Fill positions of the digi 
            vME_position_cell_[me_unit][0]->Fill(dh.row(),dh.column());
            // Fill position of the cell (maybe inside a loop)
            // --- Makes no sense --> as the the digi is charge in a pixel... 
            //const double row_cell1    = dh.row()-int(dh.row()/double(pitch.first))*pitch.first;
            //const double column_cell1 = dh.column()-int((dh.column()/pitch.second))*pitch.second;
std::cout << "Pixel [" << dh.channel() << "] row: " << dh.row() << "  col:" << dh.column() << " tot: " << dh.adc() << " Search for a link..." << std::endl;
//    << " -- " << int(dh.row()/double(pitch.first))*pitch.first << " -- " << row_cell1  
//    << std::endl;
            //vME_position_cell_[me_unit][1]->Fill(row_cell1,column_cell1);
            //vME_position_cell_[me_unit][2]->Fill(2.0*row_cell1,2.0*column_cell1);
            // -- Find the links if any
            double __prob = 0.0;
            for(const auto & dhsim: *it_digilink)
            {
                if(dh.channel() == static_cast<int>(dhsim.channel()))
                {
std::cout << " ##------> Found a sim link: " << dhsim.SimTrackId() << " -- " << dhsim.CFposition() 
    << " -- " << dhsim.TofBin() << " -- "  << dhsim.fraction() << std::endl;
                    __prob += dhsim.fraction();
                    //break;
                }
            }
std::cout << "           ----> TOTAL SUM of fraction:" << __prob << std::endl;
        }
    }*/

//      const GeomDetUnit* geomDetUnit = tkGeom->idToDetUnit(detId);
//
//      const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(geomDetUnit);
//      int nColumns = tkDetUnit->specificTopology().ncolumns();
//
//      edm::LogInfo("DQMPixelCell") << " Det Id = " << rawid;
//
//      if (layer <= 3) {
//        if (nColumns > 2)
//          moduleType = "PSP_Modules";
//        else
//          moduleType = "PSS_Modules";
//      } else
//        moduleType = "2S_Modules";
//
//      std::map<std::string, DigiMEs>::iterator pos = detMEs.find(moduleType);
//      if (pos != detMEs.end()) {
//        DigiMEs local_mes = pos->second;
//        int nDigi = 0;
//        int row_last = -1;
//        int col_last = -1;
//        int nclus = 0;
//        int width = 1;
//        int position = 0;
//        float dPhi = 9999.9;
//        for (DetSet<Phase2TrackerDigi>::const_iterator di = DSViter->begin(); di != DSViter->end(); di++) {
//          int col = di->column();  // column
//          int row = di->row();     // row
//          MeasurementPoint mp(row + 0.5, col + 0.5);
//          unsigned int channel = Phase2TrackerDigi::pixelToChannel(row, col);
//          int tkIndx = matchedSimTrackIndex(digiSimLinkHandle, simTrackHandle, detId, channel);
//
//          if (geomDetUnit && tkIndx != -1)
//            dPhi = reco::deltaPhi((*simTrackHandle)[tkIndx].momentum().phi(), geomDetUnit->position().phi());
//
//          nDigi++;
//          edm::LogInfo("DQMPixelCell") << "  column " << col << " row " << row << std::endl;
//          local_mes.PositionOfDigis->Fill(row + 1);
//
//          if (row_last == -1) {
//            width = 1;
//            position = row + 1;
//            nclus++;
//          } else {
//            if (abs(row - row_last) == 1 && col == col_last) {
//              position += row + 1;
//              width++;
//            } else {
//              position /= width;
//              fillClusterWidth(local_mes, dPhi, width);
//              local_mes.ClusterPosition->Fill(position);
//              width = 1;
//              position = row + 1;
//              nclus++;
//            }
//          }
//          edm::LogInfo("DQMPixelCell") << " row " << row << " col " << col << " row_last " << row_last << " col_last "
//                                    << col_last << " width " << width;
//          row_last = row;
//          col_last = col;
//        }
//        position /= width;
//        fillClusterWidth(local_mes, dPhi, width);
//        local_mes.ClusterPosition->Fill(position);
//        local_mes.NumberOfClusters->Fill(nclus);
//        local_mes.NumberOfDigis->Fill(nDigi);
//      }
//    }
//  }
}
//
// -- Book Histograms
//
void DQMPixelCell::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) 
{
    // Get Geometry to associate a folder to each Pixel subdetector
    edm::ESHandle<TrackerGeometry> geomHandle;
    iSetup.get<TrackerDigiGeometryRecord>().get(geomType_, geomHandle);
    const TrackerGeometry * tkGeom = geomHandle.product();
    
    // Tracker Topology (layers, modules, side, etc..)
    edm::ESHandle<TrackerTopology> tTopoHandle;
    iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
    const TrackerTopology * topo = tTopoHandle.product();

    const std::string top_folder = config_.getParameter<std::string>("TopFolderName");

    // Get all pixel subdetector, create histogram by layers
    // -- More granularity can be accomplished by modules (1 central + 5 modules: in z, 
    //    each one composed by 12 pixel modules (1 long pixel + 2 ROCs) in Phi = 108 pixel-modules
    for(auto const & dunit: tkGeom->detUnits())
    {
        if( ! isPixelSystem_(dunit) )
        {
            continue;
        }
        //if( ! dtype.isInnerTracker() || ! dtype.isTrackerPixel() )
        //{
        //    continue;
        //}
        const auto & dtype = dunit->type();
        
        const auto detId = dunit->geographicalId();
        const int layer = topo->layer(detId);
        // Create the sub-detector histogram if needed
        // Barrel_Layer/Endcap_Layer_Side
        const int me_unit = meUnit_(dtype.isBarrel(),layer,topo->side(detId));
        if(vME_position_cell_.find(me_unit) == vME_position_cell_.end() )
        {
            std::string folder_name(top_folder+"/");
            if(dtype.isBarrel())
            {
                folder_name += "Barrel";
            }
            else if(dtype.isEndcap())
            {
                folder_name += "Endcap";
            }
            else
            {
                cms::Exception("Tracker subdetector '") << dtype.subDetector() << "'"
                    << " malformed: does not belong to the Endcap neither the Barrel";
            }
            folder_name += "_Layer"+std::to_string(layer);

            if(topo->side(detId))
            {
                folder_name += "_Side"+std::to_string(topo->side(detId));
            }
            // Go the proper folder
            ibooker.cd();
            ibooker.setCurrentFolder(folder_name);

            const TrackerGeomDet * geomDetUnit = tkGeom->idToDetUnit(detId);
            const Phase2TrackerGeomDetUnit* tkDetUnit = dynamic_cast<const Phase2TrackerGeomDetUnit*>(geomDetUnit);
            const int nrows = tkDetUnit->specificTopology().nrows();
            const int ncols = tkDetUnit->specificTopology().ncolumns();
            const auto pitch = tkDetUnit->specificTopology().pitch();
            //const double x_size = nrows*pitch.first;
            //const double y_size = ncols*pitch.second;
            
            // Prepare the ranges: 0- whole sensor, 1- cell 1x1, 2-cell 2x2,
            const std::vector<std::pair<double,double>> xranges = {
                std::make_pair<double,double>(0,nrows-1),
                std::make_pair<double,double>(0,pitch.first),
                std::make_pair<double,double>(0,2.0*pitch.first) };
            const std::vector<std::pair<double,double>> yranges = {
                std::make_pair<double,double>(0,ncols-1),
                std::make_pair<double,double>(0,pitch.second),
                std::make_pair<double,double>(0,2.0*pitch.second) };

            // And create the histos
            for(unsigned int i = 0; i < xranges.size(); ++i)
            {
                // 
                vME_position_cell_[me_unit].push_back(setupH2D_(ibooker,
                            "Position_"+std::to_string(i),
                            xranges[i],
                            yranges[i]));
                vME_eff_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "Efficiency_"+std::to_string(i),
                            xranges[i],
                            yranges[i]));
                vME_clsize_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "ClusterSize_"+std::to_string(i),
                            xranges[i],
                            yranges[i]));
                vME_charge_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "Charge_"+std::to_string(i),
                            xranges[i],
                            yranges[i]));
            }
            // Per detector unit histos
            vME_clsize1D_[me_unit] = setupH1D_(ibooker,"ClusterSize1D");
            vME_charge1D_[me_unit] = setupH1D_(ibooker,"Charge1D");
            vME_track_dxdz_[me_unit] = setupH1D_(ibooker,"TrackDxdz");
            vME_track_dydz_[me_unit] = setupH1D_(ibooker,"TrackDydz");

std::cout << "DQMPixelCell" << "Booking Histograms in: " << folder_name << std::endl; 
            edm::LogInfo("DQMPixelCell") << "Booking Histograms in: " << folder_name << std::endl; 
        }
    }
}

/*int DQMPixelCell::matchedSimTrackIndex(edm::Handle<edm::DetSetVector<PixelDigiSimLink> >& linkHandle,
                                    edm::Handle<edm::SimTrackContainer>& simTkHandle,
                                    DetId detId,
                                    unsigned int& channel) {
  int simTrkIndx = -1;
  unsigned int simTrkId = 0;
  edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = linkHandle->find(detId);

  if (isearch == linkHandle->end())
    return simTrkIndx;

  edm::DetSet<PixelDigiSimLink> link_detset = (*linkHandle)[detId];
  // Loop over DigiSimLink in this det unit
  for (edm::DetSet<PixelDigiSimLink>::const_iterator it = link_detset.data.begin(); it != link_detset.data.end();
       it++) {
    if (channel == it->channel()) {
      simTrkId = it->SimTrackId();
      break;
    }
  }
  if (simTrkId == 0)
    return simTrkIndx;
  edm::SimTrackContainer sim_tracks = (*simTkHandle.product());
  for (unsigned int itk = 0; itk < sim_tracks.size(); itk++) {
    if (sim_tracks[itk].trackId() == simTrkId) {
      simTrkIndx = itk;
      break;
    }
  }
  return simTrkIndx;
}*/
/*void DQMPixelCell::fillClusterWidth(DigiMEs& mes, float dphi, float width) {
  for (unsigned int i = 0; i < phiValues.size(); i++) {
    float angle_min = (phiValues[i] - 0.1) * std::acos(-1.0) / 180.0;
    float angle_max = (phiValues[i] + 0.1) * std::acos(-1.0) / 180.0;
    if (std::fabs(dphi) > angle_min && std::fabs(dphi) < angle_max) {
      mes.ClusterWidths[i]->Fill(width);
      break;
    }
  }
}*/

bool DQMPixelCell::isPixelSystem_(const GeomDetUnit * dunit) const
{
    const auto & dtype = dunit->type();
    if( ! dtype.isInnerTracker() || ! dtype.isTrackerPixel() )
    {
        return false;
    }
    return true;
}

// Helper functions to setup 
int DQMPixelCell::meUnit_(bool isBarrel,int layer, int side) const
{
    // [isBarrel][layer][side]
    // [X][XXX][XX]
    return (static_cast<int>(isBarrel) << 6) | (layer << 3) | side;
}

DQMPixelCell::MonitorElement * DQMPixelCell::setupH1D_(DQMStore::IBooker& ibooker, 
        const std::string & histoname)
{
    // Config need to have exactly the same histo name
    edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
    return ibooker.book1D(histoname,histoname,
            params.getParameter<int32_t>("Nxbins"),
            params.getParameter<double>("xmin"),
            params.getParameter<double>("xmax"));
}

DQMPixelCell::MonitorElement * DQMPixelCell::setupH2D_(DQMStore::IBooker& ibooker, 
        const std::string & histoname,
        const std::pair<double,double> & xranges, 
        const std::pair<double,double> & yranges)
{
    // Config need to have exactly the same histo name
    edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
    return ibooker.book2D(histoname,histoname,
            params.getParameter<int32_t>("Nxbins"),
            xranges.first,
            xranges.second,
            params.getParameter<int32_t>("Nybins"),
            yranges.first,
            yranges.second);
}

DQMPixelCell::MonitorElement * DQMPixelCell::setupProf2D_(DQMStore::IBooker& ibooker, 
        const std::string & histoname,
        const std::pair<double,double> & xranges, 
        const std::pair<double,double> & yranges)
{
    // Config need to have exactly the same histo name
    edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
    return ibooker.bookProfile2D(histoname,histoname,
            params.getParameter<int32_t>("Nxbins"),
            xranges.first,
            xranges.second,
            params.getParameter<int32_t>("Nybins"),
            yranges.first,
            yranges.second,
            params.getParameter<double>("zmin"),
            params.getParameter<double>("zmax"));
}

const PixelDigi & DQMPixelCell::get_digi_from_channel_(int ch, 
        const edm::DetSetVector<PixelDigi>::const_iterator & itdigis)
{
    for(const auto & dh: *itdigis)
    {
        if(dh.channel() == ch)
        {
            return dh;
        }
    }
    // MAke sense not to find a digi?
    throw cms::Exception("Not found a PixelDigi") << " for the given channel: " << ch;
}

const SimTrack & DQMPixelCell::get_simtrack_from_index_(unsigned int idx, const edm::SimTrackContainer * st)
{
    if(idx >= st->size())
    {
        // The trackId is not the element of the SimTrackcontainer?
        throw cms::Exception("Problem extracting the SimTrack!") << " No element with index: " << idx;
    }

    return (*st)[idx];
}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMPixelCell);
