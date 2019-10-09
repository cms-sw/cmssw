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
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

// CMSSW Data formats
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"

// system
#include <algorithm>
#include <set>

// XXX - Be careful the relative position
#include "DQMPixelCell.h"


const double unit_um = 1e-4; // [cm]
const double unit_mm = 1e-1; // [cm]

using Phase2TrackerGeomDetUnit = PixelGeomDetUnit;


DQMPixelCell::DQMPixelCell(const edm::ParameterSet& iConfig) : 
    config_(iConfig),
    geomType_(iConfig.getParameter<std::string>("GeometryType")),
    //phiValues(iConfig.getParameter<std::vector<double> >("PhiAngles")),
    digiToken_(consumes<edm::DetSetVector<PixelDigi> >(
                iConfig.getParameter<edm::InputTag>("PixelDigiSource"))),
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

// -- DQM Begin Run
//
void DQMPixelCell::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) 
{
    edm::LogInfo("DQMPixelCell") << "Initialize DQMPixelCell ";
}
//
// -- Analyze
//
void DQMPixelCell::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // First clear the memoizer
    m_tId_det_simhits_.clear();

    // Get digis and the simhit links and the simhits
    edm::Handle<edm::DetSetVector<PixelDigiSimLink> > digiSimLinkHandle;
    iEvent.getByToken(digiSimLinkToken_, digiSimLinkHandle);
    const edm::DetSetVector<PixelDigiSimLink> * simdigis = digiSimLinkHandle.product();
    
    edm::Handle<edm::DetSetVector<PixelDigi> > digiHandle;
    iEvent.getByToken(digiToken_, digiHandle);
    const edm::DetSetVector<PixelDigi> * digis = digiHandle.product();
    
    // Vector of simHits
    // XXX : NOt like that, just an example
    std::vector<const edm::PSimHitContainer*> simhits;
    simhits.reserve(simHitTokens_.size());
    for(const auto & sh_token: simHitTokens_)
    {
        edm::Handle<edm::PSimHitContainer> simHitHandle;
        iEvent.getByToken(sh_token, simHitHandle);
        if(! simHitHandle.isValid())
        {
            continue;
        }
        simhits.push_back( simHitHandle.product() );
    }

    // Get SimTrack
    /*edm::Handle<edm::SimTrackContainer> simTrackHandle;
    iEvent.getByToken(simTrackToken_, simTrackHandle);
    const edm::SimTrackContainer * simtracks = simTrackHandle.product();*/
    
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
            // Create/update the list of channels created by this 
            // simtrack: remember primaries and secondaries share the same id
            stracks_channels[dhsim.SimTrackId()].insert(current_channel);
        }

        // Fill  per detector histograms
        for(const auto & st_ch: stracks_channels)
        {
            // -- Get the set of simulated hits from this trackid. 
            // -- Each Particle SimHit (PSimHit) is defining a particle passing through the detector unit
            const edm::PSimHitContainer current_psimhits = get_simhits_from_trackid_(st_ch.first,detId.rawId(),simhits);
            //const auto current_pixel(PixelDigi::channelToPixel(ch));
            
            // -- FIXME: Secondaries in the same pixel cell --> should be absorbed
            for(const auto & ps: current_psimhits)
            {
                // Fill some sim histograms
                const GlobalPoint tk_ep_gbl(dunit->surface().toGlobal(ps.entryPoint()));
                vME_track_XYMap_->Fill(tk_ep_gbl.x(),tk_ep_gbl.y());
                vME_track_RZMap_->Fill(tk_ep_gbl.z(),std::hypot(tk_ep_gbl.x(),tk_ep_gbl.y()));
                vME_track_dxdzAngle_[me_unit]->Fill(ps.thetaAtEntry());
                vME_track_dydzAngle_[me_unit]->Fill(ps.phiAtEntry());
                
                // Obtain the detected position of the sim particle: 
                // the middle point between the entry and the exit
                const auto psh_pos = tkDetUnit->specificTopology().measurementPosition(ps.localPosition()); 

                // Build the digi MC-truth clusters by matching each Particle 
                // sim hit position pixel cell. The matching condition:
                //   - a digi is created by the i-PSimHit if PsimHit_{pixel}+-1

                // Get the total charge for this cluster size
                // and obtain the center of the cluster using a charge-weighted mean
                int cluster_tot = 0;
                int cluster_size = 0;
                std::pair<std::set<int>,std::set<int>> cluster_size_xy;
                std::pair<double,double> cluster_position({0.0,0.0});
                std::set<int> used_channel;
                for(const auto & ch: st_ch.second)
                {
                    // Not re-using the digi
                    if(used_channel.find(ch) != used_channel.end())
                    {
                        continue;
                    }
                    // Digi was created by the current psimhit?
                    // Accepting +-1 pixel -- XXX: Actually the entryPoint-exitPoint 
                    // could provide the extension of the cluster
                    if( ! channel_iluminated_by_(psh_pos,ch,2.0) )
                    {
                        continue;
                    }
                    const PixelDigi & current_digi = get_digi_from_channel_(ch,it_digis);
                    used_channel.insert(ch);
                    // Fill the digi histograms
                    vME_digi_charge1D_[me_unit]->Fill(current_digi.adc());
                    // Fill maps: get the position in the sensor local frame to convert into global
                    const LocalPoint digi_local_pos(tkDetUnit->specificTopology().localPosition(
                                MeasurementPoint(current_digi.row(),current_digi.column())
                                ));
                    const GlobalPoint digi_global_pos(dunit->surface().toGlobal(digi_local_pos));
                    vME_digi_XYMap_->Fill(digi_global_pos.x(),digi_global_pos.y());
                    vME_digi_RZMap_->Fill(digi_global_pos.z(),std::hypot(digi_global_pos.x(),digi_global_pos.y()));
                    // Create the MC-cluster
                    cluster_tot += current_digi.adc();
                    // Use the center of the pixel 
                    cluster_position.first  += current_digi.adc()*(current_digi.row()+0.5);
                    cluster_position.second += current_digi.adc()*(current_digi.column()+0.5);
                    // Size
                    cluster_size_xy.first.insert(current_digi.row());
                    cluster_size_xy.second.insert(current_digi.column());
                    ++cluster_size;
                }
                vME_clsize1D_[me_unit]->Fill(cluster_size);
                vME_clsize1Dx_[me_unit]->Fill(cluster_size_xy.first.size());
                vME_clsize1Dy_[me_unit]->Fill(cluster_size_xy.second.size());
                vME_charge1D_[me_unit]->Fill(cluster_tot);
                
                // mean weighted 
                cluster_position.first  /= double(cluster_tot);
                cluster_position.second /= double(cluster_tot);
                
                // -- XXX Be careful, secondaries with already used the digis
                //        are going the be lost (then lost on efficiency)
                // Efficiency --> It was found a cluster of digis?
                const bool is_digi_present = (cluster_size > 0);

                // Get topology info of the module sensor
                //-const int n_rows = tkDetUnit->specificTopology().nrows();
                //-const int n_cols = tkDetUnit->specificTopology().ncolumns();
                const auto pitch = tkDetUnit->specificTopology().pitch();
                // Residuals, convert them to longitud units (so far, in units of row, col)
                const double dx_um = (psh_pos.x()-cluster_position.first)*pitch.first/unit_um;
                const double dy_um = (psh_pos.y()-cluster_position.second)*pitch.second/unit_um;
                if(is_digi_present)
                {
                    vME_dx1D_[me_unit]->Fill(dx_um);
                    vME_dy1D_[me_unit]->Fill(dy_um);
                }
                // Histograms per cell
                for(unsigned int i =1; i < vME_position_cell_[me_unit].size(); ++i)
                {
                    // Convert to i-cell 
                    const std::pair<double,double> icell_simhit_cluster = pixel_cell_transformation_(psh_pos,i,pitch);
                    // Efficiency? --> Any track id do not have a digi? Is there any cluster.size == 0?
                    vME_eff_cell_[me_unit][i]->Fill(icell_simhit_cluster.first/unit_um,icell_simhit_cluster.second/unit_um,double(is_digi_present));
                    if(is_digi_present)
                    {
                        // Convert to the i-cell
                        //const std::pair<double,double> icell_digi_cluster   = pixel_cell_transformation_(cluster_position,i,pitch);
                        // Position
                        vME_position_cell_[me_unit][i]->Fill(icell_simhit_cluster.first/unit_um,icell_simhit_cluster.second/unit_um);
                        // Residuals
                        vME_dx_cell_[me_unit][i]->Fill(icell_simhit_cluster.first/unit_um,icell_simhit_cluster.second/unit_um,dx_um);
                        vME_dy_cell_[me_unit][i]->Fill(icell_simhit_cluster.first/unit_um,icell_simhit_cluster.second/unit_um,dy_um);
                        // Charge
                        vME_charge_cell_[me_unit][i]->Fill(icell_simhit_cluster.first/unit_um,icell_simhit_cluster.second/unit_um,cluster_tot);
                        // Cluster size
                        vME_clsize_cell_[me_unit][i]->Fill(icell_simhit_cluster.first/unit_um,icell_simhit_cluster.second/unit_um,cluster_size);
                    }
                }
            }
        }
    }
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

    // Histograms independent of the subdetector units
    ibooker.cd();
    ibooker.setCurrentFolder(top_folder);
    vME_track_XYMap_ = setupH2D_(ibooker,"TrackXY",
            "Track entering position in the tracker system;x [cm];y [cm];N_{tracksId}");
    vME_track_RZMap_ = setupH2D_(ibooker,"TrackRZ",
            "Track entering position in the tracker system;z [cm];r [cm];N_{tracksId}");
    vME_digi_XYMap_ = setupH2D_(ibooker,"DigiXY",
            "Digi position;x [cm];y [cm];N_{digi}");
    vME_digi_RZMap_ = setupH2D_(ibooker,"DigiRZ",
            "Digi position;z [cm];r [cm];N_{digi}");

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
            folder_name += "/Layer"+std::to_string(layer);

            if(topo->side(detId))
            {
                folder_name += "/Side"+std::to_string(topo->side(detId));
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
            

            // And create the histos
            // Per detector unit histos
            vME_clsize1D_[me_unit] = setupH1D_(ibooker,"ClusterSize1D","MC-truth cluster size;Cluster size;N_{clusters}");
            vME_clsize1Dx_[me_unit] = setupH1D_(ibooker,"ClusterSize1Dx","MC-truth cluster size in X;Cluster size;N_{clusters}");
            vME_clsize1Dy_[me_unit] = setupH1D_(ibooker,"ClusterSize1Dy","MC-truth cluster size in Y;Cluster size;N_{clusters}");
            vME_charge1D_[me_unit] = setupH1D_(ibooker,"Charge1D","MC-truth charge;Cluster charge [ToT];N_{clusters}");
            vME_track_dxdzAngle_[me_unit] = setupH1D_(ibooker,"TrackAngleDxdz",
                    "Angle between the track-momentum and detector surface (X-plane);#pi/2-#theta_{x} [rad];N_{tracks}");
            vME_track_dydzAngle_[me_unit] = setupH1D_(ibooker,"TrackAngleDydz",
                    "Angle between the track-momentum and detector surface (Y-plane);#pi/2-#theta_{y} [rad];N_{tracks}");
            vME_dx1D_[me_unit] = setupH1D_(ibooker,"Dx1D",
                    "MC-truth residuals;x^{cluster}_{simhit}-x^{cluster}_{digi} [#mum];N_{digi clusters}");
            vME_dy1D_[me_unit] = setupH1D_(ibooker,"Dy1D",
                    "MC-truth residuals;y^{cluster}_{simhit}-y^{cluster}_{digi} [#mum];N_{digi clusters}");
            vME_digi_charge1D_[me_unit] = setupH1D_(ibooker,"DigiCharge1D","Digi charge;digi charge [ToT];N_{digi}");

            // The histos per cell
            // Prepare the ranges: 0- whole sensor, 1- cell 1x1, 2-cell 2x2,
            const std::vector<std::pair<double,double>> xranges = {
                std::make_pair<double,double>(0,nrows-1),
                std::make_pair<double,double>(0,pitch.first/unit_um),
                std::make_pair<double,double>(0,2.0*pitch.first/unit_um) };
            const std::vector<std::pair<double,double>> yranges = {
                std::make_pair<double,double>(0,ncols-1),
                std::make_pair<double,double>(0,pitch.second/unit_um),
                std::make_pair<double,double>(0,2.0*pitch.second/unit_um) };
            for(unsigned int i = 0; i < xranges.size(); ++i)
            {
                const std::string cell("Cell "+std::to_string(i)+"x"+std::to_string(i)+": ");
                vME_position_cell_[me_unit].push_back(setupH2D_(ibooker,
                            "Position_"+std::to_string(i),
                            cell+"Digi cluster center (charge-weighted) position;x [#mum];y [#mum];N_{clusters}",
                            xranges[i],
                            yranges[i]));
                vME_eff_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "Efficiency_"+std::to_string(i),
                            cell+"MC-truth efficiency;x [#mum];y [#mum];<#varepsilon>",
                            xranges[i],
                            yranges[i]));
                vME_clsize_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "ClusterSize_"+std::to_string(i),
                            cell+"MC-truth cluster size;x [#mum];y [#mum];<Cluster size>",
                            xranges[i],
                            yranges[i]));
                vME_charge_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "Charge_"+std::to_string(i),
                            cell+"MC-truth charge;x [#mum];y [#mum];<Cluster size>",
                            xranges[i],
                            yranges[i]));
                vME_dx_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "Dx_"+std::to_string(i),
                            cell+"MC-truth residuals;x [#mum];y [#mum];<#Deltax [#mum]>",
                            xranges[i],
                            yranges[i]));
                vME_dy_cell_[me_unit].push_back(setupProf2D_(ibooker,
                            "Dy_"+std::to_string(i),
                            cell+"MC-truth residuals;x [#mum];y [#mum];<#Deltay [#mum]>",
                            xranges[i],
                            yranges[i]));
            }

std::cout << "DQMPixelCell" << "Booking Histograms in: " << folder_name << std::endl; 
            edm::LogInfo("DQMPixelCell") << "Booking Histograms in: " << folder_name << std::endl; 
        }
    }
}


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
        const std::string & histoname,
        const std::string & title)
{
    // Config need to have exactly the same histo name
    edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
    return ibooker.book1D(histoname,title,
            params.getParameter<int32_t>("Nxbins"),
            params.getParameter<double>("xmin"),
            params.getParameter<double>("xmax"));
}

DQMPixelCell::MonitorElement * DQMPixelCell::setupH2D_(DQMStore::IBooker& ibooker, 
        const std::string & histoname,
        const std::string & title)
{
    // Config need to have exactly the same histo name
    edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
    return ibooker.book2D(histoname,title,
            params.getParameter<int32_t>("Nxbins"),
            params.getParameter<double>("xmin"),
            params.getParameter<double>("xmax"),
            params.getParameter<int32_t>("Nybins"),
            params.getParameter<double>("ymin"),
            params.getParameter<double>("ymax"));
}

DQMPixelCell::MonitorElement * DQMPixelCell::setupH2D_(DQMStore::IBooker& ibooker, 
        const std::string & histoname,
        const std::string & title,
        const std::pair<double,double> & xranges, 
        const std::pair<double,double> & yranges)
{
    // Config need to have exactly the same histo name
    edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
    return ibooker.book2D(histoname,title,
            params.getParameter<int32_t>("Nxbins"),
            xranges.first,
            xranges.second,
            params.getParameter<int32_t>("Nybins"),
            yranges.first,
            yranges.second);
}

DQMPixelCell::MonitorElement * DQMPixelCell::setupProf2D_(DQMStore::IBooker& ibooker, 
        const std::string & histoname,
        const std::string & title,
        const std::pair<double,double> & xranges, 
        const std::pair<double,double> & yranges)
{
    // Config need to have exactly the same histo name
    edm::ParameterSet params = config_.getParameter<edm::ParameterSet>(histoname);
    return ibooker.bookProfile2D(histoname,title,
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

const SimTrack * DQMPixelCell::get_simtrack_from_id_(unsigned int idx, const edm::SimTrackContainer * stc)
{
    for(const auto & st: *stc)
    {
        if(st.trackId() == idx)
        {
            return &st;
        }
    }
    // Any simtrack correspond to this trackid index
    //edm::LogWarning("DQMPixelCell::get_simtrack_from_id_")
    edm::LogInfo("DQMPixelCell::get_simtrack_from_id_")
        << "Not found any SimTrack with trackId: " << idx;
    return nullptr;
}

const edm::PSimHitContainer DQMPixelCell::get_simhits_from_trackid_(
        unsigned int tid, 
        unsigned int detid_raw,
        const std::vector<const edm::PSimHitContainer*> & psimhits)
{
    // It was already found?
    if(m_tId_det_simhits_.find(tid) != m_tId_det_simhits_.end() )
    {
        // Note that if there were no simhits in detid_raw, now it
        // is creating an empty std::vector<const edm::PSimHitContainer*>
        return m_tId_det_simhits_[tid][detid_raw];
    }

    // Otherwise, 
    // Create the new map for the track
    m_tId_det_simhits_[tid] = std::map<unsigned int,edm::PSimHitContainer>();
    // and search for it, all the PsimHit found in all detectors 
    // are going to be seeked once, therefore memoizing already
    for(const auto * sh_c: psimhits)
    {
        for(const auto & sh: *sh_c)
        {
            if(sh.trackId() == tid)
            {
                m_tId_det_simhits_[tid][sh.detUnitId()].push_back(sh);
            }
        }
    }
    // And returning what was requested
    return m_tId_det_simhits_[tid][detid_raw];
}

const std::pair<double,double> DQMPixelCell::pixel_cell_transformation_(
        const MeasurementPoint & pos,
        unsigned int icell,
        const std::pair<double,double> & pitch)
{
    return pixel_cell_transformation_(std::pair<double,double>({pos.x(),pos.y()}),icell,pitch);
}


const std::pair<double,double> DQMPixelCell::pixel_cell_transformation_(
        const std::pair<double,double> & pos,
        unsigned int icell,
        const std::pair<double,double> & pitch)
{
    // XXX - If icell == 0 -> get the nrow and ncol ?
    const double xcell = std::fmod(pos.first,icell)*pitch.first;
    const double ycell = std::fmod(pos.second,icell)*pitch.second;

    return std::pair<double,double>({xcell,ycell});
}

bool DQMPixelCell::channel_iluminated_by_(const MeasurementPoint & localpos,int channel, double tolerance) const
{
    const auto pos_channel(PixelDigi::channelToPixel(channel));
    if( std::fabs(localpos.x()-pos_channel.first) <= tolerance 
            && std::fabs(localpos.y()-pos_channel.second) <= tolerance )
    {
        return true;
    }
    return false;
}


//define this as a plug-in
DEFINE_FWK_MODULE(DQMPixelCell);
