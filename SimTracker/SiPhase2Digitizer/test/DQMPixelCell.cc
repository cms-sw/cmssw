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

// XXX - Be careful the relative position
#include "DQMPixelCell.h"


const double unit_um = 1e-4; // [cm]

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
        //const edm::PSimHitContainer * simhits = simHitHandle.product();
        simhits.push_back( simHitHandle.product() );
    }

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
        }

        // Fill  per detector histograms
        for(const auto & st_ch: stracks_channels)
        {
            // Cluster size on the detector unit, that's from the simdigi,
            //vME_clsize1D_[me_unit]->Fill(st_ch.second.size());

//std::cout << " -- Current trackId [" << st_ch.first << "] Created Digis: " << st_ch.second.size();
            // FIXME :: What happens when the same track (probably secondaries which are
            // associated to the mother track id) creates more than on digi?? So far 
            // 1 track creates 1 cluster, but what about secondaries?

            // Get the total charge for this cluster size
            // and obtain the center of the cluster using a charge-weighted mean
            int cluster_tot = 0;
            int cluster_size = 0;
            std::pair<double,double> cluster_position({0.0,0.0});
            for(const auto & ch: st_ch.second)
            {
                const PixelDigi & current_digi = get_digi_from_channel_(ch,it_digis);
                cluster_tot += current_digi.adc();
                cluster_position.first  += current_digi.adc()*current_digi.row();
                cluster_position.second += current_digi.adc()*current_digi.column();
                ++cluster_size;
            }
            vME_clsize1D_[me_unit]->Fill(cluster_size);
            vME_charge1D_[me_unit]->Fill(cluster_tot);

            // mean weighted 
            cluster_position.first  /= double(cluster_tot);
            cluster_position.second /= double(cluster_tot);
//std::cout << " ToT: " << cluster_tot << " <row>:" << cluster_position.first << " <col>:" << cluster_position.second ; 
            
            // Get The Sim tracks (once per simtrack)
            // Obtain the dz/dx y dz/dy from the track?
            const SimTrack * current_simtrack = get_simtrack_from_id_(st_ch.first,simtracks);
            if(current_simtrack == nullptr)
            {
//std::cout << std::endl;
                continue;
            }
            // Convert the momentum into the local frame in order to evaluate the incident
            // angle, but first it needs to be converted into a GlobalVector
            const GlobalVector cst_momentum(current_simtrack->momentum().x(),
                    current_simtrack->momentum().y(),
                    current_simtrack->momentum().z());
            const LocalVector cst_m_local(dunit->surface().toLocal(cst_momentum));
            // don't care about the entering direction 
            vME_track_dxdzAngle_[me_unit]->Fill(std::atan2(cst_m_local.x(),std::fabs(cst_m_local.z())));
            vME_track_dydzAngle_[me_unit]->Fill(std::atan2(cst_m_local.y(),std::fabs(cst_m_local.z())));

            // See where the track enters into the tracker and fill its histos
            const GlobalPoint cst_position(current_simtrack->trackerSurfacePosition().x(),
                    current_simtrack->trackerSurfacePosition().y(),
                    current_simtrack->trackerSurfacePosition().z());

            vME_track_XYMap_->Fill(cst_position.x(),cst_position.y());
            vME_track_RZMap_->Fill(cst_position.z(),std::hypot(cst_position.x(),cst_position.y()));

            // -- Get the set of simulated hits from this trackid
            const edm::PSimHitContainer current_psimhits = get_simhits_from_trackid_(st_ch.first,detId.rawId(),simhits);
            // Use the SimHits as the MC-truth to evaluate the digis
            //const auto current_pixel(PixelDigi::channelToPixel(ch));
//std::cout << " SimHits.size: " << current_psimhits.size() ;

            // FIXME> The same than a cluster: 1 track -> 1 cluster
            std::pair<double,double> sim_cluster_position({0.0,0.0}); 
            double eloss_total = 0.0; 
            for(const auto & ps: current_psimhits)
            {
                // Compensate the row and column center
                auto mp = tkDetUnit->specificTopology().measurementPosition(ps.localPosition())-MeasurementPoint(0.5,0.5);

                sim_cluster_position.first  += mp.x()*ps.energyLoss();
                sim_cluster_position.second += mp.y()*ps.energyLoss();
                eloss_total += ps.energyLoss();
//std::cout << " (detId.rawId: " << detId.rawId() << ")[- DetUnitId: " << ps.detUnitId() << " Entry Point: " << ps.entryPoint() << " Exit Point: " << ps.exitPoint() 
//        << " Local Position row:" << mp.x() << " col: " << mp.y() << "  p_entry=" << ps.momentumAtEntry()
//        << " Time of flight: " << ps.timeOfFlight() << "-]"; 
            }
            sim_cluster_position.first  /= eloss_total;
            sim_cluster_position.second /= eloss_total;
//std::cout << " ---> In summary: Eloss total:" << eloss_total 
//<< " <row>: " << sim_cluster_position.first << " <col>:" << sim_cluster_position.second <<  std::endl;

            // Efficiency --> It was found a cluster of digis?
            const bool is_digi_present = (cluster_size > 0);

            // Get topology info of the module sensor
            //-const int n_rows = tkDetUnit->specificTopology().nrows();
            //-const int n_cols = tkDetUnit->specificTopology().ncolumns();
            const auto pitch = tkDetUnit->specificTopology().pitch();
            // Residuals, convert them to longitud units (so far, in units of row, col)
            const double dx_um = (sim_cluster_position.first-cluster_position.first)*pitch.first/unit_um;
            const double dy_um = (sim_cluster_position.second-cluster_position.second)*pitch.second/unit_um;
            if(is_digi_present)
            {
                vME_dx1D_[me_unit]->Fill(dx_um);
                vME_dy1D_[me_unit]->Fill(dy_um);
            }
            // Histograms per cell
            for(unsigned int i =1; i < vME_position_cell_[me_unit].size(); ++i)
            {
                // Convert to i-cell 
                const std::pair<double,double> icell_simhit_cluster = pixel_cell_transformation_(sim_cluster_position,i,pitch);
                // Efficiency? --> Any track id do not have a digi? Is there any cluster.size == 0?
                vME_eff_cell_[me_unit][i]->Fill(icell_simhit_cluster.first/unit_um,icell_simhit_cluster.second/unit_um,is_digi_present);
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
            

            // And create the histos
            // Per detector unit histos
            vME_clsize1D_[me_unit] = setupH1D_(ibooker,"ClusterSize1D","MC-truth cluster size;Cluster size;N_{clusters}");
            vME_charge1D_[me_unit] = setupH1D_(ibooker,"Charge1D","MC-truth charge;Cluster charge [ToT];N_{clusters}");
            vME_track_dxdzAngle_[me_unit] = setupH1D_(ibooker,"TrackAngleDxdz",
                    "Angle between the track-momentum and detector surface (X-plane);#pi/2-#theta_{x} [rad];N_{tracks}");
            vME_track_dydzAngle_[me_unit] = setupH1D_(ibooker,"TrackAngleDydz",
                    "Angle between the track-momentum and detector surface (Y-plane);#pi/2-#theta_{y} [rad];N_{tracks}");
            vME_dx1D_[me_unit] = setupH1D_(ibooker,"Dx1D",
                    "MC-truth residuals;x^{cluster}_{simhit}-x^{cluster}_{digi} [#mum];N_{digi clusters}");
            vME_dy1D_[me_unit] = setupH1D_(ibooker,"Dy1D",
                    "MC-truth residuals;y^{cluster}_{simhit}-y^{cluster}_{digi} [#mum];N_{digi clusters}");

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
        const std::pair<double,double> & pos,
        unsigned int icell,
        const std::pair<double,double> & pitch)
{
    // XXX - If icell == 0 -> get the nrow and ncol ?
    const double xcell = std::fmod(pos.first,icell)*pitch.first;
    const double ycell = std::fmod(pos.second,icell)*pitch.second;

    return std::pair<double,double>({xcell,ycell});
}

//define this as a plug-in
DEFINE_FWK_MODULE(DQMPixelCell);
