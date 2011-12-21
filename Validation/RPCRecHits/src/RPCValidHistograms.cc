#include "Validation/RPCRecHits/interface/RPCValidHistograms.h"

#include "TAxis.h"

void RPCValidHistograms::bookHistograms(DQMStore* dbe, const std::string subDir)
{
  if ( !dbe ) 
  {
    edm::LogError("RPCValidHistograms") << "DBE not initialized\n";
    return;
  }
  if ( booked_ )
  {
    edm::LogError("RPCValidHistograms") << "Histogram is already booked\n";
    return;
  }

  const std::string pwd = dbe->pwd();
  dbe->setCurrentFolder(subDir);

  // Book histograms
  dbe->setCurrentFolder(subDir+"/HitProperty");
  clusterSize = dbe->book1D("ClusterSize", "Cluster size;Cluster size", 11, -0.5, 10.5);
  clusterSizeBarrel = dbe->book1D("ClusterSizeBarrel", "Cluster size in Barrel;Cluster size", 11, -0.5, 10.5);
  clusterSizeEndcap = dbe->book1D("ClusterSizeEndcap", "Cluster size in Endcap;Cluster size", 11, -0.5, 10.5);

  avgClusterSize = dbe->book1D("AverageClusterSize", "Average cluster size;Average clsuter size", 11, -0.5, 10.5);
  avgClusterSizeBarrel = dbe->book1D("AverageClusterSizeBarrel", "Average cluster size in Barrel;Average clsuter size", 11, -0.5, 10.5);
  avgClusterSizeEndcap = dbe->book1D("AverageClusterSizeEndcap", "Average cluster size in Endcap;Average clsuter size", 11, -0.5, 10.5);

  nRecHitBarrel = dbe->book1D("NRecHitBarrel", "Number of RPC recHits per event in Barrel;Number of RPC hits", 25, 0, 25);
  nRecHitEndcap = dbe->book1D("NRecHitEndcap", "Number of RPC recHits per event in Endcap;Number of RPC hits", 25, 0, 25);

  nRefHitBarrel = dbe->book1D("NRefHitBarrel", "Number of reference hits per event in Barrel;Number of RPC hits", 25, 0, 25);
  nRefHitEndcap = dbe->book1D("NRefHitEndcap", "Number of reference hits per event in Endcap;Number of RPC hits", 25, 0, 25);

  nMatchHitBarrel = dbe->book1D("nMatchBarrel", "Number of matched reference hits per event in Barrel;Number of RPC hits", 25, 0, 25);
  nMatchHitEndcap = dbe->book1D("nMatchEndcap", "Number of matched reference hits per event in Endcap;Number of RPC hits", 25, 0, 25);

  // Occupancy 1D
  dbe->setCurrentFolder(subDir+"/Occupancy");
  refHitOccupancyBarrel_wheel   = dbe->book1D("RefHitOccupancyBarrel_wheel"  , "Reference Hit occupancy", 5, -2.5, 2.5);
  refHitOccupancyEndcap_disk    = dbe->book1D("RefHitOccupancyEndcap_disk"   , "Reference Hit occupancy", 7, -3.5, 3.5);
  refHitOccupancyBarrel_station = dbe->book1D("RefHitOccupancyBarrel_station", "Reference Hit occupancy", 4,  0.5, 4.5);

  recHitOccupancyBarrel_wheel   = dbe->book1D("RecHitOccupancyBarrel_wheel"  , "RecHit occupancy", 5, -2.5, 2.5);
  recHitOccupancyEndcap_disk    = dbe->book1D("RecHitOccupancyEndcap_disk"   , "RecHit occupancy", 7, -3.5, 3.5);
  recHitOccupancyBarrel_station = dbe->book1D("RecHitOccupancyBarrel_station", "RecHit occupancy", 4,  0.5, 4.5);

  matchOccupancyBarrel_wheel   = dbe->book1D("MatchOccupancyBarrel_wheel"  , "Matched hit occupancy", 5, -2.5, 2.5);
  matchOccupancyEndcap_disk    = dbe->book1D("MatchOccupancyEndcap_disk"   , "Matched hit occupancy", 7, -3.5, 3.5);
  matchOccupancyBarrel_station = dbe->book1D("MatchOccupancyBarrel_station", "Matched hit occupancy", 4,  0.5, 4.5);

  umOccupancyBarrel_wheel   = dbe->book1D("UmOccupancyBarrel_wheel"  , "Un-matched hit occupancy", 5, -2.5, 2.5);
  umOccupancyEndcap_disk    = dbe->book1D("UmOccupancyEndcap_disk"   , "Un-matched hit occupancy", 7, -3.5, 3.5);
  umOccupancyBarrel_station = dbe->book1D("UmOccupancyBarrel_station", "Un-matched hit occupancy", 4,  0.5, 4.5);

  // Occupancy 2D
  refHitOccupancyBarrel_wheel_station = dbe->book2D("RefHitOccupancyBarrel_wheel_station", "Reference hit occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  refHitOccupancyEndcap_disk_ring     = dbe->book2D("RefHitOccupancyEndcap_disk_ring"    , "Reference hit occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);

  recHitOccupancyBarrel_wheel_station = dbe->book2D("RecHitOccupancyBarrel_wheel_station", "RecHit occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  recHitOccupancyEndcap_disk_ring     = dbe->book2D("RecHitOccupancyEndcap_disk_ring"    , "RecHit occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);

  matchOccupancyBarrel_wheel_station = dbe->book2D("MatchOccupancyBarrel_wheel_station", "Matched hit occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  matchOccupancyEndcap_disk_ring     = dbe->book2D("MatchOccupancyEndcap_disk_ring"    , "Matched hit occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);

  umOccupancyBarrel_wheel_station = dbe->book2D("UmOccupancyBarrel_wheel_station", "Un-matched hit occupancy", 5, -2.5, 2.5, 4, 0.5, 4.5);
  umOccupancyEndcap_disk_ring     = dbe->book2D("UmOccupancyEndcap_disk_ring"    , "Un-matched hit occupancy", 7, -3.5, 3.5, 4, 0.5, 4.5);

  // Residuals
  dbe->setCurrentFolder(subDir+"/Residual");
  resBarrel = dbe->book1D("ResBarrel", "Global Residuals for Barrel;Residual [cm]"  , 100, -8, 8);
  resEndcap = dbe->book1D("ResEndcap", "Global Residuals for Endcap;Residual [cm]"   , 100, -8, 8);

  res_wheel_res   = dbe->book2D("Res_wheel_res"  , "Residuals vs Wheel;;Residual [cm]", 5, -2.5, 2.5, 50, -8, 8);
  res_disk_res    = dbe->book2D("Res_disk_res"   , "Residuals vs Disk;;Residual [cm]", 7, -3.5, 3.5, 50, -8, 8);
  res_station_res = dbe->book2D("Res_station_res", "Redisuals vs Station;;Residual [cm]", 4, 0.5, 4.5, 50, -8, 8);
  res_ring_res    = dbe->book2D("Res_ring_res"   , "Redisuals vs Ring;;Residual [cm]", 4, 0.5, 4.5, 50, -8, 8);

  // Pulls
  pullBarrel = dbe->book1D("PullBarrel", "Global Pull for Barrel;Pull", 100, -3, 3);
  pullEndcap = dbe->book1D("PullEndcap", "Global Pull for Endcap;Pull", 100, -3, 3);

  pull_wheel_pull   = dbe->book2D("Pull_wheel_pull"  , "Pull vs Wheel;;Pull"  , 5, -2.5, 2.5, 50, -3, 3);
  pull_disk_pull    = dbe->book2D("Pull_disk_pull"   , "Pull vs Disk;;Pull"   , 7, -3.5, 3.5, 50, -3, 3);
  pull_station_pull = dbe->book2D("Pull_station_pull", "Pull vs Station;;Pull", 4,  0.5, 4.5, 50, -3, 3);
  pull_ring_pull    = dbe->book2D("Pull_ring_pull"   , "Pull vs Ring;;Pull"   , 4,  0.5, 4.5, 50, -3, 3);

  // Set plot options
  refHitOccupancyBarrel_wheel_station->getTH2F()->SetOption("COLZ");
  refHitOccupancyEndcap_disk_ring    ->getTH2F()->SetOption("COLZ");
  recHitOccupancyBarrel_wheel_station->getTH2F()->SetOption("COLZ");
  recHitOccupancyEndcap_disk_ring    ->getTH2F()->SetOption("COLZ");
  matchOccupancyBarrel_wheel_station ->getTH2F()->SetOption("COLZ");
  matchOccupancyEndcap_disk_ring     ->getTH2F()->SetOption("COLZ");
  umOccupancyBarrel_wheel_station    ->getTH2F()->SetOption("COLZ");
  umOccupancyEndcap_disk_ring        ->getTH2F()->SetOption("COLZ");

  res_wheel_res  ->getTH2F()->SetOption("COLZ");
  res_disk_res   ->getTH2F()->SetOption("COLZ");
  res_station_res->getTH2F()->SetOption("COLZ");
  res_ring_res   ->getTH2F()->SetOption("COLZ");

  pull_wheel_pull  ->getTH2F()->SetOption("COLZ");
  pull_disk_pull   ->getTH2F()->SetOption("COLZ");
  pull_station_pull->getTH2F()->SetOption("COLZ");
  pull_ring_pull   ->getTH2F()->SetOption("COLZ");

  // Set bin labels
  for ( int i=1; i<=5; ++i )
  {
    TString binLabel = Form("Wheel %d", i-3);

    refHitOccupancyBarrel_wheel->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    recHitOccupancyBarrel_wheel->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    matchOccupancyBarrel_wheel ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    umOccupancyBarrel_wheel    ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);

    refHitOccupancyBarrel_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    recHitOccupancyBarrel_wheel_station->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    matchOccupancyBarrel_wheel_station ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    umOccupancyBarrel_wheel_station    ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);

    res_wheel_res  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    pull_wheel_pull->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=7; ++i )
  {
    TString binLabel = Form("Disk %d", i-4);

    refHitOccupancyEndcap_disk->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    recHitOccupancyEndcap_disk->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    matchOccupancyEndcap_disk ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    umOccupancyEndcap_disk    ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);

    refHitOccupancyEndcap_disk_ring->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    recHitOccupancyEndcap_disk_ring->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    matchOccupancyEndcap_disk_ring ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    umOccupancyEndcap_disk_ring    ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);

    res_disk_res  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    pull_disk_pull->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Station %d", i);

    refHitOccupancyBarrel_station->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    recHitOccupancyBarrel_station->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    matchOccupancyBarrel_station ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);
    umOccupancyBarrel_station    ->getTH1F()->GetXaxis()->SetBinLabel(i, binLabel);

    refHitOccupancyBarrel_wheel_station->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    recHitOccupancyBarrel_wheel_station->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    matchOccupancyBarrel_wheel_station ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    umOccupancyBarrel_wheel_station    ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);

    res_station_res  ->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
    pull_station_pull->getTH2F()->GetXaxis()->SetBinLabel(i, binLabel);
  }

  for ( int i=1; i<=4; ++i )
  {
    TString binLabel = Form("Ring %d", i);

    refHitOccupancyEndcap_disk_ring->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    recHitOccupancyEndcap_disk_ring->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    matchOccupancyEndcap_disk_ring ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
    umOccupancyEndcap_disk_ring    ->getTH2F()->GetYaxis()->SetBinLabel(i, binLabel);
  }

  dbe->setCurrentFolder(pwd);
  booked_ = true;
}

