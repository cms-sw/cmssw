#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
#include <iomanip>
GEMStripDigiValidation::GEMStripDigiValidation(DQMStore* dbe,
                                               edm::EDGetToken& stripToken, const edm::ParameterSet& pbInfo)
:  GEMBaseValidation(dbe, stripToken, pbInfo)
{}

void GEMStripDigiValidation::bookHisto(const GEMGeometry* geom) { 
  theGEMGeometry = geom;  


  int nregions = theGEMGeometry->regions().size();
  int nstations = theGEMGeometry->regions()[0]->stations().size(); 
  int nstripsGE11  = theGEMGeometry->regions()[0]->stations()[0]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->nstrips();
  int nstripsGE21 = 0;
  
  if ( nstations > 1 ) {
    nstripsGE21  = theGEMGeometry->regions()[0]->stations()[1]->superChambers()[0]->chambers()[0]->etaPartitions()[0]->nstrips();
  }
  else LogDebug("GEMStripDIGIValidation")<<"Info : Only 1 station is existed.\n";


  int nstrips = 0;

  for( int region_num = 0 ; region_num <nregions ; region_num++ ) {
    for( int layer_num = 0 ; layer_num < 2 ; layer_num++) {
      for( int station_num = 0 ; station_num < nstations ; station_num++) {
        if ( station_num == 0 ) nstrips = nstripsGE11;
        else nstrips = nstripsGE21;
        std::string name_prefix = std::string("_r")+regionLabel[region_num]+"_st"+stationLabel[station_num] + "_l"+layerLabel[layer_num];
        std::string label_prefix = "region"+regionLabel[region_num]+" station "+stationLabel[station_num] +" layer "+layerLabel[layer_num];
        theStrip_phistrip[region_num][station_num][layer_num] = dbe_->book2D( ("strip_dg_phistrip"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+"; phi [rad];strip number").c_str(), 280, -TMath::Pi(), TMath::Pi(), nstrips/2,0,nstrips);
        theStrip[region_num][station_num][layer_num] = dbe_->book1D( ("strip_dg"+name_prefix).c_str(), ("Digi occupancy per stip number: "+label_prefix+";strip number; entries").c_str(), nstrips,0.5,nstrips+0.5);
        theStrip_bx[region_num][station_num][layer_num] = dbe_->book1D( ("strip_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5);
        theStrip_zr[region_num][station_num][layer_num] = BookHistZR("strip_dg","Strip Digi",region_num,station_num,layer_num);
        theStrip_xy[region_num][station_num][layer_num] = BookHistXY("strip_dg","Strip Digi",region_num,station_num,layer_num);
      }
    }
  }
  // All chamber XY (cm) plots
  //auto& chamber = theGEMGeometry
  theSpecific_phiz[0] = dbe_->book2D("sp_strip_re-1_st1_ch1_2_roll1","sp_strip_re-1_st1_ch1_2_roll1",500, 4.5,5.5,100,-575,-560);
  theSpecific_phiz[1] = dbe_->book2D("sp_strip_re-1_st1_ch1_2_roll4","sp_strip_re-1_st1_ch1_2_roll4",500, 4.5,5.5,100,-575,-560);
  theSpecific_phiz[2] = dbe_->book2D("sp_strip_re-1_st1_ch1_2_roll8","sp_strip_re-1_st1_ch1_2_roll8",500, 4.5,5.5,100,-575,-560);
  theSpecific_phiz[3] = dbe_->book2D("sp_strip_re-1_st1_ch1_2","sp_strip_re-1_st1_ch1_2",500, 4.5,5.5,100,-575,-560);
  for( auto& region : theGEMGeometry->regions() ) 
  for( auto& station : region->stations() ) { 
    Short_t nre=region->region();
    Short_t nst=station->station();
    TString name_suffix = TString::Format("r%d_st%s",nre, stationLabel[nst-1].c_str());
    TString title_suffix= TString::Format("Region %d Station %s",nre,stationLabel[nst-1].c_str());

    TString ss1 = TString::Format("deltaPhi_%s",name_suffix.Data() );
    TString st_title = ss1+";delta Phi(#Delta#phi);Entries";
    MonitorElement* st_temp;

    if ( nst == 1 ) st_temp = dbe_->book1D( ss1.Data(),st_title.Data(),2000.,8.5,12.5);
    if ( nst == 2 || station->station() == 3 ) st_temp = dbe_->book1D(ss1.Data(),st_title.Data(),2000.,18.5,22.5);
    theStrip_st_dphi.insert( std::map< UInt_t , MonitorElement*>::value_type(ss1.Hash(), st_temp ));
    //std::cout<< ss1.Data()<<std::endl;

    for( auto& ring : station->rings() ) 
    for( auto& sCh : ring->superChambers() ) {
     Short_t nch= sCh->id().chamber();
     Double_t zmin = 9999;
     Double_t zmax = -999;
     TString name_suffix2 = name_suffix+TString::Format("_ch%d",nch); 
     TString title_suffix2 = title_suffix+TString::Format(" Chamber %d",nch);
 
     TString name  = TString::Format("strip_phiz_%s",name_suffix2.Data());
     TString title = TString::Format("Strip's Global PHI vs Z plots %s; Azimuthia Angle(degree) ; Global Z(cm)",title_suffix2.Data());

     double step = GE11PhiStep_;
     if ( nst > 1 ) step = GE11PhiStep_*2;
     double xmin = GE11PhiBegin_ + step*(nch-1);
     double xmax = GE11PhiBegin_ + step*(nch);
     int nbin =0;

     if ( nst ==1 ) nbin = nstripsGE11; 
     else  nbin = nstripsGE21;
 
     MonitorElement* temp2 = dbe_->book2D( name.Data(), title.Data(), nbin, xmin, xmax ,100, zmin-1, zmax+1);
     theStrip_phiz_st_ch.insert( std::map<UInt_t , MonitorElement*>::value_type(name.Hash(), temp2)) ;
     //std::cout<<name.Hash()<<"  "<<name.Length()<<std::endl;
     

     for ( auto& ch : sCh->chambers() ) {
      auto& roll = ch->etaPartitions()[0];
      auto& parameters(roll->specs()->parameters());
      float nStrips = (parameters[3]); 
      GEMDetId roId( roll->id());
      Short_t nla = (Short_t)roId.layer();
      //nro = (Short_t)roId.roll();
      TString name_suffix3 = name_suffix2 +TString::Format("_la%d",nla);
      TString title_suffix3 = title_suffix2 + TString::Format(" Layer %d",nla);

      TString name = TString::Format("strip_phi_dist_%s",name_suffix3.Data());
      TString title = TString::Format("strips' phi distributio at %s ; Strip number ; #phi(Degree)",title_suffix3.Data());

      MonitorElement* temp = dbe_->book1D(name.Data(), title.Data(),nStrips,1,nStrips+1);
      theStrip_ro_phi.insert( std::map<UInt_t, MonitorElement*>::value_type(name.Hash(), temp)) ;

      const StripTopology* topology(&(roll->specificTopology()));
      LocalPoint lEdge1(topology->localPosition((float)0));
      LocalPoint lEdge2(topology->localPosition((float)nStrips));
      double x1( roll->toGlobal( lEdge1).phi().degrees() );  
      double x2( roll->toGlobal( lEdge2).phi().degrees());  
      if ( x1 == x2 )  {
        LocalPoint lEdge3(topology->localPosition((float)1));
        LocalPoint lEdge4(topology->localPosition((float)nStrips-1));
        double x3( roll->toGlobal( lEdge1).phi().degrees());  
        double x4( roll->toGlobal( lEdge2).phi().degrees());  
        LogDebug("GEMStripDIGIValidation")<<"ch : "<<ch<<"  x1: "<<x1<<"  x2: "<<x2;
        LogDebug("GEMStripDIGIValidation")<<"ch : "<<ch<<"  x3: "<<x3<<"  x4: "<<x4;
      }
      double z( roll->toGlobal( lEdge1).z()); 
      if ( zmin> z ) zmin = z;
      if ( zmax< z ) zmax = z;
    }
    
   }
 } 
}


GEMStripDigiValidation::~GEMStripDigiValidation() {
 

}
void GEMStripDigiValidation::savePhiPlot(){
  for( auto& region : theGEMGeometry->regions() ) 
  for( auto& station : region->stations() ){
    for( auto& ring : station->rings() ) 
    for( auto& sCh : ring->superChambers() ) 
    for ( auto& ch : sCh->chambers() ) { 
      GEMDetId roId;
      float nStrips;
      auto& roll = ch->etaPartitions()[0];
      const StripTopology* topology(&(roll->specificTopology()));
      auto& parameters(roll->specs()->parameters());
      nStrips = parameters[3];
      roId =  roll->id();
      TString name_prefix = TString::Format("r%d_st%s_ch%d_la%d",roId.region(),stationLabel[roId.station()-1].c_str(),roId.chamber(),roId.layer());
      TString name = TString::Format("strip_phi_dist_")+name_prefix;
      double phi_0 = 0.0;
      double phi_max = 0.0;
      double low_edge= roll->toGlobal( topology->localPosition( 0.0) ).phi().degrees();
      for( unsigned int i=0; i<=nStrips ; i++) {
        LocalPoint lEdgeN(topology->localPosition((float)i));
        double cstripN( roll->toGlobal( lEdgeN).phi().degrees());
        double d_phi = cstripN- low_edge;
        if ( TMath::Abs( d_phi ) > 2*GE11PhiStep_) {
          if ( cstripN > 0)      cstripN -= 360.;
          else if ( cstripN < 0) cstripN += 360.;
        }
        if ( theStrip_ro_phi[name.Hash()] != nullptr ) {
          theStrip_ro_phi[name.Hash()]->Fill(i,cstripN);
        }
        else std::cout<<"ro_phi error! "<<name.Data()<<std::endl;
        if ( i==0 ) phi_0 = cstripN;
        if ( i==nStrips ) phi_max = cstripN;
      }
      TString ss = TString::Format("deltaPhi_r%d_st%s",region->region(),stationLabel[roId.station()-1].c_str());
      if ( theStrip_st_dphi[ss.Hash()] != nullptr) {
      theStrip_st_dphi[ss.Hash()]->Fill( TMath::Abs(phi_max- phi_0));
      }
      else std::cout<<"st_dphi error! "<<ss.Data()<<std::endl;
    }
  }
}

void GEMStripDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{
  edm::Handle<GEMDigiCollection> gem_digis;
  e.getByToken( this->inputToken_, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMStripDigiValidation") << "Cannot get strips by Token stripToken.\n";
    return ;
  }
  for (GEMDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {
    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = theGEMGeometry->idToDet(id);
    if ( gdet == nullptr) { 
      std::cout<<"Getting DetId failed. Discard this gem strip hit.Maybe it comes from unmatched geometry."<<std::endl;
      continue; 
    }
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = theGEMGeometry->etaPartition(id);

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
    Short_t station = (Short_t) id.station();
    Short_t chamber = (Short_t) id.chamber();
    Short_t nroll = (Short_t) id.roll();

    GEMDigiCollection::const_iterator digiItr;
    for (digiItr = (*cItr ).second.first; digiItr != (*cItr ).second.second; ++digiItr)
    {
      Short_t strip = (Short_t) digiItr->strip();
      Short_t bx = (Short_t) digiItr->bx();

      LocalPoint lp = roll->centreOfStrip(digiItr->strip());

      GlobalPoint gp = surface.toGlobal(lp);
      Float_t g_r = (Float_t) gp.perp();
      //Float_t g_eta = (Float_t) gp.eta();
      Float_t g_phi = (Float_t) gp.phi();
      Float_t g_x = (Float_t) gp.x();
      Float_t g_y = (Float_t) gp.y();
      Float_t g_z = (Float_t) gp.z();
      edm::LogInfo("GEMStripDIGIValidation")<<"Global x "<<g_x<<"Global y "<<g_y<<std::endl;  
      edm::LogInfo("GEMStripDIGIValidation")<<"Global strip "<<strip<<"Global phi "<<g_phi<<std::endl;  
      edm::LogInfo("GEMStripDIGIValidation")<<"Global bx "<<bx<<std::endl;  
      // fill hist
      int region_num=0 ;
      if ( region ==-1 ) region_num = 0 ;
      else if ( region==1) region_num = 1;  
      int station_num = station-1;
      int layer_num = layer-1;
    
      if ( theStrip_xy[region_num][station_num][layer_num] != nullptr) {
        theStrip_xy[region_num][station_num][layer_num]->Fill(g_x,g_y);     
        theStrip_phistrip[region_num][station_num][layer_num]->Fill(g_phi,strip);
        theStrip[region_num][station_num][layer_num]->Fill(strip);
        theStrip_bx[region_num][station_num][layer_num]->Fill(bx);
        theStrip_zr[region_num][station_num][layer_num]->Fill(g_z,g_r);
      }
      else {
        std::cout<<"Error is occued when histograms is called."<<std::endl;
      }
    
      TString name_suffix = TString::Format("r%d_st%s_ch%d",region,stationLabel[station_num].c_str(),chamber);
      TString name = TString::Format("strip_phiz_%s",name_suffix.Data());
      Float_t digi_phi = gp.phi().degrees();
      if ( digi_phi < GE11PhiBegin_  ) digi_phi = digi_phi+360.;
      
      if ( theStrip_phiz_st_ch[name.Hash()] != nullptr) {
        theStrip_phiz_st_ch[name.Hash()]->Fill(digi_phi,g_z);
      }
      else std::cout<<"Error! nullptr dqm. "<<name.Data()<<"  "<<name.Length()<<std::endl;
      if ( region == -1 && station ==1 && ( chamber == 1 || chamber == 2 ) ) {
        theSpecific_phiz[3]->Fill(digi_phi,g_z);
        if( nroll == 1) theSpecific_phiz[0]->Fill(digi_phi,g_z);
        else if( nroll == 4) theSpecific_phiz[1]->Fill(digi_phi,g_z);
        else if( nroll == 8) theSpecific_phiz[2]->Fill(digi_phi,g_z);
      }
   }
  }
}
