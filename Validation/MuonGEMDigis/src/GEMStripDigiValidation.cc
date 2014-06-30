#include "Validation/MuonGEMDigis/interface/GEMStripDigiValidation.h"
#include <iomanip>
GEMStripDigiValidation::GEMStripDigiValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag, const edm::ParameterSet& pbInfo)
:  GEMBaseValidation(dbe, inputTag, pbInfo)
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
        theStrip_xy[region_num][station_num][layer_num] = dbe_->book2D( ("strip_dg_xy"+name_prefix).c_str(), ("Digi occupancy: "+label_prefix+";globalX [cm]; globalY[cm]").c_str(), 360, -360,360,360,-360,360);
        theStrip_bx[region_num][station_num][layer_num] = dbe_->book1D( ("strip_dg_bx"+name_prefix).c_str(), ("Bunch crossing: "+label_prefix+"; bunch crossing ; entries").c_str(), 11,-5.5,5.5);
				theStrip_zr[region_num][station_num][layer_num] = BookHistZR("strip_dg","Strip Digi",region_num,station_num,layer_num);
      }
    }
  }
  // All chamber XY (cm) plots
	//auto& chamber = theGEMGeometry
	for( auto& region : theGEMGeometry->regions() ) 
	for( auto& station : region->stations() ) { 
		std::stringstream ss1;
 	  ss1<<"deltaPhi_r"<<region->region()<<"_st"<<station->station();
		std::string st_title = std::string( ss1.str()+";delta Phi(#Delta#phi);Entries");
		std::cout<<ss1.str()<<std::endl;
		MonitorElement* st_temp;
		if ( station->station() == 1 ) st_temp = dbe_->book1D(ss1.str(),st_title,2000.,8.5,12.5);
		if ( station->station() == 2 || station->station() == 3 ) st_temp = dbe_->book1D(ss1.str(),st_title,2000.,18.5,22.5);
		theStrip_st_dphi.insert( std::map<std::string, MonitorElement*>::value_type(ss1.str(), st_temp ));
		for( auto& ring : station->rings() ) 
		for( auto& sCh : ring->superChambers() ) 
		for ( auto& ch : sCh->chambers() ) 
		for ( auto& roll : ch->etaPartitions()){
			auto& parameters(roll->specs()->parameters());
			float nStrips(parameters[3]);
			GEMDetId roId( roll->id());
			std::stringstream ss;
 		  ss<<roId;
			std::string name_prefix = "strip_phi_dist_at_"+ss.str();
			std::string title_prefix = " strips' phi distributio "+ss.str()+";# of strip ; Azimuthal angle (#phi)";
			MonitorElement* temp = dbe_->book1D(name_prefix.c_str(), title_prefix.c_str(),nStrips,1,nStrips+1);
			theStrip_ro_phi.insert( std::map<std::string, MonitorElement*>::value_type(name_prefix, temp)) ; 
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
		for ( auto& ch : sCh->chambers() ) 
		for ( auto& roll : ch->etaPartitions()){
			//const BoundPlane& bSurface(roll->surface());
			const StripTopology* topology(&(roll->specificTopology()));
			// base_bottom, base_top, height, strips, pads (all half length)
			auto& parameters(roll->specs()->parameters());
			float nStrips(parameters[3]);

			GEMDetId roId( roll->id());
			std::stringstream ss;
	    ss<<roId;
			std::string name_prefix = "strip_phi_dist_at_"+ss.str();
			double phi_0 = 0.0;
			//double phi_1 = 0.0;

			//double phi_max_1 = 0.0;
			double phi_max = 0.0;
			for( unsigned int i=0; i<=nStrips ; i++) {
				LocalPoint lEdgeN(topology->localPosition((float)i));
				double cstripN( roll->toGlobal( lEdgeN).phi().degrees());
				theStrip_ro_phi[name_prefix.c_str()]->Fill(i,cstripN);
				if ( i==0 ) phi_0 = cstripN;
				//if ( i==1 ) phi_1 = cstripN;
				if ( i==nStrips ) phi_max = cstripN;
				//if ( i==nStrips-1 ) phi_max_1 = cstripN;
			}
			/*
			std::cout<<std::fixed;
			std::cout.precision(4);	
			std::cout<<"GEMDetId :"<<roId<< "strip #0 : "<<phi_0 <<"strip #max_1 : "<<phi_max_1<<std::endl;
			std::cout<<"GEMDetId :"<<roId<< "strip #1 : "<<phi_1 <<"strip #max : "<<phi_max<<std::endl;
      */

			ss.str("");
 	    ss<<"deltaPhi_r"<<region->region()<<"_st"<<station->station();
			theStrip_st_dphi[ss.str()]->Fill( TMath::Abs(phi_max- phi_0));
		}
	}
}

void GEMStripDigiValidation::analyze(const edm::Event& e,
                                     const edm::EventSetup&)
{
  edm::Handle<GEMDigiCollection> gem_digis;
  e.getByLabel(theInputTag, gem_digis);
  if (!gem_digis.isValid()) {
    edm::LogError("GEMStripDigiValidation") << "Cannot get strips by label "
                                       << theInputTag.encode();
  }
  for (GEMDigiCollection::DigiRangeIterator cItr=gem_digis->begin(); cItr!=gem_digis->end(); cItr++) {
    GEMDetId id = (*cItr).first;

    const GeomDet* gdet = theGEMGeometry->idToDet(id);
    const BoundPlane & surface = gdet->surface();
    const GEMEtaPartition * roll = theGEMGeometry->etaPartition(id);

    Short_t region = (Short_t) id.region();
    Short_t layer = (Short_t) id.layer();
    Short_t station = (Short_t) id.station();
		//Short_t chamber = (Short_t) id.chamber();

    GEMDigiCollection::const_iterator digiItr;
    //loop over digis of given roll
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

      theStrip_xy[region_num][station_num][layer_num]->Fill(g_x,g_y);     
      theStrip_phistrip[region_num][station_num][layer_num]->Fill(g_phi,strip);
      theStrip[region_num][station_num][layer_num]->Fill(strip);
      theStrip_bx[region_num][station_num][layer_num]->Fill(bx);
      theStrip_zr[region_num][station_num][layer_num]->Fill(g_z,g_r);
   }
  }
}
