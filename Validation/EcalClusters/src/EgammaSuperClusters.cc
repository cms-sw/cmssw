#include "Validation/EcalClusters/interface/EgammaSuperClusters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DQMServices/Core/interface/DQMStore.h"

EgammaSuperClusters::EgammaSuperClusters( const edm::ParameterSet& ps )
{
	outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "");
	CMSSW_Version_ = ps.getUntrackedParameter<std::string>("CMSSW_Version", "");

	verboseDBE_ = ps.getUntrackedParameter<bool>("verboseDBE", false);

	hist_min_Size_ = ps.getParameter<double>("hist_min_Size");
	hist_max_Size_ = ps.getParameter<double>("hist_max_Size");
	hist_bins_Size_ = ps.getParameter<int>   ("hist_bins_Size");

	hist_min_NumBC_ = ps.getParameter<double>("hist_min_NumBC");
	hist_max_NumBC_ = ps.getParameter<double>("hist_max_NumBC");
	hist_bins_NumBC_ = ps.getParameter<int>   ("hist_bins_NumBC");

	hist_min_ET_ = ps.getParameter<double>("hist_min_ET");
	hist_max_ET_ = ps.getParameter<double>("hist_max_ET");
	hist_bins_ET_ = ps.getParameter<int>   ("hist_bins_ET");

	hist_min_Eta_ = ps.getParameter<double>("hist_min_Eta");
	hist_max_Eta_ = ps.getParameter<double>("hist_max_Eta");
	hist_bins_Eta_ = ps.getParameter<int>   ("hist_bins_Eta");

	hist_min_Phi_ = ps.getParameter<double>("hist_min_Phi");
	hist_max_Phi_ = ps.getParameter<double>("hist_max_Phi");
	hist_bins_Phi_ = ps.getParameter<int>   ("hist_bins_Phi");

	hist_min_S1toS9_ = ps.getParameter<double>("hist_min_S1toS9");
	hist_max_S1toS9_ = ps.getParameter<double>("hist_max_S1toS9");
	hist_bins_S1toS9_ = ps.getParameter<int>   ("hist_bins_S1toS9");

	hist_min_S25toE_ = ps.getParameter<double>("hist_min_S25toE");
	hist_max_S25toE_ = ps.getParameter<double>("hist_max_S25toE");
	hist_bins_S25toE_ = ps.getParameter<int>   ("hist_bins_S25toE");

	hist_min_EToverTruth_ = ps.getParameter<double>("hist_min_EToverTruth");
	hist_max_EToverTruth_ = ps.getParameter<double>("hist_max_EToverTruth");
	hist_bins_EToverTruth_ = ps.getParameter<int>   ("hist_bins_EToverTruth");
	
	hist_min_deltaEta_ = ps.getParameter<double>("hist_min_deltaEta");
	hist_max_deltaEta_ = ps.getParameter<double>("hist_max_deltaEta");
	hist_bins_deltaEta_ = ps.getParameter<int>   ("hist_bins_deltaEta");

	MCTruthCollection_ = ps.getParameter<edm::InputTag>("MCTruthCollection");
	barrelSuperClusterCollection_ = ps.getParameter<edm::InputTag>("barrelSuperClusterCollection");
  	endcapSuperClusterCollection_ = ps.getParameter<edm::InputTag>("endcapSuperClusterCollection");
        barrelRecHitCollection_ = ps.getParameter<edm::InputTag>("barrelRecHitCollection");
        endcapRecHitCollection_ = ps.getParameter<edm::InputTag>("endcapRecHitCollection");
}

EgammaSuperClusters::~EgammaSuperClusters() {}

void EgammaSuperClusters::beginJob(edm::EventSetup const&) 
{
  	dbe_ = edm::Service<DQMStore>().operator->();                   

  	if ( verboseDBE_ )
	{
  	dbe_->setVerbose(1);
		dbe_->showDirStructure();
	}
	else 
		dbe_->setVerbose(0);

	dbe_->setCurrentFolder("EcalClustersV/CMSSW_"+CMSSW_Version_+"/EcalClusters/SuperClusters/");

	hist_EB_SC_Size_ 
		= dbe_->book1D("hist_EB_SC_Size_","# Super Clusters in Barrel",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);
  	hist_EE_SC_Size_ 
		= dbe_->book1D("hist_EE_SC_Size_","# Super Clusters in Endcap",
			hist_bins_Size_,hist_min_Size_,hist_max_Size_);

	hist_EB_SC_NumBC_ 
		= dbe_->book1D("hist_EB_SC_NumBC_","# of Basic Clusters in Super Clusters in Barrel",
			hist_bins_NumBC_,hist_min_NumBC_,hist_max_NumBC_);
  	hist_EE_SC_NumBC_ 
		= dbe_->book1D("hist_EE_SC_NumBC_","# of Basic Clusters in Super Clusters in Endcap",
		hist_bins_NumBC_,hist_min_NumBC_,hist_max_NumBC_);

  	hist_EB_SC_ET_ 
		= dbe_->book1D("hist_EB_SC_ET_","ET of Super Clusters in Barrel",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);
  	hist_EE_SC_ET_ 
		= dbe_->book1D("hist_EE_SC_ET_","ET of Super Clusters in Endcap",
			hist_bins_ET_,hist_min_ET_,hist_max_ET_);

  	hist_EB_SC_Eta_ 
		= dbe_->book1D("hist_EB_SC_Eta_","Eta of Super Clusters in Barrel",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);
 	hist_EE_SC_Eta_ 
		= dbe_->book1D("hist_EE_SC_Eta_","Eta of Super Clusters in Endcap",
			hist_bins_Eta_,hist_min_Eta_,hist_max_Eta_);

  	hist_EB_SC_Phi_ 
		= dbe_->book1D("hist_EB_SC_Phi_","Phi of Super Clusters in Barrel",
			hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);
  	hist_EE_SC_Phi_ 
		= dbe_->book1D("hist_EE_SC_Phi_","Phi of Super Clusters in Endcap",
			hist_bins_Phi_,hist_min_Phi_,hist_max_Phi_);

  	hist_EB_SC_S1toS9_ 
		= dbe_->book1D("hist_EB_SC_S1toS9_","S1/S9 of Super Clusters in Barrel",
			hist_bins_S1toS9_,hist_min_S1toS9_,hist_max_S1toS9_);
  	hist_EE_SC_S1toS9_ 
		= dbe_->book1D("hist_EE_SC_S1toS9_","S1/S9 of Super Clusters in Endcap",
			hist_bins_S1toS9_,hist_min_S1toS9_,hist_max_S1toS9_);

  	hist_EB_SC_S25toE_ 
		= dbe_->book1D("hist_EB_SC_S25toE_","S25/E of Super Clusters in Barrel",
			hist_bins_S25toE_,hist_min_S25toE_,hist_max_S25toE_);
  	hist_EE_SC_S25toE_ 
		= dbe_->book1D("hist_EE_SC_S25toE_","S25/E of Super Clusters in Endcap",
			hist_bins_S25toE_,hist_min_S25toE_,hist_max_S25toE_);

  	hist_EB_SC_EToverTruth_ 
		= dbe_->book1D("hist_EB_SC_EToverTruth_","ET/True ET of Super Clusters in Barrel",	
			hist_bins_EToverTruth_,hist_min_EToverTruth_,hist_max_EToverTruth_);
  	hist_EE_SC_EToverTruth_ 
		= dbe_->book1D("hist_EE_SC_EToverTruth_","ET/True ET of Super Clusters in Endcap",
			hist_bins_EToverTruth_,hist_min_EToverTruth_,hist_max_EToverTruth_);

  	hist_EB_SC_deltaEta_ 
		= dbe_->book1D("hist_EB_SC_deltaEta_","Eta-True Eta of Super Clusters in Barrel",
			hist_bins_deltaEta_,hist_min_deltaEta_,hist_max_deltaEta_);
  	hist_EE_SC_deltaEta_ 
		= dbe_->book1D("hist_EE_SC_deltaEta_","Eta-True Eta of Super Clusters in Endcap",
			hist_bins_deltaEta_,hist_min_deltaEta_,hist_max_deltaEta_);
}


void EgammaSuperClusters::analyze( const edm::Event& evt, const edm::EventSetup& es )
{
  	edm::Handle<reco::SuperClusterCollection> pBarrelSuperClusters;
	evt.getByLabel(barrelSuperClusterCollection_, pBarrelSuperClusters);
	if (!pBarrelSuperClusters.isValid()) {
	  edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label " 
					       << barrelSuperClusterCollection_.label();
  	}

        EcalClusterLazyTools lazyTool( evt, es, barrelRecHitCollection_, endcapRecHitCollection_ );
        
  	const reco::SuperClusterCollection* barrelSuperClusters = pBarrelSuperClusters.product();
  	hist_EB_SC_Size_->Fill(barrelSuperClusters->size());

  	for(reco::SuperClusterCollection::const_iterator aClus = barrelSuperClusters->begin(); 
		aClus != barrelSuperClusters->end(); aClus++)
	{
		hist_EB_SC_NumBC_->Fill(aClus->clustersSize());
    		hist_EB_SC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
		hist_EB_SC_Eta_->Fill(aClus->position().eta());
		hist_EB_SC_Phi_->Fill(aClus->position().phi());

                const reco::BasicClusterRef seed = aClus->seed();
		hist_EB_SC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
		hist_EB_SC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );
  	}

  	edm::Handle<reco::SuperClusterCollection> pEndcapSuperClusters;
	evt.getByLabel(endcapSuperClusterCollection_, pEndcapSuperClusters);
	if (!pEndcapSuperClusters.isValid()) {
	  edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label " 
					       << endcapSuperClusterCollection_.label();
  	}

  	const reco::SuperClusterCollection* endcapSuperClusters = pEndcapSuperClusters.product();
  	hist_EE_SC_Size_->Fill(endcapSuperClusters->size());

  	for(reco::SuperClusterCollection::const_iterator aClus = endcapSuperClusters->begin(); 
		aClus != endcapSuperClusters->end(); aClus++)
	{
		hist_EE_SC_NumBC_->Fill(aClus->clustersSize());
    		hist_EE_SC_ET_->Fill(aClus->energy()/std::cosh(aClus->position().eta()));
		hist_EE_SC_Eta_->Fill(aClus->position().eta());
		hist_EE_SC_Phi_->Fill(aClus->position().phi());

                const reco::BasicClusterRef seed = aClus->seed();
		hist_EE_SC_S1toS9_->Fill( lazyTool.eMax( *seed ) / lazyTool.e3x3( *seed ) );
		hist_EE_SC_S25toE_->Fill( lazyTool.e5x5( *seed ) / aClus->energy() );
  	}

 	edm::Handle<edm::HepMCProduct> pMCTruth ;
	evt.getByLabel(MCTruthCollection_, pMCTruth);
	if (!pMCTruth.isValid()) {
	  edm::LogError("EgammaSuperClusters") << "Error! can't get collection with label " 
					       << MCTruthCollection_.label();
  	}

	const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();
  	for(HepMC::GenEvent::particle_const_iterator currentParticle = genEvent->particles_begin(); 
		currentParticle != genEvent->particles_end(); currentParticle++ )
  	{
	  	if((*currentParticle)->status()==1) 
		{
			HepMC::FourVector vtx = (*currentParticle)->production_vertex()->position();
			double phiTrue = (*currentParticle)->momentum().phi();
			double etaTrue = ecalEta((*currentParticle)->momentum().eta(), vtx.z()/10., vtx.perp()/10.);
			double etTrue  = (*currentParticle)->momentum().e()/cosh(etaTrue);

			if(std::fabs(etaTrue) < 1.479)
			{
				{
					double etaCurrent, etaFound = 0;
					double phiCurrent;
					double etCurrent,  etFound  = 0;

					double closestParticleDistance = 999; 
				
					for(reco::SuperClusterCollection::const_iterator aClus = barrelSuperClusters->begin(); 
						aClus != barrelSuperClusters->end(); aClus++)
					{
						etaCurrent = 	aClus->position().eta();
						phiCurrent = 	aClus->position().phi();
						etCurrent  =  aClus->energy()/std::cosh(etaCurrent);
											
						double deltaR = std::sqrt(std::pow(etaCurrent-etaTrue,2)+std::pow(phiCurrent-phiTrue,2));

						if(deltaR < closestParticleDistance)
						{
							etFound  = etCurrent;
							etaFound = etaCurrent;
							closestParticleDistance = deltaR;
						}
					}
					
					if(closestParticleDistance < 0.3)
					{
						hist_EB_SC_EToverTruth_->Fill(etFound/etTrue);
						hist_EB_SC_deltaEta_->Fill(etaFound-etaTrue);
					}
				}
			}
			else
			{
				double etaCurrent, etaFound = 0;
				double phiCurrent;
				double etCurrent,  etFound  = 0;

				double closestParticleDistance = 999; 

			  	for(reco::SuperClusterCollection::const_iterator aClus = endcapSuperClusters->begin(); 
					aClus != endcapSuperClusters->end(); aClus++)
				{
					etaCurrent = 	aClus->position().eta();
					phiCurrent = 	aClus->position().phi();
					etCurrent  =  aClus->energy()/std::cosh(etaCurrent);

					double deltaR = std::sqrt(std::pow(etaCurrent-etaTrue,2)+std::pow(phiCurrent-phiTrue,2)); 

					if(deltaR < closestParticleDistance)
					{
						etFound  = etCurrent;
						etaFound = etaCurrent;
						closestParticleDistance = deltaR;
					}
				}
				
				if(closestParticleDistance < 0.3)
				{
					hist_EE_SC_EToverTruth_->Fill(etFound/etTrue);
					hist_EE_SC_deltaEta_->Fill(etaFound-etaTrue);
				}
			}
		}
	}
}

void EgammaSuperClusters::endJob()
{
	if (outputFile_.size() != 0 && dbe_) dbe_->save(outputFile_);
}

float EgammaSuperClusters::ecalEta(float EtaParticle , float Zvertex, float plane_Radius)
{  
	const float R_ECAL           = 136.5;
	const float Z_Endcap         = 328.0;
	const float etaBarrelEndcap  = 1.479;

	if(EtaParticle != 0.)
	{
		float Theta = 0.0  ;
		float ZEcal = (R_ECAL-plane_Radius)*sinh(EtaParticle)+Zvertex;

		if(ZEcal != 0.0) Theta = atan(R_ECAL/ZEcal);
		if(Theta<0.0) Theta = Theta+Geom::pi() ;

		float ETA = - log(tan(0.5*Theta));

		if( fabs(ETA) > etaBarrelEndcap )
		{
			float Zend = Z_Endcap ;
			if(EtaParticle<0.0 )  Zend = -Zend ;
			float Zlen = Zend - Zvertex ;
			float RR = Zlen/sinh(EtaParticle);
			Theta = atan((RR+plane_Radius)/Zend);
			if(Theta<0.0) Theta = Theta+Geom::pi() ;
			ETA = - log(tan(0.5*Theta));
		}
		
		return ETA;
	}
	else
	{
		edm::LogWarning("")  << "[EgammaSuperClusters::ecalEta] Warning: Eta equals to zero, not correcting" ;
		return EtaParticle;
	}
}
