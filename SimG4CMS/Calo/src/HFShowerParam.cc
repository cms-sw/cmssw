///////////////////////////////////////////////////////////////////////////////
// File: HFShowerParam.cc
// Description: Parametrized version of HF hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerParam.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4NavigationHistory.hh"
#include "Randomize.hh"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include<iostream>

//#define DebugLog
//#define plotDebug
//#define mkdebug

HFShowerParam::HFShowerParam(std::string & name, const DDCompactView & cpv,
                             edm::ParameterSet const & p) : showerLibrary(0), 
                                                            fibre(0), gflash(0),
                                                            fillHisto(false) { 
  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  pePerGeV                = m_HF.getParameter<double>("PEPerGeV");
  trackEM                 = m_HF.getParameter<bool>("TrackEM");
  bool useShowerLibrary   = m_HF.getParameter<bool>("UseShowerLibrary");
  bool useGflash          = m_HF.getParameter<bool>("UseHFGflash");
  edMin                   = m_HF.getParameter<double>("EminLibrary");
  onlyLong                = m_HF.getParameter<bool>("OnlyLong");
  ref_index               = m_HF.getParameter<double>("RefIndex");
  double lambdaMean       = m_HF.getParameter<double>("LambdaMean");
  aperture                = cos(asin(m_HF.getParameter<double>("Aperture")));
  applyFidCut             = m_HF.getParameter<bool>("ApplyFiducialCut");
  parametrizeLast         = m_HF.getUntrackedParameter<bool>("ParametrizeLast",false);
  edm::LogInfo("HFShower") << "HFShowerParam::Use of shower library is set to "
                           << useShowerLibrary << " Use of Gflash is set to "
                           << useGflash << " P.E. per GeV " << pePerGeV
                           << ", ref. index of fibre " << ref_index 
                           << ", Track EM Flag " << trackEM << ", edMin "
                           << edMin << " GeV, use of Short fibre info in"
                           << " shower library set to " << !(onlyLong)
                           << ", use of parametrization for last part set to "
                           << parametrizeLast << ", Mean lambda " << lambdaMean
			   << ", aperture (cutoff) " << aperture
                           << ", Application of Fiducial Cut " << applyFidCut;

#ifdef plotDebug
  edm::Service<TFileService> tfile;
  if (tfile.isAvailable()) {
    fillHisto = true;
    LogDebug("HFShower") << "HFShowerParam::Save histos in directory "
			 << "ProfileFromParam";
    TFileDirectory showerDir = tfile->mkdir("ProfileFromParam");
    hzvem           = showerDir.make<TH1F>("hzvem", "Longitudinal Profile (EM Part);Number of PE",330,0.0,1650.0);
    hzvhad          = showerDir.make<TH1F>("hzvhad","Longitudinal Profile (Had Part);Number of PE",330,0.0,1650.0);
    em_2d_1         = showerDir.make<TH2F>("em_2d_1","Lateral Profile vs. Shower Depth;cm;Events",800,800.0,1600.0,100,50.0,150.0);
    em_long_1       = showerDir.make<TH1F>("em_long_1","Longitudinal Profile;Radiation Length;Number of Spots",800,800.0,1600.0);
    em_long_1_tuned = showerDir.make<TH1F>("em_long_1_tuned","Longitudinal Profile;Radiation Length;Number of Spots",800,800.0,1600.0);
    em_lateral_1    = showerDir.make<TH1F>("em_lateral_1","Lateral Profile;cm;Events",100,50.0,150.0);
    em_2d_2         = showerDir.make<TH2F>("em_2d_2","Lateral Profile vs. Shower Depth;cm;Events",800,800.0,1600.0,100,50.0,150.0);
    em_long_2       = showerDir.make<TH1F>("em_long_2","Longitudinal Profile;Radiation Length;Number of Spots",800,800.0,1600.0);
    em_lateral_2    = showerDir.make<TH1F>("em_lateral_2","Lateral Profile;cm;Events",100,50.0,150.0);
    em_long_gflash  = showerDir.make<TH1F>("em_long_gflash","Longitudinal Profile From GFlash;cm;Number of Spots",800,800.0,1600.0);
    em_long_sl      = showerDir.make<TH1F>("em_long_sl","Longitudinal Profile From Shower Library;cm;Number of Spots",800,800.0,1600.0);
  } else {
    fillHisto = false;
    edm::LogInfo("HFShower") << "HFShowerParam::No file is available for "
			     << "saving histos so the flag is set to false";
  }
#endif
  
  if (useShowerLibrary) showerLibrary = new HFShowerLibrary(name,cpv,p);
  if (useGflash)        gflash        = new HFGflash(p);
  fibre = new HFFibre(name, cpv, p);
  attLMeanInv = fibre->attLength(lambdaMean);
  edm::LogInfo("HFShower") << "att. length used for (lambda=" << lambdaMean
                           << ") = " << 1/(attLMeanInv*cm) << " cm";
}

HFShowerParam::~HFShowerParam() {
  if (fibre)         delete fibre;
  if (gflash)        delete gflash;
  if (showerLibrary) delete showerLibrary;
}

void HFShowerParam::initRun(G4ParticleTable * theParticleTable,
			    HcalDDDSimConstants* hcons) {
  emPDG = theParticleTable->FindParticle("e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle("e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle("gamma")->GetPDGEncoding();
#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShowerParam: Particle code for e- = " << emPDG
                           << " for e+ = " << epPDG << " for gamma = " << gammaPDG;
#endif
  if (showerLibrary) showerLibrary->initRun(theParticleTable, hcons);
  if (fibre)         fibre->initRun(hcons);

  //Special Geometry parameters
  gpar      = hcons->getGparHF();
  edm::LogInfo("HFShower") << "HFShowerParam: " << gpar.size() <<" gpar (cm)";
  for (unsigned int ig=0; ig<gpar.size(); ig++)
    edm::LogInfo("HFShower") << "HFShowerParam: gpar[" << ig << "] = "
                             << gpar[ig]/cm << " cm";
}

std::vector<HFShowerParam::Hit> HFShowerParam::getHits(G4Step * aStep, 
						       double weight) {
  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4Track *     track    = aStep->GetTrack();   
  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4int         particleCode = track->GetDefinition()->GetPDGEncoding();
  double        zv = std::abs(hitPoint.z()) - gpar[4] - 0.5*gpar[1];
  G4ThreeVector localPoint = G4ThreeVector(hitPoint.x(),hitPoint.y(),zv);

  double pin    = (preStepPoint->GetTotalEnergy())/GeV;
  double zint   = hitPoint.z(); 
  double zz     = std::abs(zint) - gpar[4];

#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShowerParam: getHits " 
                           << track->GetDefinition()->GetParticleName()
                           << " of energy " << pin << " GeV" 
                           << " Pos x,y,z = " << hitPoint.x() << "," 
                           << hitPoint.y() << "," << zint << " (" << zz << ","
                           << localPoint.z() << ", " 
                           << (localPoint.z()+0.5*gpar[1]) << ") Local " 
                           << localPoint;
#endif
  std::vector<HFShowerParam::Hit> hits;
  HFShowerParam::Hit hit;
  hit.position = hitPoint;

  // look for other charged particles
  bool   other = false;
  double pBeta = track->GetDynamicParticle()->GetTotalMomentum() / track->GetDynamicParticle()->GetTotalEnergy();
  double dirz  = (track->GetDynamicParticle()->GetMomentumDirection()).z();
  if (hitPoint.z() < 0) dirz *= -1.;
#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShowerParam: getHits Momentum " 
			   <<track->GetDynamicParticle()->GetMomentumDirection()
			   << " HitPoint " << hitPoint << " dirz " << dirz;
#endif  
  if (particleCode != emPDG && particleCode != epPDG && particleCode != gammaPDG ) {
    if (track->GetDefinition()->GetPDGCharge() != 0 && pBeta > (1/ref_index) &&
        aStep->GetTotalEnergyDeposit() > 0) other = true;
  }

  // take only e+-/gamma/or special particles
  if (particleCode == emPDG || particleCode == epPDG ||
      particleCode == gammaPDG || other) {
    // Leave out the last part
    double edep = 0.;
    bool   kill = false;
    if ((!trackEM) && ((zz<(gpar[1]-gpar[2])) || parametrizeLast) && (!other)){
      edep = pin;
      kill = true;
    } else if ((track->GetDefinition()->GetPDGCharge() != 0) && 
	       (pBeta > (1/ref_index)) && (dirz > aperture)) {
      edep = (aStep->GetTotalEnergyDeposit())/GeV;
    }
    std::string path = "ShowerLibrary";
#ifdef DebugLog
    edm::LogInfo("HFShower") << "HFShowerParam: getHits edep = " << edep
			     << " weight " << weight << " final " <<edep*weight
                             << ", Kill = " << kill << ", pin = " << pin 
                             << ", edMin = " << edMin << " Other " << other;
#endif
    edep *= weight;
    if (edep > 0) {
      if ((showerLibrary || gflash) && kill && pin > edMin && (!other)) {
        if (showerLibrary) {
          std::vector<HFShowerLibrary::Hit> hitSL = showerLibrary->getHits(aStep,kill,weight,onlyLong);
          for (unsigned int i=0; i<hitSL.size(); i++) {
            bool ok = true;
#ifdef DebugLog
            edm::LogInfo("HFShower") << "HFShowerParam: getHits applyFidCut = " << applyFidCut;
#endif
            if (applyFidCut) { // @@ For showerlibrary no z-cut for Short (no z)
              int npmt = HFFibreFiducial:: PMTNumber(hitSL[i].position);
              if (npmt <= 0) ok = false;
            } 
            if (ok) {
              hit.position = hitSL[i].position;
              hit.depth    = hitSL[i].depth;
              hit.time     = hitSL[i].time;
              hit.edep     = 1;
              hits.push_back(hit);
#ifdef plotDebug
              if (fillHisto) {
		double zv  = std::abs(hit.position.z()) - gpar[4];
                hzvem->Fill(zv);
                em_long_sl->Fill(hit.position.z()/cm);
                double sq = sqrt(pow(hit.position.x()/cm,2)+pow(hit.position.y()/cm,2));
                double zp = hit.position.z()/cm;
                if (hit.depth == 1) {
                  em_2d_1->Fill(zp, sq);  
                  em_lateral_1->Fill(sq);
                  em_long_1->Fill(zp);
                } else if (hit.depth == 2) {
                  em_2d_2->Fill(zp, sq);
                  em_lateral_2->Fill(sq);
                  em_long_2->Fill(zp);
                }
              }
#endif
#ifdef DebugLog
              edm::LogInfo("HFShower") << "HFShowerParam: Hit at depth " 
                                       << hit.depth << " with edep " << hit.edep
                                       << " Time " << hit.time;
#endif
            }
          }
        } else { // GFlash clusters with known z
          std::vector<HFGflash::Hit>hitSL=gflash->gfParameterization(aStep,kill, onlyLong);
          for (unsigned int i=0; i<hitSL.size(); ++i) {
            bool ok = true;
            G4ThreeVector pe_effect(hitSL[i].position.x(), hitSL[i].position.y(),
                                    hitSL[i].position.z());
            double zv  = std::abs(pe_effect.z()) - gpar[4];
            //depth
            int depth    = 1;
            int npmt     = 0;
            if (zv < 0. || zv > gpar[1]) {
#ifdef mkdebug
              std::cout<<"-#Zcut-HFShowerParam::getHits:z="<<zv<<",m="<<gpar[1]<<std::endl;
#endif
              ok = false;
            }
            if (ok && applyFidCut) {
              npmt = HFFibreFiducial:: PMTNumber(pe_effect);
#ifdef DebugLog
              edm::LogInfo("HFShower") << "HFShowerParam::getHits:#PMT= "
                                       << npmt << ",z = " << zv;
#endif
              if (npmt <= 0) {
#ifdef DebugLog
                edm::LogInfo("HFShower") << "-#PMT=0 cut-HFShowerParam::"
					 << "getHits: npmt = " << npmt;
#endif
                ok = false;
              } else if (npmt > 24) { // a short fibre
                if    (zv > gpar[0]) {
		  depth = 2; 
                } else {
#ifdef DebugLog
                  edm::LogInfo("HFShower") << "-SHORT cut-HFShowerParam::"
					   << "getHits:zMin=" << gpar[0];
#endif
                  ok = false;
                }
              }
#ifdef DebugLog
              edm::LogInfo("HFShower") << "HFShowerParam: npmt " << npmt 
                                       << " zv " << std::abs(pe_effect.z()) 
                                       << ":" << gpar[4] << ":" << zv << ":" 
                                       << gpar[0] << " ok " << ok << " depth "
				       << depth;
#endif
            } else {
              if (G4UniformRand() > 0.5) depth = 2;
              if (depth == 2 && zv < gpar[0]) ok = false;
            }
            //attenuation
            double dist = fibre->zShift(localPoint,depth,0); // distance to PMT
            double r1   = G4UniformRand();
#ifdef DebugLog
            edm::LogInfo("HFShower") << "HFShowerParam:Distance to PMT (" <<npmt
                                     << ") " << dist << ", exclusion flag " 
                                     << (r1 > exp(-attLMeanInv*zv));
#endif
            if (r1 > exp(-attLMeanInv*dist)) ok = false;
            if (ok) {
	      double r2   = G4UniformRand();
#ifdef DebugLog
	      edm::LogInfo("HFShower") << "HFShowerParam:Extra exclusion "
				       << r2 << ">" << weight << " "
				       << (r2 > weight);
#endif
	      if (r2 < weight) {
		double time = fibre->tShift(localPoint,depth,0);

		hit.position = hitSL[i].position;
		hit.depth    = depth;
		hit.time     = time + hitSL[i].time;
		hit.edep     = 1;
		hits.push_back(hit);
#ifdef plotDebug
		if (fillHisto) {
		  em_long_gflash->Fill(pe_effect.z()/cm, hitSL[i].edep);
		  hzvem->Fill(zv);
		  double sq = sqrt(pow(hit.position.x()/cm,2)+pow(hit.position.y()/cm,2));
		  double zp = hit.position.z()/cm;
		  if (hit.depth == 1) {
		    em_2d_1->Fill(zp, sq);
		    em_lateral_1->Fill(s);
		    em_long_1->Fill(zp);
		  } else if (hit.depth == 2) {
		    em_2d_2->Fill(zp, sq);
		    em_lateral_2->Fill(sq);
		    em_long_2->Fill(zp);
		  }
		}
#endif
#ifdef DebugLog
		edm::LogInfo("HFShower") << "HFShowerParam: Hit at depth " 
					 << hit.depth << " with edep " 
					 << hit.edep << " Time "  << hit.time;
#endif
	      }
            }
          }
        }
      } else {
        path          = "Rest";
        edep         *= pePerGeV;
        double tSlice = (aStep->GetPostStepPoint()->GetGlobalTime());
        double time   = fibre->tShift(localPoint,1,0); // remaining part
	bool ok = true;
	if (applyFidCut) { // @@ For showerlibrary no z-cut for Short (no z)
	  int npmt = HFFibreFiducial:: PMTNumber(hitPoint);
	  if (npmt <= 0) ok = false;
	} 
#ifdef DebugLog
	edm::LogInfo("HFShower") << "HFShowerParam: getHits hitPoint " << hitPoint << " flag " << ok;
#endif
	if (ok) {
	  hit.depth     = 1;
	  hit.time      = tSlice+time;
	  hit.edep      = edep;
	  hits.push_back(hit);
#ifdef DebugLog
	  edm::LogInfo("HFShower") << "HFShowerParam: Hit at depth 1 with edep "
				   << edep << " Time " << tSlice << ":" << time
				   << ":" << hit.time;
#endif
#ifdef plotDebug
	  double zv = std::abs(hitPoint.z()) - gpar[4];
	  if (fillHisto) {
	    hzvhad->Fill(zv);
	  }
#endif
	  if (zz >= gpar[0]) {
	    time      = fibre->tShift(localPoint,2,0);
	    hit.depth = 2;
	    hit.time  = tSlice+time;
	    hits.push_back(hit);
#ifdef DebugLog
	    edm::LogInfo("HFShower") <<"HFShowerParam: Hit at depth 2 with edep "
				     << edep << " Time " << tSlice << ":" << time
				     << hit.time;
#endif
#ifdef plotDebug
	    if (fillHisto) {
	      hzvhad->Fill(zv);
	    }
#endif
	  }
	}
      }
#ifdef DebugLog
      for (unsigned int ii=0; ii<hits.size(); ++ii) {
        double zv = std::abs(hits[ii].position.z());
        if (zv > 12790) edm::LogInfo("HFShower")<< "HFShowerParam: Abnormal hit along " 
                                                << path << " in " 
						<< preStepPoint->GetPhysicalVolume()->GetLogicalVolume()->GetName()
                                                << " at " << hits[ii].position << " zz " 
                                                << zv << " Edep " << edep << " due to " 
                                                <<track->GetDefinition()->GetParticleName()
                                                << " time " << hit.time;
      }
#endif
      if (kill) {
        track->SetTrackStatus(fStopAndKill);
        G4TrackVector tv = *(aStep->GetSecondary());
        for (unsigned int kk=0; kk<tv.size(); ++kk) {
          if (tv[kk]->GetVolume() == preStepPoint->GetPhysicalVolume())
	    tv[kk]->SetTrackStatus(fStopAndKill);
        }
      }
#ifdef DebugLog
      edm::LogInfo("HFShower") << "HFShowerParam: getHits kill (" << kill
                               << ") track " << track->GetTrackID() 
                               << " at " << hitPoint
                               << " and deposit " << edep << " " << hits.size()
                               << " times" << " ZZ " << zz << " " << gpar[0];
#endif
    }
  }
  return hits;
}

std::vector<double> HFShowerParam::getDDDArray(const std::string & str, 
                                               const DDsvalues_type & sv)
{
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerParam:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value))
  {
#ifdef DebugLog
    LogDebug("HFShower") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 2) {
      edm::LogError("HFShower") << "HFShowerParam : # of " << str 
                                << " bins " << nval << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HFShowerParam") << "nval < 2 for array "
                                                       << str << "\n";
    }
    return fvec;
  } else {
    edm::LogError("HFShower") << "HFShowerParam : cannot get array " << str;
    throw cms::Exception("Unknown", "HFShowerParam")  << "cannot get array "
                                                      << str << "\n";
  }
}
