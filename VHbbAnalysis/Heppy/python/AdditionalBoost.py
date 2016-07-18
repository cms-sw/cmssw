import itertools
import math

import ROOT
import sys
import os
import pdb

import pdb
import ctypes




from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.Heppy.physicsobjects.PhysicsObject import PhysicsObject
from PhysicsTools.Heppy.physicsobjects.PhysicsObjects import Jet
from PhysicsTools.Heppy.physicsutils.JetReCalibrator import JetReCalibrator


# Fastjet-Contrib is not in the path per default.
# We need it for n-subjettiness recalculation
ROOT.gInterpreter.AddIncludePath( "/cvmfs/cms.cern.ch/"+ os.environ['SCRAM_ARCH'] +"/external/fastjet-contrib/1.020/include" )

ROOT.gSystem.Load("libfastjet")
ROOT.gSystem.Load("libfastjetcontribfragile")
ROOT.gSystem.Load("libRecoBTagSecondaryVertex")
ROOT.gSystem.Load("libDataFormatsTauReco.so")

ROOT.gInterpreter.ProcessLine('#include "FWCore/ParameterSet/interface/ParameterSet.h"')

ROOT.gInterpreter.ProcessLine('#include "fastjet/contrib/Njettiness.hh"')
ROOT.gInterpreter.ProcessLine('#include "RecoBTag/SecondaryVertex/interface/TrackSelector.h"')
ROOT.gInterpreter.ProcessLine('#include "RecoBTag/SecondaryVertex/interface/V0Filter.h"')
ROOT.gInterpreter.ProcessLine('#include "DataFormats/BTauReco/interface/TaggingVariable.h"')

# Helper function as there seem to be no python bindings for
# DataFormats/Math/interface/deltaR.h
def deltaR2(a, b):    
    phi1 = float(a.phi())
    eta1 = float(a.eta())

    phi2 = float(b.phi())
    eta2 = float(b.eta())
    
    return pow(abs(abs(phi1 - phi2) - math.pi) - math.pi, 2) + pow(eta1 - eta2, 2)


# Test if the jet passes the loose JetID criteria defined at
# https://twiki.cern.ch/twiki/bin/view/CMS/JetID#Recommendations_for_13_TeV_data
# Furthermore we require:
#   - pT > 20 GeV
#   - |eta| < 2.4
def passesJetId(jet):
    
    if jet.pt() <= 20:
        return False
    
    if abs(jet.eta()) >= 2.4:
        return False

    if jet.neutralHadronEnergyFraction() >= 0.99:
        return False
    
    if jet.neutralEmEnergyFraction() >= 0.99:
        return False

    if jet.chargedMultiplicity()+jet.neutralMultiplicity() <= 1:
        return False
    
    if jet.chargedHadronEnergyFraction() <= 0:
        return False
        
    if jet.chargedMultiplicity() <= 0:
        return False
    
    if  jet.chargedEmEnergyFraction() >= 0.99:
        return False

    return True


def etaRelToTauAxis( vertex, tauAxis, tau_trackEtaRel) :
  direction = ROOT.Math.XYZVector(tauAxis.px(), tauAxis.py(), tauAxis.pz())
  tracks = vertex.daughterPtrVector()


  for track in tracks:
  #for(std::vector<reco::CandidatePtr>::const_iterator track = tracks.begin(); track != tracks.end(); ++track)
    tau_trackEtaRel.append(abs(ROOT.reco.btau.etaRel(direction.Unit(), track.momentum())))




def vertexKinematics(vertex, vtxKinematics):

    tracks = vertex.daughterPtrVector()
    for track in tracks:
        mytrack = track.bestTrack()
        vtxKinematics.add(mytrack, 1.0)

njettiness_08 = ROOT.fastjet.contrib.Njettiness(ROOT.fastjet.contrib.OnePass_KT_Axes(), 
                                                  ROOT.fastjet.contrib.NormalizedMeasure(1.0, 0.8))

njettiness_15 = ROOT.fastjet.contrib.Njettiness(ROOT.fastjet.contrib.OnePass_KT_Axes(), 
                                                  ROOT.fastjet.contrib.NormalizedMeasure(1.0, 1.5))

# Only needed if we want the re-calculate the bb-tag for cross-checking
#mvaID_ca15 = ROOT.MvaBoostedDoubleSecondaryVertexEstimator("/shome/gregor/VHBB-743/CMSSW_7_4_3_patch1/src/RecoBTag/SecondaryVertex/data/BoostedDoubleSV_CA15_BDT.weights.xml.gz")
#mvaID_ak08 = ROOT.MvaBoostedDoubleSecondaryVertexEstimator("/shome/gregor/VHBB-743/CMSSW_7_4_3_patch1/src/RecoBTag/SecondaryVertex/data/BoostedDoubleSV_AK8_BDT.weights.xml.gz")


# We need to instantiate a track selector object
# The construcor needs an EDM Parameter Set, so we construct that
# first.
# To do this: build a dictionary with that also contains the name of
# the c++ type and then call the templated addParameter function

# From:
# https://github.com/cms-btv-pog/cmssw/blob/BoostedDoubleSVTaggerV2_from-CMSSW_7_4_15/RecoBTag/SecondaryVertex/python/trackSelection_cff.py
# https://github.com/cms-btv-pog/cmssw/blob/BoostedDoubleSVTaggerV2_from-CMSSW_7_4_15/RecoBTag/ImpactParameter/python/variableJTA_cff.py
params = {    
    "totalHitsMin"   : [0, "unsigned int"],
    "jetDeltaRMax"   : [0.3, "double"],
    "qualityClass" :   ["any", "std::string"],
    "pixelHitsMin"   : [0, "unsigned int"],
    "maxDistToAxis"  : [0.07, "double"],
    "maxDecayLen"    : [5., "double"],
    "sip3dSigMin"    : [-99999.9, "double"],
    "sip3dSigMax"    : [99999.9 , "double"],
    "sip2dValMax"    : [99999.9 , "double"],
    "ptMin"          : [0.0, "double"],
    "sip2dSigMax"    : [99999.9, "double"],
    "sip2dSigMin"    : [-99999.9, "double"],
    "sip3dValMax"    : [99999.9, "double"],
    "sip3dValMin"    : [-99999.9, "double"],
    "sip2dValMin"    : [-99999.9, "double"],
    "normChi2Max"    : [99999.9, "double"],
    "useVariableJTA" : [False, "bool"],

    "a_dR" : [-0.001053, "double"],
    "b_dR" : [0.6263, "double"],
    "a_pT" : [0.005263, "double"],
    "b_pT" : [0.3684, "double"],
    "min_pT" : [120, "double"],
    "max_pT" : [500, "double"],
    "min_pT_dRcut" :  [0.5, "double"], 
    "max_pT_dRcut" :  [ 0.1, "double"],
    "max_pT_trackPTcut" : [ 3, "double"],

}

# Build the EDM Parameter set from dictionary
pset = ROOT.edm.ParameterSet()
for k,v in params.iteritems():
    pset.addParameter(v[1])(k, v[0])

# Alas, the TrackSelector
trackSelector = ROOT.reco.TrackSelector(pset)

# Also need a trackPairV0Filter
pset_v0filt = ROOT.edm.ParameterSet()
pset_v0filt.addParameter("double")("k0sMassWindow",0.03)
trackPairV0Filter = ROOT.reco.V0Filter(pset_v0filt)

# get TransientTrackBuilder
#  const TransientTrackRecord & transientTrackRcd = record.getRecord<TransientTrackRecord>();
#  transientTrackRcd.get("TransientTrackBuilder", trackBuilder);

# Helper function to calculate Hbb tagging input variables
def calcBBTagVariables(jet, 
                       orig_jet,
                       muonTagInfos, 
                       elecTagInfos, 
                       ipTagInfo, 
                       svTagInfo,
                       njettiness,
                       maxSVDeltaRToJet):


    DEBUG = False

    #default discriminator value
    value = -10.

    # define default values for input variables
    dummyZ_ratio             = -3.0
    dummyTrackSip3dSig       = -50.0
    dummyTrackSip2dSigAbove  = -19.0
    dummyTrackEtaRel         = -1.0
    dummyVertexMass          = -1.0
    dummyVertexEnergyRatio   = -1.0
    dummyVertexDeltaR        = -1.0
    dummyFlightDistance2dSig = -1.0

    # default variable values
    z_ratio = dummyZ_ratio
    trackSip3dSig_3 = trackSip3dSig_2 = trackSip3dSig_1 =  trackSip3dSig_0 = dummyTrackSip3dSig
    tau2_trackSip3dSig_0 =  tau1_trackSip3dSig_0 =  tau2_trackSip3dSig_1 =  tau1_trackSip3dSig_1 = dummyTrackSip3dSig
    trackSip2dSigAboveCharm_0 =  trackSip2dSigAboveBottom_0 = trackSip2dSigAboveBottom_1 = dummyTrackSip2dSigAbove
    tau1_trackEtaRel_0 =  tau1_trackEtaRel_1 = tau1_trackEtaRel_2 = dummyTrackEtaRel
    tau2_trackEtaRel_0 =  tau2_trackEtaRel_1 = tau2_trackEtaRel_2 = dummyTrackEtaRel
    tau1_vertexMass = tau1_vertexEnergyRatio =  tau1_vertexDeltaR = tau1_flightDistance2dSig = dummyFlightDistance2dSig
    tau2_vertexMass = tau2_vertexEnergyRatio =  tau2_vertexDeltaR = tau2_flightDistance2dSig = dummyFlightDistance2dSig
    jetNTracks = 0
    nSV = 0
    tau1_nSecondaryVertices = 0
    tau2_nSecondaryVertices = 0

    # Re-calculate N-subjettiness independently and get axees
    fjParticles = ROOT.std.vector("fastjet::PseudoJet")()
    for dau in jet.daughterPtrVector():
        if dau.isNonnull() and dau.isAvailable():
            fjParticles.push_back( ROOT.fastjet.PseudoJet(dau.px(), dau.py(), dau.pz(), dau.energy()))

    tau1 = njettiness.getTau(1, fjParticles)
    tau2 = njettiness.getTau(2, fjParticles)
    currentAxes = njettiness.currentAxes()

    if not tau1 == 0:
        tau_21 = tau2/tau1

    selectedTracks = ipTagInfo.selectedTracks()
    trackSize = len(selectedTracks)
    vertexRef = ipTagInfo.primaryVertex()



    vertexRef = ipTagInfo.primaryVertex()
    if vertexRef.isNonnull():
        pv = ROOT.GlobalPoint(vertexRef.x(),
                         vertexRef.y(),
                         vertexRef.z())
    else:
        pv = ROOT.GlobalPoint(0.,0.,0.)

    ipData = ipTagInfo.impactParameterData()

    allKinematics = ROOT.reco.TrackKinematics()

    if DEBUG:
        print "trackSize=", trackSize
        
    IP3Ds   = []
    IP3Ds_1 = []
    IP3Ds_2 = []
    contTrk = 0
    
    # loop over tracks associated to the jet
    for itt in range(trackSize):
        
        ptrackRef = selectedTracks[itt]
        ptrack = ROOT.reco.btag.toTrack(selectedTracks[itt])

        #const reco::CandidatePtr ptrackRef = selectedTracks[itt];
        #const reco::Track * ptrackPtr = reco::btag::toTrack(ptrackRef);
        #const reco::Track & ptrack = *ptrackPtr;

        # Assume MiniAOD and do setTracksPV(ptrackRef, vertexRef, track_PVweight)
        # directly
        track_PVweight = 0.
        pcand = ptrackRef.get()
        if (pcand.fromPV() == ROOT.pat.PackedCandidate.PVUsedInFit):
            track_PVweight = 1.

        if (track_PVweight>0.5):
            allKinematics.add(ptrack, track_PVweight)

        data = ipData[itt]
        isSelected = False



        if (trackSelector(ptrack, data, orig_jet, pv)):
            isSelected = True

        # check if the track is from V0
        isfromV0 = False
        isfromV0Tight = False


        trackPairV0Test = [None, None]
        trackPairV0Test[0] = ROOT.reco.btag.toTrack(ptrackRef)

        for jtt in range(trackSize):
 
            if (itt == jtt):
                continue
 
            pairTrackData = ipData[jtt]
            pairTrackRef = selectedTracks[jtt]
            pairTrackPtr = ROOT.reco.btag.toTrack(pairTrackRef)
            pairTrack = pairTrackPtr
 
            trackPairV0Test[1] = pairTrackPtr
            
            # Convert python list to std vector
            trackPairV0Test_vec = ROOT.std.vector("reco::Track")()
            for x in trackPairV0Test:
                trackPairV0Test_vec.push_back(x)

            if ( not trackPairV0Filter(trackPairV0Test_vec)):
                isfromV0 = True
                if trackSelector(pairTrack, pairTrackData, orig_jet, pv):
                    isfromV0Tight = True
 
            if isfromV0 and isfromV0Tight:
                break
        # End of second track loop


        if isSelected and not isfromV0Tight :
            jetNTracks += 1.
            
        # Not available!    
        # reco::TransientTrack transientTrack = trackBuilder->build(ptrack);
        #GlobalVector direction(jet->px(), jet->py(), jet->pz());

        # int index = 0;
        # if (currentAxes.size() > 1 && reco::deltaR2(ptrack,currentAxes[1]) < reco::deltaR2(ptrack,currentAxes[0])):
        #     index = 1;
        # direction = GlobalVector(currentAxes[index].px(), currentAxes[index].py(), currentAxes[index].pz());
        #
        # # decay distance and track distance wrt to the closest tau axis
        # float decayLengthTau=-1;
        # float distTauAxis=-1;
        #
        # TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(transientTrack.impactPointState(), *vertexRef , direction, transientTrack.field());
        # if (closest.isValid()):
        #    decayLengthTau =  (closest.globalPosition() - RecoVertex::convertPos(vertexRef->position())).mag();
        #
        # distTauAxis = std::abs(IPTools::jetTrackDistance(transientTrack, direction, *vertexRef ).second.value());
        # float IP3Dsig = ipTagInfo.impactParameterData()[itt].ip3d.significance();
        #
        # if( !isfromV0 && decayLengthTau<maxDecayLen_ && distTauAxis<maxDistToAxis_ )
        # {
        #   IP3Ds.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
        #   ++contTrk;
        #   if (currentAxes.size() > 1)
        #   {
        #     if (reco::deltaR2(ptrack,currentAxes[0]) < reco::deltaR2(ptrack,currentAxes[1]))
        #       IP3Ds_1.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
        #     else
        #       IP3Ds_2.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
        #   }
        #   else
        #     IP3Ds_1.push_back( IP3Dsig<-50. ? -50. : IP3Dsig );
        # }
        # 

    # End of loop over tracks

    jetDir = jet.momentum().Unit()
    tau1Kinematics = ROOT.reco.TrackKinematics()
    tau2Kinematics = ROOT.reco.TrackKinematics()
    tau1_trackEtaRels = []
    tau2_trackEtaRels = []
    VTXmap            = {}
    
    for vtx in range(svTagInfo.nVertices()):

        vertexKinematic = ROOT.reco.TrackKinematics()

        # get the vertex kinematics
        vertex = svTagInfo.secondaryVertex(vtx)
        vertexKinematics(vertex, vertexKinematic)

        if len(currentAxes) > 1:
        
                if math.sqrt(deltaR2(svTagInfo.flightDirection(vtx),currentAxes[1])) < math.sqrt(deltaR2(svTagInfo.flightDirection(vtx),currentAxes[0])) :
                
                        tau2Kinematics = tau2Kinematics + vertexKinematic
                        if tau2_flightDistance2dSig < 0 :
                        
                          tau2_flightDistance2dSig = svTagInfo.flightDistance(vtx,True).significance()
                          tau2_vertexDeltaR = math.sqrt(deltaR2(svTagInfo.flightDirection(vtx),currentAxes[1]))
                        
                        etaRelToTauAxis(vertex, currentAxes[1], tau2_trackEtaRels)
                        tau2_nSecondaryVertices += 1.
                
                else:
                
                        tau1Kinematics = tau1Kinematics + vertexKinematic
                        if tau1_flightDistance2dSig < 0 :
                        
                          tau1_flightDistance2dSig =svTagInfo.flightDistance(vtx,True).significance()
                          tau1_vertexDeltaR = math.sqrt(deltaR2(svTagInfo.flightDirection(vtx),currentAxes[0]))
                        
                        etaRelToTauAxis(vertex, currentAxes[0], tau1_trackEtaRels)
                        tau1_nSecondaryVertices += 1.
                

        
        elif len(currentAxes) > 0:
        
                tau1Kinematics = tau1Kinematics + vertexKinematic
                if tau1_flightDistance2dSig < 0 :
                
                  tau1_flightDistance2dSig =svTagInfo.flightDistance(vtx,True).significance()
                  tau1_vertexDeltaR = math.sqrt(deltaR2(svTagInfo.flightDirection(vtx),currentAxes[0]))
                
                etaRelToTauAxis(vertex, currentAxes[1], tau1_trackEtaRels)
                tau1_nSecondaryVertices += 1.
        

        flightDir = svTagInfo.flightDirection(vtx);
        if (math.sqrt(deltaR2(flightDir, jetDir)<(0.7*0.7))) :
          VTXmap[svTagInfo.flightDistance(vtx).error()]=vtx
  # end loop over vertices
    nSV = len(VTXmap)

    #allSum=ROOT.math.XYZTLorentzVector()
    allSum = allKinematics.weightedVectorSum()
    if tau1_nSecondaryVertices > 0. :
     #tau1_vertexSum=ROOT.math.XYZTLorentzVector()	
      tau1_vertexSum = tau1Kinematics.weightedVectorSum()
      tau1_vertexEnergyRatio = tau1_vertexSum.E() / allSum.E() if allSum.E() > 0 else 0.
      if tau1_vertexEnergyRatio > 50. :
		 tau1_vertexEnergyRatio = 50.	

      tau1_vertexMass = tau1_vertexSum.M()
  

    if  tau2_nSecondaryVertices > 0. :
     tau2_vertexSum=ROOT.math.XYZTLorentzVector()  
     tau2_vertexSum = tau2Kinematics.weightedVectorSum()
     tau2_vertexEnergyRatio = tau2_vertexSum.E() / allSum.E() if allSum.E() > 0 else 0.
     if tau2_vertexEnergyRatio > 50. :
	 tau2_vertexEnergyRatio = 50.

     tau2_vertexMass= tau2_vertexSum.M()
  


    dummyEtaRel = -1.;

    #std::sort( tau1_trackEtaRels.begin(),tau1_trackEtaRels.end() );
    #std::sort( tau2_trackEtaRels.begin(),tau2_trackEtaRels.end() );
    tau1_trackEtaRels.sort()	
    tau2_trackEtaRels.sort()
 
    if len(tau1_trackEtaRels) >2  :
      tau1_trackEtaRel_0 = tau1_trackEtaRels[0]
      tau1_trackEtaRel_1 = tau1_trackEtaRels[1]
      tau1_trackEtaRel_2 = tau1_trackEtaRels[2]
    elif len(tau1_trackEtaRels) ==0 : 	
      tau1_trackEtaRel_0 = dummyEtaRel
      tau1_trackEtaRel_1 = dummyEtaRel
      tau1_trackEtaRel_2 = dummyEtaRel	
    elif len(tau1_trackEtaRels) ==1:
      tau1_trackEtaRel_0 = tau1_trackEtaRels[0]
      tau1_trackEtaRel_1 = dummyEtaRel
      tau1_trackEtaRel_2 = dummyEtaRel
    elif len(tau1_trackEtaRels) ==2: 
      tau1_trackEtaRel_0 = tau1_trackEtaRels[0]
      tau1_trackEtaRel_1 = tau1_trackEtaRels[1]
      tau1_trackEtaRel_2 = dummyEtaRel
     
    if len(tau2_trackEtaRels) >2 :
       tau2_trackEtaRel_0 = tau2_trackEtaRels[0]
       tau2_trackEtaRel_1 = tau2_trackEtaRels[1]
       tau2_trackEtaRel_2 = tau2_trackEtaRels[2]
    elif len(tau2_trackEtaRels) == 0:
       tau2_trackEtaRel_0 = dummyEtaRel
       tau2_trackEtaRel_1 = dummyEtaRel
       tau2_trackEtaRel_2 = dummyEtaRel
    elif len(tau2_trackEtaRels) == 1:
       tau2_trackEtaRel_0 = tau2_trackEtaRels[0]
       tau2_trackEtaRel_1 = dummyEtaRel
       tau2_trackEtaRel_2 = dummyEtaRel
    elif len(tau2_trackEtaRels) == 2:
       tau2_trackEtaRel_0 = tau2_trackEtaRels[0]
       tau2_trackEtaRel_1 = tau2_trackEtaRels[1]
       tau2_trackEtaRel_2 = dummyEtaRel

    
    cont = 0
		 
    for vtx in (sorted(VTXmap.keys())):
       cont += 1
       index = VTXmap[vtx]
       vertex = svTagInfo.secondaryVertex(index)
 
       if cont==1:
	   flightDir_0 = svTagInfo.flightDirection(index)
           SV_p4_0 = vertex.p4()
           vtxMass = SV_p4_0.mass()

           if vtxMass > 0.:
              z_ratio = math.sqrt(deltaR2(currentAxes[1],currentAxes[0]))*SV_p4_0.pt()/vtxMass
 
       if cont==2:         
           flightDir_1 = svTagInfo.flightDirection(index)
           SV_p4_1 = vertex.p4()
           vtxMass = (SV_p4_1+SV_p4_0).mass()
            
           if vtxMass > 0.:
	        z_ratio = math.sqrt(deltaR2(flightDir_0,flightDir_1))*SV_p4_1.pt()/vtxMass
   
           break


  # when only one tau axis has SVs assigned, they are all assigned to the 1st tau axis
  # in the special case below need to swap values
    if tau1_vertexMass<0 and tau2_vertexMass>0 :
  
     temp = tau1_trackEtaRel_0
     tau1_trackEtaRel_0= tau2_trackEtaRel_0
     tau2_trackEtaRel_0= temp

     temp = tau1_trackEtaRel_1
     tau1_trackEtaRel_1= tau2_trackEtaRel_1
     tau2_trackEtaRel_1= temp

     temp = tau1_trackEtaRel_2
     tau1_trackEtaRel_2= tau2_trackEtaRel_2
     tau2_trackEtaRel_2= temp

     temp = tau1_flightDistance2dSig
     tau1_flightDistance2dSig= tau2_flightDistance2dSig;
     tau2_flightDistance2dSig= temp

     tau1_vertexDeltaR= tau2_vertexDeltaR

     temp = tau1_vertexEnergyRatio
     tau1_vertexEnergyRatio= tau2_vertexEnergyRatio
     tau2_vertexEnergyRatio= temp

     temp = tau1_vertexMass
     tau1_vertexMass= tau2_vertexMass
     tau2_vertexMass= temp
  
    nSM = muonTagInfos.leptons()
    nSE = elecTagInfos.leptons()
    nSL = nSM + nSE

    if DEBUG:
        print "SL SM SE", nSL, nSM, nSE

    # Lepton pT-rel and IP2D
    PFLepton_ptrel = -1.
    PFLepton_IP2D  = -1.

  # PFMuon information
    for i_lepton in range(muonTagInfos.leptons()):

        PFMuon_ptrel  = muonTagInfos.properties(i_lepton).ptRel

        if (PFMuon_ptrel > PFLepton_ptrel ):
            PFLepton_ptrel = PFMuon_ptrel
            PFLepton_IP2D  = muonTagInfos.properties(i_lepton).sip2d
            # End of loop over leptons

    # PFElectron information
    for i_lepton in range(elecTagInfos.leptons()):

        PFElectron_ptrel  = elecTagInfos.properties(i_lepton).ptRel

        if (PFElectron_ptrel > PFLepton_ptrel ):
            PFLepton_ptrel = PFElectron_ptrel
            PFLepton_IP2D  = elecTagInfos.properties(i_lepton).sip2d
    # End of loop over leptons

##    #jet.bbtag_recalc = mvaID.mvaValue(PFLepton_ptrel, 
##    #                                  z_ratio, 
##    #                                  tau_dot, 
##    #                                  SV_mass_0, 
##    #                                  SV_EnergyRatio_0, 
##    #                                  SV_EnergyRatio_1, 
##    #                                  PFLepton_IP2D, 
##    #                                  tau_21, 
##    #                                  nSL, 
##    #                                  vertexNTracks)      
##    




    jet.PFLepton_ptrel       = PFLepton_ptrel
    jet.PFLepton_IP2D        = PFLepton_IP2D
    jet.nSL                  = nSL
 
    jet.z_ratio = z_ratio
    #jet.trackSip3dSig_3 = trackSip3dSig_3
    #jet.trackSip3dSig_2 = trackSip3dSig_2
    #jet.trackSip3dSig_1 = trackSip3dSig_1
    #jet.trackSip3dSig_0 = trackSip3dSig_0
    #jet.tau2_trackSip3dSig_0 = tau2_trackSip3dSig_0
    #jet.tau1_trackSip3dSig_0 = tau1_trackSip3dSig_0
    #jet.tau2_trackSip3dSig_1 = tau2_trackSip3dSig_1
    #jet.tau1_trackSip3dSig_1 = tau1_trackSip3dSig_1
    #jet.trackSip2dSigAboveCharm_0 = trackSip2dSigAboveCharm_0
    #jet.trackSip2dSigAboveBottom_0 = trackSip2dSigAboveBottom_0
    #jet.trackSip2dSigAboveBottom_1 = trackSip2dSigAboveBottom_1
    jet.tau2_trackEtaRel_0 = tau2_trackEtaRel_0
    jet.tau2_trackEtaRel_1 = tau2_trackEtaRel_1
    jet.tau2_trackEtaRel_2 = tau2_trackEtaRel_2
    jet.tau1_trackEtaRel_0 = tau1_trackEtaRel_0
    jet.tau1_trackEtaRel_1 = tau1_trackEtaRel_1
    jet.tau1_trackEtaRel_2 = tau1_trackEtaRel_2
    jet.tau1_vertexMass = tau1_vertexMass
    jet.tau1_vertexEnergyRatio = tau1_vertexEnergyRatio
    jet.tau1_vertexDeltaR = tau1_vertexDeltaR
    jet.tau1_flightDistance2dSig = tau1_flightDistance2dSig
    jet.tau2_vertexMass = tau2_vertexMass
    jet.tau2_vertexEnergyRatio = tau2_vertexEnergyRatio
    jet.tau2_flightDistance2dSig = tau2_flightDistance2dSig
    jet.jetNTracks = jetNTracks
    jet.nSV = nSV

# End of calcBBTagVariables

def calcBBTagVariables_dummy(jet):
    dummy_val = -9999

    jet.PFLepton_ptrel       = dummy_val
    jet.PFLepton_IP2D        = dummy_val
    jet.nSL                  = dummy_val

    jet.z_ratio                  = dummy_val
    jet.tau2_trackEtaRel_0       = dummy_val
    jet.tau2_trackEtaRel_1       = dummy_val
    jet.tau2_trackEtaRel_2       = dummy_val
    jet.tau1_trackEtaRel_0       = dummy_val
    jet.tau1_trackEtaRel_1       = dummy_val
    jet.tau1_trackEtaRel_2       = dummy_val
    jet.tau1_vertexMass          = dummy_val
    jet.tau1_vertexEnergyRatio   = dummy_val
    jet.tau1_vertexDeltaR        = dummy_val
    jet.tau1_flightDistance2dSig = dummy_val
    jet.tau2_vertexMass          = dummy_val
    jet.tau2_vertexEnergyRatio   = dummy_val
    jet.tau2_flightDistance2dSig = dummy_val
    jet.jetNTracks               = dummy_val
    jet.nSV                      = dummy_val

    #jet.PFLepton_ptrel       = dummy_val
    #jet.z_ratio              = dummy_val
    #jet.tau_dot              = dummy_val
    #jet.SV_mass_0            = dummy_val
    #jet.SV_EnergyRatio_0     = dummy_val
    #jet.SV_EnergyRatio_1     = dummy_val
    #jet.PFLepton_IP2D        = dummy_val
    #jet.tau_21               = dummy_val
    #jet.nSL                  = dummy_val
    #jet.vertexNTracks        = dummy_val
    
    
class AdditionalBoost( Analyzer ):

    def __init__(self, cfg_ana, cfg_comp, looperName):
        
        super(AdditionalBoost,self).__init__(cfg_ana, cfg_comp, looperName)
        
        # Get the config parameters
        skip_ca15 = cfg_ana.skip_ca15 if hasattr(cfg_ana,'skip_ca15') else False
        GT        = cfg_ana.GT if hasattr(cfg_ana,'GT')   else "Summer15_25nsV6_DATA"
        jecPath   = cfg_ana.jecPath if hasattr(cfg_ana,'jecPath') else "."
        isMC      = cfg_ana.isMC if hasattr(cfg_ana,'isMC') else False

        self.skip_ca15 = skip_ca15

        # Prepare re-calibrator
        recalibrationTypeAK8 = "AK8PFchs"        
        recalibrationTypeAK4 = "AK4PFchs"        

        # Following instructions from:
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetWtagging
        # L2L3
        if isMC:
            doResidual = False
        # L2L3 Residual
        else:
            doResidual = True


        self.jetReCalibratorAK8L2L3 = JetReCalibrator(GT,
                                                      recalibrationTypeAK8, 
                                                      doResidual, 
                                                      jecPath,
                                                      skipLevel1=True)

	self.jetReCalibratorAK8L1L2L3 = JetReCalibrator(GT,
                                                       recalibrationTypeAK8,
                                                       doResidual,
                                                       jecPath,
                                                       skipLevel1=False)


        self.jetReCalibratorAK4 = JetReCalibrator(GT,
                                                  recalibrationTypeAK4, 
                                                  doResidual, 
                                                  jecPath,
                                                  skipLevel1=False)



    
    def declareHandles(self):
        super(AdditionalBoost, self).declareHandles()
        
        self.handles['rho'] = AutoHandle( ('fixedGridRhoFastjetAll',""), 'double' )

        self.handles['ak08']     = AutoHandle( ("slimmedJetsAK8",""), "std::vector<pat::Jet>")

        self.handles['ak08softdropsubjets'] = AutoHandle( ("slimmedJetsAK8PFCHSSoftDropPacked","SubJets"), "std::vector<pat::Jet>")

        self.handles['ak08bbtag'] = AutoHandle( ("slimmedJetsAK8pfBoostedDoubleSecondaryVertexBJetTags","","EX"), 
                                                "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")
        self.handles['ak08ipTagInfos']     = AutoHandle( ("slimmedJetsAK8ImpactParameterTagInfos","","EX"), "vector<reco::IPTagInfo<vector<edm::Ptr<reco::Candidate> >,reco::JetTagInfo> >")

        self.handles['ak08svTagInfos']     = AutoHandle( ("slimmedJetsAK8pfInclusiveSecondaryVertexFinderTagInfos", "","EX"), "vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<vector<edm::Ptr<reco::Candidate> >,reco::JetTagInfo>,reco::VertexCompositePtrCandidate> >")

        self.handles['ak08muonTagInfos']     = AutoHandle( ("slimmedJetsAK8softPFMuonsTagInfos", "","EX"), "vector<reco::TemplatedSoftLeptonTagInfo<edm::Ptr<reco::Candidate> > >")

        self.handles['ak08elecTagInfos']     = AutoHandle( ("slimmedJetsAK8softPFElectronsTagInfos", "","EX"), "vector<reco::TemplatedSoftLeptonTagInfo<edm::Ptr<reco::Candidate> > >")


        if not self.skip_ca15:
        
            self.handles['ca15ipTagInfos']     = AutoHandle( ("ca15PFJetsCHSImpactParameterTagInfos","","EX"), "vector<reco::IPTagInfo<vector<edm::Ptr<reco::Candidate> >,reco::JetTagInfo> >")

            self.handles['ca15svTagInfos']     = AutoHandle( ("ca15PFJetsCHSpfInclusiveSecondaryVertexFinderTagInfos", "","EX"), "vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<vector<edm::Ptr<reco::Candidate> >,reco::JetTagInfo>,reco::VertexCompositePtrCandidate> >")

            self.handles['ca15muonTagInfos']     = AutoHandle( ("ca15PFJetsCHSsoftPFMuonsTagInfos", "","EX"), "vector<reco::TemplatedSoftLeptonTagInfo<edm::Ptr<reco::Candidate> > >")

            self.handles['ca15elecTagInfos']     = AutoHandle( ("ca15PFJetsCHSsoftPFElectronsTagInfos", "","EX"), "vector<reco::TemplatedSoftLeptonTagInfo<edm::Ptr<reco::Candidate> > >")

            self.handles['ca15ungroomed']           = AutoHandle( ("ca15PFJetsCHS","","EX"), "std::vector<reco::PFJet>")
            self.handles['ca15trimmed']             = AutoHandle( ("ca15PFTrimmedJetsCHS","","EX"), "std::vector<reco::PFJet>")
            self.handles['ca15pruned']              = AutoHandle( ("ca15PFPrunedJetsCHS","","EX"), "std::vector<reco::BasicJet>")
            self.handles['ca15softdrop']            = AutoHandle( ("ca15PFSoftdropJetsCHS","","EX"), "std::vector<reco::BasicJet>")
            self.handles['ca15softdropz2b1']        = AutoHandle( ("ca15PFSoftdropZ2B1JetsCHS","","EX"), "std::vector<reco::BasicJet>")

            self.handles['ca15subjetfiltered']        = AutoHandle( ("ca15PFSubjetFilterCHS","filtercomp","EX"), "std::vector<reco::BasicJet>")

            self.handles['ca15prunedsubjets']       = AutoHandle( ("ca15PFPrunedJetsCHS","SubJets","EX"), "std::vector<reco::PFJet>")
            self.handles['ca15softdropsubjets']     = AutoHandle( ("ca15PFSoftdropJetsCHS","SubJets","EX"), "std::vector<reco::PFJet>")
            self.handles['ca15softdropz2b1subjets'] = AutoHandle( ("ca15PFSoftdropZ2B1JetsCHS","SubJets","EX"), "std::vector<reco::PFJet>")

            # we call them subjets, even though they are technically the filterjets in BDRS lingo
            self.handles['ca15subjetfilteredsubjets']  = AutoHandle( ("ca15PFSubjetFilterCHS","filter","EX"), "std::vector<reco::PFJet>")

            self.handles['ca15tau1'] = AutoHandle( ("ca15PFJetsCHSNSubjettiness","tau1","EX"), "edm::ValueMap<float>")
            self.handles['ca15tau2'] = AutoHandle( ("ca15PFJetsCHSNSubjettiness","tau2","EX"), "edm::ValueMap<float>")
            self.handles['ca15tau3'] = AutoHandle( ("ca15PFJetsCHSNSubjettiness","tau3","EX"), "edm::ValueMap<float>")

            self.handles['ca15softdropz2b1tau1'] = AutoHandle( ("ca15PFSoftdropZ2B1JetsCHSNSubjettiness","tau1","EX"), "edm::ValueMap<float>")
            self.handles['ca15softdropz2b1tau2'] = AutoHandle( ("ca15PFSoftdropZ2B1JetsCHSNSubjettiness","tau2","EX"), "edm::ValueMap<float>")
            self.handles['ca15softdropz2b1tau3'] = AutoHandle( ("ca15PFSoftdropZ2B1JetsCHSNSubjettiness","tau3","EX"), "edm::ValueMap<float>")

            self.handles['httCandJets']  = AutoHandle( ("looseOptRHTT","","EX"), "std::vector<reco::BasicJet>")
            self.handles['httCandInfos'] = AutoHandle( ("looseOptRHTT","","EX"), "vector<reco::HTTTopJetTagInfo>")


            self.handles['httSubjetBtags'] = AutoHandle( ("looseOptRHTTpfCombinedInclusiveSecondaryVertexV2BJetTags","","EX"), 
                                                         "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")
            

            self.handles['ca15bbtag'] = AutoHandle( ("ca15PFJetsCHSpfBoostedDoubleSecondaryVertexBJetTags","","EX"), 
                                                    "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")

            self.handles['ca15prunedsubjetbtag'] = AutoHandle( ("ca15PFPrunedJetsCHSpfCombinedInclusiveSecondaryVertexV2BJetTags","","EX"), 
                                                               "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")

            self.handles['ca15softdropsubjetbtag'] = AutoHandle( ("ca15PFSoftdropJetsCHSpfCombinedInclusiveSecondaryVertexV2BJetTags","","EX"), 
                                                               "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")

            self.handles['ca15softdropz2b1subjetbtag'] = AutoHandle( ("ca15PFSoftdropZ2B1JetsCHSpfCombinedInclusiveSecondaryVertexV2BJetTags","","EX"), 
                                                               "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")

            self.handles['ca15subjetfilteredsubjetbtag'] = AutoHandle( ("ca15PFSubjetFilterCHSpfCombinedInclusiveSecondaryVertexV2BJetTags","","EX"), 
                                                                       "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")


    def process(self, event):
 
        run = event.input.eventAuxiliary().id().run()
        lumi = event.input.eventAuxiliary().id().luminosityBlock()
        eventId = event.input.eventAuxiliary().id().event()
        
        self.readCollections( event.input )
        
        # Will need who for jet calibration later
        rho =  self.handles["rho"].product()[0]

        ######## 
        # Softdrop Fatjets + NSubjettiness
        ########

        for fj_name in ["ca15softdropz2b1"]:

            if self.skip_ca15 and ("ca15" in fj_name):
                continue
                
            # Set the four-vector
            setattr(event, fj_name, map(PhysicsObject, self.handles[fj_name].product()))

            # N-Subjettiness
            tau1 = self.handles[fj_name+'tau1'].product()
            tau2 = self.handles[fj_name+'tau2'].product()
            tau3 = self.handles[fj_name+'tau3'].product()

            # Loop over jets                        
            for ij, jet in enumerate(getattr(event, fj_name)):

                # Fill N-Subjettiness
                jet.tau1 = tau1.get(ij)
                jet.tau2 = tau2.get(ij)
                jet.tau3 = tau3.get(ij)
                                    
            # end of loop over jets



                                                                
        ######## 
        # Groomed Uncalibrated Fatjets
        ########

        for fj_name in ['ca15trimmed', 'ca15softdrop', 'ca15pruned', 'ca15subjetfiltered']:            
                setattr(event, fj_name, map(PhysicsObject, self.handles[fj_name].product()))

#
#        ######## 
#        # Groomed Fatjets to calibrate
#        ########
#
#        pruned_cal_jets = []
#
#        for groomed_fj in self.handles['ak08pruned'].product():                        
#
#            # We need the closest ungroomed fatjet to get the JEC:            
#            # - Make a list of pairs: deltaR(ungroomed fj, groomed fj) for all ungroomed fatjets
#            # - Sort by deltaR
#            # - And take the minimum
#            
#            if len(getattr(event, "ak08")):
#                closest_ung_fj_and_dr = sorted(
#                    [(ung_fj, deltaR2(ung_fj, groomed_fj)) for ung_fj in getattr(event, "ak08")], 
#                    key=lambda x:x[1])[0]
#            else:
#                print "WARNING: No ungroomed fatjets found in event with groomed fatjet. Skipping"
#                continue
#
#            # Use the jet cone size for matching
#            minimal_dr_groomed_ungroomed = 0.8
#            if closest_ung_fj_and_dr[1] > minimal_dr_groomed_ungroomed:
#                print "WARNING: No ungroomed fatjet found close to groomed fatjet. Skipping"
#                continue
#
#            ungroomed_jet = Jet(closest_ung_fj_and_dr[0])        
#
#            c = self.jetReCalibratorAK8L2L3.getCorrection(ungroomed_jet, rho)
#
#                        
#            # Need to do a deep-copy. Otherwise the original jet will be modified
#            cal_groomed_fj = PhysicsObject(groomed_fj).__copy__() 
#            cal_groomed_fj.scaleEnergy(c)
#            
#            pruned_cal_jets.append(cal_groomed_fj)
#
#        setattr(event, 'ak08prunedcal', pruned_cal_jets)
#



            
        ######## 
        # Subjets 
        ########

        for fj_name in ['ca15pruned', 'ca15softdrop', 'ca15softdropz2b1', 'ca15subjetfiltered']:

            if self.skip_ca15 and ("ca15" in fj_name):
                continue

            # Get the 4-vectors
            setattr(event, fj_name + "subjets", map(PhysicsObject, self.handles[fj_name+"subjets"].product()))
            
            # Add b-tag information
            newtags =  self.handles[fj_name+'subjetbtag'].product()
            for i in xrange(0,len(newtags)) :
                for j in getattr(event, fj_name+"subjets"):
                    if  j.physObj == newtags.key(i).get():
                        j.btag = newtags.value(i)                        
                        # Map dummy value to -0.1
                        if j.btag == -10.:
                            j.btag = -0.1
                                                    
            # Add information from which FJ the subjet comes
            # Loop over subjets
            for j in getattr(event, fj_name+"subjets"):

                j.fromFJ = -1

                # Loop over fatjets
                for i_fj, fj in enumerate(getattr(event, fj_name)):

                    # Loop over daughters (and see if they correspond to the subjet)
                    # (Unfortunately the object == fails, so we have to use kinematics)
                    for i_daughter in range(fj.numberOfDaughters()):

                        if not fj_name == "ca15subjetfiltered":
                            daughter = fj.daughter(i_daughter)
                        else:
                            daughter = fj.daughterPtr(i_daughter).get()                            

                        if (daughter.pt() == j.pt() and 
                            daughter.eta() == j.eta() and 
                            daughter.phi() == j.phi() and
                            daughter.mass() == j.mass()):

                            j.fromFJ = i_fj
                            break

                    if j.fromFJ > -1:
                        break

            # Calibrate subjets
            for j in getattr(event, fj_name+"subjets"):

                # Calibrate the subjet
                sj_uncal = Jet(j)        
                # isHttSubjet is set to true
                # it can be used for all jets that are uncalibrated and have no RAW factor
                cal = self.jetReCalibratorAK4.getCorrection(sj_uncal, rho, isHttSubjet=True)            
                
                j.scaleEnergy(cal)
 

            # Add jetID information 
            for j in getattr(event, fj_name+"subjets"):
                j.jetID = passesJetId(j)


            
            # Sort subjets by pT
            setattr(event, fj_name + "subjets", sorted(getattr(event, fj_name + "subjets"), key = lambda x:-x.pt()))


        ######## 
        # HEPTopTagger
        ########

        if not self.skip_ca15:
            candJets = self.handles['httCandJets'].product()
            candInfos = self.handles['httCandInfos'].product()

            event.httCandidates = map(PhysicsObject, candJets) 
            sjbtags = self.handles['httSubjetBtags'].product()

            for i in xrange(0, len(candJets)):            

                event.httCandidates[i].fRec = candInfos[i].properties().fRec
                event.httCandidates[i].Ropt = candInfos[i].properties().Ropt
                event.httCandidates[i].RoptCalc = candInfos[i].properties().RoptCalc
                event.httCandidates[i].ptForRoptCalc = candInfos[i].properties().ptForRoptCalc

                # HTT return the subjet-pair closest to the W-mass as W-subjets
                # Could be improved by b-tagging if we run into a problem
                [sj_w1, sj_w2, sj_nonw] = [con.__deref__() for con in candJets[i].getJetConstituents() if not con.isNull()]

                # Calibrate the subjets: W1
                sj_w1_uncal = Jet(sj_w1)        
                c = self.jetReCalibratorAK4.getCorrection(sj_w1_uncal, rho, isHttSubjet=True)            
                sj_w1_cal = PhysicsObject(sj_w1).__copy__() 
                sj_w1_cal.scaleEnergy(c)

                # Calibrate the subjets: W2
                sj_w2_uncal = Jet(sj_w2)        
                c = self.jetReCalibratorAK4.getCorrection(sj_w2_uncal, rho, isHttSubjet=True)            
                sj_w2_cal = PhysicsObject(sj_w2).__copy__() 
                sj_w2_cal.scaleEnergy(c)

                # Calibrate the subjets: NonW
                sj_nonw_uncal = Jet(sj_nonw)        
                c = self.jetReCalibratorAK4.getCorrection(sj_nonw_uncal, rho, isHttSubjet=True)            
                sj_nonw_cal = PhysicsObject(sj_nonw).__copy__() 
                sj_nonw_cal.scaleEnergy(c)

                # Do all subjets pass the JetID requirements
                event.httCandidates[i].subjetIDPassed = all([passesJetId(x) for x in [sj_w1_cal, sj_w2_cal, sj_nonw_cal]])

                # Make TLVs so we can add them and get the top quark
                # candidate                
                tlv_sj_w1_cal   = ROOT.TLorentzVector()
                tlv_sj_w2_cal   = ROOT.TLorentzVector()
                tlv_sj_nonw_cal = ROOT.TLorentzVector()

                tlv_sj_w1_cal.SetPtEtaPhiM(sj_w1_cal.pt(),
                                           sj_w1_cal.eta(),
                                           sj_w1_cal.phi(),
                                           sj_w1_cal.mass())
                tlv_sj_w2_cal.SetPtEtaPhiM(sj_w2_cal.pt(),
                                           sj_w2_cal.eta(),
                                           sj_w2_cal.phi(),
                                           sj_w2_cal.mass())
                tlv_sj_nonw_cal.SetPtEtaPhiM(sj_nonw_cal.pt(),
                                             sj_nonw_cal.eta(),
                                             sj_nonw_cal.phi(),
                                             sj_nonw_cal.mass())
                
                tlv_top_cal =  tlv_sj_w1_cal + tlv_sj_w2_cal + tlv_sj_nonw_cal

                # Store calibrated top candidate variables
                event.httCandidates[i].ptcal    = tlv_top_cal.Pt()
                event.httCandidates[i].etacal   = tlv_top_cal.Eta()
                event.httCandidates[i].phical   = tlv_top_cal.Phi()
                event.httCandidates[i].masscal  = tlv_top_cal.M()
                                            
                # Store SJ W1 Variables
                event.httCandidates[i].sjW1ptcal   = sj_w1_cal.pt()
                event.httCandidates[i].sjW1masscal = sj_w1_cal.mass()
                event.httCandidates[i].sjW1pt      = sj_w1.pt()
                event.httCandidates[i].sjW1eta     = sj_w1.eta()
                event.httCandidates[i].sjW1phi     = sj_w1.phi()
                event.httCandidates[i].sjW1mass    = sj_w1.mass()

                # Get the correct b-tag
                for ib in xrange(0, len(sjbtags)) :
                    if  sj_w1 == sjbtags.key(ib).get():
                        event.httCandidates[i].sjW1btag = sjbtags.value(ib)
                        # Map dummy for -10 to -0.1 
                        if event.httCandidates[i].sjW1btag == -10.:
                            event.httCandidates[i].sjW1btag = -0.1
                            

                # Store SJ W2 Variables
                event.httCandidates[i].sjW2ptcal   = sj_w2_cal.pt()
                event.httCandidates[i].sjW2masscal = sj_w2_cal.mass()
                event.httCandidates[i].sjW2pt      = sj_w2.pt()  
                event.httCandidates[i].sjW2eta     = sj_w2.eta() 
                event.httCandidates[i].sjW2phi     = sj_w2.phi() 
                event.httCandidates[i].sjW2mass    = sj_w2.mass()

                # Get the correct b-tag
                for ib in xrange(0, len(sjbtags)) :
                    if  sj_w2 == sjbtags.key(ib).get():
                        event.httCandidates[i].sjW2btag = sjbtags.value(ib)
                        # Map dummy for -10 to -0.1 
                        if event.httCandidates[i].sjW2btag == -10.:
                            event.httCandidates[i].sjW2btag = -0.1


                # Store SJ Non W Variables
                event.httCandidates[i].sjNonWptcal   = sj_nonw_cal.pt()  
                event.httCandidates[i].sjNonWmasscal = sj_nonw_cal.mass()  
                event.httCandidates[i].sjNonWpt      = sj_nonw.pt()  
                event.httCandidates[i].sjNonWeta     = sj_nonw.eta() 
                event.httCandidates[i].sjNonWphi     = sj_nonw.phi() 
                event.httCandidates[i].sjNonWmass    = sj_nonw.mass()

                # Get the correct b-tag
                for ib in xrange(0, len(sjbtags)) :
                    if  sj_nonw == sjbtags.key(ib).get():
                        event.httCandidates[i].sjNonWbtag = sjbtags.value(ib)
                        # Map dummy for -10 to -0.1 
                        if event.httCandidates[i].sjNonWbtag == -10.:
                            event.httCandidates[i].sjNonWbtag = -0.1


        ######## 
        # AK8 Jets from MiniAOD + Subjet btags
        ########

        setattr(event, "ak08", map(PhysicsObject, self.handles["ak08"].product()))
        setattr(event, "ak08softdropsubjets", map(PhysicsObject, self.handles["ak08softdropsubjets"].product()))

        do_calc_bb = False
        # Calc BB tag  
        # -- if there is a HTT candidate and at least one lepton
        if len(event.selectedLeptons):
            for cand in event.httCandidates:
                if cand.fRec < 0.2 and cand.mass() > 100 and cand.mass() < 200:
                    do_calc_bb = True                
        # -- or if there are at least two fatjets with pT > 250 GeV and the scalar sum of the jet pTs is > 600
        fj_pt250 = [j for j in event.ak08 if j.pt() > 250]
        if len(fj_pt250) >= 2 and sum([j.pt() for j in fj_pt250]) > 600:
            do_calc_bb = True            
                
        # bb-tag Output
        newtags =  self.handles['ak08bbtag'].product()

        # Loop over jets                        
        for ij, jet in enumerate(getattr(event, "ak08")):

            # Fill bb-tag
            for i in xrange(len(newtags)) :
                if jet.physObj == newtags.key(i).get():
                    jet.bbtag = newtags.value(i)

                tmp_jet = Jet(jet)
		corr = self.jetReCalibratorAK8L2L3.getCorrection(tmp_jet,rho)
                jet.mprunedcorr= jet.userFloat("ak8PFJetsCHSPrunedMass")*corr	
		jet.JEC_L2L3 = corr                
                jet.JEC_L2L3Unc = tmp_jet.jetEnergyCorrUncertainty
		jet.JEC_L1L2L3 = self.jetReCalibratorAK8L1L2L3.getCorrection(tmp_jet,rho)
                jet.JEC_L1L2L3Unc = tmp_jet.jetEnergyCorrUncertainty

            # bb-tag Inputs
            muonTagInfos = self.handles['ak08muonTagInfos'].product()[ij]
            elecTagInfos = self.handles['ak08elecTagInfos'].product()[ij]
            ipTagInfo    = self.handles['ak08ipTagInfos'].product()[ij]
            svTagInfo    = self.handles['ak08svTagInfos'].product()[ij]

            orig_jet = self.handles["ak08"].product()[ij]

            
            # Commented out so rest of code can run
            # TODO: FIX!
            #if do_calc_bb:
            #    calcBBTagVariables(jet, 
            #                       orig_jet,
            #                       muonTagInfos, 
            #                       elecTagInfos, 
            #                       ipTagInfo, 
            #                       svTagInfo,
            #                       njettiness_08,
            #                       maxSVDeltaRToJet = 0.7,)
            #else:
            calcBBTagVariables_dummy(jet)


        # end of loop over jets

        ######## 
        # Ungroomed Fatjets + NSubjettiness + Hbb Tagging
        ########

        # So far only applied to ungroomed CA15 jet 
        # TODO: also add for other collections
        max_fatjet_eta = 2.0

        for prefix in ["ca15"]:

            if self.skip_ca15 and ("ca15" in prefix):
                continue

            # N-Subjettiness
            tau1 = self.handles[prefix+'tau1'].product()
            tau2 = self.handles[prefix+'tau2'].product()
            tau3 = self.handles[prefix+'tau3'].product()

            # bb-tag Output
            newtags =  self.handles[prefix+'bbtag'].product()
                
            # Four Vector
            tmp = map(PhysicsObject, self.handles[prefix+'ungroomed'].product())
            # assign the original index (so we can do a proper lookup of original jet even after eta cut)
            for ij, j in enumerate(tmp):
                j.original_index = ij
                
            # And apply eta cut
            setattr(event, prefix+"ungroomed", [x for x in tmp if abs(x.eta()) < max_fatjet_eta])
                
            # Loop over jets                        
            for ij, jet in enumerate(getattr(event, prefix+"ungroomed")):

                # Fill N-Subjettiness
                jet.tau1 = tau1.get(ij)
                jet.tau2 = tau2.get(ij)
                jet.tau3 = tau3.get(ij)

                # Fill bb-tag
                for i in xrange(len(newtags)) :
                    if jet.physObj == newtags.key(i).get():
                        jet.bbtag = newtags.value(i)

                # bb-tag Inputs
                muonTagInfos = self.handles['ca15muonTagInfos'].product()[ij]
                elecTagInfos = self.handles['ca15elecTagInfos'].product()[ij]
                ipTagInfo    = self.handles['ca15ipTagInfos'].product()[ij]
                svTagInfo    = self.handles['ca15svTagInfos'].product()[ij]

                orig_jet = self.handles[prefix+'ungroomed'].product()[jet.original_index]

                # Commented out so rest of code can run
                # TODO: FIX!
                #if do_calc_bb:
                #    calcBBTagVariables(jet, 
                #                       orig_jet,
                #                       muonTagInfos, 
                #                       elecTagInfos, 
                #                       ipTagInfo, 
                #                       svTagInfo,
                #                       njettiness_15,
                #                       maxSVDeltaRToJet = 1.3)
                #else:
                calcBBTagVariables_dummy(jet)

                                    
            # end of loop over jets


            



        return True


