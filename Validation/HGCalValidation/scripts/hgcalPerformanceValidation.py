#! /usr/bin/env python3

#------------------------------------------------------------------------------------------
# Description: This script is used to produce the results of the regurarly announced RelVal campaings.
#              Essentially, this is a wrapper around the basic HGCal Validation script:
#              Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py
# Documentation: Full documentation on this script and details on how to run it are in the section 
#                "Campaign Validation" in the HGCal DPG website:
#                http://hgcal.web.cern.ch/hgcal/Validation/RelVals/
#                Information on the CMSSW HGCalValidation package and relevant objects can be 
#                found in the "Validation" section.  
#------------------------------------------------------------------------------------------

import sys
import os
import subprocess
import optparse
import pandas as pd

from collections import OrderedDict

from Validation.RecoTrack.plotting.validation import Sample, Validation
from Validation.HGCalValidation.hgcalHtml import _sampleName,_pageNameMap,_summary,_summobj,_MatBudSections,_geoPageNameMap,_individualmaterials,_matPageNameMap,_individualmatplots,_individualMatPlotsDesc,_hideShowFun,_allmaterialsplots,_allmaterialsPlotsDesc, _fromvertexplots, _fromVertexPlotsDesc

from Validation.HGCalValidation.PostProcessorHGCAL_cfi import tracksterLabels as trackstersIters

#------------------------------------------------------------------------------------------
#Parsing input options
def parseOptions():

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)

    parser.add_option('', '--Obj', dest='OBJ',  type='string', default=None, help='Object to run. Options are: Geometry, SimHits, Digis, RecHits, Calibrations, CaloParticles, hgcalLayerClusters')
    parser.add_option('', '--html-validation-name', dest='HTMLVALNAME', type='string', default='', help='Could be either be hgcalLayerClusters or hgcalMultiClusters')
    parser.add_option('-d', '--download', action='store_true', dest='DOWNLOAD', default=False, help='Download DQM files from RelVals')
    parser.add_option('-g', '--gather', dest='GATHER',  type='string', default=None, help='Objects to gather: hitValidation, hitCalibration, hgcalLayerClusters, hgcalMultiClusters, ticlMultiClustersFromTrackstersEM, ticlMultiClustersFromTrackstersHAD')
    parser.add_option('-w', '--wwwarea', dest='WWWAREA',  type='string', default='/eos/project/h/hgcaldpg/www', help='Objects to gather: hitValidation, hitCalibration, hgcalLayerClusters, hgcalMultiClusters, ticlMultiClustersFromTrackstersEM, ticlMultiClustersFromTrackstersHAD')
    parser.add_option('-y', '--dry-run', action='store_true', dest='DRYRUN', default=False, help='perform a dry run (nothing is lauched).')
    parser.add_option('-i', '--inputeosarea', dest='INPUT',  type='string', default='/eos/cms/store/group/dpg_hgcal/comm_hgcal/apsallid/RelVals', help='Eos area where we will place all DQM files of the new and reference release campaign')
    parser.add_option('', '--geometry', action='store_true', dest='GEOMETRY', default=False, help='Geometry validation section')
    parser.add_option('', '--copyhtml', action='store_true', dest='COPYHTML', default=False, help='If used the main index.html file will be copied to the www area. Useful in case of experimenting to avoid surprises.')

    # store options and arguments as global variables
    global opt, args
    (opt, args) = parser.parse_args()

parseOptions()

#------------------------------------------------------------------------------------------
#Some helpful functions
#Processing the external os subprocess
def processCmd(cmd, quite = 0):
    print(cmd)
    status, output = subprocess.getstatusoutput(cmd)
    if (status !=0 and not quite):
        print('Error in processing command:\n   ['+cmd+']')
        print('Output:\n   ['+output+'] \n')
    return output

#PUtype
def putype(t):
    if "_pmx" in NewRelease:
        if "_pmx" in RefRelease:
            return {"default": "pmx"+t}
        return {"default": t, NewRelease: "pmx"+t}
    return t

#------------------------------------------------------------------------------------------
#Input section: Each time a new RelVal campaign is announced the following variables should 
#               be updated:
#               NewRelease: The new release being validated.
#               RefRelease: The reference release we want to test the new one against.
#               thereleases: All releases validated for which we have validation results 
#                            including the NewRelease.
#               NotNormalRelease , NotNormalRefRelease: If one of the releases to be validated has
#                            used the DIGI on 11_0_0 put "raw". Otherwise put "normal". 
#               phase2samples_noPU: These are the noPU phase 2 RelVal samples that are regurarly 
#                                   produced.   
#               phase2samples_PU: These are the PU phase 2 RelVal samples that are regurarly 
#                                 produced.  
#               RefRepository, NewRepository: The path where the DQM files of the campaign 
#                                             will be placed.
#------------------------------------------------------------------------------------------
#thereleases = { "CMSSW 11_1_X" : ["CMSSW_11_1_0_pre4_GEANT4","CMSSW_11_1_0_pre3","CMSSW_11_1_0_pre2"] }
thereleases = OrderedDict()
thereleases = { "CMSSW 12_4_X" : [
    "CMSSW_12_4_0_pre3_DD4HEP_vs_CMSSW_12_4_0_pre3_DDD",
    "CMSSW_12_4_0_pre3_vs_CMSSW_12_4_0_pre2",
    "CMSSW_12_4_0_pre2_vs_CMSSW_12_3_0_pre6"
                ],
                "CMSSW 12_3_X" : [
    "CMSSW_12_3_1_vs_CMSSW_12_3_0_pre6",
    "CMSSW_12_3_0_pre6_vs_CMSSW_12_3_0_pre5",
    "CMSSW_12_3_0_pre5_D88_vs_CMSSW_12_3_0_pre5_D77",
    "CMSSW_12_3_0_pre5_D77_vs_CMSSW_12_3_0_pre3_D77",
    "CMSSW_12_3_0_pre4_vs_CMSSW_12_3_0_pre3",
    "CMSSW_12_3_0_pre3_vs_CMSSW_12_3_0_pre2"
                ],
                "CMSSW 12_2_X" : [
    "CMSSW_12_2_0_vs_CMSSW_12_2_0_pre3",
    "CMSSW_12_2_0_pre3_D88_vs_CMSSW_12_2_0_pre3_D77",
    "CMSSW_12_2_0_pre3_vs_CMSSW_12_2_0_pre2",
    "CMSSW_12_2_0_pre2_vs_CMSSW_12_1_0_pre5"
                 ],
                "CMSSW 12_1_X" : [
    "CMSSW_12_1_0_pre5_vs_CMSSW_12_1_0_pre4",
    "CMSSW_12_1_0_pre5_D77_vs_CMSSW_12_1_0_pre4_D76",
    "CMSSW_12_1_0_pre4_ROOT624_vs_CMSSW_12_1_0_pre4",
    "CMSSW_12_1_0_pre4_vs_CMSSW_12_1_0_pre3",
    "CMSSW_12_1_0_pre3_vs_CMSSW_12_1_0_pre2",
    "CMSSW_12_1_0_pre2_vs_CMSSW_12_0_0_pre6",
    "CMSSW_12_1_0_pre2_D77_vs_CMSSW_12_1_0_pre2_D76"
                 ],
                "CMSSW 12_0_X" : [
    "CMSSW_12_0_1_vs_CMSSW_12_0_0_pre4",
    "CMSSW_12_0_0_pre6_vs_CMSSW_12_0_0_pre4",
    "CMSSW_12_0_0_pre4_vs_CMSSW_12_0_0_pre3",
    "CMSSW_12_0_0_pre3_vs_CMSSW_12_0_0_pre2",
    "CMSSW_12_0_0_pre2_vs_CMSSW_12_0_0_pre1",
    "CMSSW_12_0_0_pre1_vs_CMSSW_11_3_0_pre6"
                 ],
                "CMSSW 11_3_X" : [
    "CMSSW_11_3_0_vs_CMSSW_11_3_0_pre6",
    "CMSSW_11_3_0_pre6_vs_CMSSW_11_3_0_pre5",
    "CMSSW_11_3_0_pre5_vs_CMSSW_11_3_0_pre4",
    "CMSSW_11_3_0_pre4_vs_CMSSW_11_3_0_pre3",
    "CMSSW_11_3_0_pre3_G4VECGEOM_vs_CMSSW_11_3_0_pre3",
    "CMSSW_11_3_0_pre3_D76_vs_CMSSW_11_3_0_pre3",
    "CMSSW_11_3_0_pre3_vs_CMSSW_11_3_0_pre2",
    "CMSSW_11_3_0_pre2_vs_CMSSW_11_3_0_pre1",
    "CMSSW_11_3_0_pre1_vs_CMSSW_11_2_0_pre10",
                ],
                "CMSSW 11_2_X" : [
    "CMSSW_11_2_0_vs_CMSSW_11_2_0_pre10",
    "CMSSW_11_2_0_pre10_vs_CMSSW_11_2_0_pre9",
    "CMSSW_11_2_0_pre9_vs_CMSSW_11_2_0_pre8",
    "CMSSW_11_2_0_pre8_vs_CMSSW_11_2_0_pre7",
    "CMSSW_11_2_0_pre7_vs_CMSSW_11_2_0_pre6",
    "CMSSW_11_2_0_pre6_ROOT622_vs_CMSSW_11_2_0_pre6",
    "CMSSW_11_2_0_pre6_vs_CMSSW_11_2_0_pre5",
    "CMSSW_11_2_0_pre5_GEANT106_vs_CMSSW_11_2_0_pre5",
    "CMSSW_11_2_0_pre5_vs_CMSSW_11_2_0_pre3",
    "CMSSW_11_2_0_pre3_vs_CMSSW_11_2_0_pre1",
    "CMSSW_11_2_0_pre1_vs_CMSSW_11_1_0_pre8"
                ],
                "CMSSW 11_1_X" : [
    "CMSSW_11_1_0_pre8_raw1100_vs_CMSSW_11_1_0_pre8",
    "CMSSW_11_1_0_pre8_raw1100_vs_CMSSW_11_1_0_pre7_raw1100",
    "CMSSW_11_1_0_pre8_vs_CMSSW_11_1_0_pre7",
    "CMSSW_11_1_0_pre7_raw1100_vs_CMSSW_11_1_0_pre7",
    "CMSSW_11_1_0_pre7_raw1100_vs_CMSSW_11_1_0_pre6_raw1100",
    "CMSSW_11_1_0_pre7_vs_CMSSW_11_1_0_pre6",
    "CMSSW_11_1_0_pre6_raw1100_vs_CMSSW_11_1_0_pre6",
    "CMSSW_11_1_0_pre6_raw1100_vs_CMSSW_11_1_0_pre5_raw1100",
    "CMSSW_11_1_0_pre6_vs_CMSSW_11_1_0_pre5",
    "CMSSW_11_1_0_pre5_vs_CMSSW_11_1_0_pre4",
    "CMSSW_11_1_0_pre5_raw1100_vs_CMSSW_11_1_0_pre5",
    "CMSSW_11_1_0_pre5_raw1100_vs_CMSSW_11_1_0_pre4_raw1100",
    "CMSSW_11_1_0_pre4_raw1100_vs_CMSSW_11_1_0_pre4",
    "CMSSW_11_1_0_pre4_raw1100_vs_CMSSW_11_1_0_pre3_raw1100",
    "CMSSW_11_1_0_pre4_GEANT4","CMSSW_11_1_0_pre4"
               ] 
}

geometryTests = OrderedDict()
geometryTests = { "Material budget" : [
                #"Extended2026D49_vs_Extended2026D71",
                "Extended2026D49_vs_Extended2026D76",
                "Extended2026D76_vs_Extended2026D83",
                "Extended2026D83_vs_Extended2026D86",
                "Extended2026D77_vs_Extended2026D88"
                ]
}

GeoScenario = "Extended2026D77_vs_Extended2026D88"

RefRelease='CMSSW_12_3_0_pre6'

NewRelease='CMSSW_12_3_1'

NotNormalRelease = "normal"
NotNormalRefRelease = "normal"
#NotNormalRefRelease = "raw"

if ( os.path.isdir('%s/%s' %(opt.WWWAREA, NewRelease))) : 
    print("The campaign you are trying to validate has already an existing validation folder in the official www area.")
    print("Make sure you are not overwriting anything and try again.")
    exit()

if "raw" in NotNormalRelease: 
    #   appendglobaltag = "_2026D49noPU_raw1100_rsb"
    #   appendglobaltag = "_2026D49noPU_raw1100"
    #   appendglobaltag = "_2026D49noPU_gcc900"
    #appendglobaltag = "_2026D77noPU"
    appendglobaltag = "_2026D88noPU"
    #appendglobaltag = "_2026D88noPU_DDD"
    #appendglobaltag = "_2026D88noPU_DD4HEP"
else: 
    #   appendglobaltag = "_2026D49noPU"
    #appendglobaltag = "_2026D76noPU"
    #appendglobaltag = "_2026D77noPU"
    appendglobaltag = "_2026D88noPU"
    #appendglobaltag = "_2026D88noPU_DDD"
    #appendglobaltag = "_2026D88noPU_DD4HEP"

#Until the final list of RelVals settles down the following sample list is under constant review
'''
phase2samples_noPU_oldnaming = [
#    Sample("RelValCloseByParticleGun_CE_H_Fine_300um", dqmVersion="0002", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_H_Fine_300um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_H_Fine_200um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_H_Fine_120um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_H_Coarse_Scint", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_H_Coarse_300um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_E_Front_300um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_E_Front_200um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByParticleGun_CE_E_Front_120um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValCloseByParticleGun_CE_E_Front_120um", scenario="2026D49", appendGlobalTag=appendglobaltag , version="v2"),
    Sample("RelValTTbar", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValZMM", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleGammaFlatPt8To150", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleMuPt10", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleMuPt100", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleMuPt1000", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleMuFlatPt2To100", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleMuFlatPt0p7To10", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleEFlatPt2To100", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleTauFlatPt2To150", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSinglePiFlatPt0p7To10", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValQCD_Pt20toInfMuEnrichPt15", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValQCD_Pt15To7000_Flat", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValZTT", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValZMM", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValZEE", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValB0ToKstarMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValBsToEleEle", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValBsToMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValBsToJpsiGamma", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValBsToJpsiPhi_mumuKK", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValBsToPhiPhi_KKKK", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValDisplacedMuPt30To100", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValDisplacedMuPt2To10", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValDisplacedMuPt10To30", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValTauToMuMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValMinBias", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValH125GGgluonfusion", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValNuGun", scenario="2026D49", appendGlobalTag=appendglobaltag )
    ]
'''

#Main workflow RelVals
phase2samples_noPU = [

    #------------------------------
    #version v2 campaign
    #Sample("RelValZpTT_1500", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValZpTT_1500", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValZTT", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValZMM", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValZEE", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValTenTau_15_500_Eta3p1", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2"  ),
    #Sample("RelValTenTau_15_500", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2"  ),
    #Sample("RelValTTbar", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValQCD_Pt15To7000_Flat", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValQCD_Pt15To7000_Flat", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValNuGun", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValMinBias", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValMinBias", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValH125GGgluonfusion", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" )

    #------------------------------
    #NORMAL version v1 campaign
    Sample("RelValZpTT_1500", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValZpTT_1500", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValZTT", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValZMM", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValZMM", midfix="14", scenario="2026D49", dqmVersion="0002", appendGlobalTag=appendglobaltag ),
    Sample("RelValZEE", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValTenTau_15_500_Eta3p1", scenario="2026D49", appendGlobalTag=appendglobaltag  ),
    #Sample("RelValTenTau_15_500", scenario="2026D49", appendGlobalTag=appendglobaltag  ),
    Sample("RelValTTbar", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValQCD_Pt15To7000_Flat", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValQCD_Pt15To7000_Flat", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValNuGun", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValMinBias", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValMinBias", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValH125GGgluonfusion", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag )
    #------------------------------


]


#More workflows 
phase2samples_noPU_extend = [

    #Sample("RelValSingleMuPt10", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2"),
    #Sample("RelValSingleMuPt100", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValSingleMuPt1000", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" )

    Sample("RelValSingleMuPt10", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleMuPt100", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleMuPt1000", scenario="2026D49", appendGlobalTag=appendglobaltag )

]

#These workflows were added in CMSSW_11_1_0_pre6 but there were missing from CMSSW_11_1_0_pre5.
#So, I am only download them to be reary for pre7. Then, I comment them out
#For the moment I cannot find these in pre7.
phase2samples_noPU_extend_more = [

    #------------------------------
    #version v3 campaign
    #Sample("RelValCloseByPGun_CE_H_Fine_300um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValCloseByPGun_CE_H_Fine_200um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValCloseByPGun_CE_H_Fine_120um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValCloseByPGun_CE_H_Coarse_Scint", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValCloseByPGun_CE_H_Coarse_300um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValCloseByPGun_CE_E_Front_300um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValCloseByPGun_CE_E_Front_200um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValCloseByPGun_CE_E_Front_120um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValSingleGammaFlatPt8To150", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValSingleEFlatPt2To100", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" ),
    #Sample("RelValSinglePiFlatPt0p7To10", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v3" )

    #------------------------------
    #version v2 campaign
    #Sample("RelValCloseByPGun_CE_H_Fine_300um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValCloseByPGun_CE_H_Fine_200um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValCloseByPGun_CE_H_Fine_120um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValCloseByPGun_CE_H_Coarse_Scint", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValCloseByPGun_CE_H_Coarse_300um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValCloseByPGun_CE_E_Front_300um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValCloseByPGun_CE_E_Front_200um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValCloseByPGun_CE_E_Front_120um", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValSingleGammaFlatPt8To150", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValSingleEFlatPt2To100", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" ),
    #Sample("RelValSinglePiFlatPt0p7To10", scenario="2026D49", appendGlobalTag=appendglobaltag, version="v2" )



    #------------------------------
    #NORMAL version v1 campaign
    Sample("RelValCloseByPGun_CE_H_Fine_300um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByPGun_CE_H_Fine_200um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByPGun_CE_H_Fine_120um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByPGun_CE_H_Coarse_Scint", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByPGun_CE_H_Coarse_300um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByPGun_CE_E_Front_300um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByPGun_CE_E_Front_200um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValCloseByPGun_CE_E_Front_120um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleGammaFlatPt8To150", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSingleEFlatPt2To100", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    Sample("RelValSinglePiFlatPt0p7To10", scenario="2026D49", appendGlobalTag=appendglobaltag )
    #------------------------------

    #Sample("RelValCloseByPGun_CE_H_Fine_300um", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValCloseByPGun_CE_H_Fine_200um", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValCloseByPGun_CE_H_Fine_120um", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValCloseByPGun_CE_H_Coarse_Scint", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValCloseByPGun_CE_H_Coarse_300um", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValCloseByPGun_CE_E_Front_300um", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValCloseByPGun_CE_E_Front_200um", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValCloseByPGun_CE_E_Front_120um", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValSingleGammaFlatPt8To150", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValSingleEFlatPt2To100", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" ),
    #Sample("RelValSinglePiFlatPt0p7To10", scenario="2026D49", appendGlobalTag=appendglobaltag + "_HGCal" )


    #Sample("RelValQCD_Pt20toInfMuEnrichPt15", midfix="14", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValDisplacedMuPt30To100", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValDisplacedMuPt2To10", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValDisplacedMuPt10To30", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValB0ToKstarMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValBsToEleEle", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValBsToMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValBsToJpsiGamma", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValBsToJpsiPhi_mumuKK", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValBsToPhiPhi_KKKK", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValTauToMuMuMu", midfix="14TeV", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValSingleTauFlatPt2To150", scenario="2026D49", appendGlobalTag=appendglobaltag ),
    #Sample("RelValSingleMuFlatPt0p7To10", scenario="2026D49", appendGlobalTag=appendglobaltag )

]

phase2samples_noPU.extend(phase2samples_noPU_extend)
phase2samples_noPU.extend(phase2samples_noPU_extend_more)
#phase2samples_noPU = phase2samples_noPU_extend_more
#phase2samples_noPU.extend(phase2samples_noPU_oldnaming)

#phase2samples_noPU = [
#    Sample("RelValCloseByPGun_CE_E_Front_300um", scenario="2026D49", appendGlobalTag=appendglobaltag ),
#    Sample("RelValCloseByPGun_CE_E_Front_200um", scenario="2026D49", appendGlobalTag=appendglobaltag )
#]

#For the PU samples 
phase2samples_PU = [
    Sample("RelValTTbar", midfix="14TeV", scenario="2026D49", putype=putype("25ns"), punum=200, appendGlobalTag="_2026D49PU200", version="v2"),
]

# Reference and new repository
RefRepository = '%s' %(opt.INPUT)
NewRepository = '%s' %(opt.INPUT)

#------------------------------------------------------------------------------------------
#Download section: The basic HGCal validation object is created and the DQM files of the 
#                  DQM files of the RelVals are downloaded. 
#------------------------------------------------------------------------------------------

#Create basic object for the HGCal validation plots
val = Validation(
    fullsimSamples = phase2samples_noPU,
    fastsimSamples = [],
    refRelease=RefRelease, refRepository=RefRepository,
    newRelease=NewRelease, newRepository=NewRepository
)

#------------------------------------------------------------------------------------------
#Download the DQM files of the RelVals. 
if(opt.DOWNLOAD): 
    val.download()

    #Keep them in eos, save afs space. 
    if (not os.path.isdir(RefRepository+'/'+NewRelease)) :
        processCmd('mkdir -p '+RefRepository+'/'+NewRelease)

    for infi in phase2samples_noPU:
        if "_HGCal" in infi.filename(NewRelease): 
            processCmd('mv ' + infi.filename(NewRelease) + ' ' + infi.filename(NewRelease).replace("_HGCal",""))
            processCmd('mv ' + infi.filename(NewRelease).replace("_HGCal","") + ' ' + RefRepository+'/'+NewRelease)
        else: 
            #processCmd('mv ' + infi.filename(NewRelease) + ' ' + infi.filename(NewRelease).replace("2026D49noPU-v2","2026D49noPU-v1"))
            #processCmd('mv ' + infi.filename(NewRelease).replace("2026D49noPU-v2","2026D49noPU-v1")  + ' ' + RefRepository+'/'+NewRelease)
            processCmd('mv ' + infi.filename(NewRelease)  + ' ' + RefRepository+'/'+NewRelease)

#------------------------------------------------------------------------------------------
#Objects processing section: The objects defined in --Obj are analyzed here. 
#------------------------------------------------------------------------------------------
if (opt.OBJ == 'layerClusters' or opt.OBJ == 'hitCalibration' or opt.OBJ == 'hitValidation' or opt.OBJ == 'tracksters' or opt.OBJ == 'simulation'):
    fragments = []
    #In the case of simulation we want to split the plots in specific folder
    if opt.OBJ == 'simulation': processCmd('mkdir HGCValid_SimClusters_Plots HGCValid_CaloParticles_Plots')
    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        #samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").split("__CMSSW_10_6_0_pre4",1)[0]
        #samplename = samplename + infi.pileup()
        if infi.pileup() == "PU":
            samplename = samplename + str(infi.pileupNumber())

        #print( infi.name()  )
        print(_sampleName[infi.name()])
        print("="*40)
        print(samplename)
        print("="*40)

        #In the case of tracksters. We want to split the results.
        if opt.OBJ == 'tracksters':
           for tracksterCollection in trackstersIters:
               processCmd('mkdir -p HGCValid_Tracksters_Plots/plots_%s_%s HGCValid_Test-TICL_Plots/plots_%s_%s HGCValid_TICL-patternRecognition_Plots/plots_%s_%s' %(samplename,tracksterCollection,samplename,tracksterCollection,samplename,tracksterCollection) )

        inputpathRef = ""
        if RefRelease != None: inputpathRef = RefRepository +'/' + RefRelease +'/'
        inputpathNew = NewRepository +'/' + NewRelease+ '/'

        if RefRelease == None:
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename)+ ' --collection %s' %(opt.HTMLVALNAME)
        elif "raw" in NotNormalRelease and "normal" in NotNormalRefRelease:
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v3_2026D76noPU-v1","mcRun4_realistic_v3_2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v3_2026D49noPU_raw1100_rsb-v1","mcRun4_realistic_v3_2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
        elif "normal" in NotNormalRelease and "raw" in NotNormalRefRelease:
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v7_2026D77noPU-v1","mcRun4_realistic_v7_2026D76noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v3_2026D49noPU_raw1100_rsb-v1","mcRun4_realistic_v3_2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
        elif "raw" in NotNormalRelease and "raw" in NotNormalRefRelease:
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease) + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("_raw1100","_raw1100_rsb") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
        elif "normal" in NotNormalRelease and "normal" in NotNormalRefRelease:
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease) + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("2026D88noPU_DD4HEP-v1","2026D88noPU_DDD-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
        else: 
            #print inputpathRef, infi.filename(RefRelease).replace("D49","D41")
            #YOU SHOULD INSPECT EACH TIME THIS COMMAND AND THE REPLACE
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("D49","D41").replace("200-v2","200-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME) .replace("v2__", "v1__")
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v2-v1", "mcRun4_realistic_v2_2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME) 
            print(cmd)

        if(opt.DRYRUN):
            print('Dry-run: ['+cmd+']')
        else:
            output = processCmd(cmd)
            if opt.OBJ == 'layerClusters':
                processCmd('mv HGCValid_%s_Plots/plots_%s_Layer\ Clusters.html HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME,samplename,opt.HTMLVALNAME))
                processCmd('awk \'NR>=6&&NR<=396\' HGCValid_%s_Plots/index.html > HGCValid_%s_Plots/index_%s.html '% (opt.HTMLVALNAME,opt.HTMLVALNAME, samplename))
                processCmd('echo "  <br/>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )
                processCmd('echo "  <hr>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )

            if opt.OBJ == 'hitCalibration':
                #processCmd('indexname=`ls HGCValid_%s_Plots/plots_*.html`; mv ${indexname} HGCValid_%s_Plots/index.html;'%(opt.HTMLVALNAME,opt.HTMLVALNAME))
                processCmd('mv HGCValid_%s_Plots/plots_%s_Calibrated\ RecHits.html HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME,samplename,opt.HTMLVALNAME))
                processCmd('sed -i \'s/Calibrated\ RecHits//g\' HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME) )
                processCmd('awk \'NR>=6&&NR<=27\' HGCValid_%s_Plots/index.html > HGCValid_%s_Plots/index_%s.html '% (opt.HTMLVALNAME,opt.HTMLVALNAME, samplename))
                processCmd('echo "  <br/>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )
                processCmd('echo "  <hr>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )

            if opt.OBJ == 'hitValidation':
                processCmd('mv HGCValid_%s_Plots/plots_%s_Hits.html HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME,samplename,opt.HTMLVALNAME))
                processCmd('awk \'NR>=6&&NR<=184\' HGCValid_%s_Plots/index.html > HGCValid_%s_Plots/index_%s.html '% (opt.HTMLVALNAME,opt.HTMLVALNAME, samplename))
                processCmd('echo "  <br/>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )  
                processCmd('echo "  <hr>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )
                
            if opt.OBJ == 'tracksters':
                processCmd('mv HGCValid_%s_Plots/plots_%s_Tracksters.html HGCValid_Tracksters_Plots/index.html'%(opt.HTMLVALNAME,samplename))
                processCmd('mv HGCValid_%s_Plots/plots_%s_Test-TICL.html HGCValid_Test-TICL_Plots/index.html'%(opt.HTMLVALNAME,samplename))
                processCmd('mv HGCValid_%s_Plots/plots_%s_TICL-patternRecognition.html HGCValid_TICL-patternRecognition_Plots/index.html'%(opt.HTMLVALNAME,samplename))
                processCmd('awk \'NR>=6&&NR<=135\' HGCValid_Tracksters_Plots/index.html > HGCValid_Tracksters_Plots/index_%s.html ' %(samplename))
                processCmd('awk \'NR>=6&&NR<=117\' HGCValid_Test-TICL_Plots/index.html > HGCValid_Test-TICL_Plots/index_%s.html '% (samplename))
                processCmd('awk \'NR>=6&&NR<=117\' HGCValid_TICL-patternRecognition_Plots/index.html > HGCValid_TICL-patternRecognition_Plots/index_%s.html '% (samplename))
                processCmd('echo "  <br/>" >> HGCValid_Tracksters_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <br/>" >> HGCValid_Test-TICL_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <br/>" >> HGCValid_TICL-patternRecognition_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <hr>" >> HGCValid_Tracksters_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <hr>" >> HGCValid_Test-TICL_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <hr>" >> HGCValid_TICL-patternRecognition_Plots/index_%s.html '%(samplename) )
                #Now move the plots also to the relevant folders
                for tracksterCollection in trackstersIters:
                    #Linking
                    processCmd('mv HGCValid_%s_Plots/plots_%s_%s/*_Link HGCValid_Test-TICL_Plots/plots_%s_%s/.'%(opt.HTMLVALNAME,samplename,tracksterCollection,samplename,tracksterCollection))
                    processCmd('mv HGCValid_%s_Plots/plots_%s_%s/*CaloParticle*Trackster* HGCValid_Test-TICL_Plots/plots_%s_%s/.'%(opt.HTMLVALNAME,samplename,tracksterCollection,samplename,tracksterCollection))
                    processCmd('mv HGCValid_%s_Plots/plots_%s_%s/*Trackster*CaloParticle* HGCValid_Test-TICL_Plots/plots_%s_%s/.'%(opt.HTMLVALNAME,samplename,tracksterCollection,samplename,tracksterCollection))
                    #Pattern recognition
                    processCmd('mv HGCValid_%s_Plots/plots_%s_%s/*_PR HGCValid_TICL-patternRecognition_Plots/plots_%s_%s/.'%(opt.HTMLVALNAME,samplename,tracksterCollection,samplename,tracksterCollection))
                    processCmd('mv HGCValid_%s_Plots/plots_%s_%s/*SimTrackster*Trackster* HGCValid_TICL-patternRecognition_Plots/plots_%s_%s/.'%(opt.HTMLVALNAME,samplename,tracksterCollection,samplename,tracksterCollection))
                    processCmd('mv HGCValid_%s_Plots/plots_%s_%s/*Trackster*SimTrackster* HGCValid_TICL-patternRecognition_Plots/plots_%s_%s/.'%(opt.HTMLVALNAME,samplename,tracksterCollection,samplename,tracksterCollection))
                    #Tracksters
                    for gr in ['EtaPhiPtEnergy','XYZ','TotalNumberofTracksters','NumberofLayerClustersinTrackster','NumberofLayerClustersinTracksterPerLayer','NumberofLayerClustersinTracksterPerLayer_zminus_EE','NumberofLayerClustersinTracksterPerLayer_zminus_FH','NumberofLayerClustersinTracksterPerLayer_zminus_BH','NumberofLayerClustersinTracksterPerLayer_zplus_EE','NumberofLayerClustersinTracksterPerLayer_zplus_FH','NumberofLayerClustersinTracksterPerLayer_zplus_BH','LayerNumbersOfTrackster','MultiplicityofLCinTST']:
                        processCmd('mv HGCValid_%s_Plots/plots_%s_%s/%s HGCValid_Tracksters_Plots/plots_%s_%s/.'%(opt.HTMLVALNAME,samplename,tracksterCollection,gr,samplename,tracksterCollection))


            if  opt.OBJ == 'simulation':              

                processCmd('mv HGCValid_%s_Plots/plots_%s_SimClusters.html HGCValid_SimClusters_Plots/index.html'%(opt.HTMLVALNAME,samplename))
                processCmd('mv HGCValid_%s_Plots/plots_%s_CaloParticles.html HGCValid_CaloParticles_Plots/index.html'%(opt.HTMLVALNAME,samplename))
                processCmd('awk \'NR>=6&&NR<=157\' HGCValid_SimClusters_Plots/index.html > HGCValid_SimClusters_Plots/index_%s.html '% (samplename))
                processCmd('awk \'NR>=6&&NR<=331\' HGCValid_CaloParticles_Plots/index.html > HGCValid_CaloParticles_Plots/index_%s.html '% (samplename))
                processCmd('echo "  <br/>" >> HGCValid_SimClusters_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <br/>" >> HGCValid_CaloParticles_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <hr>" >> HGCValid_SimClusters_Plots/index_%s.html '%(samplename) )
                processCmd('echo "  <hr>" >> HGCValid_CaloParticles_Plots/index_%s.html '%(samplename) )
                #Now move the plots also to the relevant folders
                processCmd('mv HGCValid_%s_Plots/plots_%s_ClusterLevel HGCValid_SimClusters_Plots/.'%(opt.HTMLVALNAME,samplename))
                processCmd('mv HGCValid_%s_Plots/plots_%s_ticlSimTracksters HGCValid_SimClusters_Plots/.'%(opt.HTMLVALNAME,samplename))
                processCmd('mv HGCValid_%s_Plots/plots_%s_CaloParticles_* HGCValid_CaloParticles_Plots/.'%(opt.HTMLVALNAME,samplename))


        if opt.OBJ == 'simulation': 
            fragments.append( 'HGCValid_SimClusters_Plots/index_%s.html'% (samplename) )
            fragments.append( 'HGCValid_CaloParticles_Plots/index_%s.html'% (samplename) )
        elif opt.OBJ == 'tracksters':
            fragments.append( 'HGCValid_Tracksters_Plots/index_%s.html'% (samplename) )
            fragments.append( 'HGCValid_Test-TICL_Plots/index_%s.html'% (samplename) )
            fragments.append( 'HGCValid_TICL-patternRecognition_Plots/index_%s.html'% (samplename) )
        else:
            fragments.append( 'HGCValid_%s_Plots/index_%s.html'% (opt.HTMLVALNAME, samplename) )


    #Let's also create the final index xml file(s). 
    indexfiles = []
    if opt.OBJ == 'simulation': 
        indexfiles = ["SimClusters","CaloParticles"]
    elif opt.OBJ == 'tracksters':
        indexfiles = ["Tracksters","Test-TICL","TICL-patternRecognition"]
    else: 
        indexfiles = [opt.HTMLVALNAME]

    for ind in indexfiles:        
        processCmd('mv HGCValid_%s_Plots/index.html HGCValid_%s_Plots/test.html' %(ind,ind) )
        index_file = open('HGCValid_%s_Plots/index.html'%(ind),'w')            
        #Write preamble
        index_file.write('<html>\n')
        index_file.write(' <head>\n')
        index_file.write('  <title>HGCAL validation %s </title>\n' %(ind) )
        index_file.write(' </head>\n')
        index_file.write(' <body>\n')

        for frag in fragments:   
            if ind not in frag: continue
            with open(frag,'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    print(line)
                    index_file.write(line + '\n')
                    #processCmd( 'cat ' + frag + ' >> HGCalValidationPlots/index.html '   )
                    #index_file.write(frag)

        #Writing postamble"
        index_file.write(' </body>\n')
        index_file.write('</html>\n')
        index_file.close()

#------------------------------------------------------------------------------------------
#This is the SimHits part
if (opt.OBJ == 'SimHits'):
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("hgcalSimHitStudy")) :
        processCmd('mkdir -p hgcalSimHitStudy')
    #Prepare for www
    processCmd('cp %s/../public/index.php hgcalSimHitStudy/.'%(opt.WWWAREA) )

    #The input to this is for the moment 100 GeV muon from runnin cmsRun runHGCalSimHitStudy_cfg.py 
    #Input: hgcSimHits.root
    cmd = 'root.exe -b -q Validation/HGCalValidation/macros/validationplots.C\(\\"hgcSimHit.root' +  '\\",\\"'+ opt.OBJ + '\\"\)'
    if(opt.DRYRUN):
        print('Dry-run: ['+cmd+']')
    else:
        output = processCmd(cmd)

#------------------------------------------------------------------------------------------
'''
if (opt.OBJ == 'hitValidation'):
    fragments = []
    #Now  that we have them in eos lets produce plots
    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        #samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").split("__CMSSW_10_6_0_pre4",1)[0]
        #samplename = samplename + infi.pileup()
        if infi.pileup() == "PU":
            samplename = samplename + str(infi.pileupNumber())

        print("="*40)
        print(samplename)
        print("="*40)

        inputpathRef = ""
        if RefRelease != None: inputpathRef = RefRepository +'/' + RefRelease +'/'
        inputpathNew = NewRepository +'/' + NewRelease+ '/'

        if RefRelease == None:
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename)+ ' --collection %s' %(opt.HTMLVALNAME)
        elif "raw" in NotNormalRelease and "normal" in NotNormalRefRelease:
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v3_2026D76noPU-v1","mcRun4_realistic_v3_2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v3_2026D49noPU_raw1100_rsb-v1","mcRun4_realistic_v3_2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
        elif "raw" in NotNormalRelease and "raw" in NotNormalRefRelease:
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease) + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("_raw1100","_raw1100_rsb") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
        elif "normal" in NotNormalRelease and "normal" in NotNormalRefRelease:
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease) + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("2026D49noPU-v2","2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
        else: 
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("D49","D41").replace("200-v2","200-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME) 
            cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease).replace("mcRun4_realistic_v2-v1", "mcRun4_realistic_v2_2026D49noPU-v1") + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)
            #cmd = 'python3 Validation/HGCalValidation/scripts/makeHGCalValidationPlots.py ' +  inputpathRef + infi.filename(RefRelease) + ' ' +  inputpathNew + infi.filename(NewRelease) + ' --outputDir HGCValid_%s_Plots --no-ratio --png --separate --html-sample "%s" ' %(opt.HTMLVALNAME, _sampleName[infi.name()] ) + ' --html-validation-name %s --subdirprefix ' %(opt.HTMLVALNAME) + ' plots_%s' % (samplename) + ' --collection %s' %(opt.HTMLVALNAME)


        if(opt.DRYRUN):
            print('Dry-run: ['+cmd+']')
        else:
            output = processCmd(cmd)
            processCmd('mv HGCValid_%s_Plots/plots_%s_Hits.html HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME,samplename,opt.HTMLVALNAME))
            processCmd('awk \'NR>=6&&NR<=184\' HGCValid_%s_Plots/index.html > HGCValid_%s_Plots/index_%s.html '% (opt.HTMLVALNAME,opt.HTMLVALNAME, samplename))
            processCmd('echo "  <br/>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )  
            processCmd('echo "  <hr>" >> HGCValid_%s_Plots/index_%s.html '%(opt.HTMLVALNAME, samplename) )

        fragments.append( 'HGCValid_%s_Plots/index_%s.html'% (opt.HTMLVALNAME, samplename) )


    #Let's also create the final index xml file. 
    processCmd('mv HGCValid_%s_Plots/index.html HGCValid_%s_Plots/test.html' %(opt.HTMLVALNAME,opt.HTMLVALNAME) )
    index_file = open('HGCValid_%s_Plots/index.html'%(opt.HTMLVALNAME),'w')            
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')
    index_file.write('  <title>HGCal validation %s </title>\n' %(opt.HTMLVALNAME) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')

    for frag in fragments:   
        with open(frag,'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                print(line)
                index_file.write(line + '\n')
                #processCmd( 'cat ' + frag + ' >> HGCalValidationPlots/index.html '   )
                #index_file.write(frag)


    #Writing postamble"
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()
'''

#-------------------------------------------------------------------------------------------
#This is the Digis part
if (opt.OBJ == 'Digis'):
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("hgcalDigiStudy")) :
        processCmd('mkdir -p hgcalDigiStudy')
        processCmd('mkdir -p hgcalDigiStudyEE')
        processCmd('mkdir -p hgcalDigiStudyHEF')
        processCmd('mkdir -p hgcalDigiStudyHEB')
    #Prepare for www
    processCmd('cp %s/../public/index.php hgcalDigiStudy/.'%(opt.WWWAREA) )
    processCmd('cp %s/../public/index.php hgcalDigiStudyEE/.'%(opt.WWWAREA) )
    processCmd('cp %s/../public/index.php hgcalDigiStudyHEF/.'%(opt.WWWAREA) )
    processCmd('cp %s/../public/index.php hgcalDigiStudyHEB/.'%(opt.WWWAREA) )
    #The input here is from running cmsRun runHGCalDigiStudy_cfg.py, to which 
    #we usually give ttbar noPU as input 
    #Input: hgcDigi.root
    cmd = 'root.exe -b -q Validation/HGCalValidation/macros/validationplots.C\(\\"hgcDigi.root' +  '\\",\\"'+ opt.OBJ + '\\"\)'
    if(opt.DRYRUN):
        print('Dry-run: ['+cmd+']')
    else:
        output = processCmd(cmd)
        #mv the output under the main directory
        processCmd('mv hgcalDigiStudyEE hgcalDigiStudy/.')
        processCmd('mv hgcalDigiStudyHEF hgcalDigiStudy/.')
        processCmd('mv hgcalDigiStudyHEB hgcalDigiStudy/.')

#-------------------------------------------------------------------------------------------
#This is the RecHits part
if (opt.OBJ == 'RecHits'):
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("hgcalRecHitStudy")) :
        processCmd('mkdir -p hgcalRecHitStudy')
        processCmd('mkdir -p hgcalRecHitStudyEE')
        processCmd('mkdir -p hgcalRecHitStudyHEF')
        processCmd('mkdir -p hgcalRecHitStudyHEB')
    #Prepare for www
    processCmd('cp %s/../public/index.php hgcalRecHitStudy/.'%(opt.WWWAREA) )
    processCmd('cp %s/../public/index.php hgcalRecHitStudyEE/.'%(opt.WWWAREA) )
    processCmd('cp %s/../public/index.php hgcalRecHitStudyHEF/.'%(opt.WWWAREA) )
    processCmd('cp %s/../public/index.php hgcalRecHitStudyHEB/.'%(opt.WWWAREA) )
    #The input here is from running cmsRun runHGCalRecHitStudy_cfg.py, to which 
    #we usually give ttbar noPU as input 
    #Input: hgcRecHit.root
    cmd = 'root.exe -b -q Validation/HGCalValidation/macros/validationplots.C\(\\"hgcRecHit.root' +  '\\",\\"'+ opt.OBJ + '\\"\)'
    if(opt.DRYRUN):
        print('Dry-run: ['+cmd+']')
    else:
        output = processCmd(cmd)
        #mv the output under the main directory
        processCmd('mv hgcalRecHitStudyEE hgcalRecHitStudy/.')
        processCmd('mv hgcalRecHitStudyHEF hgcalRecHitStudy/.')
        processCmd('mv hgcalRecHitStudyHEB hgcalRecHitStudy/.')

#-------------------------------------------------------------------------------------------
## TODO #This is the CaloParticles part
if (opt.OBJ == 'CaloParticles'):
    particletypes = ["-11","-13","-211","-321","11","111","13","211","22","321"]
    #This is where we will save the final output pngs: 
    if (not os.path.isdir("CaloParticles")) :
        processCmd('mkdir -p CaloParticles')
    #Prepare for www
    processCmd('cp %s/../public/index.php CaloParticles/.'%(opt.WWWAREA) )

    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        #samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").split("__"+NewRelease,1)[0]
        samplename = samplename + infi.pileup()
        if infi.pileup() == "PU":
            samplename = samplename + str(infi.pileupNumber())

        print("="*40)
        print(samplename)
        print("="*40)
        if (not os.path.isdir(samplename)) :
            processCmd('mkdir -p ' + samplename )
            processCmd('cp %s/RelVals/index.php '%(opt.WWWAREA) + samplename + '/.')
            for part in particletypes: 
                processCmd('mkdir -p ' + samplename + '/' +part )
                #Prepare for www
                processCmd('cp %s/RelVals/index.php '%(opt.WWWAREA) + samplename + '/' +part + '/.')

        inputpathRef = ""
        if RefRelease != None: inputpathRef = RefRepository +'/' + RefRelease +'/'
        inputpathNew = NewRepository +'/' + NewRelease+ '/'
        cmd = 'root.exe -b -q Validation/HGCalValidation/macros/validationplots.C\(\\"'+ inputpathNew + infi.filename(NewRelease) +  '\\",\\"'+ opt.OBJ + '\\",\\"'+ samplename + '\\"\\)'
        if(opt.DRYRUN):
            print('Dry-run: ['+cmd+']')
        else:
            output = processCmd(cmd)
            processCmd('mv ' +samplename+ ' CaloParticles/.' )

#------------------------------------------------------------------------------------------
#Summary section: After processing all the objects the results are gathered, webpages are 
#                 created and a summary page is added. 
#-------------------------------------------------------------------------------------------
#Here we will gather all results. 
if (opt.GATHER != None) :

    #First we need the top folder to contain all validation releases. 
    index_file = open('index.html','w')            
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')
    index_file.write('  <title>HGCAL validation results </title>\n'  )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')
    index_file.write(' <h1>\n')
    index_file.write(' HGCAL Validation Results \n'  )
    index_file.write(' </h1>\n')
    index_file.write(' <hr/>\n' )
    index_file.write(' <h2>\n')
    index_file.write(' Release Validation Campaigns \n'  )
    index_file.write(' </h2>\n')
    index_file.write('  <ul>\n' )

    for trel in thereleases.keys():
        index_file.write('   <li>\n' )
        index_file.write('   %s\n' %(trel) )
        for rel in thereleases[trel]: 
            index_file.write('  <ul>\n' )
            index_file.write('   <li><a href="%s/index.html">%s</a></li>\n' %(rel, rel ) )
            index_file.write('  </ul>\n' )
        index_file.write('   </li>\n' )
        index_file.write('  <br>\n' )
        index_file.write('  <br>\n' )
        index_file.write('  <br>\n' )

    index_file.write('  </ul>\n' )
    index_file.write(' <hr/>\n' )

    #New section : Geometry Validation
    #Regardless of the release validation, the top html menu should contain the geometry section.
    #we put this in the "gather" step.
    index_file.write(' <h2>\n')
    index_file.write(' Geometry Validation \n'  )
    index_file.write(' </h2>\n')
    index_file.write('  <ul>\n' )

    for tgeo in geometryTests.keys():
        index_file.write('   <li>\n' )
        index_file.write('   %s\n' %(tgeo) )
        for geo in geometryTests[tgeo]:
            #We need the directory for the geometry related results 
            if (not os.path.isdir(geo)):
                processCmd('mkdir -p %s/%s' %(opt.WWWAREA,geo) )
                processCmd('mkdir -p %s' %(geo) )
                for mats in _individualmaterials:
                    processCmd('mkdir -p %s/%s/indimat/%s' %(opt.WWWAREA,geo,mats) )
                    processCmd('mkdir -p %s/indimat/%s' %(geo,mats) )

            index_file.write('  <ul>\n' )
            index_file.write('   <li><a href="%s/index.html">%s</a></li>\n' %(geo, geo ) )
            index_file.write('  </ul>\n' )
        index_file.write('   </li>\n' )
        index_file.write('  <br>\n' )
        index_file.write('  <br>\n' )
        index_file.write('  <br>\n' )

    #Writing postamble"
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()

    #This is the main html file for the validation webpage. In order to avoid 
    #surprises when experimenting, in order to copy it automatically to the 
    #www area you should have activated the relevant flag:  
    if (opt.COPYHTML) : processCmd('cp index.html %s/.' %(opt.WWWAREA) )

    #Let's make also the summary folder
    if (not os.path.isdir("HGCValid_summary_Plots")):  
        processCmd('mkdir -p HGCValid_summary_Plots')	

    #To avoid the nans transpose later                          
    df = pd.DataFrame.from_dict(_summary, orient = 'index').transpose()
    #Make a specific order in columns
    df = df[_summobj]

    index_file = open('HGCValid_summary_Plots/index.html','w')
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <body>\n')

    #Let's loop through RelVals
    for infi in phase2samples_noPU:
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        index_file.write( '<h2> %s </h2> \n' %(_sampleName[infi.name()]) )
        #table here with summary objects
        index_file.write('<table> \n')
        index_file.write('  <tr>\n')
        #This is the row with the headers. So, the objects for us.
        for obj in _summobj:
            index_file.write('    <th>%s</th>\n' %(_pageNameMap[obj]) )
        index_file.write('  </tr>\n')

        for i, row in df.iterrows():
            index_file.write('  <tr>\n')
            for j, column in row.iteritems():
                print(column)  
                index_file.write('    <td>\n')
                index_file.write('    <ul>\n')

#                if df[obj][ind] == None: 
                if column == None:  
                    index_file.write('    </ul>\n')
                    index_file.write('    </td>\n')
                    continue
                    #index_file.write(' \n')
                else:
                    #print(df[obj][ind])          
                    print(j)
                    #index_file.write(' <li><a href="plots_%s_%s">%s</a></li>   \n' %(samplename, df[obj][ind], df[obj][ind].partition("/")[2] ))
                    if "Tracksters" in j or "Test-TICL" in j or "TICL-patternRecognition" in j:
                        index_file.write(' <li><a href="../HGCValid_%s_Plots/plots_%s_%s">%s</a></li>   \n' %(j, samplename, column, column.replace("ticlTracksters","") ))
                    else:
                        index_file.write(' <li><a href="../HGCValid_%s_Plots/plots_%s_%s">%s</a></li>   \n' %(j, samplename, column, column.partition("/")[2] ))

                index_file.write('    </ul>\n')                        
                index_file.write('    </td>\n')

            index_file.write('  </tr>\n')

        index_file.write(' </table>\n')
        index_file.write('  <br/>\n' )
        index_file.write('  <br/>\n' )
        index_file.write('  <br/>\n' )

        #Writing postamble"
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()

    objects = opt.GATHER.split(",")

    localoutputdir = ""
    if "raw" in NotNormalRelease and "raw" in NotNormalRefRelease: 
        localoutputdir = NewRelease + "_raw1100" + "_vs_" + RefRelease + "_raw1100"
    elif "raw" in NotNormalRelease and "normal" in NotNormalRefRelease: 
        #localoutputdir = NewRelease + "_raw1100" + "_vs_" + RefRelease
        localoutputdir = NewRelease + "_D76" + "_vs_" + RefRelease
    elif "normal" in NotNormalRelease and "normal" in NotNormalRefRelease: 
        localoutputdir = NewRelease + "_vs_" + RefRelease
    else: 
        localoutputdir = NewRelease

    #make the structure to hold the objects
    for obj in objects:
        #This is where we will save the final output per campaing: 
        if (not os.path.isdir('%s/standalone' %(localoutputdir))) :
            processCmd('mkdir -p %s/standalone' %(localoutputdir))
        if (obj!="standalone"): processCmd('mv HGCValid_%s_Plots %s'%(obj, localoutputdir) )
        else : 
            processCmd('mv hgcalSimHitStudy %s/standalone/.'%(localoutputdir) )
            processCmd('mv hgcalDigiStudy %s/standalone/.'%(localoutputdir) )
            processCmd('mv hgcalRecHitStudy %s/standalone/.'%(localoutputdir) )
            processCmd('cp %s/../public/index.php %s/standalone/.'%(opt.WWWAREA, localoutputdir) )

    '''
    #Let's also copy to the summary folder what we need. 
    for infi in phase2samples_noPU:
        samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
        for obj in _summobj:
            #print obj 
            #if obj == "hitValidation" : samplename = samplename + infi.pileup()
            #else : samplename = infi.filename(NewRelease).replace("DQM_V0001_R000000001__","").replace("__DQMIO.root","")
            for ind in df.index:
                if df[obj][ind] == None: continue
                else: processCmd('cp -r %s/HGCValid_%s_Plots/plots_%s_%s %s/HGCValid_summary_Plots ' %(NewRelease, obj, samplename, df[obj][ind].partition("/")[0], NewRelease ) )
    '''

    #html file of the relval campaign we are validating
    index_file = open('%s/index.html'%(localoutputdir),'w')            
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')
    index_file.write('  <title> <h2> HGCAL validation results for %s </h2> </title>\n' %(localoutputdir) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')
    index_file.write(' <h2> HGCAL validation results for %s </h2> \n' %(localoutputdir) )

    for obj in objects:
        print(obj)
        if (obj!="standalone"):
            index_file.write('  <br/>\n' )
            index_file.write('  <ul>\n' )
            index_file.write('   <li><a href="HGCValid_%s_Plots/index.html">%s</a></li>\n' %(obj, _pageNameMap[obj] ) )
            index_file.write('  </ul>\n' )
            index_file.write('  <br/>\n' )
        else : 
            index_file.write('  <br/>\n' )
            index_file.write('  <ul>\n' )
            index_file.write('   <li><a href="%s/index.php">%s</a></li>\n' %(obj, _pageNameMap[obj] ) )
            index_file.write('  </ul>\n' )
            index_file.write('  <br/>\n' )


    #Writing postamble
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()

    #We choose to zip in uncompressed form all the files for two reasons:
    #1. Copying to eos so many files is really slow. It is faster to
    #   create one uncompressed file, copy that and unzip there.
    #2. Inevitably, we will have to do some cleanup of the older campaigns,
    #   since we will reach the number of files limit quite easily. 
    #   It will be easier to have already save the zip file and just delete
    #   the directory content, leaving inside only the zip file.

    # This will take some time. 
#    processCmd('zip -0 -r %s.zip %s' %(localoutputdir,localoutputdir) )
#    processCmd('cp %s.zip %s/.' %(localoutputdir,opt.WWWAREA) )
#    processCmd('cd %s' %(opt.WWWAREA) )
#    processCmd('unzip -q %s.zip' %(localoutputdir) )
#    processCmd('mv %s.zip %s/.' %(localoutputdir,localoutputdir) )
#    processCmd('cd -')


#------------------------------------------------------------------------------------------
#Geometry section: Here we gather results from geometry related validation packages.
#-------------------------------------------------------------------------------------------
#Keep in mind that the gne
if (opt.GEOMETRY) :
    #html file of the geometry scenario we are estimating the material budget
    index_file = open('%s/index.html'%(GeoScenario),'w')
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')
    index_file.write('  <title> <h2> HGCAL material budget results for %s </h2> </title>\n' %(GeoScenario) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')
    index_file.write(' <h2> HGCAL material budget results for %s </h2> \n' %(GeoScenario) )

    for obj in _MatBudSections:
        print(obj)
        #We need the directory for the geometry related results 
        if (not os.path.isdir('%s/%s/%s' %(opt.WWWAREA,GeoScenario,obj))):
            processCmd('mkdir -p %s/%s/%s' %(opt.WWWAREA,GeoScenario,obj) )
            processCmd('mkdir -p %s/%s' %(GeoScenario,obj) )

        index_file.write('  <br/>\n' )
        index_file.write('  <ul>\n' )
        index_file.write('   <li><a href="%s/index.html">%s</a></li>\n' %(obj, _geoPageNameMap[obj] ) )
        index_file.write('  </ul>\n' )
        index_file.write('  <br/>\n' )

    #Writing postamble
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()

    #Copy the material budget menu file in the current geometry scenario
    processCmd('cp %s/index.html %s/%s/.' %(GeoScenario, opt.WWWAREA,GeoScenario) )

    #html file for the menu of the individual materials
    index_file = open('%s/indimat/index.html'%(GeoScenario),'w')
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')
    index_file.write('  <title> <h2> HGCAL material budget results for individual materials for  %s </h2> </title>\n' %(GeoScenario) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')
    index_file.write(' <h2> HGCAL material budget results for individual materials for %s </h2> \n' %(GeoScenario) )
    for mats in _individualmaterials:
        print(mats)
        #index_file.write('  <br/>\n' )
        index_file.write('  <ul>\n' )
        index_file.write('   <li><a href="%s/index.html">%s</a></li>\n' %(mats, _matPageNameMap[mats] ) )
        index_file.write('  </ul>\n' )
        #index_file.write('  <br/>\n' )

    #Writing postamble
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()

    #Copy the menu html file for the individual materials
    processCmd('cp %s/indimat/index.html %s/%s/indimat/.' %(GeoScenario, opt.WWWAREA,GeoScenario) )

    #html file for all HGCal stack plots materials
    index_file = open('%s/allhgcal/index.html'%(GeoScenario),'w')
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')

    index_file.write(' <style>img.Reference{margin: 20px auto 20px auto; border: 10px solid green; border-radius: 10px;}img.New{margin: 20px auto 20px auto; border: 10px solid red; border-radius: 10px;} </style> \n')

    index_file.write(_hideShowFun["thestyle"])

    index_file.write('  <title> <h2> HGCAL material budget results for all materials for  %s </h2> </title>\n' %(GeoScenario) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')

    index_file.write(' <h2> HGCAL material budget results for : <span style="color:red;font-size:120%%" >All Materials </span></h2> \n' )

    index_file.write('<p> %s plots have a green border followed by the %s plots which features a red border. </p>\n' % (GeoScenario.split("_")[0], GeoScenario.split("_")[2]) )

    index_file.write('<h2> Geometry: <span style="color:green;" > %s</span>_vs_<span style="color:red;" >%s </span> </h2>\n' % (GeoScenario.split("_")[0], GeoScenario.split("_")[2]) )

    index_file.write('<hr/>\n')

    index_file.write(_hideShowFun["divTabs"])

    for region in ["_AllHGCAL", "_ZminusZoom", "_ZplusZoom"]:

        index_file.write('<div id="%s" class="tabcontent"> \n' %(region))
        pngnamestring = ""
        if region == "_AllHGCAL": pngnamestring = ""
        else: pngnamestring = region

        for allmatplot in _allmaterialsplots:
            if region == "_AllHGCAL":
                index_file.write('<p> %s <a href="../%s/%s%s.pdf" class="TMLlink">Click to enlarge %s plot</a></p>\n' %(_allmaterialsPlotsDesc[allmatplot], GeoScenario.split("_")[2],allmatplot,pngnamestring,GeoScenario.split("_")[2]))
                index_file.write('<img class="Reference" src="../%s/%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[0],allmatplot,pngnamestring) )
                index_file.write('<img class="New" src="../%s/%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[2],allmatplot,pngnamestring))
                index_file.write('<hr/>\n')
            elif region != "_AllHGCAL" and "HGCal_l_vs_z_vs_R" in allmatplot:
                index_file.write('<p> %s <a href="../%s/%s/%s%s.pdf" class="TMLlink">Click to enlarge %s plot</a></p>\n' %(_allmaterialsPlotsDesc[allmatplot], GeoScenario.split("_")[2],region.replace("_Zminus","ZMinus").replace("_Zplus","ZPlus"),allmatplot,pngnamestring,GeoScenario.split("_")[2]))
                index_file.write('<img class="Reference" src="../%s/%s/%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[0],region.replace("_Zminus","ZMinus").replace("_Zplus","ZPlus"),allmatplot,pngnamestring) )
                index_file.write('<img class="New" src="../%s/%s/%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[2],region.replace("_Zminus","ZMinus").replace("_Zplus","ZPlus"),allmatplot,pngnamestring))
                index_file.write('<hr/>\n')


        index_file.write('</div>\n')

    index_file.write(_hideShowFun["buttonandFunction"])
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()

    #Copy all materials budget file
    processCmd('cp %s/allhgcal/index.html %s/%s/allhgcal/.' %(GeoScenario, opt.WWWAREA,GeoScenario) )

    #html file of the individual materials for the material budget analysis
    for mats in _individualmaterials:
        index_file = open('%s/indimat/%s/index.html'%(GeoScenario,mats),'w')
        #Write preamble
        index_file.write('<html>\n')
        index_file.write(' <head>\n')

        index_file.write(' <style>img.Reference{margin: 20px auto 20px auto; border: 10px solid green; border-radius: 10px;}img.New{margin: 20px auto 20px auto; border: 10px solid red; border-radius: 10px;} </style> \n')

        index_file.write(_hideShowFun["thestyle"])

        index_file.write('  <title> <h2> HGCAL material budget results for individual materials for  %s </h2> </title>\n' %(GeoScenario) )
        index_file.write(' </head>\n')
        index_file.write(' <body>\n')
        index_file.write(' <h2> HGCAL material budget results for : <span style="color:red;font-size:120%%" >%s </span></h2> \n' %(_matPageNameMap[mats]) )

        index_file.write('<p> %s plots have a green border followed by the %s plots which features a red border. </p>\n' % (GeoScenario.split("_")[0], GeoScenario.split("_")[2]) )

        index_file.write('<h2> Geometry: <span style="color:green;" > %s</span>_vs_<span style="color:red;" >%s </span> </h2>\n' % (GeoScenario.split("_")[0], GeoScenario.split("_")[2]) )

        index_file.write('<hr/>\n')

        #--------------------------------------------------------------
        #This one below is a solution using a table with 3 columns: 
        #Two for the plots and the third for the text. 

        #index_file.write('<table style=\'font-size:120%%\' border="1" cellspacing="1" cellpadding="0">\n')
        #index_file.write('<tbody>\n')

        #for indiplots in _individualmatplots:
        #    index_file.write('<tr>\n')
        #    index_file.write('<td> <img class="Reference" src="../../%s/%s/%s%s.png" width="375"/> </td>\n' %(GeoScenario.split("_")[0],mats,indiplots,mats) )
        #    index_file.write('<td> <img class="New" src="../../%s/%s/%s%s.png" width="375"/> </td>\n' %(GeoScenario.split("_")[2],mats,indiplots,mats))
        #    index_file.write('<td> %s <a href="../../%s/%s/%s%s.pdf" class="TMLlink">Click to enlarge %s plot</a></td>\n' %(_individualMatPlotsDesc[indiplots].replace("THEMAT",_matPageNameMap[mats]), GeoScenario.split("_")[2],mats,indiplots,mats,GeoScenario.split("_")[2]))
        #    index_file.write('</tr>\n')

        #Writing postamble
        #index_file.write('</tbody>\n')
        #index_file.write('</table>\n')
        #--------------------------------------------------------------
        index_file.write(_hideShowFun["divTabs"])

        #Individual material here for: All HGCAL, Zminus, Zplus
        for region in ["_AllHGCAL", "_ZminusZoom", "_ZplusZoom"]:
            #The hide/show button
            #index_file.write(_hideShowFun["buttonandFunction%s"%(region)])

            index_file.write('<div id="%s" class="tabcontent"> \n' %(region))
            pngnamestring = ""
            if region == "_AllHGCAL": pngnamestring = ""
            else: pngnamestring = region 
            for indiplots in _individualmatplots: 
                if region == "_AllHGCAL":
                    index_file.write('<p> %s <a href="../../%s/%s/%s%s%s.pdf" class="TMLlink">Click to enlarge %s plot</a></p>\n' %(_individualMatPlotsDesc[indiplots].replace("THEMAT",_matPageNameMap[mats]), GeoScenario.split("_")[2],mats,indiplots,mats,pngnamestring,GeoScenario.split("_")[2]))
                    index_file.write('<img class="Reference" src="../../%s/%s/%s%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[0],mats,indiplots,mats,pngnamestring) )
                    index_file.write('<img class="New" src="../../%s/%s/%s%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[2],mats,indiplots,mats,pngnamestring))
                    index_file.write('<hr/>\n')
                else: 
                    index_file.write('<p> %s <a href="../../%s/%s/%s/%s%s%s.pdf" class="TMLlink">Click to enlarge %s plot</a></p>\n' %(_individualMatPlotsDesc[indiplots].replace("THEMAT",_matPageNameMap[mats]), GeoScenario.split("_")[2],mats,region.replace("_Zminus","ZMinus").replace("_Zplus","ZPlus"),indiplots,mats,pngnamestring,GeoScenario.split("_")[2]))
                    index_file.write('<img class="Reference" src="../../%s/%s/%s/%s%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[0],mats,region.replace("_Zminus","ZMinus").replace("_Zplus","ZPlus"),indiplots,mats,pngnamestring) )
                    index_file.write('<img class="New" src="../../%s/%s/%s/%s%s%s.png" width="375"/> \n' %(GeoScenario.split("_")[2],mats,region.replace("_Zminus","ZMinus").replace("_Zplus","ZPlus"),indiplots,mats,pngnamestring))
                    index_file.write('<hr/>\n')


            index_file.write('</div>\n')          

        index_file.write(_hideShowFun["buttonandFunction"]) 
        index_file.write(' </body>\n')
        index_file.write('</html>\n')
        index_file.close()

        #Copy the individual materials budget file
        processCmd('cp %s/indimat/%s/index.html %s/%s/indimat/%s/.' %(GeoScenario, mats, opt.WWWAREA,GeoScenario,mats) )

    #html file for from vertex up to muon stations
    index_file = open('%s/fromvertex/index.html'%(GeoScenario),'w')
    #Write preamble
    index_file.write('<html>\n')
    index_file.write(' <head>\n')

    index_file.write(' <style>img.Reference{margin: 20px auto 20px auto; border: 10px solid green; border-radius: 10px;}img.New{margin: 20px auto 20px auto; border: 10px solid red; border-radius: 10px;} </style> \n')

    index_file.write(_hideShowFun["thestyle"])

    index_file.write('  <title> <h2> HGCAL material budget results from vertex up to in front of muon stations for  %s </h2> </title>\n' %(GeoScenario) )
    index_file.write(' </head>\n')
    index_file.write(' <body>\n')

    index_file.write(' <h2> HGCAL material budget results from vertex up to in front of muon stations: <span style="color:red;font-size:120%%" >All detectors </span></h2> \n' )

    index_file.write('<p> %s plots have a green border followed by the %s plots which features a red border. </p>\n' % (GeoScenario.split("_")[0], GeoScenario.split("_")[2]) )

    index_file.write('<h2> Geometry: <span style="color:green;" > %s</span>_vs_<span style="color:red;" >%s </span> </h2>\n' % (GeoScenario.split("_")[0], GeoScenario.split("_")[2]) )

    index_file.write('<hr/>\n')

    #index_file.write(_hideShowFun["divTabs"])

    for vertexplots in _fromvertexplots:
        index_file.write('<p> %s </p>\n' %(_fromVertexPlotsDesc[vertexplots]))
        index_file.write('<img class="Reference" src="%s/Figures/MaterialBdg_FromVertexToBackOf%s.png" width="375"/> \n' %(GeoScenario.split("_")[0],vertexplots) )
        index_file.write('<img class="New" src="%s/Figures/MaterialBdg_FromVertexToBackOf%s.png" width="375"/> \n' %(GeoScenario.split("_")[2],vertexplots) )
        index_file.write('<hr/>\n')

    #index_file.write(_hideShowFun["buttonandFunction"])
    index_file.write(' </body>\n')
    index_file.write('</html>\n')
    index_file.close()

    #Copy all materials budget file
    processCmd('cp %s/fromvertex/index.html %s/%s/fromvertex/.' %(GeoScenario, opt.WWWAREA,GeoScenario) )

