from __future__ import print_function
import os
import sys
import copy
import collections

import ROOT
from ROOT import TFile, TString
from ROOT import gDirectory
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from Validation.RecoTrack.plotting.plotting import Plot, PlotGroup, PlotFolder, Plotter, PlotOnSideGroup
from Validation.RecoTrack.plotting.html import PlotPurpose
import Validation.RecoTrack.plotting.plotting as plotting
import Validation.RecoTrack.plotting.validation as validation
import Validation.RecoTrack.plotting.html as html

from Validation.HGCalValidation.HGCalValidator_cfi import hgcalValidator
from Validation.HGCalValidation.PostProcessorHGCAL_cfi import lcToCP_linking, simDict, tsToCP_linking, tsToSTS_patternRec, variables

hgcVal_dqm = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/"
#The number of layers per endcap in the current default geometry scenario. 
geometryscenario = 47

#To be able to spot any issues both in -z and +z a layer id was introduced
#that spans from 0 to 103 for hgcal_v9 geometry. The mapping for hgcal_v9 is:
#-z: 0->51
#+z: 52->103
#while for V10 is:
#-z: 0->49
#+z: 50->99
'''
layerscheme = { 'lastLayerEEzm': 0, 'lastLayerFHzm': 0, 'maxlayerzm': 0, 'lastLayerEEzp': 0, 'lastLayerFHzp': 0, 'maxlayerzp': 0 }

#Let's take the relevant values of layerscheme from the dqm file.
theDQMfile =  "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root"
if not os.path.isfile(theDQMfile):
    print("Error: file", theDQMfile, "not found, exit")
    sys.exit(0)


#Take general info from the first file is sufficient.
thefile = TFile( theDQMfile )
GeneralInfoDirectory = hgcVal_dqm + 'GeneralInfo'

if not gDirectory.GetDirectory( GeneralInfoDirectory ):
  print("Error: GeneralInfo directory not found in DQM file, exit")
  sys.exit(0)

keys = gDirectory.GetDirectory( GeneralInfoDirectory ).GetListOfKeys()
key = keys[0]
layvalue = 0
while key:
    obj = key.ReadObj()
    for laykey in layerscheme.keys():
      if laykey in obj.GetName():
        layvalue = obj.GetName()[len("<"+laykey+">i="):-len("</"+laykey+">")]
        layerscheme[laykey] = layvalue
        #print(layvalue)
    key = keys.After(key)

thefile.Close()

print(layerscheme)
#TODO: Anticipating the fine/coarse layer information in CMSSW we overwrite values from DQM file
#For now values returned for
# 'lastLayerFHzp': '104', 'lastLayerFHzm': '52'
#are not the one expected. Will come back to this when there will be info in CMSSW to put in DQM file.
#For V9:
#layerscheme = { 'lastLayerEEzm': 28, 'lastLayerFHzm': 40, 'maxlayerzm': 52, 'lastLayerEEzp': 80, 'lastLayerFHzp': 92, 'maxlayerzp': 104 }
#For V10:
layerscheme = { 'lastLayerEEzm': 28, 'lastLayerFHzm': 40, 'maxlayerzm': 50, 'lastLayerEEzp': 78, 'lastLayerFHzp': 90, 'maxlayerzp': 100 }
#For V16
layerscheme = { 'lastLayerEEzm': 26, 'lastLayerFHzm': 37, 'maxlayerzm': 47, 'lastLayerEEzp': 73, 'lastLayerFHzp': 84, 'maxlayerzp': 94 }
'''
#print(layerscheme)

layerscheme = {}

if geometryscenario == 52:
   layerscheme = { 'lastLayerEEzm': 28, 'lastLayerFHzm': 40, 'maxlayerzm': 52, 'lastLayerEEzp': 80, 'lastLayerFHzp': 92, 'maxlayerzp': 104 }
elif geometryscenario == 50:
   layerscheme = { 'lastLayerEEzm': 28, 'lastLayerFHzm': 40, 'maxlayerzm': 50, 'lastLayerEEzp': 78, 'lastLayerFHzp': 90, 'maxlayerzp': 100 }
elif geometryscenario == 47:
   layerscheme = { 'lastLayerEEzm': 26, 'lastLayerFHzm': 37, 'maxlayerzm': 47, 'lastLayerEEzp': 73, 'lastLayerFHzp': 84, 'maxlayerzp': 94 }
#print(layerscheme)

lastLayerEEzm = layerscheme['lastLayerEEzm']  # last layer of EE -z
lastLayerFHzm = layerscheme['lastLayerFHzm']  # last layer of FH -z
maxlayerzm = layerscheme['maxlayerzm'] # last layer of BH -z
lastLayerEEzp = layerscheme['lastLayerEEzp']  # last layer of EE +z
lastLayerFHzp = layerscheme['lastLayerFHzp']  # last layer of FH +z
maxlayerzp = layerscheme['maxlayerzp'] # last layer of BH +z

hitlayerscheme = { 'EE_min': 1,'EE_max': 28, 'HESilicon_min': 1, 'HESilicon_max': 22, 'HEScintillator_min': 9 , 'HEScintillator_max': 22 }
#print(hitlayerscheme)

EE_min = hitlayerscheme['EE_min']
EE_max = hitlayerscheme['EE_max']
HESilicon_min = hitlayerscheme['HESilicon_min']
HESilicon_max = hitlayerscheme['HESilicon_max']
HEScintillator_min = hitlayerscheme['HEScintillator_min']
HEScintillator_max = hitlayerscheme['HEScintillator_max']

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }
_legend_common = {"legendDx": -0.3,
                  "legendDy": -0.05,
                  "legendDw": 0.1}

_SelectedCaloParticles = PlotGroup("SelectedCaloParticles", [
        Plot("num_caloparticle_eta", xtitle="", **_common),
        Plot("caloparticle_energy", xtitle="", **_common),
        Plot("caloparticle_pt", xtitle="", **_common),
        Plot("caloparticle_phi", xtitle="", **_common),
        Plot("Eta vs Zorigin", xtitle="", **_common),
       ])

#Need to adjust the statbox to see better the plot
_common = {"stat": True, "drawStyle": "hist", "statx": 0.38, "staty": 0.68 }
_num_reco_cluster_eta = PlotGroup("num_reco_cluster_eta", [
  Plot("num_reco_cluster_eta", xtitle="", **_common),
],ncols=1)
#Back to normal
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

_mixedhitsclusters = PlotGroup("mixedhitsclusters", [
  Plot("mixedhitscluster_zminus", xtitle="", **_common),
  Plot("mixedhitscluster_zplus", xtitle="", **_common),
],ncols=2)

_mixedhitssimclusters = PlotGroup("mixedhitssimclusters", [
  Plot("mixedhitssimcluster_zminus", xtitle="", **_common),
  Plot("mixedhitssimcluster_zplus", xtitle="", **_common),
],ncols=2)

#Just to prevent the stabox covering the plot
_common = {"stat": True, "drawStyle": "hist", "statx": 0.45, "staty": 0.65 }

_energyclustered = PlotGroup("energyclustered", [
  Plot("energyclustered_zminus", xtitle="", **_common),
  Plot("energyclustered_zplus", xtitle="", **_common),
],ncols=2)

#Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

_longdepthbarycentre = PlotGroup("longdepthbarycentre", [
  Plot("longdepthbarycentre_zminus", xtitle="", **_common),
  Plot("longdepthbarycentre_zplus", xtitle="", **_common),
],ncols=2)

_common_layerperthickness = {}
_common_layerperthickness.update(_common)
_common_layerperthickness['xmin'] = 0.
_common_layerperthickness['xmax'] = 100

_totclusternum_thick = PlotGroup("totclusternum_thick", [
  Plot("totclusternum_thick_120", xtitle="", **_common_layerperthickness),
  Plot("totclusternum_thick_200", xtitle="", **_common_layerperthickness),
  Plot("totclusternum_thick_300", xtitle="", **_common_layerperthickness),
  Plot("totclusternum_thick_-1", xtitle="", **_common_layerperthickness),
  Plot("mixedhitscluster", xtitle="", **_common_layerperthickness),
])

_totsimclusternum_thick = PlotGroup("totsimclusternum_thick", [
  Plot("totsimclusternum_thick_120", xtitle="", **_common_layerperthickness),
  Plot("totsimclusternum_thick_200", xtitle="", **_common_layerperthickness),
  Plot("totsimclusternum_thick_300", xtitle="", **_common_layerperthickness),
  Plot("totsimclusternum_thick_-1", xtitle="", **_common_layerperthickness),
  Plot("mixedhitssimcluster", xtitle="", **_common_layerperthickness),
])

#We will plot the density in logy scale.
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ylog": True}

_cellsenedens_thick =  PlotGroup("cellsenedens_thick", [
  Plot("cellsenedens_thick_120", xtitle="", **_common),
  Plot("cellsenedens_thick_200", xtitle="", **_common),
  Plot("cellsenedens_thick_300", xtitle="", **_common),
  Plot("cellsenedens_thick_-1", xtitle="", **_common),
])

#Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }


#--------------------------------------------------------------------------------------------
# z-
#--------------------------------------------------------------------------------------------
_totclusternum_layer_EE_zminus = PlotGroup("totclusternum_layer_EE", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_totclusternum_layer_FH_zminus = PlotGroup("totclusternum_layer_FH", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_totclusternum_layer_BH_zminus = PlotGroup("totclusternum_layer_BH", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

_totsimclusternum_layer_EE_zminus = PlotGroup("totsimclusternum_layer_EE_zminus", [
  Plot("totsimclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_totsimclusternum_layer_FH_zminus = PlotGroup("totsimclusternum_layer_FH_zminus", [
  Plot("totsimclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_totsimclusternum_layer_BH_zminus = PlotGroup("totsimclusternum_layer_BH_zminus", [
  Plot("totsimclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

_energyclustered_perlayer_EE_zminus = PlotGroup("energyclustered_perlayer_EE", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_energyclustered_perlayer_FH_zminus = PlotGroup("energyclustered_perlayer_FH", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_energyclustered_perlayer_BH_zminus = PlotGroup("energyclustered_perlayer_BH", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_common_cells = {}
_common_cells.update(_common)
_common_cells["xmin"] = 0
_common_cells["xmax"] = 50
_common_cells["ymin"] = 0.1
_common_cells["ymax"] = 10000
_common_cells["ylog"] = True
_cellsnum_perthick_perlayer_120_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_120_EE", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=7)

_cellsnum_perthick_perlayer_120_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_120_FH", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_cellsnum_perthick_perlayer_120_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_120_BH", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#200 um
_cellsnum_perthick_perlayer_200_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_200_EE", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=7)

_cellsnum_perthick_perlayer_200_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_200_FH", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_cellsnum_perthick_perlayer_200_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_200_BH", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#300 um
_cellsnum_perthick_perlayer_300_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_300_EE", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=7)

_cellsnum_perthick_perlayer_300_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_300_FH", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_cellsnum_perthick_perlayer_300_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_300_BH", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#scint um
_cellsnum_perthick_perlayer_scint_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_Sci_EE", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=7)

_cellsnum_perthick_perlayer_scint_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_Sci_FH", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_cellsnum_perthick_perlayer_scint_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_Sci_BH", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_common_distance = {}
_common_distance.update(_common)
_common_distance.update(_legend_common)
_common_distance["xmax"] = 150
_common_distance["stat"] = False
_common_distance["ymin"] = 1e-3
_common_distance["ymax"] = 10000
_common_distance["ylog"] = True

_distancetomaxcell_perthickperlayer_120_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_EE", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_120_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_FH", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_120_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_BH", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#200 um
_distancetomaxcell_perthickperlayer_200_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_EE", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_200_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_FH", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_200_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_BH", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#300 um
_distancetomaxcell_perthickperlayer_300_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_EE", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_300_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_FH", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_300_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_BH", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#scint um
_distancetomaxcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_Sci_EE", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_Sci_FH", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_Sci_BH", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcell_perthickperlayer_120_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_120_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_120_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#200 um
_distancebetseedandmaxcell_perthickperlayer_200_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_200_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_200_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#300 um
_distancebetseedandmaxcell_perthickperlayer_300_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_300_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_300_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#scint um
_distancebetseedandmaxcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_Sci_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_Sci_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_Sci_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#200 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#300 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#scint um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_Sci_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_Sci_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_Sci_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_120_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_EE", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_120_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_FH", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_120_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_BH", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#200 um
_distancetoseedcell_perthickperlayer_200_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_EE", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_200_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_FH", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_200_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_BH", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#300 um
_distancetoseedcell_perthickperlayer_300_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_EE", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_300_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_FH", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_300_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_BH", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#scint um
_distancetoseedcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_Sci_EE", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_Sci_FH", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_Sci_BH", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#=====================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#We need points for the weighted plots
_common = {"stat": True, "drawStyle": "EP", "staty": 0.65 }
#120 um
_distancetomaxcell_perthickperlayer_eneweighted_120_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_120_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_120_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#200 um
_distancetomaxcell_perthickperlayer_eneweighted_200_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_200_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_200_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#300 um
_distancetomaxcell_perthickperlayer_eneweighted_300_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_300_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_300_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)
#scint um
_distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_Sci_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_Sci_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_Sci_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)


#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_eneweighted_120_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_120_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_120_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#200 um
_distancetoseedcell_perthickperlayer_eneweighted_200_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_200_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_200_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#300 um
_distancetoseedcell_perthickperlayer_eneweighted_300_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_300_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_300_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#scint um
_distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_Sci_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_Sci_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_Sci_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#Coming back to the usual definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_totclusternum_layer_EE_zplus = PlotGroup("totclusternum_layer_EE", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_totclusternum_layer_FH_zplus = PlotGroup("totclusternum_layer_FH", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_totclusternum_layer_BH_zplus = PlotGroup("totclusternum_layer_BH", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

_totsimclusternum_layer_EE_zplus = PlotGroup("totsimclusternum_layer_EE_zplus", [
  Plot("totsimclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_totsimclusternum_layer_FH_zplus = PlotGroup("totsimclusternum_layer_FH_zplus", [
  Plot("totsimclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_totsimclusternum_layer_BH_zplus = PlotGroup("totsimclusternum_layer_BH_zplus", [
  Plot("totsimclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

_energyclustered_perlayer_EE_zplus = PlotGroup("energyclustered_perlayer_EE", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_energyclustered_perlayer_FH_zplus = PlotGroup("energyclustered_perlayer_FH", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_energyclustered_perlayer_BH_zplus = PlotGroup("energyclustered_perlayer_BH", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_cellsnum_perthick_perlayer_120_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_120_EE", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_cellsnum_perthick_perlayer_120_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_120_FH", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)
_cellsnum_perthick_perlayer_120_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_120_BH", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#200 um
_cellsnum_perthick_perlayer_200_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_200_EE", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_cellsnum_perthick_perlayer_200_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_200_FH", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_cellsnum_perthick_perlayer_200_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_200_BH", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)
#300 um
_cellsnum_perthick_perlayer_300_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_300_EE", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_cellsnum_perthick_perlayer_300_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_300_FH", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)
_cellsnum_perthick_perlayer_300_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_300_BH", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#scint um
_cellsnum_perthick_perlayer_scint_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_Sci_EE", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_cellsnum_perthick_perlayer_scint_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_Sci_FH", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_cellsnum_perthick_perlayer_scint_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_Sci_BH", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_common_distance = {}
_common_distance.update(_common)
_common_distance.update(_legend_common)
_common_distance["xmax"] = 150
_common_distance["stat"] = False
_common_distance["ymin"] = 1e-3
_common_distance["ymax"] = 10000
_common_distance["ylog"] = True

_distancetomaxcell_perthickperlayer_120_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_EE", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_120_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_FH", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_120_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_BH", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#200 um
_distancetomaxcell_perthickperlayer_200_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_EE", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_200_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_FH", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_200_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_BH", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#300 um
_distancetomaxcell_perthickperlayer_300_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_EE", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_300_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_FH", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_300_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_BH", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#scint um
_distancetomaxcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_Sci_EE", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_Sci_FH", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_Sci_BH", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcell_perthickperlayer_120_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_120_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_120_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#200 um
_distancebetseedandmaxcell_perthickperlayer_200_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_200_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_200_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#300 um
_distancebetseedandmaxcell_perthickperlayer_300_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_300_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_300_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#scint um
_distancebetseedandmaxcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_Sci_EE", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_Sci_FH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_Sci_BH", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#200 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#300 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#scint um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_Sci_EE", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_Sci_FH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_Sci_BH", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)


#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_120_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_EE", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_120_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_FH", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_120_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_BH", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#200 um
_distancetoseedcell_perthickperlayer_200_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_EE", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_200_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_FH", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_200_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_BH", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#300 um
_distancetoseedcell_perthickperlayer_300_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_EE", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_300_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_FH", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_300_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_BH", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#scint um
_distancetoseedcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_Sci_EE", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_Sci_FH", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_Sci_BH", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#=====================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#We need points for the weighted plots
_common = {"stat": True, "drawStyle": "EP", "staty": 0.65 }

#120 um
_distancetomaxcell_perthickperlayer_eneweighted_120_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_120_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_120_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#200 um
_distancetomaxcell_perthickperlayer_eneweighted_200_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)
_distancetomaxcell_perthickperlayer_eneweighted_200_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_200_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#300 um
_distancetomaxcell_perthickperlayer_eneweighted_300_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_300_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_300_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#scint um
_distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_Sci_EE", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_Sci_FH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_Sci_BH", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_eneweighted_120_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_120_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_120_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#200 um
_distancetoseedcell_perthickperlayer_eneweighted_200_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_200_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_200_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#300 um
_distancetoseedcell_perthickperlayer_eneweighted_300_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_300_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_300_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#scint um
_distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_Sci_EE", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_Sci_FH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_Sci_BH", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)
#Just in case we add some plots below to be on the safe side.
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#--------------------------------------------------------------------------------------------
# z-
#--------------------------------------------------------------------------------------------

_common_score = {"title": "Score CaloParticle to LayerClusters in z-",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 1000,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_score.update(_legend_common)
_score_caloparticle_to_layerclusters_zminus = PlotGroup("score_caloparticle_to_layercluster", [
        Plot("Score_caloparticle2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_score) for i in range(0,maxlayerzm)
        ], ncols=10 )

_common_score = {"title": "Score LayerCluster to CaloParticles in z-",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 1000,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_score.update(_legend_common)
_score_layercluster_to_caloparticles_zminus = PlotGroup("score_layercluster_to_caloparticle", [
        Plot("Score_layercl2caloparticle_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_score) for i in range(0,maxlayerzm)
        ], ncols=8 )

_common_shared= {"title": "Shared Energy CaloParticle To Layer Cluster in z-",
                 "stat": False,
                 "legend": False,
                }
_common_shared.update(_legend_common)
_shared_plots_zminus = [Plot("SharedEnergy_caloparticle2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(0,maxlayerzm)]
_shared_plots_zminus.extend([Plot("SharedEnergy_caloparticle2layercl_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(0,maxlayerzm)])
_shared_plots_zminus.extend([Plot("SharedEnergy_caloparticle2layercl_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(0,maxlayerzm)])
_sharedEnergy_caloparticle_to_layercluster_zminus = PlotGroup("sharedEnergy_caloparticle_to_layercluster", _shared_plots_zminus, ncols=8)

_common_shared= {"title": "Shared Energy Layer Cluster To CaloParticle in z-",
                 "stat": False,
                 "legend": False,
                }
_common_shared.update(_legend_common)
_shared_plots2_zminus = [Plot("SharedEnergy_layercluster2caloparticle_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(0,maxlayerzm)]
_common_shared= {"title": "Shared Energy Layer Cluster To Best CaloParticle in z-",
                 "stat": False,
                 "legend": False,
                 "ymin": 0,
                 "ymax": 1
                }
_common_shared.update(_legend_common)
_shared_plots2_zminus.extend([Plot("SharedEnergy_layercl2caloparticle_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(0,maxlayerzm)])
_shared_plots2_zminus.extend([Plot("SharedEnergy_layercl2caloparticle_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(0,maxlayerzm)])
_sharedEnergy_layercluster_to_caloparticle_zminus = PlotGroup("sharedEnergy_layercluster_to_caloparticle", _shared_plots2_zminus, ncols=8)


_common_assoc = {#"title": "Cell Association Table in z-",
                 "stat": False,
                 "legend": False,
                 "xbinlabels": ["", "TN(pur)", "FN(ineff.)", "FP(fake)", "TP(eff)"],
                 "xbinlabeloption": "h",
                 "drawStyle": "hist",
                 "ymin": 0.1,
                 "ymax": 10000,
                 "ylog": True}
_common_assoc.update(_legend_common)
_cell_association_table_zminus = PlotGroup("cellAssociation_table", [
        Plot("cellAssociation_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_assoc) for i in range(0,maxlayerzm)
        ], ncols=8 )

_bin_count = 0
_xbinlabels = [ "{:02d}".format(i+1) for i in range(0,maxlayerzm) ]
_xtitle = "Layer Numbers in z-"
_common_eff = {"stat": False, "legend": False, "ymin": 0.0, "ymax": 1.1, "xbinlabeloption": "d"}
_effplots_zminus_eta = [Plot("effic_eta_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(0,maxlayerzm)]
_effplots_zminus_phi = [Plot("effic_phi_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(0,maxlayerzm)]
_common_eff = {"stat": False, "legend": False, "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_eff["xmin"] = _bin_count
_common_eff["xmax"] = maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_effplots_zminus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Efficiency", **_common_eff)]
_efficiencies_zminus_eta = PlotGroup("Efficiencies_vs_eta", _effplots_zminus_eta, ncols=10)
_efficiencies_zminus_phi = PlotGroup("Efficiencies_vs_phi", _effplots_zminus_phi, ncols=10)
_efficiencies_zminus     = PlotGroup("Efficiencies_vs_layer", _effplots_zminus, ncols=1)

_common_dup = {"stat": False, "legend": False, "ymin":0.0, "ymax":1.1}
_dupplots_zminus_eta = [Plot("duplicate_eta_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(0,maxlayerzm)]
_dupplots_zminus_phi = [Plot("duplicate_phi_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(0,maxlayerzm)]
_common_dup = {"stat": False, "legend": False, "title": "Global Duplicates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_dup["xmin"] = _bin_count
_common_dup["xmax"] = _common_dup["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_dupplots_zminus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Duplicates", **_common_dup)]
_duplicates_zminus_eta = PlotGroup("Duplicates_vs_eta", _dupplots_zminus_eta, ncols=10)
_duplicates_zminus_phi = PlotGroup("Duplicates_vs_phi", _dupplots_zminus_phi, ncols=10)
_duplicates_zminus     = PlotGroup("Duplicates_vs_layer", _dupplots_zminus, ncols=1)

_common_fake = {"stat": False, "legend": False, "ymin":0.0, "ymax":1.1}
_fakeplots_zminus_eta = [Plot("fake_eta_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(0,maxlayerzm)]
_fakeplots_zminus_phi = [Plot("fake_phi_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(0,maxlayerzm)]
_common_fake = {"stat": False, "legend": False, "title": "Global Fake Rates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_fake["xmin"] = _bin_count
_common_fake["xmax"] = _common_fake["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_common_fake["xbinlabelsize"] = 10.
_fakeplots_zminus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Fake Rate", **_common_fake)]
_fakes_zminus_eta = PlotGroup("FakeRate_vs_eta", _fakeplots_zminus_eta, ncols=10)
_fakes_zminus_phi = PlotGroup("FakeRate_vs_phi", _fakeplots_zminus_phi, ncols=10)
_fakes_zminus     = PlotGroup("FakeRate_vs_layer", _fakeplots_zminus, ncols=1)

_common_merge = {"stat": False, "legend": False, "ymin":0.0, "ymax":1.1}
_mergeplots_zminus_eta = [Plot("merge_eta_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(0,maxlayerzm)]
_mergeplots_zminus_phi = [Plot("merge_phi_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(0,maxlayerzm)]
_common_merge = {"stat": False, "legend": False, "title": "Global Merge Rates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_merge["xmin"] = _bin_count
_common_merge["xmax"] = _common_merge["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_common_merge["xbinlabelsize"] = 10.
_mergeplots_zminus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Merge Rate", **_common_merge)]
_merges_zminus_eta = PlotGroup("MergeRate_vs_eta", _mergeplots_zminus_eta, ncols=10)
_merges_zminus_phi = PlotGroup("MergeRate_vs_phi", _mergeplots_zminus_phi, ncols=10)
_merges_zminus     = PlotGroup("MergeRate_vs_layer", _mergeplots_zminus, ncols=1)


_common_energy_score = dict(removeEmptyBins=False, xbinlabelsize=10,
    stat=True,
    xbinlabeloption="d",
    ncols=1,
    xmin=0.001,
    xmax=1.,
    ymin=0.01,
    ymax=1.)
_energyscore_cp2lc_zminus = PlotGroup("Energy_vs_Score_CP2LC", [Plot("Energy_vs_Score_caloparticle2layer_perlayer{:02d}".format(i), title="Energy_vs_Score_CP2LC", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(0, maxlayerzm)
                                                     ], ncols=10)

_energyscore_cp2lc_zplus = PlotGroup("Energy_vs_Score_CP2LC", [Plot("Energy_vs_Score_caloparticle2layer_perlayer{:02d}".format(i), title="Energy_vs_Score_CP2LC", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(maxlayerzm,maxlayerzp)
                                                     ], ncols=10)
_common_energy_score["xmin"]=-0.1
_energyscore_lc2cp_zminus = PlotGroup("Energy_vs_Score_LC2CP", [Plot("Energy_vs_Score_layer2caloparticle_perlayer{:02d}".format(i), title="Energy_vs_Score_LC2CP", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(0, maxlayerzm)
                                                     ], ncols=10)
_energyscore_lc2cp_zplus = PlotGroup("Energy_vs_Score_LC2CP", [Plot("Energy_vs_Score_layer2caloparticle_perlayer{:02d}".format(i), title="Energy_vs_Score_LC2CP", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(maxlayerzm,maxlayerzp)
                                                     ], ncols=10)

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_common_score = {"title": "Score CaloParticle to LayerClusters in z+",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 1000,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_score.update(_legend_common)
_score_caloparticle_to_layerclusters_zplus = PlotGroup("score_caloparticle_to_layercluster", [
        Plot("Score_caloparticle2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_score) for i in range(maxlayerzm,maxlayerzp)
        ], ncols=10 )

_common_score = {"title": "Score LayerCluster to CaloParticles in z+",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 1000,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_score.update(_legend_common)
_score_layercluster_to_caloparticles_zplus = PlotGroup("score_layercluster_to_caloparticle", [
        Plot("Score_layercl2caloparticle_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_score) for i in range(maxlayerzm,maxlayerzp)
        ], ncols=8 )

_common_shared= {"title": "Shared Energy CaloParticle To Layer Cluster in z+",
                 "stat": False,
                 "legend": False,
                }
_common_shared.update(_legend_common)
_shared_plots_zplus = [Plot("SharedEnergy_caloparticle2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(maxlayerzm,maxlayerzp)]
_shared_plots_zplus.extend([Plot("SharedEnergy_caloparticle2layercl_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(maxlayerzm,maxlayerzp)])
_shared_plots_zplus.extend([Plot("SharedEnergy_caloparticle2layercl_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(maxlayerzm,maxlayerzp)])
_sharedEnergy_caloparticle_to_layercluster_zplus = PlotGroup("sharedEnergy_caloparticle_to_layercluster", _shared_plots_zplus, ncols=8)

_common_shared= {"title": "Shared Energy Layer Cluster To CaloParticle in z+",
                 "stat": False,
                 "legend": False,
                }
_common_shared.update(_legend_common)
_shared_plots2_zplus = [Plot("SharedEnergy_layercluster2caloparticle_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(maxlayerzm,maxlayerzp)]
_common_shared= {"title": "Shared Energy Layer Cluster To Best CaloParticle in z+",
                 "stat": False,
                 "legend": False,
                 "ymin": 0,
                 "ymax": 1,
                }
_common_shared.update(_legend_common)
_shared_plots2_zplus.extend([Plot("SharedEnergy_layercl2caloparticle_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(maxlayerzm,maxlayerzp)])
_shared_plots2_zplus.extend([Plot("SharedEnergy_layercl2caloparticle_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_shared) for i in range(maxlayerzm,maxlayerzp)])
_sharedEnergy_layercluster_to_caloparticle_zplus = PlotGroup("sharedEnergy_layercluster_to_caloparticle", _shared_plots2_zplus, ncols=8)


_common_assoc = {#"title": "Cell Association Table in z+",
                 "stat": False,
                 "legend": False,
                 "xbinlabels": ["", "TN(pur)", "FN(ineff.)", "FP(fake)", "TP(eff)"],
                 "xbinlabeloption": "h",
                 "drawStyle": "hist",
                 "ymin": 0.1,
                 "ymax": 10000,
                 "ylog": True}
_common_assoc.update(_legend_common)
_cell_association_table_zplus = PlotGroup("cellAssociation_table", [
        Plot("cellAssociation_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_assoc) for i in range(maxlayerzm,maxlayerzp)
        ], ncols=8 )


_bin_count = maxlayerzm
_xtitle = "Layer Numbers in z+"
_common_eff = {"stat": False, "legend": False, "ymin":0.0, "ymax":1.1}
_effplots_zplus_eta = [Plot("effic_eta_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(maxlayerzm,maxlayerzp)]
_effplots_zplus_phi = [Plot("effic_phi_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(maxlayerzm,maxlayerzp)]
_common_eff = {"stat": False, "legend": False, "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_eff["xmin"] = _bin_count
_common_eff["xmax"] = _common_eff["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_effplots_zplus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Efficiency", **_common_eff)]
_efficiencies_zplus_eta = PlotGroup("Efficiencies_vs_eta", _effplots_zplus_eta, ncols=10)
_efficiencies_zplus_phi = PlotGroup("Efficiencies_vs_phi", _effplots_zplus_phi, ncols=10)
_efficiencies_zplus = PlotGroup("Efficiencies_vs_layer", _effplots_zplus, ncols=1)

_common_dup = {"stat": False, "legend": False, "ymin": 0.0, "ymax": 1.1}
_dupplots_zplus_eta = [Plot("duplicate_eta_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(maxlayerzm,maxlayerzp)]
_dupplots_zplus_phi = [Plot("duplicate_phi_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(maxlayerzm,maxlayerzp)]
_common_dup = {"stat": False, "legend": False, "title": "Global Duplicates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_dup["xmin"] = _bin_count
_common_dup["xmax"] = _common_dup["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_dupplots_zplus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Duplicates", **_common_dup)]
_duplicates_zplus_eta = PlotGroup("Duplicates_vs_eta", _dupplots_zplus_eta, ncols=10)
_duplicates_zplus_phi = PlotGroup("Duplicates_vs_phi", _dupplots_zplus_phi, ncols=10)
_duplicates_zplus = PlotGroup("Duplicates_vs_layer", _dupplots_zplus, ncols=1)

_common_fake = {"stat": False, "legend": False, "ymin": 0.0, "ymax": 1.1}
_fakeplots_zplus_eta = [Plot("fake_eta_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(maxlayerzm,maxlayerzp)]
_fakeplots_zplus_phi = [Plot("fake_phi_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(maxlayerzm,maxlayerzp)]
_common_fake = {"stat": False, "legend": False, "title": "Global Fake Rates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_fake["xmin"] = _bin_count
_common_fake["xmax"] = _common_fake["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_fakeplots_zplus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Fake Rate", **_common_fake)]
_fakes_zplus_eta = PlotGroup("FakeRate_vs_eta", _fakeplots_zplus_eta, ncols=10)
_fakes_zplus_phi = PlotGroup("FakeRate_vs_phi", _fakeplots_zplus_phi, ncols=10)
_fakes_zplus = PlotGroup("FakeRate_vs_layer", _fakeplots_zplus, ncols=1)

_common_merge = {"stat": False, "legend": False, "ymin": 0.0, "ymax": 1.1}
_mergeplots_zplus_eta = [Plot("merge_eta_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(maxlayerzm,maxlayerzp)]
_mergeplots_zplus_phi = [Plot("merge_phi_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(maxlayerzm,maxlayerzp)]
_common_merge = {"stat": False, "legend": False, "title": "Global Merge Rates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloption": "v", "ymin": 0.0, "ymax": 1.1}
_common_merge["xmin"] = _bin_count
_common_merge["xmax"] = _common_merge["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_mergeplots_zplus = [Plot("globalEfficiencies", xtitle=_xtitle, ytitle="Merge Rate", **_common_merge)]
_merges_zplus_eta = PlotGroup("MergeRate_vs_eta", _mergeplots_zplus_eta, ncols=10)
_merges_zplus_phi = PlotGroup("MergeRate_vs_phi", _mergeplots_zplus_phi, ncols=10)
_merges_zplus = PlotGroup("MergeRate_vs_layer", _mergeplots_zplus, ncols=1)

#--------------------------------------------------------------------------------------------
# SimClusters
#--------------------------------------------------------------------------------------------

_common_sc_score = {"title": "Score SimCluster to LayerClusters in z-",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 10**6,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_sc_score.update(_legend_common)
_score_simcluster_to_layerclusters_zminus = PlotGroup("score_simcluster_to_layercluster_zminus", [
        Plot("Score_simcluster2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_score) for i in range(0,maxlayerzm)
        ], ncols=10 )

_common_sc_score = {"title": "Score LayerCluster to SimClusters in z-",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 10**6,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_sc_score.update(_legend_common)
_score_layercluster_to_simclusters_zminus = PlotGroup("score_layercluster_to_simcluster_zminus", [
        Plot("Score_layercl2simcluster_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_score) for i in range(0,maxlayerzm)
        ], ncols=8 )

_common_sc_shared= {"title": "Shared Energy SimCluster To Layer Cluster in z-",
                 "stat": False,
                 "legend": False,
                }
_common_sc_shared.update(_legend_common)
_shared_sc_plots_zminus = [Plot("SharedEnergy_simcluster2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(0,maxlayerzm)]
_shared_sc_plots_zminus.extend([Plot("SharedEnergy_simcluster2layercl_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(0,maxlayerzm)])
_shared_sc_plots_zminus.extend([Plot("SharedEnergy_simcluster2layercl_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(0,maxlayerzm)])
_sharedEnergy_simcluster_to_layercluster_zminus = PlotGroup("sharedEnergy_simcluster_to_layercluster_zminus", _shared_sc_plots_zminus, ncols=8)

_common_sc_shared= {"title": "Shared Energy Layer Cluster To SimCluster in z-",
                 "stat": False,
                 "legend": False,
                }
_common_sc_shared.update(_legend_common)
_shared_plots2_sc_zminus = [Plot("SharedEnergy_layercluster2simcluster_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(0,maxlayerzm)]
_common_sc_shared= {"title": "Shared Energy Layer Cluster To Best SimCluster in z-",
                 "stat": False,
                 "legend": False,
                 "ymin": 0,
                 "ymax": 1
                }
_common_sc_shared.update(_legend_common)
_shared_plots2_sc_zminus.extend([Plot("SharedEnergy_layercl2simcluster_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(0,maxlayerzm)])
_shared_plots2_sc_zminus.extend([Plot("SharedEnergy_layercl2simcluster_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(0,maxlayerzm)])
_sharedEnergy_layercluster_to_simcluster_zminus = PlotGroup("sharedEnergy_layercluster_to_simcluster_zminus", _shared_plots2_sc_zminus, ncols=8)

_bin_count = 0
_xbinlabels = [ "L{:02d}".format(i+1) for i in range(0,maxlayerzm) ]
_common_eff = {"stat": False, "legend": False}
_effplots_sc_zminus_eta = [Plot("effic_eta_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(0,maxlayerzm)]
_effplots_sc_zminus_phi = [Plot("effic_phi_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(0,maxlayerzm)]
_common_eff = {"stat": False, "legend": False, "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_eff["xmin"] = _bin_count
_common_eff["xmax"] = maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_effplots_sc_zminus = [Plot("globalEfficiencies_zminus", xtitle="Global Efficiencies in z-", **_common_eff)]
_efficiencies_sc_zminus_eta = PlotGroup("Efficiencies_vs_eta_zminus", _effplots_sc_zminus_eta, ncols=10)
_efficiencies_sc_zminus_phi = PlotGroup("Efficiencies_vs_phi_zminus", _effplots_sc_zminus_phi, ncols=10)
_efficiencies_sc_zminus     = PlotGroup("Eff_Dup_Fake_Merge_Global_zminus", _effplots_sc_zminus, ncols=4)

_common_dup = {"stat": False, "legend": False}
_dupplots_sc_zminus_eta = [Plot("duplicate_eta_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(0,maxlayerzm)]
_dupplots_sc_zminus_phi = [Plot("duplicate_phi_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(0,maxlayerzm)]
_common_dup = {"stat": False, "legend": False, "title": "Global Duplicates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_dup["xmin"] = _bin_count
_common_dup["xmax"] = _common_dup["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_dupplots_sc_zminus = [Plot("globalDublicates_zminus", xtitle="Global Duplicates in z-", **_common_dup)]
_duplicates_sc_zminus_eta = PlotGroup("Duplicates_vs_eta_zminus", _dupplots_sc_zminus_eta, ncols=10)
_duplicates_sc_zminus_phi = PlotGroup("Duplicates_vs_phi_zminus", _dupplots_sc_zminus_phi, ncols=10)
_duplicates_sc_zminus     = PlotGroup("Eff_Dup_Fake_Merge_Global_zminus", _dupplots_sc_zminus, ncols=4)

_common_fake = {"stat": False, "legend": False}
_fakeplots_sc_zminus_eta = [Plot("fake_eta_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(0,maxlayerzm)]
_fakeplots_sc_zminus_phi = [Plot("fake_phi_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(0,maxlayerzm)]
_common_fake = {"stat": False, "legend": False, "title": "Global Fake Rates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_fake["xmin"] = _bin_count
_common_fake["xmax"] = _common_fake["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_common_fake["xbinlabels"] = [ "L{:02d}".format(i+1) for i in range(0,maxlayerzm) ]
_common_fake["xbinlabelsize"] = 10.
_fakeplots_sc_zminus = [Plot("globalFakes_zminus", xtitle="Global Fake Rate in z-", **_common_fake)]
_fakes_sc_zminus_eta = PlotGroup("FakeRate_vs_eta_zminus", _fakeplots_sc_zminus_eta, ncols=10)
_fakes_sc_zminus_phi = PlotGroup("FakeRate_vs_phi_zminus", _fakeplots_sc_zminus_phi, ncols=10)
_fakes_sc_zminus     = PlotGroup("Eff_Dup_Fake_Merge_Global_zminus", _fakeplots_sc_zminus, ncols=4)

_common_merge = {"stat": False, "legend": False}
_mergeplots_sc_zminus_eta = [Plot("merge_eta_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(0,maxlayerzm)]
_mergeplots_sc_zminus_phi = [Plot("merge_phi_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(0,maxlayerzm)]
_common_merge = {"stat": False, "legend": False, "title": "Global Merge Rates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_merge["xmin"] = _bin_count
_common_merge["xmax"] = _common_merge["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_common_merge["xbinlabels"] = [ "L{:02d}".format(i+1) for i in range(0,maxlayerzm) ]
_common_merge["xbinlabelsize"] = 10.
_mergeplots_sc_zminus = [Plot("globalMergeRate_zminus", xtitle="Global merge Rate in z-", **_common_merge)]
_merges_sc_zminus_eta = PlotGroup("MergeRate_vs_eta_zminus", _mergeplots_sc_zminus_eta, ncols=10)
_merges_sc_zminus_phi = PlotGroup("MergeRate_vs_phi_zminus", _mergeplots_sc_zminus_phi, ncols=10)
_merges_sc_zminus     = PlotGroup("Eff_Dup_Fake_Merge_Global_zminus", _mergeplots_sc_zminus, ncols=4)

_common_energy_score = dict(removeEmptyBins=False, xbinlabelsize=10,
    stat=True,
    xbinlabeloption="d",
    ncols=1,
    ylog=True,
    xlog=True,
    xmin=0.001,
    xmax=1.,
    ymin=0.01,
    ymax=1.)
_energyscore_sc2lc_zminus = PlotGroup("Energy_vs_Score_SC2LC_zminus", [Plot("Energy_vs_Score_simcluster2layer_perlayer{:02d}".format(i), title="Energy_vs_Score_SC2LC",
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(0, maxlayerzm)
                                                     ], ncols=10)

_energyscore_sc2lc_zplus = PlotGroup("Energy_vs_Score_SC2LC_zplus", [Plot("Energy_vs_Score_simcluster2layer_perlayer{:02d}".format(i), title="Energy_vs_Score_SC2LC",
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(maxlayerzm,maxlayerzp)
                                                     ], ncols=10)

_common_energy_score["xlog"]=False
_common_energy_score["ylog"]=False
_common_energy_score["xmin"]=-0.1
_energyscore_lc2sc_zminus = PlotGroup("Energy_vs_Score_LC2SC_zminus", [Plot("Energy_vs_Score_layer2simcluster_perlayer{:02d}".format(i), title="Energy_vs_Score_LC2SC",
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(0, maxlayerzm)
                                                     ], ncols=10)
_energyscore_lc2sc_zplus = PlotGroup("Energy_vs_Score_LC2SC_zplus", [Plot("Energy_vs_Score_layer2simcluster_perlayer{:02d}".format(i), title="Energy_vs_Score_LC2SC",
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score) for i in range(maxlayerzm,maxlayerzp)
                                                     ], ncols=10)

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_common_sc_score = {"title": "Score SimCluster to LayerClusters in z+",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 1000,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_sc_score.update(_legend_common)
_score_simcluster_to_layerclusters_zplus = PlotGroup("score_simcluster_to_layercluster_zplus", [
        Plot("Score_simcluster2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_score) for i in range(maxlayerzm,maxlayerzp)
        ], ncols=10 )

_common_sc_score = {"title": "Score LayerCluster to SimClusters in z+",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 1000,
                 "xmin": 0,
                 "xmax": 1,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_sc_score.update(_legend_common)
_score_layercluster_to_simclusters_zplus = PlotGroup("score_layercluster_to_simcluster_zplus", [
        Plot("Score_layercl2simcluster_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_score) for i in range(maxlayerzm,maxlayerzp)
        ], ncols=8 )

_common_sc_shared= {"title": "Shared Energy SimCluster To Layer Cluster in z+",
                 "stat": False,
                 "legend": False,
                }
_common_sc_shared.update(_legend_common)
_shared_sc_plots_zplus = [Plot("SharedEnergy_simcluster2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(maxlayerzm,maxlayerzp)]
_shared_sc_plots_zplus.extend([Plot("SharedEnergy_simcluster2layercl_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(maxlayerzm,maxlayerzp)])
_shared_sc_plots_zplus.extend([Plot("SharedEnergy_simcluster2layercl_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(maxlayerzm,maxlayerzp)])
_sharedEnergy_simcluster_to_layercluster_zplus = PlotGroup("sharedEnergy_simcluster_to_layercluster_zplus", _shared_sc_plots_zplus, ncols=8)

_common_sc_shared= {"title": "Shared Energy Layer Cluster To SimCluster in z+",
                 "stat": False,
                 "legend": False,
                }
_common_sc_shared.update(_legend_common)
_shared_plots2_sc_zplus = [Plot("SharedEnergy_layercluster2simcluster_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(maxlayerzm,maxlayerzp)]
_common_sc_shared= {"title": "Shared Energy Layer Cluster To Best SimCluster in z+",
                 "stat": False,
                 "legend": False,
                 "ymin": 0,
                 "ymax": 1,
                }
_common_sc_shared.update(_legend_common)
_shared_plots2_sc_zplus.extend([Plot("SharedEnergy_layercl2simcluster_vs_eta_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(maxlayerzm,maxlayerzp)])
_shared_plots2_sc_zplus.extend([Plot("SharedEnergy_layercl2simcluster_vs_phi_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_sc_shared) for i in range(maxlayerzm,maxlayerzp)])
_sharedEnergy_layercluster_to_simcluster_zplus = PlotGroup("sharedEnergy_layercluster_to_simcluster_zplus", _shared_plots2_sc_zplus, ncols=8)


_bin_count = maxlayerzm
_common_eff = {"stat": False, "legend": False}
_effplots_sc_zplus_eta = [Plot("effic_eta_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(maxlayerzm,maxlayerzp)]
_effplots_sc_zplus_phi = [Plot("effic_phi_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(maxlayerzm,maxlayerzp)]
_common_eff = {"stat": False, "legend": False, "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_eff["xmin"] = _bin_count
_common_eff["xmax"] = _common_eff["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_effplots_sc_zplus = [Plot("globalEfficiencies_zplus", xtitle="Global Efficiencies in z+", **_common_eff)]
_efficiencies_sc_zplus_eta = PlotGroup("Efficiencies_vs_eta_zplus", _effplots_sc_zplus_eta, ncols=10)
_efficiencies_sc_zplus_phi = PlotGroup("Efficiencies_vs_phi_zplus", _effplots_sc_zplus_phi, ncols=10)
_efficiencies_sc_zplus = PlotGroup("Eff_Dup_Fake_Merge_Global_zplus", _effplots_sc_zplus, ncols=4)

_common_dup = {"stat": False, "legend": False}
_dupplots_sc_zplus_eta = [Plot("duplicate_eta_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(maxlayerzm,maxlayerzp)]
_dupplots_sc_zplus_phi = [Plot("duplicate_phi_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(maxlayerzm,maxlayerzp)]
_common_dup = {"stat": False, "legend": False, "title": "Global Duplicates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_dup["xmin"] = _bin_count
_common_dup["xmax"] = _common_dup["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_dupplots_sc_zplus = [Plot("globalDuplicates_zplus", xtitle="Global Duplicates in z+", **_common_dup)]
_duplicates_sc_zplus_eta = PlotGroup("Duplicates_vs_eta_zplus", _dupplots_sc_zplus_eta, ncols=10)
_duplicates_sc_zplus_phi = PlotGroup("Duplicates_vs_phi_zplus", _dupplots_sc_zplus_phi, ncols=10)
_duplicates_sc_zplus = PlotGroup("Eff_Dup_Fake_Merge_Global_zplus", _dupplots_sc_zplus, ncols=4)

_common_fake = {"stat": False, "legend": False}
_fakeplots_sc_zplus_eta = [Plot("fake_eta_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(maxlayerzm,maxlayerzp)]
_fakeplots_sc_zplus_phi = [Plot("fake_phi_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(maxlayerzm,maxlayerzp)]
_common_fake = {"stat": False, "legend": False, "title": "Global Fake Rates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_fake["xmin"] = _bin_count
_common_fake["xmax"] = _common_fake["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_fakeplots_sc_zplus = [Plot("globalFakeRate_zplus", xtitle="Global Fake Rate in z+", **_common_fake)]
_fakes_sc_zplus_eta = PlotGroup("FakeRate_vs_eta_zplus", _fakeplots_sc_zplus_eta, ncols=10)
_fakes_sc_zplus_phi = PlotGroup("FakeRate_vs_phi_zplus", _fakeplots_sc_zplus_phi, ncols=10)
_fakes_sc_zplus = PlotGroup("Eff_Dup_Fake_Merge_Global_zplus", _fakeplots_sc_zplus, ncols=4)

_common_merge = {"stat": False, "legend": False}
_mergeplots_sc_zplus_eta = [Plot("merge_eta_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(maxlayerzm,maxlayerzp)]
_mergeplots_sc_zplus_phi = [Plot("merge_phi_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(maxlayerzm,maxlayerzp)]
_common_merge = {"stat": False, "legend": False, "title": "Global Merge Rates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_merge["xmin"] = _bin_count
_common_merge["xmax"] = _common_merge["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_mergeplots_sc_zplus = [Plot("globalMergeRate_zplus", xtitle="Global merge Rate in z+", **_common_merge)]
_merges_sc_zplus_eta = PlotGroup("MergeRate_vs_eta_zplus", _mergeplots_sc_zplus_eta, ncols=10)
_merges_sc_zplus_phi = PlotGroup("MergeRate_vs_phi_zplus", _mergeplots_sc_zplus_phi, ncols=10)
_merges_sc_zplus = PlotGroup("Eff_Dup_Fake_Merge_Global_zplus", _mergeplots_sc_zplus, ncols=4)

#Just in case we add some plots below to be on the safe side.
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#--------------------------------------------------------------------------------------------
# TRACKSTERS
#--------------------------------------------------------------------------------------------
_common_score = {"stat": False, "legend": False
                 ,"ymin": 0.1
                 ,"ymax": 100000
                 ,"xmin": 0
                 #,"xmax": 1.0
                 ,"drawStyle": "hist"
                 ,"lineWidth": 1
                 ,"ylog": True
                 ,"xlog": True
                 ,"xtitle": "Default"
                }
_common_score.update(_legend_common)

score_to_trackster = ["","Pur","Dupl"]
_score_caloparticle_to_tracksters = PlotGroup("ScoreCaloParticlesToTracksters", [], ncols=len(score_to_trackster))
_score_simtrackster_to_tracksters = PlotGroup("ScoreSimTrackstersToTracksters", [], ncols=len(score_to_trackster))
for score in score_to_trackster:
    _score_caloparticle_to_tracksters.append(Plot("Score"+score+"_caloparticle2trackster", **_common_score))
    _score_simtrackster_to_tracksters.append(Plot("Score"+score+"_simtrackster2trackster", **_common_score))

score_trackster_to = ["","Fake","Merge"]
_score_trackster_to_caloparticles = PlotGroup("ScoreTrackstersToCaloParticles", [], ncols=len(score_trackster_to))
_score_trackster_to_simtracksters = PlotGroup("ScoreTrackstersToSimTracksters", [], ncols=len(score_trackster_to))
for score in score_trackster_to:
    _score_trackster_to_caloparticles.append(Plot("Score"+score+"_trackster2caloparticle", **_common_score))
    _score_trackster_to_simtracksters.append(Plot("Score"+score+"_trackster2simtrackster", **_common_score))


_common_shared = {"stat": False, "legend": False, "xtitle": 'Default', "ytitle": 'Default'}
_common_shared.update(_legend_common)
_common_energy_score = dict(removeEmptyBins=True, xbinlabelsize=10, xbinlabeloption="d", drawStyle="COLZ", adjustMarginRight=0.1, legend=False, xtitle='Default', ytitle='Default')
#_common_energy_score["ymax"] = 1.
#_common_energy_score["xmax"] = 1.0

_sharedEnergy_to_trackster = []
_sharedEnergy_trackster_to = []
versions = ["", "_assoc", "_assoc_vs_eta", "_assoc_vs_phi"]

_energyscore_to_trackster = []
_energyscore_trackster_to = []
en_vs_score = ["","best","secBest"]
for val in simDict:
    _sharedEnergy_to_trackster.append(PlotGroup("SharedEnergy_"+val+"ToTrackster", [], ncols=2))
    _sharedEnergy_trackster_to.append(PlotGroup("SharedEnergy_TracksterTo"+val, [], ncols=2))
    for ver in versions:
        _sharedEnergy_to_trackster[-1].append(Plot("SharedEnergy_"+val.lower()+"2trackster"+ver, **_common_shared))
        _sharedEnergy_trackster_to[-1].append(Plot("SharedEnergy_trackster2"+val.lower()+ver, **_common_shared))

    _energyscore_to_trackster.append(PlotGroup("Energy_vs_Score_"+val+"ToTracksters", [], ncols=len(en_vs_score)))
    _energyscore_trackster_to.append(PlotGroup("Energy_vs_Score_TrackstersTo"+val, [], ncols=len(en_vs_score)))
    for ver in en_vs_score:
        _energyscore_to_trackster[-1].append(Plot("Energy_vs_Score_"+val.lower()+"2"+ver+"Trackster", **_common_energy_score))
        _energyscore_trackster_to[-1].append(Plot("Energy_vs_Score_trackster2"+ver+val, **_common_energy_score))

_common_assoc = {#"title": "Cell Association Table",
                 "stat": False,
                 "legend": False,
                 "xbinlabels": ["", "TN(pur)", "FN(ineff.)", "FP(fake)", "TP(eff)"],
                 "xbinlabeloption": "h",
                 "drawStyle": "hist",
                 "ymin": 0.1,
                 "ymax": 10000000,
                 "ylog": True}
_common_assoc.update(_legend_common)
_cell_association_table = PlotGroup("cellAssociation_table", [
        Plot("cellAssociation_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_assoc) for i in range(0,maxlayerzm)
        ], ncols=8 )

# Trackster plots
_common_metric = {"stat": False, "legend": False, "xbinlabelsize": 14, "xbinlabeloption": "d", "ymin": 0.0, "ymax": 1.1}
_common_metric_logx = _common_metric.copy()
_common_metric_logx["xlog"] = True

_efficiencies = []
_purities = []
_duplicates = []
_fakes = []
_merges = []
for val in simDict:
    _effplots = [Plot("globalEfficiencies", xtitle="", **_common_metric)]
    _purityplots = [Plot("globalEfficiencies", xtitle="", **_common_metric)]
    _dupplots = [Plot("globalEfficiencies", xtitle="", **_common_metric)]
    _fakeplots = [Plot("globalEfficiencies", xtitle="", **_common_metric)]
    _mergeplots = [Plot("globalEfficiencies", xtitle="", **_common_metric)]

    for v in variables:
        kwargs = _common_metric_logx if v in ["energy","pt"] else _common_metric
        _effplots.extend([Plot("effic_"+v+simDict[val], xtitle = variables[v][0]+variables[v][1], **kwargs)])
        _purityplots.extend([Plot("purity_"+v+simDict[val], xtitle = variables[v][0]+variables[v][1], **kwargs)])
        _dupplots.extend([Plot("duplicate_"+v+simDict[val], xtitle = variables[v][0]+variables[v][1], **kwargs)])
        _fakeplots.extend([Plot("fake_"+v+simDict[val], xtitle = variables[v][0]+variables[v][1], **kwargs)])
        _mergeplots.extend([Plot("merge_"+v+simDict[val], xtitle = variables[v][0]+variables[v][1], **kwargs)])

    _efficiencies.append(PlotGroup("Efficiencies"+simDict[val], _effplots, ncols=3))
    _purities.append(PlotGroup("Purities"+simDict[val], _purityplots, ncols=3))
    _duplicates.append(PlotGroup("Duplicates"+simDict[val], _dupplots, ncols=3))
    _fakes.append(PlotGroup("FakeRate"+simDict[val], _fakeplots, ncols=3))
    _merges.append(PlotGroup("MergeRate"+simDict[val], _mergeplots, ncols=3))


#Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "xtitle": "Default"}

_tottracksternum = PlotGroup("TotalNumberofTracksters", [
  Plot("tottracksternum", **_common)
],ncols=1)

_trackster_layernum_plots = [Plot("trackster_firstlayer", **_common)]
_trackster_layernum_plots.extend([Plot("trackster_lastlayer", **_common)])
_trackster_layernum_plots.extend([Plot("trackster_layersnum", **_common)])
_trackster_layernum = PlotGroup("LayerNumbersOfTrackster", _trackster_layernum_plots, ncols=3)

_common["xmax"] = 50
_clusternum_in_trackster = PlotGroup("NumberofLayerClustersinTrackster",[
  Plot("clusternum_in_trackster", **_common)
],ncols=1)

_common = {"stat": True, "drawStyle": "pcolz", "staty": 0.65, "xtitle": "Default", "ytitle": "Default"}

_clusternum_in_trackster_vs_layer = PlotGroup("NumberofLayerClustersinTracksterPerLayer",[
  Plot("clusternum_in_trackster_vs_layer", **_common)
],ncols=1)

_common["scale"] = 100.
#, ztitle = "% of clusters" normalizeToUnitArea=True
_multiplicity_numberOfEventsHistogram = hgcVal_dqm + "ticlTrackstersMerge/multiplicity_numberOfEventsHistogram"
_multiplicity_zminus_numberOfEventsHistogram = hgcVal_dqm + "ticlTrackstersMerge/multiplicity_zminus_numberOfEventsHistogram"
_multiplicity_zplus_numberOfEventsHistogram = hgcVal_dqm + "ticlTrackstersMerge/multiplicity_zplus_numberOfEventsHistogram"

_multiplicityOfLCinTST_plots = [Plot("multiplicityOfLCinTST",
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)]
_multiplicityOfLCinTST_plots.extend([Plot("multiplicityOfLCinTST_vs_layerclusterenergy",
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)]) 
_multiplicityOfLCinTST_plots.extend([Plot("multiplicityOfLCinTST_vs_layercluster_zplus",
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)])
_multiplicityOfLCinTST_plots.extend([Plot("multiplicityOfLCinTST_vs_layercluster_zminus",
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)])
_multiplicityOfLCinTST = PlotGroup("MultiplicityofLCinTST", _multiplicityOfLCinTST_plots, ncols=2)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}
#--------------------------------------------------------------------------------------------
# z-
#--------------------------------------------------------------------------------------------
_clusternum_in_trackster_perlayer_zminus_EE = PlotGroup("NumberofLayerClustersinTracksterPerLayer_zminus_EE", [
  Plot("clusternum_in_trackster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_clusternum_in_trackster_perlayer_zminus_FH = PlotGroup("NumberofLayerClustersinTracksterPerLayer_zminus_FH", [
  Plot("clusternum_in_trackster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_clusternum_in_trackster_perlayer_zminus_BH = PlotGroup("NumberofLayerClustersinTracksterPerLayer_zminus_BH", [
  Plot("clusternum_in_trackster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_clusternum_in_trackster_perlayer_zplus_EE = PlotGroup("NumberofLayerClustersinTracksterPerLayer_zplus_EE", [
  Plot("clusternum_in_trackster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_clusternum_in_trackster_perlayer_zplus_FH = PlotGroup("NumberofLayerClustersinTracksterPerLayer_zplus_FH", [
  Plot("clusternum_in_trackster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_clusternum_in_trackster_perlayer_zplus_BH = PlotGroup("NumberofLayerClustersinTracksterPerLayer_zplus_BH", [
  Plot("clusternum_in_trackster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

# Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "xtitle": "Default"}

# Some tracksters quantities
_trackster_eppe_plots = [Plot("trackster_eta", **_common)]
_trackster_eppe_plots.extend([Plot("trackster_phi", **_common)])
_trackster_eppe_plots.extend([Plot("trackster_pt", **_common)])
_trackster_eppe_plots.extend([Plot("trackster_energy", **_common)])
_trackster_eppe = PlotGroup("EtaPhiPtEnergy", _trackster_eppe_plots, ncols=2)

_trackster_xyz_plots = [Plot("trackster_x", **_common)]
_trackster_xyz_plots.extend([Plot("trackster_y", **_common)])
_trackster_xyz_plots.extend([Plot("trackster_z", **_common)])
_trackster_xyz = PlotGroup("XYZ", _trackster_xyz_plots, ncols=3)

#--------------------------------------------------------------------------------------------
# SIMHITS, DIGIS, RECHITS
#--------------------------------------------------------------------------------------------

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": True}

_HitValidation = PlotGroup("HitValidation", [
                                             Plot("heeEnSim", title="SimHits_EE_Energy", **_common),
                                             Plot("hebEnSim", title="SimHits_HE_Silicon_Energy", **_common),
                                             Plot("hefEnSim", title="SimHits_HE_Scintillator_Energy", **_common),
                                             ])

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}

_Occupancy_EE_zplus = PlotGroup("Occupancy_EE_zplus", [Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="Occupancy_EE_zplus", 
                                                        xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                        ], ncols=7)

_Occupancy_HE_Silicon_zplus = PlotGroup("Occupancy_HE_Silicon_zplus", [Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="Occupancy_HE_zplus", 
                                                                       xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                       ], ncols=7)

_Occupancy_HE_Scintillator_zplus = PlotGroup("Occupancy_HE_Scintillator_zplus", [Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="Occupancy_HE_Scintillator_zplus", 
                                                                                         xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                         ], ncols=7)

_Occupancy_EE_zminus = PlotGroup("Occupancy_EE_zminus", [Plot("HitOccupancy_Minus_layer_{:02d}".format(i), title="Occupancy_EE_zminus", 
                                                         xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                         ], ncols=7)

_Occupancy_HE_Silicon_zminus = PlotGroup("Occupancy_HE_Silicon_zminus", [Plot("HitOccupancy_Minus_layer_{:02d}".format(i), title="Occupancy_HE_Silicon_zminus", 
                                                                         xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                         ], ncols=7)

_Occupancy_HE_Scintillator_zminus = PlotGroup("Occupancy_HE_Scintillator_zminus", [Plot("HitOccupancy_Minus_layer_{:02d}".format(i), title="Occupancy_HE_Scintillator_zminus", 
                                                                                   xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                   ], ncols=7)

_common_etaphi = dict(removeEmptyBins=False, xbinlabelsize=10, xbinlabeloption="d", ymin=None)

_EtaPhi_EE_zplus = PlotGroup("EtaPhi_EE_zplus", [Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="EtaPhi_EE_zplus", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi) for i in range(EE_min,EE_max+1)
                                                     ], ncols=7)

_EtaPhi_HE_Silicon_zplus = PlotGroup("EtaPhi_HE_Silicon_zplus", [Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="EtaPhi_HE_Silicon_zplus", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi) for i in range(HESilicon_min,HESilicon_max+1)
                                                     ], ncols=7)

_EtaPhi_HE_Scintillator_zplus = PlotGroup("EtaPhi_HE_Scintillator_zplus", [Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="EtaPhi_HE_Scintillator_zplus", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                     ], ncols=7)

_EtaPhi_EE_zminus = PlotGroup("EtaPhi_EE_zminus", [Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="EtaPhi_EE_zminus", 
                                                      xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi) for i in range(EE_min,EE_max+1)
                                                      ], ncols=7)

_EtaPhi_HE_Silicon_zminus = PlotGroup("EtaPhi_HE_Silicon_zminus", [Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="EtaPhi_HE_Silicon_zminus", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi) for i in range(HESilicon_min,HESilicon_max+1)
                                                     ], ncols=7)

_EtaPhi_HE_Scintillator_zminus = PlotGroup("EtaPhi_HE_Scintillator_zminus", [Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="EtaPhi_HE_Scintillator_zminus", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                     ], ncols=7)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": True}

_Energy_EE_0 = PlotGroup("Energy_Time_0_EE", [Plot("energy_time_0_layer_{:02d}".format(i), title="Energy_Time_0_EE", 
                                              xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                              ], ncols=7)

_Energy_HE_Silicon_0 = PlotGroup("Energy_Time_0_HE_Silicon", [Plot("energy_time_0_layer_{:02d}".format(i), title="Energy_Time_0_HE_Silicon", 
                                                              xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                              ], ncols=7)

_Energy_HE_Scintillator_0 = PlotGroup("Energy_Time_0_HE_Scintillator", [Plot("energy_time_0_layer_{:02d}".format(i), title="Energy_Time_0_HE_Scintillator", 
                                                                        xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                        ], ncols=7)

_Energy_EE_1 = PlotGroup("Energy_Time_1_EE", [Plot("energy_time_1_layer_{:02d}".format(i), title="Energy_Time_1_EE", 
                                              xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                              ], ncols=7)

_Energy_HE_Silicon_1 = PlotGroup("Energy_Time_1_HE_Silicon", [Plot("energy_time_1_layer_{:02d}".format(i), title="Energy_Time_1_HE_Silicon", 
                                                              xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                              ], ncols=7)

_Energy_HE_Scintillator_1 = PlotGroup("Energy_Time_1_HE_Scintillator", [Plot("energy_time_1_layer_{:02d}".format(i), title="Energy_Time_1_HE_Scintillator", 
                                                                        xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                        ], ncols=7)

_Energy_EE = PlotGroup("Energy_EE", [Plot("energy_layer_{:02d}".format(i), title="Energy_EE", 
                                             xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                             ], ncols=7)

_Energy_HE_Silicon = PlotGroup("Energy_HE_Silicon", [Plot("energy_layer_{:02d}".format(i), title="Energy_HE_Silicon", 
                                                             xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                             ], ncols=7)

_Energy_HE_Scintillator = PlotGroup("Energy_HE_Scintillator", [Plot("energy_layer_{:02d}".format(i), title="Energy_HE_Scintillator", 
                                                                               xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                               ], ncols=7)

_DigiHits_ADC_EE = PlotGroup("ADC_EE", [Plot("ADC_layer_{:02d}".format(i), title="DigiHits_ADC_EE", 
                                        xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                        ], ncols=7)

_DigiHits_ADC_HE_Silicon = PlotGroup("ADC_HE_Silicon", [Plot("ADC_layer_{:02d}".format(i), title="DigiHits_ADC_HE_Silicon", 
                                                       xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                       ], ncols=7)

_DigiHits_ADC_HE_Scintillator = PlotGroup("ADC_HE_Scintillator", [Plot("ADC_layer_{:02d}".format(i), title="DigiHits_ADC_HE_Scintillator", 
                                                                  xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                  ], ncols=7)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}

_DigiHits_Occupancy_EE_zplus = PlotGroup("Occupancy_EE_zplus", [Plot("DigiOccupancy_Plus_layer_{:02d}".format(i), title="DigiHits_Occupancy_EE_zplus", 
                                                                xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                ], ncols=7)

_DigiHits_Occupancy_HE_Silicon_zplus = PlotGroup("Occupancy_HE_Silicon_zplus", [Plot("DigiOccupancy_Plus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Silicon_zplus", 
                                                                                 xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                 ], ncols=7)

_DigiHits_Occupancy_HE_Scintillator_zplus = PlotGroup("Occupancy_HE_Scintillator_zplus", [Plot("DigiOccupancy_Plus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Scintillator_zplus", 
                                                                                          xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                           ], ncols=7)

_DigiHits_Occupancy_EE_zminus = PlotGroup("Occupancy_EE_zminus", [Plot("DigiOccupancy_Minus_layer_{:02d}".format(i), title="DigiHits_Occupancy_EE_zminus", 
                                                                  xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                  ], ncols=7)

_DigiHits_Occupancy_HE_Silicon_zminus = PlotGroup("Occupancy_HE_Silicon_zminus", [Plot("DigiOccupancy_Minus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Silicon_zminus", 
                                                                                  xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                  ], ncols=7)

_DigiHits_Occupancy_HE_Scintillator_zminus = PlotGroup("Occupancy_HE_Scintillator_zminus", [Plot("DigiOccupancy_Minus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Scintillator_zminus", 
                                                                                            xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                            ], ncols=7)

_common_XY = dict(removeEmptyBins=True, xbinlabelsize=10, xbinlabeloption="d", ymin=None)

_DigiHits_Occupancy_XY_EE = PlotGroup("Occupancy_XY_EE", [Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_EE", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY) for i in range(EE_min,EE_max+1)
                                                     ], ncols=7)

_DigiHits_Occupancy_XY_HE_Silicon = PlotGroup("Occupancy_XY_HE_Silicon", [Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_HE_Silicon", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY) for i in range(HESilicon_min,HESilicon_max+1)
                                                     ], ncols=7)

_DigiHits_Occupancy_XY_HE_Scintillator = PlotGroup("Occupancy_XY_HE_Scintillator", [Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_HE_Scintillator", 
                                                     xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                     ], ncols=7)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": True}

_DigiHits_TOA_EE = PlotGroup("TOA_EE", [
                                                 Plot("TOA_layer_{:02d}".format(i), title="TOA_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                 ], ncols=7)

_DigiHits_TOA_HE_Silicon = PlotGroup("TOA_HE_Silicon", [
                                                                 Plot("TOA_layer_{:02d}".format(i), title="TOA_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                 ], ncols=7)

_DigiHits_TOA_HE_Scintillator = PlotGroup("TOA_HE_Scintillator", [
                                                                           Plot("TOA_layer_{:02d}".format(i), title="TOA_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                           ], ncols=7)

_DigiHits_TOT_EE = PlotGroup("TOT_EE", [
                                                 Plot("TOT_layer_{:02d}".format(i), title="TOT_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                 ], ncols=7)

_DigiHits_TOT_HE_Silicon = PlotGroup("TOT_HE_Silicon", [
                                                                 Plot("TOT_layer_{:02d}".format(i), title="TOT_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                 ], ncols=7)

_DigiHits_TOT_HE_Scintillator = PlotGroup("TOT_HE_Scintillator", [
                                                                           Plot("TOT_layer_{:02d}".format(i), title="TOT_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                           ], ncols=7)

#===================================================================================================================
#Plot definition for HitCalibration
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": False}

_LayerOccupancy = PlotGroup("LayerOccupancy", [
                                               Plot("LayerOccupancy", title="LayerOccupancy", **_common)], ncols=1)

_ReconstructableEnergyOverCPenergy = PlotGroup("ReconstructableEnergyOverCPenergy", [
  Plot("h_EoP_CPene_100_calib_fraction", title="EoP_CPene_100_calib_fraction", **_common),
  Plot("h_EoP_CPene_200_calib_fraction", title="EoP_CPene_200_calib_fraction", **_common),
  Plot("h_EoP_CPene_300_calib_fraction", title="EoP_CPene_300_calib_fraction", **_common),
  Plot("h_EoP_CPene_scint_calib_fraction", title="EoP_CPene_scint_calib_fraction", **_common),
])

_ParticleFlowClusterHGCalFromTrackster_Closest_EoverCPenergy = PlotGroup("ParticleFlowClusterHGCalFromTrackster", [
  Plot("hgcal_EoP_CPene_100_calib_fraction", title="hgcal_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_200_calib_fraction", title="hgcal_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_300_calib_fraction", title="hgcal_EoP_CPene_300_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_scint_calib_fraction", title="hgcal_EoP_CPene_scint_calib_fraction", **_common),
])

_EcalDrivenGsfElectronsFromTrackster_Closest_EoverCPenergy = PlotGroup("EcalDrivenGsfElectronsFromTrackster", [
  Plot("hgcal_ele_EoP_CPene_100_calib_fraction", title="hgcal_ele_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_200_calib_fraction", title="hgcal_ele_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_300_calib_fraction", title="hgcal_ele_EoP_CPene_300_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_scint_calib_fraction", title="hgcal_ele_EoP_CPene_scint_calib_fraction", **_common),
])

_PhotonsFromTrackster_Closest_EoverCPenergy = PlotGroup("PhotonsFromTrackster", [
  Plot("hgcal_photon_EoP_CPene_100_calib_fraction", title="hgcal_photon_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_200_calib_fraction", title="hgcal_photon_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_300_calib_fraction", title="hgcal_photon_EoP_CPene_300_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_scint_calib_fraction", title="hgcal_photon_EoP_CPene_scint_calib_fraction", **_common),
])

#=================================================================================================
hgcalLayerClustersPlotter = Plotter()
layerClustersLabel = 'Layer Clusters'

lc_general_clusterlevel = [
  # number of layer clusters per event in a) 120um, b) 200um, c) 300um, d) scint
  # (one entry per event in each of the four histos)
  _totclusternum_thick,
  # Miscellaneous plots:
  # longdepthbarycentre: The longitudinal depth barycentre. One entry per event.
  # mixedhitscluster: Number of clusters per event with hits in different thicknesses.
  # num_reco_cluster_eta: Number of reco clusters vs eta
  _num_reco_cluster_eta,
  _energyclustered,
  _mixedhitsclusters,
  _longdepthbarycentre,
  # calculated "energy density" for cells in a) 120um, b) 200um, c) 300um, d) scint
  # (one entry per rechit, in the appropriate histo)
  _cellsenedens_thick
]

lc_clusterlevel_zminus = [
  # number of layer clusters per layer (one entry per event in each histo)
  _totclusternum_layer_EE_zminus,
  _totclusternum_layer_FH_zminus,
  _totclusternum_layer_BH_zminus,
  # Looking at the fraction of true energy that has been clustered; by layer and overall
  _energyclustered_perlayer_EE_zminus,
  _energyclustered_perlayer_FH_zminus,
  _energyclustered_perlayer_BH_zminus
]

lc_cellevel_zminus = [
  # For each layer cluster:
  # number of cells in layer cluster, by layer - separate histos in each layer for 120um Si, 200/300um Si, Scint
  # NB: not all combinations exist; e.g. no 120um Si in layers with scint.
  # (One entry in the appropriate histo per layer cluster).
  _cellsnum_perthick_perlayer_120_EE_zminus,
  _cellsnum_perthick_perlayer_120_FH_zminus,
  _cellsnum_perthick_perlayer_120_BH_zminus,
  _cellsnum_perthick_perlayer_200_EE_zminus,
  _cellsnum_perthick_perlayer_200_FH_zminus,
  _cellsnum_perthick_perlayer_200_BH_zminus,
  _cellsnum_perthick_perlayer_300_EE_zminus,
  _cellsnum_perthick_perlayer_300_FH_zminus,
  _cellsnum_perthick_perlayer_300_BH_zminus,
  _cellsnum_perthick_perlayer_scint_EE_zminus,
  _cellsnum_perthick_perlayer_scint_FH_zminus,
  _cellsnum_perthick_perlayer_scint_BH_zminus,
  # Cell Association per Layer
  _cell_association_table_zminus
]

lc_cp_association_zminus = [
  # Efficiency Plots
  _efficiencies_zminus,
  _efficiencies_zminus_eta,
  _efficiencies_zminus_phi,
  # Duplicate Plots
  _duplicates_zminus,
  _duplicates_zminus_eta,
  _duplicates_zminus_phi,
  # Fake Rate Plots
  _fakes_zminus,
  _fakes_zminus_eta,
  _fakes_zminus_phi,
  # Merge Rate Plots
  _merges_zminus,
  _merges_zminus_eta,
  _merges_zminus_phi,
  # Score of CaloParticles wrt Layer Clusters
  _score_caloparticle_to_layerclusters_zminus,
  # Score of LayerClusters wrt CaloParticles
  _score_layercluster_to_caloparticles_zminus,
  # Shared Energy between CaloParticle and LayerClusters
  _sharedEnergy_caloparticle_to_layercluster_zminus,
  # Shared Energy between LayerClusters and CaloParticle
  _sharedEnergy_layercluster_to_caloparticle_zminus,
  # Energy vs Score 2D plots CP to LC
  _energyscore_cp2lc_zminus,
  # Energy vs Score 2D plots LC to CP
  _energyscore_lc2cp_zminus
]

lc_zminus_extended = [
  # For each layer cluster:
  # distance of cells from a) seed cell, b) max cell; and c), d): same with entries weighted by cell energy
  # separate histos in each layer for 120um Si, 200/300um Si, Scint
  # NB: not all combinations exist; e.g. no 120um Si in layers with scint.
  # (One entry in each of the four appropriate histos per cell in a layer cluster)
  _distancetomaxcell_perthickperlayer_120_EE_zminus,
  _distancetomaxcell_perthickperlayer_120_FH_zminus,
  _distancetomaxcell_perthickperlayer_120_BH_zminus,
  _distancetomaxcell_perthickperlayer_200_EE_zminus,
  _distancetomaxcell_perthickperlayer_200_FH_zminus,
  _distancetomaxcell_perthickperlayer_200_BH_zminus,
  _distancetomaxcell_perthickperlayer_300_EE_zminus,
  _distancetomaxcell_perthickperlayer_300_FH_zminus,
  _distancetomaxcell_perthickperlayer_300_BH_zminus,
  _distancetomaxcell_perthickperlayer_scint_EE_zminus,
  _distancetomaxcell_perthickperlayer_scint_FH_zminus,
  _distancetomaxcell_perthickperlayer_scint_BH_zminus,
  _distancetoseedcell_perthickperlayer_120_EE_zminus,
  _distancetoseedcell_perthickperlayer_120_FH_zminus,
  _distancetoseedcell_perthickperlayer_120_BH_zminus,
  _distancetoseedcell_perthickperlayer_200_EE_zminus,
  _distancetoseedcell_perthickperlayer_200_FH_zminus,
  _distancetoseedcell_perthickperlayer_200_BH_zminus,
  _distancetoseedcell_perthickperlayer_300_EE_zminus,
  _distancetoseedcell_perthickperlayer_300_FH_zminus,
  _distancetoseedcell_perthickperlayer_300_BH_zminus,
  _distancetoseedcell_perthickperlayer_scint_EE_zminus,
  _distancetoseedcell_perthickperlayer_scint_FH_zminus,
  _distancetoseedcell_perthickperlayer_scint_BH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_120_EE_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_120_FH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_120_BH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_200_EE_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_200_FH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_200_BH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_300_EE_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_300_FH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_300_BH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zminus,
  _distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_120_EE_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_120_FH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_120_BH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_200_EE_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_200_FH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_200_BH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_300_EE_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_300_FH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_300_BH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zminus,
  _distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_120_EE_zminus,
  _distancebetseedandmaxcell_perthickperlayer_120_FH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_120_BH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_200_EE_zminus,
  _distancebetseedandmaxcell_perthickperlayer_200_FH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_200_BH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_300_EE_zminus,
  _distancebetseedandmaxcell_perthickperlayer_300_FH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_300_BH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_scint_EE_zminus,
  _distancebetseedandmaxcell_perthickperlayer_scint_FH_zminus,
  _distancebetseedandmaxcell_perthickperlayer_scint_BH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zminus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zminus
]

lc_clusterlevel_zplus = [
  # number of layer clusters per layer (one entry per event in each histo)
  _totclusternum_layer_EE_zplus,
  _totclusternum_layer_FH_zplus,
  _totclusternum_layer_BH_zplus,
  # Looking at the fraction of true energy that has been clustered; by layer and overall
  _energyclustered_perlayer_EE_zplus,
  _energyclustered_perlayer_FH_zplus,
  _energyclustered_perlayer_BH_zplus
]

lc_cellevel_zplus = [
  # number of cells in layer cluster, by layer - separate histos in each layer for 120um Si, 200/300um Si, Scint
  _cellsnum_perthick_perlayer_120_EE_zplus,
  _cellsnum_perthick_perlayer_120_FH_zplus,
  _cellsnum_perthick_perlayer_120_BH_zplus,
  _cellsnum_perthick_perlayer_200_EE_zplus,
  _cellsnum_perthick_perlayer_200_FH_zplus,
  _cellsnum_perthick_perlayer_200_BH_zplus,
  _cellsnum_perthick_perlayer_300_EE_zplus,
  _cellsnum_perthick_perlayer_300_FH_zplus,
  _cellsnum_perthick_perlayer_300_BH_zplus,
  _cellsnum_perthick_perlayer_scint_EE_zplus,
  _cellsnum_perthick_perlayer_scint_FH_zplus,
  _cellsnum_perthick_perlayer_scint_BH_zplus,
  # Cell Association per Layer
  _cell_association_table_zplus
]

lc_cp_association_zplus = [
  # Efficiency Plots
  _efficiencies_zplus,
  _efficiencies_zplus_eta,
  _efficiencies_zplus_phi,
  # Duplicate Plots
  _duplicates_zplus,
  _duplicates_zplus_eta,
  _duplicates_zplus_phi,
  # Fake Rate Plots
  _fakes_zplus,
  _fakes_zplus_eta,
  _fakes_zplus_phi,
  # Merge Rate Plots
  _merges_zplus,
  _merges_zplus_eta,
  _merges_zplus_phi,  
  # Score of CaloParticles wrt Layer Clusters
  _score_caloparticle_to_layerclusters_zplus,
  # Score of LayerClusters wrt CaloParticles
  _score_layercluster_to_caloparticles_zplus,
  # Shared Energy between CaloParticle and LayerClusters
  _sharedEnergy_caloparticle_to_layercluster_zplus,
  # Shared Energy between LayerClusters and CaloParticle
  _sharedEnergy_layercluster_to_caloparticle_zplus,
  _energyscore_cp2lc_zplus,
  _energyscore_lc2cp_zplus
]

lc_zplus_extended = [
  # distance of cells from a) seed cell, b) max cell; and c), d): same with entries weighted by cell energy
  _distancetomaxcell_perthickperlayer_120_EE_zplus,
  _distancetomaxcell_perthickperlayer_120_FH_zplus,
  _distancetomaxcell_perthickperlayer_120_BH_zplus,
  _distancetomaxcell_perthickperlayer_200_EE_zplus,
  _distancetomaxcell_perthickperlayer_200_FH_zplus,
  _distancetomaxcell_perthickperlayer_200_BH_zplus,
  _distancetomaxcell_perthickperlayer_300_EE_zplus,
  _distancetomaxcell_perthickperlayer_300_FH_zplus,
  _distancetomaxcell_perthickperlayer_300_BH_zplus,
  _distancetomaxcell_perthickperlayer_scint_EE_zplus,
  _distancetomaxcell_perthickperlayer_scint_FH_zplus,
  _distancetomaxcell_perthickperlayer_scint_BH_zplus,
  _distancetoseedcell_perthickperlayer_120_EE_zplus,
  _distancetoseedcell_perthickperlayer_120_FH_zplus,
  _distancetoseedcell_perthickperlayer_120_BH_zplus,
  _distancetoseedcell_perthickperlayer_200_EE_zplus,
  _distancetoseedcell_perthickperlayer_200_FH_zplus,
  _distancetoseedcell_perthickperlayer_200_BH_zplus,
  _distancetoseedcell_perthickperlayer_300_EE_zplus,
  _distancetoseedcell_perthickperlayer_300_FH_zplus,
  _distancetoseedcell_perthickperlayer_300_BH_zplus,
  _distancetoseedcell_perthickperlayer_scint_EE_zplus,
  _distancetoseedcell_perthickperlayer_scint_FH_zplus,
  _distancetoseedcell_perthickperlayer_scint_BH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_120_EE_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_120_FH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_120_BH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_200_EE_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_200_FH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_200_BH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_300_EE_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_300_FH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_300_BH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zplus,
  _distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_120_EE_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_120_FH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_120_BH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_200_EE_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_200_FH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_200_BH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_300_EE_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_300_FH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_300_BH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zplus,
  _distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_120_EE_zplus,
  _distancebetseedandmaxcell_perthickperlayer_120_FH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_120_BH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_200_EE_zplus,
  _distancebetseedandmaxcell_perthickperlayer_200_FH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_200_BH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_300_EE_zplus,
  _distancebetseedandmaxcell_perthickperlayer_300_FH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_300_BH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_scint_EE_zplus,
  _distancebetseedandmaxcell_perthickperlayer_scint_FH_zplus,
  _distancebetseedandmaxcell_perthickperlayer_scint_BH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zplus,
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zplus
]

def append_hgcalLayerClustersPlots(collection = hgcalValidator.label_layerClusterPlots._InputTag__moduleLabel, name_collection = layerClustersLabel, extended = False):
  print('extended : ',extended)
  regions_ClusterLevel       = ["General: Cluster Level", "Z-minus: Cluster Level", "Z-plus: Cluster Level"]
  regions_CellLevel          = ["Z-minus: Cell Level", "Z-plus: Cell Level"]
  regions_LCtoCP_association = ["Z-minus: LC_CP association", "Z-plus: LC_CP association"]
  
  plots_lc_general_clusterlevel  = lc_general_clusterlevel
  plots_lc_clusterlevel_zminus   = lc_clusterlevel_zminus 
  plots_lc_cellevel_zminus       = lc_cellevel_zminus 
  plots_lc_clusterlevel_zplus    = lc_clusterlevel_zplus
  plots_lc_cellevel_zplus        = lc_cellevel_zplus
  plots_lc_cp_association_zminus = lc_cp_association_zminus
  plots_lc_cp_association_zplus  = lc_cp_association_zplus

  if extended :
    #plots_lc_clusterlevel_zminus   = lc_clusterlevel_zminus 
    #plots_lc_clusterlevel_zplus    = lc_clusterlevel_zplus 
    plots_lc_cellevel_zminus       = lc_cellevel_zminus + lc_zminus_extended
    plots_lc_cellevel_zplus        = lc_cellevel_zplus + lc_zplus_extended
    #plots_lc_cp_association_zminus = lc_cp_association_zminus 
    #plots_lc_cp_association_zplus  = lc_cp_association_zplus 

  setPlots_ClusterLevel       = [plots_lc_general_clusterlevel, plots_lc_clusterlevel_zminus, plots_lc_clusterlevel_zplus]
  setPlots_CellLevel          = [plots_lc_cellevel_zminus, plots_lc_cellevel_zplus]
  setPlots_LCtoCP_association = [plots_lc_cp_association_zminus, plots_lc_cp_association_zplus]
  for reg, setPlot in zip(regions_ClusterLevel, setPlots_ClusterLevel):
    hgcalLayerClustersPlotter.append(collection+"_"+reg, [
                _hgcalFolders(collection + "/ClusterLevel")
                ], PlotFolder(
                *setPlot,
                loopSubFolders=False,
                purpose=PlotPurpose.Timing, page=layerClustersLabel, section=reg))
  for reg, setPlot in zip(regions_CellLevel, setPlots_CellLevel):
    hgcalLayerClustersPlotter.append(collection+"_"+reg, [
                _hgcalFolders(collection + "/CellLevel")
                ], PlotFolder(
                *setPlot,
                loopSubFolders=False,
                purpose=PlotPurpose.Timing, page=layerClustersLabel, section=reg))
  for reg, setPlot in zip(regions_LCtoCP_association, setPlots_LCtoCP_association):
    hgcalLayerClustersPlotter.append(collection+"_"+reg, [
                _hgcalFolders(collection + "/" + lcToCP_linking)
                ], PlotFolder(
                *setPlot,
                loopSubFolders=False,
                purpose=PlotPurpose.Timing, page=layerClustersLabel, section=reg))

#=================================================================================================

sc_clusterlevel = [
  # number of layer clusters per event in a) 120um, b) 200um, c) 300um, d) scint
  # (one entry per event in each of the four histos) ([B] above)
  _totsimclusternum_thick,
  # number of simclusters per layer (one entry per event in each histo) ([C] above)
  # z-
  _totsimclusternum_layer_EE_zminus,
  _totsimclusternum_layer_FH_zminus,
  _totsimclusternum_layer_BH_zminus,
  # z+
  _totsimclusternum_layer_EE_zplus,
  _totsimclusternum_layer_FH_zplus,
  _totsimclusternum_layer_BH_zplus,
  # Miscellaneous plots ([G] above):
  # mixedhitscluster: Number of clusters per event with hits in different thicknesses.
  _mixedhitssimclusters,
]

sc_ticltracksters = [
  # Score of SimClusters wrt Layer Clusters
  # z-
  _score_simcluster_to_layerclusters_zminus,
  # z+
  _score_simcluster_to_layerclusters_zplus,
  # Score of LayerClusters wrt SimClusters
  # z-
  _score_layercluster_to_simclusters_zminus,
  # z+
  _score_layercluster_to_simclusters_zplus,
  # Shared Energy between SimCluster and LayerClusters
  # z-
  _sharedEnergy_simcluster_to_layercluster_zminus,
  # z+
  _sharedEnergy_simcluster_to_layercluster_zplus,
  # Shared Energy between LayerClusters and SimCluster
  # z-
  _sharedEnergy_layercluster_to_simcluster_zminus,
  # z+
  _sharedEnergy_layercluster_to_simcluster_zplus,
  # Efficiency Plots
  # z-
  _efficiencies_sc_zminus,
  _duplicates_sc_zminus,
  _fakes_sc_zminus,
  _merges_sc_zminus,
  _efficiencies_sc_zminus_eta,
  _efficiencies_sc_zminus_phi,
  # z+
  _efficiencies_sc_zplus,
  _duplicates_sc_zplus,
  _fakes_sc_zplus,
  _merges_sc_zplus,
  _efficiencies_sc_zplus_eta,
  _efficiencies_sc_zplus_phi,
   # Duplicate Plots
  # z-
  _duplicates_sc_zminus_eta,
  _duplicates_sc_zminus_phi,
   # z+
  _duplicates_sc_zplus_eta,
  _duplicates_sc_zplus_phi,
  # Fake Rate Plots
  # z-
  _fakes_sc_zminus_eta,
  _fakes_sc_zminus_phi,
  # z+
  _fakes_sc_zplus_eta,
  _fakes_sc_zplus_phi,
  # Merge Rate Plots
  # z-
  _merges_sc_zminus_eta,
  _merges_sc_zminus_phi,
  # z+
  _merges_sc_zplus_eta,
  _merges_sc_zplus_phi,
  # Energy vs Score 2D plots SC to LC
  # z-
  _energyscore_sc2lc_zminus,
  # z+
  _energyscore_sc2lc_zplus,
  # Energy vs Score 2D plots LC to SC
  # z-
  _energyscore_lc2sc_zminus,
  # z+
  _energyscore_lc2sc_zplus
]

hgcalSimClustersPlotter = Plotter()

def append_hgcalSimClustersPlots(collection, name_collection):
  if collection == hgcalValidator.label_SimClustersLevel._InputTag__moduleLabel:
      hgcalSimClustersPlotter.append(collection, [
                  _hgcalFolders(hgcalValidator.label_SimClusters._InputTag__moduleLabel +"/"+ collection)
                  ], PlotFolder(
                  *sc_clusterlevel,
                  loopSubFolders=False,
                  purpose=PlotPurpose.Timing, page="SimClusters", section=name_collection))
  else:
      hgcalSimClustersPlotter.append(collection, [
                  _hgcalFolders(hgcalValidator.label_SimClusters._InputTag__moduleLabel +"/"+collection)
                  ], PlotFolder(
                  *sc_ticltracksters,
                  loopSubFolders=False,
                  purpose=PlotPurpose.Timing, page="SimClusters", section=name_collection))


#=================================================================================================
def _hgcalFolders(lastDirName="hgcalLayerClusters"):
    return hgcVal_dqm + lastDirName

_trackstersPlots = [
  _trackster_eppe,
  _trackster_xyz,
  _tottracksternum,
  _clusternum_in_trackster,
  _clusternum_in_trackster_vs_layer,
  _clusternum_in_trackster_perlayer_zminus_EE,
  _clusternum_in_trackster_perlayer_zminus_FH,
  _clusternum_in_trackster_perlayer_zminus_BH,
  _clusternum_in_trackster_perlayer_zplus_EE,
  _clusternum_in_trackster_perlayer_zplus_FH,
  _clusternum_in_trackster_perlayer_zplus_BH,
  _trackster_layernum,
  _multiplicityOfLCinTST,
]

_trackstersToCPLinkPlots = [
  _efficiencies[0],
  _purities[0],
  _duplicates[0],
  _fakes[0],
  _merges[0],
  _score_caloparticle_to_tracksters,
  _score_trackster_to_caloparticles,
  _sharedEnergy_to_trackster[0],
  _sharedEnergy_trackster_to[0],
  _energyscore_to_trackster[0],
  _energyscore_trackster_to[0],
]

_trackstersToSTSPRPlots = [
  _efficiencies[1],
  _purities[1],
  _duplicates[1],
  _fakes[1],
  _merges[1],
  _score_simtrackster_to_tracksters,
  _score_trackster_to_simtracksters,
  _sharedEnergy_to_trackster[1],
  _sharedEnergy_trackster_to[1],
  _energyscore_to_trackster[1],
  _energyscore_trackster_to[1],
]
hgcalTrackstersPlotter = Plotter()
def append_hgcalTrackstersPlots(collection = 'ticlTrackstersMerge', name_collection = "TrackstersMerge"):
  # Appending generic plots for Tracksters
  hgcalTrackstersPlotter.append(collection, [
              _hgcalFolders(collection+ "/" + hgcalValidator.label_TS.value())
              ], PlotFolder(
              *_trackstersPlots,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing, page="Tracksters", section=name_collection))

  # Appending plots for Tracksters-CP linking
  hgcalTrackstersPlotter.append(collection, [
              _hgcalFolders(collection + "/" + tsToCP_linking)
              ], PlotFolder(
              *_trackstersToCPLinkPlots,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing
              #,page=tsToCP_linking.replace('TSToCP_','TICL-')
              ,page=tsToCP_linking.replace('TSToCP_','Test-TICL').replace('linking','')
              ,section=name_collection)
              )

  # Appending plots for Tracksters Pattern Recognition
  hgcalTrackstersPlotter.append(collection, [
              _hgcalFolders(collection + "/" + tsToSTS_patternRec)
              ], PlotFolder(
              *_trackstersToSTSPRPlots,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing, page=tsToSTS_patternRec.replace('TSToSTS_','TICL-'), section=name_collection))

  #We append here two PlotFolder because we want the text to be in percent
  #and the number of events are different in zplus and zminus
  #hgcalTrackstersPlotter.append("Multiplicity", [
  #            dqmfolder
  #            ], PlotFolder(
  #            _multiplicityOfLCinTST_vs_layercluster_zminus,
  #            loopSubFolders=False,
  #            purpose=PlotPurpose.Timing, page=collection,
  #            numberOfEventsHistogram=_multiplicity_zminus_numberOfEventsHistogram
  #            ))
  #
  #hgcalTrackstersPlotter.append("Multiplicity", [
  #            dqmfolder
  #            ], PlotFolder(
  #            _multiplicityOfLCinTST_vs_layercluster_zplus,
  #            loopSubFolders=False,
  #            purpose=PlotPurpose.Timing, page=collection,
  #            numberOfEventsHistogram=_multiplicity_zplus_numberOfEventsHistogram
  #            ))

#=================================================================================================
_common_Calo = {"stat": False, "drawStyle": "hist", "staty": 0.65, "ymin": 0.0, "ylog": False, "xtitle": "Default", "ytitle": "Default"}

hgcalCaloParticlesPlotter = Plotter()
def append_hgcalCaloParticlesPlots(files, collection = '-211', name_collection = "pion-"):

  list_2D_histos = ["Energy of Rec-matched Hits vs layer",
                    "Energy of Rec-matched Hits vs layer (1SC)",
                    "Rec-matched Hits Sum Energy vs layer"]

  dqmfolder = hgcVal_dqm + "SelectedCaloParticles/" + collection
  templateFile = ROOT.TFile.Open(files[0]) # assuming all files have same structure
  if not gDirectory.GetDirectory(dqmfolder):
    print("Error: GeneralInfo directory %s not found in DQM file, exit"%dqmfolder)
    return hgcalTrackstersPlotter

  keys = gDirectory.GetDirectory(dqmfolder,True).GetListOfKeys()
  key = keys[0]
  while key:
    obj = key.ReadObj()
    name = obj.GetName()
    fileName = TString(name)
    fileName.ReplaceAll(" ","_")
    pg = PlotGroup(fileName.Data(),[
                  Plot(name,
                       drawCommand = "",
                       normalizeToNumberOfEvents = True, **_common_Calo)
                  ],
                  ncols=1)

    if name in list_2D_histos :
        pg = PlotOnSideGroup(plotName.Data(),
                      Plot(name,
                           drawCommand = "COLZ",
                           normalizeToNumberOfEvents = True, **_common_Calo)
                      ,
                      ncols=1)

    hgcalCaloParticlesPlotter.append("CaloParticles_"+name_collection, [
              dqmfolder
              ], PlotFolder(
                pg,
                loopSubFolders=False,
                purpose=PlotPurpose.Timing, page="CaloParticles", section=name_collection)
              )

    key = keys.After(key)

  templateFile.Close()

  return hgcalCaloParticlesPlotter

#=================================================================================================
def create_hgcalTrackstersPlotter(files, collection = 'ticlTrackstersMerge', name_collection = "TrackstersMerge"):
  grouped = {"cosAngle Beta": PlotGroup("cosAngle_Beta_per_layer",[],ncols=10), "cosAngle Beta Weighted": PlotGroup("cosAngle_Beta_Weighted_per_layer",[],ncols=10)}
  groupingFlag = " on Layer "

  hgcalTrackstersPlotter = Plotter()
  dqmfolder = hgcVal_dqm + collection
  #_multiplicity_tracksters_numberOfEventsHistogram = dqmfolder+"/Number of Trackster per Event"

  _common["ymin"] = 0.0
  _common["staty"] = 0.85
  templateFile = ROOT.TFile.Open(files[0]) # assuming all files have same structure
  if not gDirectory.GetDirectory(dqmfolder):
    print("Error: GeneralInfo directory %s not found in DQM file, exit"%dqmfolder)
    return hgcalTrackstersPlotter

  keys = gDirectory.GetDirectory(dqmfolder,True).GetListOfKeys()
  key = keys[0]
  while key:
    obj = key.ReadObj()
    name = obj.GetName()
    plotName = TString(name)
    plotName.ReplaceAll(" ","_")

    if groupingFlag in name:
        for group in grouped:
            if group+groupingFlag in name:
                grouped[group].append(Plot(name,
                                           xtitle="Default", ytitle="Default",
                                           **_common)
                                     )
    else:
        pg = None
        if obj.InheritsFrom("TH2"):
            pg = PlotOnSideGroup(plotName.Data(),
                                 Plot(name,
                                      xtitle="Default", ytitle="Default",
                                      drawCommand = "COLZ",
                                      **_common),
                                 ncols=1)
        elif obj.InheritsFrom("TH1"):
            pg = PlotGroup(plotName.Data(),
                           [Plot(name,
                                 xtitle="Default", ytitle="Default",
                                 drawCommand = "COLZ", # ineffective for TH1
                                 **_common)
                           ],
                           ncols=1, legendDh=-0.03 * len(files))

        if (pg is not None):
            hgcalTrackstersPlotter.append(name_collection+"_TICLDebugger",
                [dqmfolder], PlotFolder(pg,
                                        loopSubFolders=False,
                                        purpose=PlotPurpose.Timing, page="Tracksters", section=name_collection)
                #numberOfEventsHistogram=_multiplicity_tracksters_numberOfEventsHistogram)
                )

    key = keys.After(key)

  for group in grouped:
      hgcalTrackstersPlotter.append(name_collection+"_TICLDebugger",
          [dqmfolder], PlotFolder(grouped[group],
                                  loopSubFolders=False,
                                  purpose=PlotPurpose.Timing, page="Tracksters", section=name_collection)
          #numberOfEventsHistogram=_multiplicity_tracksters_numberOfEventsHistogram)
          )

  templateFile.Close()

  return hgcalTrackstersPlotter

#=================================================================================================
_common_Calo = {"stat": False, "drawStyle": "hist", "staty": 0.65, "ymin": 0.0, "ylog": False, "xtitle": "Default", "ytitle": "Default"}

hgcalCaloParticlesPlotter = Plotter()

def append_hgcalCaloParticlesPlots(files, collection = '-211', name_collection = "pion-"):
  dqmfolder = hgcVal_dqm + "SelectedCaloParticles/" + collection
  print(dqmfolder)
#  _common["ymin"] = 0.0
  templateFile = ROOT.TFile.Open(files[0]) # assuming all files have same structure
  keys = gDirectory.GetDirectory(dqmfolder,True).GetListOfKeys()
  key = keys[0]
  while key:
    obj = key.ReadObj()
    name = obj.GetName()
    plotName = TString(name)
    plotName.ReplaceAll(" ","_")

    pg = None
    if obj.InheritsFrom("TH2"):
        pg = PlotOnSideGroup(plotName.Data(),
                      Plot(name,
                           drawCommand = "COLZ",
                           normalizeToNumberOfEvents = True, **_common_Calo),
                      ncols=1)
    elif obj.InheritsFrom("TH1"):
        pg = PlotGroup(plotName.Data(),[
                      Plot(name,
                           drawCommand = "", # may want to customize for TH2 (colz, etc.)
                           normalizeToNumberOfEvents = True, **_common_Calo)
                      ],
                      ncols=1)

    if (pg is not None):
        hgcalCaloParticlesPlotter.append("CaloParticles_"+name_collection, [
                  dqmfolder
                  ], PlotFolder(
                    pg,
                    loopSubFolders=False,
                    purpose=PlotPurpose.Timing, page="CaloParticles", section=name_collection)
                  )

    key = keys.After(key)

  templateFile.Close()

  return hgcalCaloParticlesPlotter

#=================================================================================================
# hitValidation
def _hgcalHitFolders(dirName="HGCalSimHitsV/HGCalEESensitive"):
    return "DQMData/Run 1/HGCAL/Run summary/"+dirName

hgcalHitPlotter = Plotter()
hitsLabel = 'Hits'
simHitsLabel = 'Simulated Hits'

hgcalHitPlotter.append("SimHits_Validation", [
                                              "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HitValidation",
                                              ], PlotFolder(
                                                            _HitValidation,
                                                            loopSubFolders=False,
                                                            purpose=PlotPurpose.Timing, page=hitsLabel, section=simHitsLabel
                                                            ))

def append_hgcalHitsPlots(collection = "HGCalSimHitsV", name_collection = "Simulated Hits"):
  _hitsCommonPlots_EE = [
    _Occupancy_EE_zplus,
    _Occupancy_EE_zminus, 
    _EtaPhi_EE_zminus,
    _EtaPhi_EE_zplus
  ]
  _hitsCommonPlots_HE_Sil = [
    _Occupancy_HE_Silicon_zplus,
    _Occupancy_HE_Silicon_zminus,
    _EtaPhi_HE_Silicon_zminus,
    _EtaPhi_HE_Silicon_zplus
  ]
  _hitsCommonPlots_HE_Sci = [
    _Occupancy_HE_Scintillator_zplus,
    _Occupancy_HE_Scintillator_zminus,
    _EtaPhi_HE_Scintillator_zminus,
    _EtaPhi_HE_Scintillator_zplus
  ]

  regions = ["HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"]
  setPlots = [_hitsCommonPlots_EE, _hitsCommonPlots_HE_Sil, _hitsCommonPlots_HE_Sci]
  if "SimHits" in collection :
    _hitsCommonPlots_EE.append(_Energy_EE_0)
    _hitsCommonPlots_EE.append(_Energy_EE_1)
    _hitsCommonPlots_HE_Sil.append(_Energy_HE_Silicon_0)
    _hitsCommonPlots_HE_Sil.append( _Energy_HE_Silicon_1)
    _hitsCommonPlots_HE_Sil.append(_Energy_HE_Scintillator_0)
    _hitsCommonPlots_HE_Sil.append(_Energy_HE_Scintillator_1)
  if "RecHits" in collection :
    _hitsCommonPlots_EE.append(_Energy_EE)
    _hitsCommonPlots_HE_Sil.append(_Energy_HE_Silicon)
    _hitsCommonPlots_HE_Sil.append(_Energy_HE_Scintillator)

  for reg, setPlot in zip(regions, setPlots):
    dirName = collection+"/"+reg
    print(dirName)
    hgcalHitPlotter.append(collection, [
                _hgcalHitFolders(dirName)
                ], PlotFolder(
                *setPlot,
                loopSubFolders=False,
                purpose=PlotPurpose.Timing, page=hitsLabel, section=name_collection))

_digisCommonPlots_EE = [
  _DigiHits_Occupancy_EE_zplus,
  _DigiHits_Occupancy_EE_zminus,
  _DigiHits_Occupancy_XY_EE,
  _DigiHits_ADC_EE,
  _DigiHits_TOA_EE,
  _DigiHits_TOT_EE,
]
_digisCommonPlots_HE_Sil = [
  _DigiHits_Occupancy_HE_Silicon_zplus,
  _DigiHits_Occupancy_HE_Silicon_zminus,
  _DigiHits_Occupancy_XY_HE_Silicon,
  _DigiHits_ADC_HE_Silicon,
  _DigiHits_TOA_HE_Silicon,
  _DigiHits_TOT_HE_Silicon,
]
_digisCommonPlots_HE_Sci = [
  _DigiHits_Occupancy_HE_Scintillator_zplus,
  _DigiHits_Occupancy_HE_Scintillator_zminus,
  _DigiHits_Occupancy_XY_HE_Scintillator,
  _DigiHits_ADC_HE_Scintillator,
  _DigiHits_TOA_HE_Scintillator,
  _DigiHits_TOT_HE_Scintillator,
]

def append_hgcalDigisPlots(collection = "HGCalDigisV", name_collection = "Digis"):
  regions = ["HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"]
  setPlots = [_digisCommonPlots_EE, _digisCommonPlots_HE_Sil, _digisCommonPlots_HE_Sci]
  for reg, setPlot in zip(regions, setPlots):
    dirName = collection+"/"+reg
    print(dirName)
    hgcalHitPlotter.append(name_collection, [
                _hgcalHitFolders(dirName)
                ], PlotFolder(
                *setPlot,
                loopSubFolders=False,
                purpose=PlotPurpose.Timing, page=hitsLabel, section=name_collection))

#=================================================================================================
# hitCalibration
hgcalHitCalibPlotter = Plotter()
hitCalibrationLabel = 'Calibrated RecHits'

hgcalHitCalibPlotter.append("Layer_Occupancy", [
                                                "DQMData/Run 1/HGCalHitCalibration/Run summary",
                                                ], PlotFolder(
                                                              _LayerOccupancy,
                                                              loopSubFolders=False,
                                                              purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
                                                              ))
hgcalHitCalibPlotter.append("ReconstructableEnergyOverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _ReconstructableEnergyOverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
        ))

hgcalHitCalibPlotter.append("ParticleFlowClusterHGCalFromTrackster_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _ParticleFlowClusterHGCalFromTrackster_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
        ))

hgcalHitCalibPlotter.append("PhotonsFromTrackster_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _PhotonsFromTrackster_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
        ))

hgcalHitCalibPlotter.append("EcalDrivenGsfElectronsFromTrackster_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _EcalDrivenGsfElectronsFromTrackster_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
        ))
