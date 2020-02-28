from __future__ import print_function
import os
import sys
import copy
import collections

import six
import ROOT
from ROOT import TFile
from ROOT import gDirectory
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from Validation.RecoTrack.plotting.plotting import Plot, PlotGroup, PlotFolder, Plotter, PlotOnSideGroup
from Validation.RecoTrack.plotting.html import PlotPurpose
import Validation.RecoTrack.plotting.plotting as plotting
import Validation.RecoTrack.plotting.validation as validation
import Validation.RecoTrack.plotting.html as html

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
GeneralInfoDirectory = 'DQMData/Run 1/HGCAL/Run summary/HGCalValidator/GeneralInfo'

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
'''
layerscheme = { 'lastLayerEEzm': 28, 'lastLayerFHzm': 40, 'maxlayerzm': 50, 'lastLayerEEzp': 78, 'lastLayerFHzp': 90, 'maxlayerzp': 100 }
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
_totclusternum_layer_EE_zminus = PlotGroup("totclusternum_layer_EE_zminus", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_totclusternum_layer_FH_zminus = PlotGroup("totclusternum_layer_FH_zminus", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_totclusternum_layer_BH_zminus = PlotGroup("totclusternum_layer_BH_zminus", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

_energyclustered_perlayer_EE_zminus = PlotGroup("energyclustered_perlayer_EE_zminus", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_energyclustered_perlayer_FH_zminus = PlotGroup("energyclustered_perlayer_FH_zminus", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_energyclustered_perlayer_BH_zminus = PlotGroup("energyclustered_perlayer_BH_zminus", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_common_cells = {}
_common_cells.update(_common)
_common_cells["xmin"] = 0
_common_cells["xmax"] = 50
_common_cells["ymin"] = 0.1
_common_cells["ymax"] = 10000
_common_cells["ylog"] = True
_cellsnum_perthick_perlayer_120_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_120_EE_zminus", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=4)

_cellsnum_perthick_perlayer_120_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_120_FH_zminus", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_cellsnum_perthick_perlayer_120_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_120_BH_zminus", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#200 um
_cellsnum_perthick_perlayer_200_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_200_EE_zminus", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=4)

_cellsnum_perthick_perlayer_200_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_200_FH_zminus", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_cellsnum_perthick_perlayer_200_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_200_BH_zminus", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#300 um
_cellsnum_perthick_perlayer_300_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_300_EE_zminus", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=4)

_cellsnum_perthick_perlayer_300_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_300_FH_zminus", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_cellsnum_perthick_perlayer_300_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_300_BH_zminus", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#scint um
_cellsnum_perthick_perlayer_scint_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_-1_EE_zminus", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm)
], ncols=4)

_cellsnum_perthick_perlayer_scint_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_-1_FH_zminus", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_cellsnum_perthick_perlayer_scint_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_-1_BH_zminus", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

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

_distancetomaxcell_perthickperlayer_120_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_120_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_120_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#200 um
_distancetomaxcell_perthickperlayer_200_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_200_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_200_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#300 um
_distancetomaxcell_perthickperlayer_300_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_300_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_300_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#scint um
_distancetomaxcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_-1_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_-1_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_-1_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcell_perthickperlayer_120_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_EE_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_120_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_FH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_120_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_BH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#200 um
_distancebetseedandmaxcell_perthickperlayer_200_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_EE_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_200_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_FH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_200_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_BH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#300 um
_distancebetseedandmaxcell_perthickperlayer_300_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_EE_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_300_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_FH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_300_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_BH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#scint um
_distancebetseedandmaxcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_EE_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_FH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_BH_zminus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#200 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#300 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#scint um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_EE_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_FH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_BH_zminus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_120_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_120_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_120_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#200 um
_distancetoseedcell_perthickperlayer_200_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_200_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_200_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#300 um
_distancetoseedcell_perthickperlayer_300_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_300_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_300_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#scint um
_distancetoseedcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_-1_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_-1_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_-1_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#=====================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#We need points for the weighted plots
_common = {"stat": True, "drawStyle": "EP", "staty": 0.65 }
#120 um
_distancetomaxcell_perthickperlayer_eneweighted_120_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_120_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_120_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#200 um
_distancetomaxcell_perthickperlayer_eneweighted_200_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_200_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_200_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#300 um
_distancetomaxcell_perthickperlayer_eneweighted_300_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_300_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_300_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)
#scint um
_distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_EE_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_FH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_BH_zminus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)


#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_eneweighted_120_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_120_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_120_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#200 um
_distancetoseedcell_perthickperlayer_eneweighted_200_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_200_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_200_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#300 um
_distancetoseedcell_perthickperlayer_eneweighted_300_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_300_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_300_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#scint um
_distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_EE_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_FH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_BH_zminus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#Coming back to the usual definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_totclusternum_layer_EE_zplus = PlotGroup("totclusternum_layer_EE_zplus", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_totclusternum_layer_FH_zplus = PlotGroup("totclusternum_layer_FH_zplus", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_totclusternum_layer_BH_zplus = PlotGroup("totclusternum_layer_BH_zplus", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

_energyclustered_perlayer_EE_zplus = PlotGroup("energyclustered_perlayer_EE_zplus", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_energyclustered_perlayer_FH_zplus = PlotGroup("energyclustered_perlayer_FH_zplus", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_energyclustered_perlayer_BH_zplus = PlotGroup("energyclustered_perlayer_BH_zplus", [
  Plot("energyclustered_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_cellsnum_perthick_perlayer_120_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_120_EE_zplus", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_cellsnum_perthick_perlayer_120_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_120_FH_zplus", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)
_cellsnum_perthick_perlayer_120_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_120_BH_zplus", [
  Plot("cellsnum_perthick_perlayer_120_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#200 um
_cellsnum_perthick_perlayer_200_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_200_EE_zplus", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_cellsnum_perthick_perlayer_200_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_200_FH_zplus", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_cellsnum_perthick_perlayer_200_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_200_BH_zplus", [
  Plot("cellsnum_perthick_perlayer_200_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)
#300 um
_cellsnum_perthick_perlayer_300_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_300_EE_zplus", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_cellsnum_perthick_perlayer_300_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_300_FH_zplus", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)
_cellsnum_perthick_perlayer_300_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_300_BH_zplus", [
  Plot("cellsnum_perthick_perlayer_300_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#scint um
_cellsnum_perthick_perlayer_scint_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_-1_EE_zplus", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_cellsnum_perthick_perlayer_scint_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_-1_FH_zplus", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_cellsnum_perthick_perlayer_scint_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_-1_BH_zplus", [
  Plot("cellsnum_perthick_perlayer_-1_{:02d}".format(i), xtitle="", **_common_cells) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

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

_distancetomaxcell_perthickperlayer_120_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_120_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_120_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#200 um
_distancetomaxcell_perthickperlayer_200_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_200_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_200_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#300 um
_distancetomaxcell_perthickperlayer_300_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_300_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_300_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#scint um
_distancetomaxcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_-1_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_-1_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_-1_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcell_perthickperlayer_120_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_EE_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_120_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_FH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_120_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_BH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#200 um
_distancebetseedandmaxcell_perthickperlayer_200_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_EE_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_200_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_FH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_200_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_BH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#300 um
_distancebetseedandmaxcell_perthickperlayer_300_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_EE_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_300_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_FH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_300_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_BH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#scint um
_distancebetseedandmaxcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_EE_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_FH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_BH_zplus", [
  Plot("distancebetseedandmaxcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#200 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#300 um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#scint um
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_EE_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_FH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_BH_zplus", [
  Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)


#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_120_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_120_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_120_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#200 um
_distancetoseedcell_perthickperlayer_200_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_200_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_200_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#300 um
_distancetoseedcell_perthickperlayer_300_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_300_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_300_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#scint um
_distancetoseedcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_-1_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_-1_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_-1_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#=====================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#We need points for the weighted plots
_common = {"stat": True, "drawStyle": "EP", "staty": 0.65 }

#120 um
_distancetomaxcell_perthickperlayer_eneweighted_120_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_120_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_120_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#200 um
_distancetomaxcell_perthickperlayer_eneweighted_200_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)
_distancetomaxcell_perthickperlayer_eneweighted_200_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_200_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#300 um
_distancetomaxcell_perthickperlayer_eneweighted_300_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_300_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_300_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#scint um
_distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_EE_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_FH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_BH_zplus", [
  Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#----------------------------------------------------------------------------------------------------------------
#120 um
_distancetoseedcell_perthickperlayer_eneweighted_120_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_120_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_120_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_120_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#200 um
_distancetoseedcell_perthickperlayer_eneweighted_200_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_200_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_200_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_200_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#300 um
_distancetoseedcell_perthickperlayer_eneweighted_300_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_300_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_300_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_300_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#scint um
_distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_EE_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_FH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_BH_zplus", [
  Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)
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
_score_caloparticle_to_layerclusters_zminus = PlotGroup("score_caloparticle_to_layercluster_zminus", [
        Plot("Score_caloparticle2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_score) for i in range(0,maxlayerzm)
        ], ncols=8 )

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
_score_layercluster_to_caloparticles_zminus = PlotGroup("score_layercluster_to_caloparticle_zminus", [
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
_sharedEnergy_caloparticle_to_layercluster_zminus = PlotGroup("sharedEnergy_caloparticle_to_layercluster_zminus", _shared_plots_zminus, ncols=8)

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
_sharedEnergy_layercluster_to_caloparticle_zminus = PlotGroup("sharedEnergy_layercluster_to_caloparticle_zminus", _shared_plots2_zminus, ncols=8)


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
_cell_association_table_zminus = PlotGroup("cellAssociation_table_zminus", [
        Plot("cellAssociation_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_assoc) for i in range(0,maxlayerzm)
        ], ncols=8 )

_bin_count = 0
_xbinlabels = [ "L{:02d}".format(i+1) for i in range(0,maxlayerzm) ]
_common_eff = {"stat": False, "legend": False}
_effplots_zminus = [Plot("effic_eta_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(0,maxlayerzm)]
_effplots_zminus.extend([Plot("effic_phi_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(0,maxlayerzm)])
_common_eff = {"stat": False, "legend": False, "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_eff["xmin"] = _bin_count
_common_eff["xmax"] = maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_effplots_zminus.extend([Plot("globalEfficiencies", xtitle="Global Efficiencies in z-", **_common_eff)])
_efficiencies_zminus = PlotGroup("Efficiencies_zminus", _effplots_zminus, ncols=8)

_common_dup = {"stat": False, "legend": False}
_dupplots_zminus = [Plot("duplicate_eta_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(0,maxlayerzm)]
_dupplots_zminus.extend([Plot("duplicate_phi_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(0,maxlayerzm)])
_common_dup = {"stat": False, "legend": False, "title": "Global Duplicates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_dup["xmin"] = _bin_count
_common_dup["xmax"] = _common_dup["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_dupplots_zminus.extend([Plot("globalEfficiencies", xtitle="Global Duplicates in z-", **_common_dup)])
_duplicates_zminus = PlotGroup("Duplicates_zminus", _dupplots_zminus, ncols=8)

_common_fake = {"stat": False, "legend": False}
_fakeplots_zminus = [Plot("fake_eta_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(0,maxlayerzm)]
_fakeplots_zminus.extend([Plot("fake_phi_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(0,maxlayerzm)])
_common_fake = {"stat": False, "legend": False, "title": "Global Fake Rates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_fake["xmin"] = _bin_count
_common_fake["xmax"] = _common_fake["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_common_fake["xbinlabels"] = [ "L{:02d}".format(i+1) for i in range(0,maxlayerzm) ]
_common_fake["xbinlabelsize"] = 10.
_fakeplots_zminus.extend([Plot("globalEfficiencies", xtitle="Global Fake Rate in z-", **_common_fake)])
_fakes_zminus = PlotGroup("FakeRate_zminus", _fakeplots_zminus, ncols=8)

_common_merge = {"stat": False, "legend": False}
_mergeplots_zminus = [Plot("merge_eta_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(0,maxlayerzm)]
_mergeplots_zminus.extend([Plot("merge_phi_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(0,maxlayerzm)])
_common_merge = {"stat": False, "legend": False, "title": "Global Merge Rates in z-", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_merge["xmin"] = _bin_count
_common_merge["xmax"] = _common_merge["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_common_merge["xbinlabels"] = [ "L{:02d}".format(i+1) for i in range(0,maxlayerzm) ]
_common_merge["xbinlabelsize"] = 10.
_mergeplots_zminus.extend([Plot("globalEfficiencies", xtitle="Global merge Rate in z-", **_common_merge)])
_merges_zminus = PlotGroup("MergeRate_zminus", _mergeplots_zminus, ncols=8)


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
_energyscore_cp2lc_zminus = []
for i in range(0, maxlayerzm):
  _energyscore_cp2lc_zminus.append(PlotOnSideGroup("Energy_vs_Score_Layer{:02d}".format(i), Plot("Energy_vs_Score_caloparticle2layer_perlayer{:02d}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1))

_energyscore_lc2cp_zminus = []
_common_energy_score["xlog"]=False
_common_energy_score["ylog"]=False
_common_energy_score["xmin"]=-0.1
for i in range(0, maxlayerzm):
  _energyscore_lc2cp_zminus.append(PlotOnSideGroup("Energy_vs_Score_Layer{:02d}".format(i), Plot("Energy_vs_Score_layer2caloparticle_perlayer{:02d}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1))
#_energyclustered =

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
_score_caloparticle_to_layerclusters_zplus = PlotGroup("score_caloparticle_to_layercluster_zplus", [
        Plot("Score_caloparticle2layercl_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_score) for i in range(maxlayerzm,maxlayerzp)
        ], ncols=8 )

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
_score_layercluster_to_caloparticles_zplus = PlotGroup("score_layercluster_to_caloparticle_zplus", [
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
_sharedEnergy_caloparticle_to_layercluster_zplus = PlotGroup("sharedEnergy_caloparticle_to_layercluster_zplus", _shared_plots_zplus, ncols=8)

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
_sharedEnergy_layercluster_to_caloparticle_zplus = PlotGroup("sharedEnergy_layercluster_to_caloparticle_zplus", _shared_plots2_zplus, ncols=8)


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
_cell_association_table_zplus = PlotGroup("cellAssociation_table_zplus", [
        Plot("cellAssociation_perlayer{:02d}".format(i), xtitle="Layer {:02d} in z-".format(i%maxlayerzm+1) if (i<maxlayerzm) else "Layer {:02d} in z+".format(i%maxlayerzm+1), **_common_assoc) for i in range(maxlayerzm,maxlayerzp)
        ], ncols=8 )


_bin_count = 50
_common_eff = {"stat": False, "legend": False}
_effplots_zplus = [Plot("effic_eta_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(maxlayerzm,maxlayerzp)]
_effplots_zplus.extend([Plot("effic_phi_layer{:02d}".format(i), xtitle="", **_common_eff) for i in range(maxlayerzm,maxlayerzp)])
_common_eff = {"stat": False, "legend": False, "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_eff["xmin"] = _bin_count
_common_eff["xmax"] = _common_eff["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_effplots_zplus.extend([Plot("globalEfficiencies", xtitle="Global Efficiencies in z+", **_common_eff)])
_efficiencies_zplus = PlotGroup("Efficiencies_zplus", _effplots_zplus, ncols=8)


_common_dup = {"stat": False, "legend": False}
_dupplots_zplus = [Plot("duplicate_eta_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(maxlayerzm,maxlayerzp)]
_dupplots_zplus.extend([Plot("duplicate_phi_layer{:02d}".format(i), xtitle="", **_common_dup) for i in range(maxlayerzm,maxlayerzp)])
_common_dup = {"stat": False, "legend": False, "title": "Global Duplicates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_dup["xmin"] = _bin_count
_common_dup["xmax"] = _common_dup["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_dupplots_zplus.extend([Plot("globalEfficiencies", xtitle="Global Duplicates in z+", **_common_dup)])
_duplicates_zplus = PlotGroup("Duplicates_zplus", _dupplots_zplus, ncols=8)

_common_fake = {"stat": False, "legend": False}
_fakeplots_zplus = [Plot("fake_eta_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(maxlayerzm,maxlayerzp)]
_fakeplots_zplus.extend([Plot("fake_phi_layer{:02d}".format(i), xtitle="", **_common_fake) for i in range(maxlayerzm,maxlayerzp)])
_common_fake = {"stat": False, "legend": False, "title": "Global Fake Rates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_fake["xmin"] = _bin_count
_common_fake["xmax"] = _common_fake["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_fakeplots_zplus.extend([Plot("globalEfficiencies", xtitle="Global Fake Rate in z+", **_common_fake)])
_fakes_zplus = PlotGroup("FakeRate_zplus", _fakeplots_zplus, ncols=8)

_common_merge = {"stat": False, "legend": False}
_mergeplots_zplus = [Plot("merge_eta_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(maxlayerzm,maxlayerzp)]
_mergeplots_zplus.extend([Plot("merge_phi_layer{:02d}".format(i), xtitle="", **_common_merge) for i in range(maxlayerzm,maxlayerzp)])
_common_merge = {"stat": False, "legend": False, "title": "Global Merge Rates in z+", "xbinlabels": _xbinlabels, "xbinlabelsize": 12, "xbinlabeloptions": "v"}
_common_merge["xmin"] = _bin_count
_common_merge["xmax"] = _common_merge["xmin"] + maxlayerzm
_bin_count += 4*maxlayerzm # 2 for the eta{-,+} and 2 for phi{+,-}
_mergeplots_zplus.extend([Plot("globalEfficiencies", xtitle="Global merge Rate in z+", **_common_merge)])
_merges_zplus = PlotGroup("MergeRate_zplus", _mergeplots_zplus, ncols=8)


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
_energyscore_cp2lc_zplus = []
for i in range(maxlayerzm,maxlayerzp):
  _energyscore_cp2lc_zplus.append(PlotOnSideGroup("Energy_vs_Score_Layer{:02d}".format(i), Plot("Energy_vs_Score_caloparticle2layer_perlayer{:02d}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1))

_common_energy_score["xlog"]=False
_common_energy_score["ylog"]=False
_common_energy_score["xmin"]=-0.1
_energyscore_lc2cp_zplus = []
for i in range(maxlayerzm,maxlayerzp):
  _energyscore_lc2cp_zplus.append(PlotOnSideGroup("Energy_vs_Score_Layer{:02d}".format(i), Plot("Energy_vs_Score_layer2caloparticle_perlayer{:02d}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1))
#_energyclustered =

#--------------------------------------------------------------------------------------------
# MULTICLUSTERS
#--------------------------------------------------------------------------------------------
_common_score = {#"title": "Score CaloParticle to MultiClusters",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 100000,
                 "xmin": 0,
                 "xmax": 1.0,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_score.update(_legend_common)
_score_caloparticle_to_multiclusters = PlotGroup("score_caloparticle_to_multicluster", [
        Plot("Score_caloparticle2multicl", **_common_score)
        ], ncols=1)

_common_score = {#"title": "Score MultiCluster to CaloParticles",
                 "stat": False,
                 "ymin": 0.1,
                 "ymax": 100000,
                 "xmin": 0,
                 "xmax": 1.0,
                 "drawStyle": "hist",
                 "lineWidth": 1,
                 "ylog": True
                }
_common_score.update(_legend_common)
_score_multicluster_to_caloparticles = PlotGroup("score_multicluster_to_caloparticle", [
        Plot("Score_multicl2caloparticle", **_common_score)
        ])

_common_shared= {"title": "Shared Energy CaloParticle To Multi Cluster ",
                 "stat": False,
                 "legend": True,
                 "xmin": 0,
                 "xmax": 2.0,
               }
_common_shared.update(_legend_common)
_shared_plots = [ Plot("SharedEnergy_caloparticle2multicl", **_common_shared) ]
_common_shared["xmin"] = -4.0
_common_shared["xmax"] = 4.0
_shared_plots.extend([Plot("SharedEnergy_caloparticle2multicl_vs_eta", **_common_shared)])
_shared_plots.extend([Plot("SharedEnergy_caloparticle2multicl_vs_phi", **_common_shared)])
_sharedEnergy_caloparticle_to_multicluster = PlotGroup("sharedEnergy_caloparticle_to_multicluster", _shared_plots, ncols=3)

_common_shared= {"title": "Shared Energy Multi Cluster To CaloParticle ",
                 "stat": False,
                 "legend": True,
                 "xmin": 0,
                 "xmax": 2.0,
                }
_common_shared.update(_legend_common)
_shared_plots2 = [Plot("SharedEnergy_multicluster2caloparticle", **_common_shared)]
_common_shared["xmin"] = -4.0
_common_shared["xmax"] = 4.0
_shared_plots2.extend([Plot("SharedEnergy_multicl2caloparticle_vs_eta", **_common_shared)])
_shared_plots2.extend([Plot("SharedEnergy_multicl2caloparticle_vs_phi", **_common_shared)])
_sharedEnergy_multicluster_to_caloparticle = PlotGroup("sharedEnergy_multicluster_to_caloparticle", _shared_plots2, ncols=3)


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

_common_eff = {"stat": False, "legend": False}
_effplots = [Plot("effic_eta", xtitle="", **_common_eff)]
_effplots.extend([Plot("effic_phi", xtitle="", **_common_eff)])
_effplots.extend([Plot("globalEfficiencies", xtitle="", **_common_eff)])
_efficiencies = PlotGroup("Efficiencies", _effplots, ncols=3)


_common_dup = {"stat": False, "legend": False}
_dupplots = [Plot("duplicate_eta", xtitle="", **_common_dup)]
_dupplots.extend([Plot("duplicate_phi", xtitle="", **_common_dup)])
_dupplots.extend([Plot("globalEfficiencies", xtitle="", **_common_dup)])
_duplicates = PlotGroup("Duplicates", _dupplots, ncols=3)

_common_fake = {"stat": False, "legend": False}
_fakeplots = [Plot("fake_eta", xtitle="", **_common_fake)]
_fakeplots.extend([Plot("fake_phi", xtitle="", **_common_fake)])
_fakeplots.extend([Plot("globalEfficiencies", xtitle="", **_common_fake)])
_fakes = PlotGroup("FakeRate", _fakeplots, ncols=3)

_common_merge = {"stat": False, "legend": False}
_mergeplots = [Plot("merge_eta", xtitle="", **_common_merge)]
_mergeplots.extend([Plot("merge_phi", xtitle="", **_common_merge)])
_mergeplots.extend([Plot("globalEfficiencies", xtitle="", **_common_merge)])
_merges = PlotGroup("MergeRate", _mergeplots, ncols=3)

_common_energy_score = dict(removeEmptyBins=True, xbinlabelsize=10, xbinlabeloption="d")
_common_energy_score["ymax"] = 1.
_common_energy_score["xmax"] = 1.0
_energyscore_cp2mcl = PlotOnSideGroup("_energyscore_cp2mcl", Plot("Energy_vs_Score_caloparticle2multi", drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1)
_common_energy_score["ymax"] = 1.
_common_energy_score["xmax"] = 1.0
_energyscore_mcl2cp = PlotOnSideGroup("_energyscore_mcl2cp", Plot("Energy_vs_Score_multi2caloparticle", drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1)

#Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

_totmulticlusternum = PlotGroup("totmulticlusternum", [
  Plot("totmulticlusternum", xtitle="", **_common)
],ncols=1)

_multicluster_firstlayer = PlotGroup("multicluster_firstlayer", [
  Plot("multicluster_firstlayer", xtitle="Layer number", **_common)
],ncols=1)

_multicluster_lastlayer = PlotGroup("multicluster_lastlayer", [
  Plot("multicluster_lastlayer", xtitle="Layer number", **_common)
],ncols=1)

_multicluster_layersnum = PlotGroup("multicluster_layersnum", [
  Plot("multicluster_layersnum", xtitle="", **_common)
],ncols=1)

_common["xmax"] = 50
_clusternum_in_multicluster = PlotGroup("clusternum_in_multicluster",[
  Plot("clusternum_in_multicluster", xtitle="", **_common)
],ncols=1)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}
_common = {"stat": True, "drawStyle": "pcolz", "staty": 0.65}

_clusternum_in_multicluster_vs_layer = PlotGroup("clusternum_in_multicluster_vs_layer",[
  Plot("clusternum_in_multicluster_vs_layer", xtitle="Layer number", ytitle = "<2d Layer Clusters in Multicluster>",  **_common)
],ncols=1)

_common["scale"] = 100.
#, ztitle = "% of clusters" normalizeToUnitArea=True
_multiplicity_numberOfEventsHistogram = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA/multiplicity_numberOfEventsHistogram"
_multiplicity_zminus_numberOfEventsHistogram = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA/multiplicity_zminus_numberOfEventsHistogram"
_multiplicity_zplus_numberOfEventsHistogram = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA/multiplicity_zplus_numberOfEventsHistogram"
_multiplicityOfLCinMCL = PlotGroup("multiplicityOfLCinMCL",[
  Plot("multiplicityOfLCinMCL", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Cluster size (n_{hit})         ", drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)
],ncols=1)

_multiplicityOfLCinMCL_vs_layercluster_zminus = PlotGroup("multiplicityOfLCinMCL_vs_layercluster_zminus",[
  Plot("multiplicityOfLCinMCL_vs_layercluster_zminus", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Layer Number         ", drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)
],ncols=1)

_multiplicityOfLCinMCL_vs_layercluster_zplus = PlotGroup("multiplicityOfLCinMCL_vs_layercluster_zplus",[
  Plot("multiplicityOfLCinMCL_vs_layercluster_zplus", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Layer Number         ", drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)
],ncols=1)

_multiplicityOfLCinMCL_vs_layerclusterenergy = PlotGroup("multiplicityOfLCinMCL_vs_layerclusterenergy",[
  Plot("multiplicityOfLCinMCL_vs_layerclusterenergy", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Cluster Energy (GeV)         ", drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)
],ncols=1)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}
#--------------------------------------------------------------------------------------------
# z-
#--------------------------------------------------------------------------------------------
_clusternum_in_multicluster_perlayer_zminus_EE = PlotGroup("clusternum_in_multicluster_perlayer_zminus_EE", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=4)

_clusternum_in_multicluster_perlayer_zminus_FH = PlotGroup("clusternum_in_multicluster_perlayer_zminus_FH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=4)

_clusternum_in_multicluster_perlayer_zminus_BH = PlotGroup("clusternum_in_multicluster_perlayer_zminus_BH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=4)

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_clusternum_in_multicluster_perlayer_zplus_EE = PlotGroup("clusternum_in_multicluster_perlayer_zplus_EE", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=4)

_clusternum_in_multicluster_perlayer_zplus_FH = PlotGroup("clusternum_in_multicluster_perlayer_zplus_FH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=4)

_clusternum_in_multicluster_perlayer_zplus_BH = PlotGroup("clusternum_in_multicluster_perlayer_zplus_BH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=4)

#Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#Some multiclusters quantities
_multicluster_pt = PlotGroup("multicluster_pt", [
  Plot("multicluster_pt", xtitle="", **_common)
],ncols=1)

_multicluster_eta = PlotGroup("multicluster_eta", [
  Plot("multicluster_eta", xtitle="", **_common)
],ncols=1)

_multicluster_phi = PlotGroup("multicluster_phi", [
  Plot("multicluster_phi", xtitle="", **_common)
],ncols=1)

_multicluster_energy = PlotGroup("multicluster_energy", [
  Plot("multicluster_energy", xtitle="", **_common)
],ncols=1)

_multicluster_x = PlotGroup("multicluster_x", [
  Plot("multicluster_x", xtitle="", **_common)
],ncols=1)

_multicluster_y = PlotGroup("multicluster_y", [
  Plot("multicluster_y", xtitle="", **_common)
],ncols=1)

_multicluster_z = PlotGroup("multicluster_z", [
  Plot("multicluster_z", xtitle="", **_common)
],ncols=1)

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

_SimHits_Occupancy_EE_zplus = PlotGroup("SimHits_Occupancy_EE_zplus", [
                                                                       Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="SimHits_Occupancy_EE_zplus", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                       ], ncols=4)

_SimHits_Occupancy_HE_Silicon_zplus = PlotGroup("SimHits_Occupancy_HE_Silicon_zplus", [
                                                                                       Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="SimHits_HE_Occupancy_zplus", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                       ], ncols=4)

_SimHits_Occupancy_HE_Scintillator_zplus = PlotGroup("SimHits_Occupancy_HE_Scintillator_zplus", [
                                                                                                 Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="SimHits_Occupancy_HE_Scintillator_zplus", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                                 ], ncols=4)

_SimHits_Occupancy_EE_zminus = PlotGroup("SimHits_Occupancy_EE_zminus", [
                                                                         Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="SimHits_Occupancy_EE_zminus", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                         ], ncols=4)

_SimHits_Occupancy_HE_Silicon_zminus = PlotGroup("SimHits_Occupancy_HE_Silicon_zminus", [
                                                                                         Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="SimHits_Occupancy_HE_Silicon_zminus", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                         ], ncols=4)

_SimHits_Occupancy_HE_Scintillator_zminus = PlotGroup("SimHits_Occupancy_HE_Scintillator_zminus", [
                                                                                                   Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="SimHits_Occupancy_HE_Scintillator_zminus", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                                   ], ncols=4)

_common_etaphi = dict(removeEmptyBins=False, xbinlabelsize=10, xbinlabeloption="d", ymin=None)

_SimHits_EtaPhi_EE_zplus=[]
for i in range(EE_min,EE_max+1):
    _SimHits_EtaPhi_EE_zplus.append(PlotOnSideGroup("EE_EtaPhi_zplus_layer_{:02d}".format(i), Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="SimHits_EtaPhi_EE_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_SimHits_EtaPhi_HE_Silicon_zplus=[]
for i in range(HESilicon_min,HESilicon_max+1):
    _SimHits_EtaPhi_HE_Silicon_zplus.append(PlotOnSideGroup("HE_Silicon_EtaPhi_zplus_layer_{:02d}".format(i), Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="SimHits_EtaPhi_HE_Silicon_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_SimHits_EtaPhi_HE_Scintillator_zplus=[]
for i in range(HEScintillator_min,HEScintillator_max+1):
    _SimHits_EtaPhi_HE_Scintillator_zplus.append(PlotOnSideGroup("HE_Scintillator_EtaPhi_zplus_layer_{:02d}".format(i), Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="SimHits_EtaPhi_HE_Scintillator_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_SimHits_EtaPhi_EE_zminus=[]
for i in range(EE_min,EE_max+1):
    _SimHits_EtaPhi_EE_zminus.append(PlotOnSideGroup("EE_EtaPhi_zminus_layer_{:02d}".format(i), Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="SimHits_EtaPhi_EE_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_SimHits_EtaPhi_HE_Silicon_zminus=[]
for i in range(HESilicon_min,HESilicon_max+1):
    _SimHits_EtaPhi_HE_Silicon_zminus.append(PlotOnSideGroup("HE_Silicon_EtaPhi_zminus_layer_{:02d}".format(i), Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="SimHits_EtaPhi_HE_Silicon_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_SimHits_EtaPhi_HE_Scintillator_zminus=[]
for i in range(HEScintillator_min,HEScintillator_max+1):
    _SimHits_EtaPhi_HE_Scintillator_zminus.append(PlotOnSideGroup("HE_Scintillator_EtaPhi_zminuslayer_{:02d}".format(i), Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="SimHits_EtaPhi_HE_Scintillator_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": True}

_SimHits_Energy_EE_0 = PlotGroup("SimHits_Energy_Time_0_EE", [
                                                              Plot("energy_time_0_layer_{:02d}".format(i), title="SimHits_Energy_Time_0_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                              ], ncols=4)

_SimHits_Energy_HE_Silicon_0 = PlotGroup("SimHits_Energy_Time_0_HE_Silicon", [
                                                                              Plot("energy_time_0_layer_{:02d}".format(i), title="SimHits_Energy_Time_0_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                              ], ncols=4)

_SimHits_Energy_HE_Scintillator_0 = PlotGroup("SimHits_Energy_Time_0_HE_Scintillator", [
                                                                                        Plot("energy_time_0_layer_{:02d}".format(i), title="SimHits_Energy_Time_0_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                        ], ncols=4)

_SimHits_Energy_EE_1 = PlotGroup("SimHits_Energy_Time_1_EE", [
                                                              Plot("energy_time_1_layer_{:02d}".format(i), title="SimHits_Energy_Time_1_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                              ], ncols=4)

_SimHits_Energy_HE_Silicon_1 = PlotGroup("SimHits_Energy_Time_1_HE_Silicon", [
                                                                              Plot("energy_time_1_layer_{:02d}".format(i), title="SimHits_Energy_Time_1_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                              ], ncols=4)

_SimHits_Energy_HE_Scintillator_1 = PlotGroup("SimHits_Energy_Time_1_HE_Scintillator", [
                                                                                        Plot("energy_time_1_layer_{:02d}".format(i), title="SimHits_Energy_Time_1_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                        ], ncols=4)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}

_RecHits_Occupancy_EE_zplus = PlotGroup("RecHits_Occupancy_EE_zplus", [
                                                                       Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="RecHits_Occupancy_EE_zplus", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                       ], ncols=4)

_RecHits_Occupancy_HE_Silicon_zplus = PlotGroup("RecHits_Occupancy_HE_Silicon_zplus", [
                                                                                       Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="RecHits_Occupancy_HE_Silicon_zplus", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                       ], ncols=4)

_RecHits_Occupancy_HE_Scintillator_zplus = PlotGroup("RecHits_Occupancy_HE_Scintillator_zplus", [
                                                                                                 Plot("HitOccupancy_Plus_layer_{:02d}".format(i), title="RecHits_Occupancy_HE_Scintillator_zplus", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                                 ], ncols=4)

_RecHits_Occupancy_EE_zminus = PlotGroup("RecHits_Occupancy_EE_zminus", [
                                                                         Plot("HitOccupancy_Minus_layer_{:02d}".format(i), title="RecHits_Occupancy_EE_zminus", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                         ], ncols=4)

_RecHits_Occupancy_HE_Silicon_zminus = PlotGroup("RecHits_Occupancy_HE_Silicon_zminus", [
                                                                                         Plot("HitOccupancy_Minus_layer_{:02d}".format(i), title="RecHits_Occupancy_HE_Silicon_zminus", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                         ], ncols=4)

_RecHits_Occupancy_HE_Scintillator_zminus = PlotGroup("RecHits_Occupancy_HE_Scintillator_zminus", [
                                                                                                   Plot("HitOccupancy_Minus_layer_{:02d}".format(i),  title="RecHits_Occupancy_HE_Scintillator_zminus", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                                   ], ncols=4)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": True}

_RecHits_Energy_EE = PlotGroup("RecHits_Energy_EE", [
                                                     Plot("energy_layer_{:02d}".format(i), title="RecHits_Energy_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                     ], ncols=4)

_RecHits_Energy_HE_Silicon = PlotGroup("RecHits_Energy_HE_Silicon", [
                                                                     Plot("energy_layer_{:02d}".format(i), title="RecHits_Energy_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                     ], ncols=4)

_RecHits_Energy_HE_Scintillator = PlotGroup("RecHits_Energy_HE_Scintillator", [
                                                                               Plot("energy_layer_{:02d}".format(i), title="RecHits_Energy_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                               ], ncols=4)

_RecHits_EtaPhi_EE_zplus=[]
for i in range(EE_min,EE_max+1):
    _RecHits_EtaPhi_EE_zplus.append(PlotOnSideGroup("RecHits_EtaPhi_EE_zplus_layer_{:02d}".format(i), Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="RecHits_EtaPhi_EE_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_RecHits_EtaPhi_HE_Silicon_zplus=[]
for i in range(HESilicon_min,HESilicon_max+1):
    _RecHits_EtaPhi_HE_Silicon_zplus.append(PlotOnSideGroup("RecHits_EtaPhi_HE_Silicon_zplus_layer_{:02d}".format(i), Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="RecHits_EtaPhi_HE_Silicon_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_RecHits_EtaPhi_HE_Scintillator_zplus=[]
for i in range(HEScintillator_min,HEScintillator_max+1):
    _RecHits_EtaPhi_HE_Scintillator_zplus.append(PlotOnSideGroup("RecHits_EtaPhi_HE_Scintillator_zplus_layer_{:02d}".format(i), Plot("EtaPhi_Plus_layer_{:02d}".format(i), title="RecHits_EtaPhi_HE_Scintillator_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_RecHits_EtaPhi_EE_zminus=[]
for i in range(EE_min,EE_max+1):
    _RecHits_EtaPhi_EE_zminus.append(PlotOnSideGroup("RecHits_EtaPhi_EE_zminus_layer_{:02d}".format(i), Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="RecHits_EtaPhi_EE_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_RecHits_EtaPhi_HE_Silicon_zminus=[]
for i in range(HESilicon_min,HESilicon_max+1):
    _RecHits_EtaPhi_HE_Silicon_zminus.append(PlotOnSideGroup("RecHits_EtaPhi_HE_Silicon_zminus_layer_{:02d}".format(i), Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="RecHits_EtaPhi_HE_Silicon_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_RecHits_EtaPhi_HE_Scintillator_zminus=[]
for i in range(HEScintillator_min,HEScintillator_max+1):
    _RecHits_EtaPhi_HE_Scintillator_zminus.append(PlotOnSideGroup("RecHits_EtaPhi_HE_Scintillator_zminus_layer_{:02d}".format(i), Plot("EtaPhi_Minus_layer_{:02d}".format(i), title="RecHits_EtaPhi_HE_Scintillator_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_etaphi), ncols=1))

_DigiHits_ADC_EE = PlotGroup("DigiHits_ADC_EE", [
                                                 Plot("ADC_layer_{:02d}".format(i), title="DigiHits_ADC_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                 ], ncols=4)

_DigiHits_ADC_HE_Silicon = PlotGroup("DigiHits_ADC_HE_Silicon", [
                                                                 Plot("ADC_layer_{:02d}".format(i), title="DigiHits_ADC_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                 ], ncols=4)

_DigiHits_ADC_HE_Scintillator = PlotGroup("DigiHits_ADC_HE_Scintillator", [
                                                                           Plot("ADC_layer_{:02d}".format(i), title="DigiHits_ADC_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                           ], ncols=4)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}

_DigiHits_Occupancy_EE_zplus = PlotGroup("DigiHits_Occupancy_EE_zplus", [
                                                                         Plot("DigiOccupancy_Plus_layer_{:02d}".format(i), title="DigiHits_Occupancy_EE_zplus", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                         ], ncols=4)

_DigiHits_Occupancy_HE_Silicon_zplus = PlotGroup("DigiHits_Occupancy_HE_Silicon_zplus", [
                                                                                         Plot("DigiOccupancy_Plus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Silicon_zplus", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                         ], ncols=4)

_DigiHits_Occupancy_HE_Scintillator_zplus = PlotGroup("DigiHits_Occupancy_HE_Scintillator_zplus", [
                                                                                                   Plot("DigiOccupancy_Plus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Scintillator_zplus", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                                   ], ncols=4)

_DigiHits_Occupancy_EE_zminus = PlotGroup("DigiHits_Occupancy_EE_zminus", [
                                                                           Plot("DigiOccupancy_Minus_layer_{:02d}".format(i), title="DigiHits_Occupancy_EE_zminus", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                                           ], ncols=4)

_DigiHits_Occupancy_HE_Silicon_zminus = PlotGroup("DigiHits_Occupancy_HE_Silicon_zminus", [
                                                                                           Plot("DigiOccupancy_Minus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Silicon_zminus", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                                           ], ncols=4)

_DigiHits_Occupancy_HE_Scintillator_zminus = PlotGroup("DigiHits_Occupancy_HE_Scintillator_zminus", [
                                                                                                     Plot("DigiOccupancy_Minus_layer_{:02d}".format(i), title="DigiHits_Occupancy_HE_Scintillator_zminus", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                                                     ], ncols=4)

_common_XY = dict(removeEmptyBins=True, xbinlabelsize=10, xbinlabeloption="d", ymin=None)

_DigiHits_Occupancy_XY_EE_zplus=[]
for i in range(EE_min,EE_max+1):
    _DigiHits_Occupancy_XY_EE_zplus.append(PlotOnSideGroup("DigiHits_Occupancy_XY_EE_zplus_layer_{:02d}".format(i), Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_EE_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY), ncols=1))

_DigiHits_Occupancy_XY_HE_Silicon_zplus=[]
for i in range(HESilicon_min,HESilicon_max+1):
    _DigiHits_Occupancy_XY_HE_Silicon_zplus.append(PlotOnSideGroup("DigiHits_Occupancy_XY_HE_Silicon_zplus_layer_{:02d}".format(i), Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_HE_Silicon_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY), ncols=1))

_DigiHits_Occupancy_XY_HE_Scintillator_zplus=[]
for i in range(HEScintillator_min,HEScintillator_max+1):
    _DigiHits_Occupancy_XY_HE_Scintillator_zplus.append(PlotOnSideGroup("DigiHits_Occupancy_XY_HE_Scintillator_zplus_layer_{:02d}".format(i), Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_HE_Scintillator_zplus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY), ncols=1))

_DigiHits_Occupancy_XY_EE_zminus=[]
for i in range(EE_min,EE_max+1):
    _DigiHits_Occupancy_XY_EE_zminus.append(PlotOnSideGroup("DigiHits_Occupancy_XY_EE_zminus_layer_{:02d}".format(i), Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_EE_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY), ncols=1))

_DigiHits_Occupancy_XY_HE_Silicon_zminus=[]
for i in range(HESilicon_min,HESilicon_max+1):
    _DigiHits_Occupancy_XY_HE_Silicon_zminus.append(PlotOnSideGroup("DigiHits_Occupancy_XY_HE_Silicon_zminus_layer_{:02d}".format(i), Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_HE_Silicon_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY), ncols=1))

_DigiHits_Occupancy_XY_HE_Scintillator_zminus=[]
for i in range(HEScintillator_min,HEScintillator_max+1):
    _DigiHits_Occupancy_XY_HE_Scintillator_zminus.append(PlotOnSideGroup("DigiHits_Occupancy_XY_HE_Scintillator_zminus_layer_{:02d}".format(i), Plot("DigiOccupancy_XY_layer_{:02d}".format(i), title="DigiHits_Occupancy_XY_HE_Scintillator_zminus", xtitle="Layer {}".format(i), drawStyle="COLZ", adjustMarginRight=0.1, **_common_XY), ncols=1))

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": True}

_DigiHits_TOA_EE = PlotGroup("DigiHits_TOA_EE", [
                                                 Plot("TOA_layer_{:02d}".format(i), title="DigiHits_TOA_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                 ], ncols=4)

_DigiHits_TOA_HE_Silicon = PlotGroup("DigiHits_TOA_HE_Silicon", [
                                                                 Plot("TOA_layer_{:02d}".format(i), title="DigiHits_TOA_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                 ], ncols=4)

_DigiHits_TOA_HE_Scintillator = PlotGroup("DigiHits_TOA_HE_Scintillator", [
                                                                           Plot("TOA_layer_{:02d}".format(i), title="DigiHits_TOA_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                           ], ncols=4)

_DigiHits_TOT_EE = PlotGroup("DigiHits_TOT_EE", [
                                                 Plot("TOT_layer_{:02d}".format(i), title="DigiHits_TOT_EE", xtitle="Layer {}".format(i), **_common) for i in range(EE_min,EE_max+1)
                                                 ], ncols=4)

_DigiHits_TOT_HE_Silicon = PlotGroup("DigiHits_TOT_HE_Silicon", [
                                                                 Plot("TOT_layer_{:02d}".format(i), title="DigiHits_TOT_HE_Silicon", xtitle="Layer {}".format(i), **_common) for i in range(HESilicon_min,HESilicon_max+1)
                                                                 ], ncols=4)

_DigiHits_TOT_HE_Scintillator = PlotGroup("DigiHits_TOT_HE_Scintillator", [
                                                                           Plot("TOT_layer_{:02d}".format(i), title="DigiHits_TOT_HE_Scintillator", xtitle="Layer {}".format(i), **_common) for i in range(HEScintillator_min,HEScintillator_max+1)
                                                                           ], ncols=4)

#===================================================================================================================
#Plot definition for HitCalibration
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ymin": 0.1, "ylog": False}

_LayerOccupancy = PlotGroup("LayerOccupancy", [
                                               Plot("LayerOccupancy", title="LayerOccupancy", **_common)], ncols=1)

_ReconstructableEnergyOverCPenergy = PlotGroup("ReconstructableEnergyOverCPenergy", [
  Plot("h_EoP_CPene_100_calib_fraction", title="EoP_CPene_100_calib_fraction", **_common),
  Plot("h_EoP_CPene_200_calib_fraction", title="EoP_CPene_200_calib_fraction", **_common),
  Plot("h_EoP_CPene_300_calib_fraction", title="EoP_CPene_300_calib_fraction", **_common),
])

_ParticleFlowClusterHGCalFromMultiCl_Closest_EoverCPenergy = PlotGroup("ParticleFlowClusterHGCalFromMultiCl", [
  Plot("hgcal_EoP_CPene_100_calib_fraction", title="hgcal_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_200_calib_fraction", title="hgcal_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_300_calib_fraction", title="hgcal_EoP_CPene_300_calib_fraction", **_common),
])

_EcalDrivenGsfElectronsFromMultiCl_Closest_EoverCPenergy = PlotGroup("EcalDrivenGsfElectronsFromMultiCl", [
  Plot("hgcal_ele_EoP_CPene_100_calib_fraction", title="hgcal_ele_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_200_calib_fraction", title="hgcal_ele_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_300_calib_fraction", title="hgcal_ele_EoP_CPene_300_calib_fraction", **_common),
])

_PhotonsFromMultiCl_Closest_EoverCPenergy = PlotGroup("PhotonsFromMultiCl", [
  Plot("hgcal_photon_EoP_CPene_100_calib_fraction", title="hgcal_photon_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_200_calib_fraction", title="hgcal_photon_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_300_calib_fraction", title="hgcal_photon_EoP_CPene_300_calib_fraction", **_common),
])

#=================================================================================================
hgcalLayerClustersPlotter = Plotter()
#We follow Chris categories in folders
# [A] calculated "energy density" for cells in a) 120um, b) 200um, c) 300um, d) scint
# (one entry per rechit, in the appropriate histo)
hgcalLayerClustersPlotter.append("CellsEnergyDensityPerThickness", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
        _cellsenedens_thick,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="CellsEnergyDensityPerThickness"
        ))

# [B] number of layer clusters per event in a) 120um, b) 200um, c) 300um, d) scint
# (one entry per event in each of the four histos)
hgcalLayerClustersPlotter.append("TotalNumberofLayerClustersPerThickness", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
        _totclusternum_thick,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="TotalNumberofLayerClustersPerThickness"
        ))

# [C] number of layer clusters per layer (one entry per event in each histo)
# z-
hgcalLayerClustersPlotter.append("NumberofLayerClustersPerLayer_zminus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
        _totclusternum_layer_EE_zminus,
        _totclusternum_layer_FH_zminus,
        _totclusternum_layer_BH_zminus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="NumberofLayerClustersPerLayer_zminus"
        ))

# z+
hgcalLayerClustersPlotter.append("NumberofLayerClustersPerLayer_zplus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
        _totclusternum_layer_EE_zplus,
        _totclusternum_layer_FH_zplus,
        _totclusternum_layer_BH_zplus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="NumberofLayerClustersPerLayer_zplus"
        ))

# [D] For each layer cluster:
# number of cells in layer cluster, by layer - separate histos in each layer for 120um Si, 200/300um Si, Scint
# NB: not all combinations exist; e.g. no 120um Si in layers with scint.
# (One entry in the appropriate histo per layer cluster).
# z-
hgcalLayerClustersPlotter.append("CellsNumberPerLayerPerThickness_zminus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
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
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="CellsNumberPerLayerPerThickness_zminus"
        ))

# z+
hgcalLayerClustersPlotter.append("CellsNumberPerLayerPerThickness_zplus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
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
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="CellsNumberPerLayerPerThickness_zplus"
        ))

# [E] For each layer cluster:
# distance of cells from a) seed cell, b) max cell; and c), d): same with entries weighted by cell energy
# separate histos in each layer for 120um Si, 200/300um Si, Scint
# NB: not all combinations exist; e.g. no 120um Si in layers with scint.
# (One entry in each of the four appropriate histos per cell in a layer cluster)
# z-
hgcalLayerClustersPlotter.append("CellsDistanceToSeedAndMaxCellPerLayerPerThickness_zminus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
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
        _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zminus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="CellsDistanceToSeedAndMaxCellPerLayerPerThickness_zminus"
        ))

# z+
hgcalLayerClustersPlotter.append("CellsDistanceToSeedAndMaxCellPerLayerPerThickness_zplus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
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
        _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zplus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="CellsDistanceToSeedAndMaxCellPerLayerPerThickness_zplus"
        ))

# [F] Looking at the fraction of true energy that has been clustered; by layer and overall
# z-
hgcalLayerClustersPlotter.append("EnergyClusteredByLayerAndOverall_zminus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
        _energyclustered_perlayer_EE_zminus,
        _energyclustered_perlayer_FH_zminus,
        _energyclustered_perlayer_BH_zminus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="EnergyClusteredByLayerAndOverall_zminus"
        ))
# z+
hgcalLayerClustersPlotter.append("EnergyClusteredByLayerAndOverall_zplus", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
        _energyclustered_perlayer_EE_zplus,
        _energyclustered_perlayer_FH_zplus,
        _energyclustered_perlayer_BH_zplus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="EnergyClusteredByLayerAndOverall_zplus"
        ))

# [G] Miscellaneous plots:
# longdepthbarycentre: The longitudinal depth barycentre. One entry per event.
# mixedhitscluster: Number of clusters per event with hits in different thicknesses.
# num_reco_cluster_eta: Number of reco clusters vs eta

hgcalLayerClustersPlotter.append("Miscellaneous", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _num_reco_cluster_eta,
            _energyclustered,
            _mixedhitsclusters,
            _longdepthbarycentre,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Miscellaneous"
            ))

# [H] SelectedCaloParticles plots
hgcalLayerClustersPlotter.append("SelectedCaloParticles_Photons", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/SelectedCaloParticles/22",
            ], PlotFolder(
            _SelectedCaloParticles,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="SelectedCaloParticles_Photons"
            ))

# [I] Score of CaloParticles wrt Layer Clusters
# z-
hgcalLayerClustersPlotter.append("ScoreCaloParticlesToLayerClusters_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _score_caloparticle_to_layerclusters_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="ScoreCaloParticlesToLayerClusters_zminus"))

# z+
hgcalLayerClustersPlotter.append("ScoreCaloParticlesToLayerClusters_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _score_caloparticle_to_layerclusters_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="ScoreCaloParticlesToLayerClusters_zplus"))

# [J] Score of LayerClusters wrt CaloParticles
# z-
hgcalLayerClustersPlotter.append("ScoreLayerClustersToCaloParticles_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _score_layercluster_to_caloparticles_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="ScoreLayerClustersToCaloParticles_zminus"))

# z+
hgcalLayerClustersPlotter.append("ScoreLayerClustersToCaloParticles_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _score_layercluster_to_caloparticles_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="ScoreLayerClustersToCaloParticles_zplus"))

# [K] Shared Energy between CaloParticle and LayerClusters
# z-
hgcalLayerClustersPlotter.append("SharedEnergyC2L_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _sharedEnergy_caloparticle_to_layercluster_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="SharedEnergyCaloParticleToLayerCluster_zminus"))

# z+
hgcalLayerClustersPlotter.append("SharedEnergyC2L_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _sharedEnergy_caloparticle_to_layercluster_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="SharedEnergyCaloParticleToLayerCluster_zplus"))

# [K2] Shared Energy between LayerClusters and CaloParticle
# z-
hgcalLayerClustersPlotter.append("SharedEnergyL2C_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _sharedEnergy_layercluster_to_caloparticle_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="SharedEnergyLayerClusterToCaloParticle_zminus"))

# z+
hgcalLayerClustersPlotter.append("SharedEnergyL2C_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _sharedEnergy_layercluster_to_caloparticle_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="SharedEnergyLayerClusterToCaloParticle_zplus"))

# [L] Cell Association per Layer
# z-
hgcalLayerClustersPlotter.append("CellAssociation_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _cell_association_table_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="CellAssociation_zminus"))

# z+
hgcalLayerClustersPlotter.append("CellAssociation_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _cell_association_table_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="CellAssociation_zplus"))

# [M] Efficiency Plots
# z-
hgcalLayerClustersPlotter.append("Efficiencies_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _efficiencies_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Efficiencies_zminus"))

# z+
hgcalLayerClustersPlotter.append("Efficiencies_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _efficiencies_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Efficiencies_zplus"))

# [L] Duplicate Plots
# z-
hgcalLayerClustersPlotter.append("Duplicates_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _duplicates_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Duplicates_zminus"))

# z+
hgcalLayerClustersPlotter.append("Duplicates_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _duplicates_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Duplicates_zplus"))

# [M] Fake Rate Plots
# z-
hgcalLayerClustersPlotter.append("FakeRate_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _fakes_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Fakes_zminus"))

# z+
hgcalLayerClustersPlotter.append("FakeRate_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _fakes_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Fakes_zplus"))

# [N] Merge Rate Plots
# z-
hgcalLayerClustersPlotter.append("MergeRate_zminus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _merges_zminus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Merges_zminus"))

# z+
hgcalLayerClustersPlotter.append("MergeRate_zplus", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
            ], PlotFolder(
            _merges_zplus,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Merges_zplus"))

# [O] Energy vs Score 2D plots CP to LC
# z-
for i,item in enumerate(_energyscore_cp2lc_zminus, start=1):
  hgcalLayerClustersPlotter.append("Energy_vs_Score_CP2LC_zminus", [
              "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
              ], PlotFolder(
              item,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing, page="Energy_vs_Score_CP2LC_zminus"))

# z+
for i,item in enumerate(_energyscore_cp2lc_zplus, start=1):
  hgcalLayerClustersPlotter.append("Energy_vs_Score_CP2LC_zplus", [
              "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
              ], PlotFolder(
              item,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing, page="Energy_vs_Score_CP2LC_zplus"))

# [P] Energy vs Score 2D plots LC to CP
# z-
for i,item in enumerate(_energyscore_lc2cp_zminus, start=1):
  hgcalLayerClustersPlotter.append("Energy_vs_Score_LC2CP_zminus", [
              "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
              ], PlotFolder(
              item,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing, page="Energy_vs_Score_LC2CP_zminus"))

# z+
for i,item in enumerate(_energyscore_lc2cp_zplus, start=1):
  hgcalLayerClustersPlotter.append("Energy_vs_Score_LC2CP_zplus", [
              "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
              ], PlotFolder(
              item,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing, page="Energy_vs_Score_LC2CP_zplus"))

#=================================================================================================
hgcalMultiClustersPlotter = Plotter()
# [A] Score of CaloParticles wrt Multi Clusters
hgcalMultiClustersPlotter.append("ScoreCaloParticlesToMultiClusters", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
            ], PlotFolder(
            _score_caloparticle_to_multiclusters,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="ScoreCaloParticlesToMultiClusters"))

# [B] Score of MultiClusters wrt CaloParticles
hgcalMultiClustersPlotter.append("ScoreMultiClustersToCaloParticles", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _score_multicluster_to_caloparticles,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="ScoreMultiClustersToCaloParticles"))

# [C] Shared Energy between CaloParticle and MultiClusters
hgcalMultiClustersPlotter.append("SharedEnergy_CP2MCL", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _sharedEnergy_caloparticle_to_multicluster,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="SharedEnergyCaloParticleToMultiCluster"))

# [C2] Shared Energy between MultiClusters and CaloParticle
hgcalMultiClustersPlotter.append("SharedEnergy_MCL2CP", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _sharedEnergy_multicluster_to_caloparticle,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="SharedEnergyMultiClusterToCaloParticle"))

# [E] Efficiency Plots
hgcalMultiClustersPlotter.append("Efficiencies", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _efficiencies,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Efficiencies"))

# [F] Duplicate Plots
hgcalMultiClustersPlotter.append("Duplicates", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _duplicates,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Duplicates"))

# [G] Fake Rate Plots
hgcalMultiClustersPlotter.append("FakeRate", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _fakes,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Fakes"))

# [H] Merge Rate Plots
hgcalMultiClustersPlotter.append("MergeRate", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _merges,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Merges"))

# [I] Energy vs Score 2D plots CP to MCL and MCL to CP
hgcalMultiClustersPlotter.append("Energy_vs_Score_CP2MCL_MCL2CP", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            #_energyscore_cp2mcl_mcl2cp,
            _energyscore_cp2mcl,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Energy_vs_Score_CP2MCL"))

# [J] Energy vs Score 2D plots MCL to CP
hgcalMultiClustersPlotter.append("Energy_vs_Score_MCL2CP", [
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
            "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
             ], PlotFolder(
            _energyscore_mcl2cp,
            loopSubFolders=False,
            purpose=PlotPurpose.Timing, page="Energy_vs_Score_MCL2CP"))

#[K] Number of multiclusters per event.
hgcalMultiClustersPlotter.append("NumberofMultiClusters", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
        ], PlotFolder(
        _totmulticlusternum,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="NumberofMultiClusters"
        ))

#[L] total number of layer clusters in multicluster per event and per layer
hgcalMultiClustersPlotter.append("NumberofLayerClustersinMultiClusterPerEventAndPerLayer", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
        ], PlotFolder(
        _clusternum_in_multicluster,
        _clusternum_in_multicluster_vs_layer,
        _clusternum_in_multicluster_perlayer_zminus_EE,
        _clusternum_in_multicluster_perlayer_zminus_FH,
        _clusternum_in_multicluster_perlayer_zminus_BH,
        _clusternum_in_multicluster_perlayer_zplus_EE,
        _clusternum_in_multicluster_perlayer_zplus_FH,
        _clusternum_in_multicluster_perlayer_zplus_BH,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="NumberofLayerClustersinMultiClusterPerEventAndPerLayer"
        ))

#[M] For each multicluster: pt, eta, phi, energy, x, y, z.
hgcalMultiClustersPlotter.append("MultiClustersPtEtaPhiEneXYZ", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
        ], PlotFolder(
        _multicluster_pt,
        _multicluster_eta,
        _multicluster_phi,
        _multicluster_energy,
        _multicluster_x,
        _multicluster_y,
        _multicluster_z,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="MultiClustersPtEtaPhiEneXYZ"
        ))

#[N] Multicluster first, last, total number of layers
hgcalMultiClustersPlotter.append("NumberofMultiClusters_First_Last_NLayers", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
        ], PlotFolder(
        _multicluster_firstlayer,
        _multicluster_lastlayer,
        _multicluster_layersnum,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="NumberofMultiClusters_First_Last_NLayers"
        ))

#[O] Multiplicity of layer clusters in multicluster
hgcalMultiClustersPlotter.append("Multiplicity", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
       ], PlotFolder(
        _multiplicityOfLCinMCL,
        _multiplicityOfLCinMCL_vs_layerclusterenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="Multiplicity",
        numberOfEventsHistogram=_multiplicity_numberOfEventsHistogram
        ))

#We append here two PlotFolder because we want the text to be in percent
#and the number of events are different in zplus and zminus
hgcalMultiClustersPlotter.append("Multiplicity", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
        ], PlotFolder(
        _multiplicityOfLCinMCL_vs_layercluster_zminus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="Multiplicity",
        numberOfEventsHistogram=_multiplicity_zminus_numberOfEventsHistogram
        ))

hgcalMultiClustersPlotter.append("Multiplicity", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersMIP_MIPMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersTrk_TrkMultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersEM_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/multiClustersFromTrackstersHAD_MultiClustersFromTracksterByCA",
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalMultiClusters",
        ], PlotFolder(
        _multiplicityOfLCinMCL_vs_layercluster_zplus,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="Multiplicity",
        numberOfEventsHistogram=_multiplicity_zplus_numberOfEventsHistogram
        ))

#=================================================================================================
# hitValidation
hgcalHitPlotter = Plotter()

hgcalHitPlotter.append("SimHits_Validation", [
                                              "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HitValidation",
                                              ], PlotFolder(
                                                            _HitValidation,
                                                            loopSubFolders=False,
                                                            purpose=PlotPurpose.Timing, page="SimHits_Validation"
                                                            ))

hgcalHitPlotter.append("SimHits_Occupancy_zplus", [
                                                   "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalEESensitive"
                                                   ], PlotFolder(
                                                                 _SimHits_Occupancy_EE_zplus,
                                                                 loopSubFolders=False,
                                                                 purpose=PlotPurpose.Timing, page="SimHits_Occupancy_zplus"
                                                                 ))

hgcalHitPlotter.append("SimHits_Occupancy_zplus", [
                                                   "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHESiliconSensitive"
                                                   ], PlotFolder(
                                                                 _SimHits_Occupancy_HE_Silicon_zplus,
                                                                 loopSubFolders=False,
                                                                 purpose=PlotPurpose.Timing, page="SimHits_Occupancy_zplus"
                                                                 ))

hgcalHitPlotter.append("SimHits_Occupancy_zplus", [
                                                   "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHEScintillatorSensitive"
                                                   ], PlotFolder(
                                                                 _SimHits_Occupancy_HE_Scintillator_zplus,
                                                                 loopSubFolders=False,
                                                                 purpose=PlotPurpose.Timing, page="SimHits_Occupancy_zplus"
                                                                 ))

hgcalHitPlotter.append("SimHits_Occupancy_zminus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalEESensitive"
                                                    ], PlotFolder(
                                                                  _SimHits_Occupancy_EE_zminus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="SimHits_Occupancy_zminus"
                                                                  ))

hgcalHitPlotter.append("SimHits_Occupancy_zminus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHESiliconSensitive"
                                                    ], PlotFolder(
                                                                  _SimHits_Occupancy_HE_Silicon_zminus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="SimHits_Occupancy_zminus"
                                                                  ))

hgcalHitPlotter.append("SimHits_Occupancy_zminus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHEScintillatorSensitive"
                                                    ], PlotFolder(
                                                                  _SimHits_Occupancy_HE_Scintillator_zminus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="SimHits_Occupancy_zminus"
                                                                  ))

for i,item in enumerate(_SimHits_EtaPhi_EE_zplus, start=1):
    hgcalHitPlotter.append("SimHits_EtaPhi_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalEESensitive"
                                                    ], PlotFolder(
                                                                  item,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="SimHits_EtaPhi_zplus"
                                                                  ))

for i,item in enumerate(_SimHits_EtaPhi_HE_Silicon_zplus, start=1):
    hgcalHitPlotter.append("SimHits_EtaPhi_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHESiliconSensitive"
                                                    ], PlotFolder(
                                                                  item,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="SimHits_EtaPhi_zplus"
                                                                  ))

for i,item in enumerate(_SimHits_EtaPhi_HE_Scintillator_zplus, start=1):
    hgcalHitPlotter.append("SimHits_EtaPhi_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHEScintillatorSensitive",
                                                    ], PlotFolder(
                                                                  item,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="SimHits_EtaPhi_zplus"))

for i,item in enumerate(_SimHits_EtaPhi_EE_zminus, start=1):
    hgcalHitPlotter.append("SimHits_EtaPhi_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalEESensitive"
                                                     ], PlotFolder(
                                                                   item,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="SimHits_EtaPhi_zminus"
                                                                   ))

for i,item in enumerate(_SimHits_EtaPhi_HE_Silicon_zminus, start=1):
    hgcalHitPlotter.append("SimHits_EtaPhi_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHESiliconSensitive"
                                                     ], PlotFolder(
                                                                   item,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="SimHits_EtaPhi_zminus"
                                                                   ))

for i,item in enumerate(_SimHits_EtaPhi_HE_Scintillator_zminus, start=1):
    hgcalHitPlotter.append("SimHits_EtaPhi_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHEScintillatorSensitive"
                                                     ], PlotFolder(
                                                                   item,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="SimHits_EtaPhi_zminus"
                                                                   ))

hgcalHitPlotter.append("SimHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalEESensitive"
                                          ], PlotFolder(
                                                        _SimHits_Energy_EE_0,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="SimHits_Energy"
                                                        ))

hgcalHitPlotter.append("SimHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHESiliconSensitive"
                                          ], PlotFolder(
                                                        _SimHits_Energy_HE_Silicon_0,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="SimHits_Energy"
                                                        ))

hgcalHitPlotter.append("SimHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHEScintillatorSensitive"
                                          ], PlotFolder(
                                                        _SimHits_Energy_HE_Scintillator_0,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="SimHits_Energy"
                                                        ))

hgcalHitPlotter.append("SimHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalEESensitive"
                                          ], PlotFolder(
                                                        _SimHits_Energy_EE_1,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="SimHits_Energy"
                                                        ))

hgcalHitPlotter.append("SimHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHESiliconSensitive"
                                          ], PlotFolder(
                                                        _SimHits_Energy_HE_Silicon_1,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="SimHits_Energy"
                                                        ))

hgcalHitPlotter.append("SimHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalSimHitsV/HGCalHEScintillatorSensitive"
                                          ], PlotFolder(
                                                        _SimHits_Energy_HE_Scintillator_1,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="SimHits_Energy"
                                                        ))

hgcalHitPlotter.append("RecHits_Occupancy_zplus", [
                                                   "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalEESensitive"
                                                   ], PlotFolder(
                                                                 _RecHits_Occupancy_EE_zplus,
                                                                 loopSubFolders=False,
                                                                 purpose=PlotPurpose.Timing, page="RecHits_Occupancy_zplus"
                                                                 ))

hgcalHitPlotter.append("RecHits_Occupancy_zplus", [
                                                   "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHESiliconSensitive"
                                                   ], PlotFolder(
                                                                 _RecHits_Occupancy_HE_Silicon_zplus,
                                                                 loopSubFolders=False,
                                                                 purpose=PlotPurpose.Timing, page="RecHits_Occupancy_zplus"
                                                                 ))

hgcalHitPlotter.append("RecHits_Occupancy_zplus", [
                                                   "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHEScintillatorSensitive"
                                                   ], PlotFolder(
                                                                 _RecHits_Occupancy_HE_Scintillator_zplus,
                                                                 loopSubFolders=False,
                                                                 purpose=PlotPurpose.Timing, page="RecHits_Occupancy_zplus"
                                                                 ))

hgcalHitPlotter.append("RecHits_Occupancy_zminus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalEESensitive"
                                                    ], PlotFolder(
                                                                  _RecHits_Occupancy_EE_zminus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="RecHits_Occupancy_zminus"
                                                                  ))

hgcalHitPlotter.append("RecHits_Occupancy_zminus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHESiliconSensitive"
                                                    ], PlotFolder(
                                                                  _RecHits_Occupancy_HE_Silicon_zminus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="RecHits_Occupancy_zminus"
                                                                  ))

hgcalHitPlotter.append("RecHits_Occupancy_zminus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHEScintillatorSensitive"
                                                    ], PlotFolder(
                                                                  _RecHits_Occupancy_HE_Scintillator_zminus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="RecHits_Occupancy_zminus"
                                                                  ))

hgcalHitPlotter.append("RecHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalEESensitive"
                                          ], PlotFolder(
                                                        _RecHits_Energy_EE,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="RecHits_Energy"
                                                        ))

hgcalHitPlotter.append("RecHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHESiliconSensitive"
                                          ], PlotFolder(
                                                        _RecHits_Energy_HE_Silicon,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="RecHits_Energy"
                                                        ))

hgcalHitPlotter.append("RecHits_Energy", [
                                          "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHEScintillatorSensitive"
                                          ], PlotFolder(
                                                        _RecHits_Energy_HE_Scintillator,
                                                        loopSubFolders=False,
                                                        purpose=PlotPurpose.Timing, page="RecHits_Energy"
                                                        ))

for i,item in enumerate(_RecHits_EtaPhi_EE_zplus, start=1):
    hgcalHitPlotter.append("RecHits_EtaPhi_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalEESensitive"
                                                    ], PlotFolder(
                                                                  item,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="RecHits_EtaPhi_zplus"
                                                                  ))

for i,item in enumerate(_RecHits_EtaPhi_HE_Silicon_zplus, start=1):
    hgcalHitPlotter.append("RecHits_EtaPhi_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHESiliconSensitive"
                                                    ], PlotFolder(
                                                                  item,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="RecHits_EtaPhi_zplus"
                                                                  ))

for i,item in enumerate(_RecHits_EtaPhi_HE_Scintillator_zplus, start=1):
    hgcalHitPlotter.append("RecHits_EtaPhi_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHEScintillatorSensitive",
                                                    ], PlotFolder(
                                                                  item,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="RecHits_EtaPhi_zplus"))

for i,item in enumerate(_RecHits_EtaPhi_EE_zminus, start=1):
    hgcalHitPlotter.append("RecHits_EtaPhi_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalEESensitive"
                                                     ], PlotFolder(
                                                                   item,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="RecHits_EtaPhi_zminus"
                                                                   ))

for i,item in enumerate(_RecHits_EtaPhi_HE_Silicon_zminus, start=1):
    hgcalHitPlotter.append("RecHits_EtaPhi_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHESiliconSensitive"
                                                     ], PlotFolder(
                                                                   item,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="RecHits_EtaPhi_zminus"
                                                                   ))

for i,item in enumerate(_RecHits_EtaPhi_HE_Scintillator_zminus, start=1):
    hgcalHitPlotter.append("RecHits_EtaPhi_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalRecHitsV/HGCalHEScintillatorSensitive"
                                                     ], PlotFolder(
                                                                   item,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="RecHits_EtaPhi_zminus"
                                                                   ))

hgcalHitPlotter.append("DigiHits_ADC", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalEESensitive"
                                        ], PlotFolder(
                                                      _DigiHits_ADC_EE,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_ADC"
                                                      ))

hgcalHitPlotter.append("DigiHits_ADC", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHESiliconSensitive"
                                        ], PlotFolder(
                                                      _DigiHits_ADC_HE_Silicon,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_ADC"
                                                      ))

hgcalHitPlotter.append("DigiHits_ADC", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHEScintillatorSensitive"
                                        ], PlotFolder(
                                                      _DigiHits_ADC_HE_Scintillator,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_ADC"
                                                      ))

hgcalHitPlotter.append("DigiHits_Occupancy_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalEESensitive"
                                                    ], PlotFolder(
                                                                  _DigiHits_Occupancy_EE_zplus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_zplus"
                                                                  ))

hgcalHitPlotter.append("DigiHits_Occupancy_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHESiliconSensitive"
                                                    ], PlotFolder(
                                                                  _DigiHits_Occupancy_HE_Silicon_zplus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_zplus"
                                                                  ))

hgcalHitPlotter.append("DigiHits_Occupancy_zplus", [
                                                    "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHEScintillatorSensitive"
                                                    ], PlotFolder(
                                                                  _DigiHits_Occupancy_HE_Scintillator_zplus,
                                                                  loopSubFolders=False,
                                                                  purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_zplus"
                                                                  ))

hgcalHitPlotter.append("DigiHits_Occupancy_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalEESensitive"
                                                     ], PlotFolder(
                                                                   _DigiHits_Occupancy_EE_zminus,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_zminus"
                                                                   ))

hgcalHitPlotter.append("DigiHits_Occupancy_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHESiliconSensitive"
                                                     ], PlotFolder(
                                                                   _DigiHits_Occupancy_HE_Silicon_zminus,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_zminus"
                                                                   ))

hgcalHitPlotter.append("DigiHits_Occupancy_zminus", [
                                                     "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHEScintillatorSensitive"
                                                     ], PlotFolder(
                                                                   _DigiHits_Occupancy_HE_Scintillator_zminus,
                                                                   loopSubFolders=False,
                                                                   purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_zminus"
                                                                   ))

for i,item in enumerate(_DigiHits_Occupancy_XY_EE_zplus, start=1):
    hgcalHitPlotter.append("DigiHits_Occupancy_XY_zplus", [
                                                           "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalEESensitive"
                                                           ], PlotFolder(
                                                                         item,
                                                                         loopSubFolders=False,
                                                                         purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_XY_zplus"
                                                                         ))

for i,item in enumerate(_DigiHits_Occupancy_XY_HE_Silicon_zplus, start=1):
    hgcalHitPlotter.append("DigiHits_Occupancy_XY_zplus", [
                                                           "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHESiliconSensitive"
                                                           ], PlotFolder(
                                                                         item,
                                                                         loopSubFolders=False,
                                                                         purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_XY_zplus"
                                                                         ))

for i,item in enumerate(_DigiHits_Occupancy_XY_HE_Scintillator_zplus, start=1):
    hgcalHitPlotter.append("DigiHits_Occupancy_XY_zplus", [
                                                           "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHEScintillatorSensitive",
                                                           ], PlotFolder(
                                                                         item,
                                                                         loopSubFolders=False,
                                                                         purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_XY_zplus"))

for i,item in enumerate(_DigiHits_Occupancy_XY_EE_zminus, start=1):
    hgcalHitPlotter.append("DigiHits_Occupancy_XY_zminus", [
                                                            "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalEESensitive"
                                                            ], PlotFolder(
                                                                          item,
                                                                          loopSubFolders=False,
                                                                          purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_XY_zminus"
                                                                          ))

for i,item in enumerate(_DigiHits_Occupancy_XY_HE_Silicon_zminus, start=1):
    hgcalHitPlotter.append("DigiHits_Occupancy_XY_zminus", [
                                                            "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHESiliconSensitive"
                                                            ], PlotFolder(
                                                                          item,
                                                                          loopSubFolders=False,
                                                                          purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_XY_zminus"
                                                                          ))

for i,item in enumerate(_DigiHits_Occupancy_XY_HE_Scintillator_zminus, start=1):
    hgcalHitPlotter.append("DigiHits_Occupancy_XY_zminus", [
                                                            "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHEScintillatorSensitive",
                                                            ], PlotFolder(
                                                                          item,
                                                                          loopSubFolders=False,
                                                                          purpose=PlotPurpose.Timing, page="DigiHits_Occupancy_XY_zminus"))

hgcalHitPlotter.append("DigiHits_TOA", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalEESensitive"
                                        ], PlotFolder(
                                                      _DigiHits_TOA_EE,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_TOA"
                                                      ))

hgcalHitPlotter.append("DigiHits_TOA", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHESiliconSensitive"
                                        ], PlotFolder(
                                                      _DigiHits_TOA_HE_Silicon,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_TOA"
                                                      ))

hgcalHitPlotter.append("DigiHits_TOA", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHEScintillatorSensitive"
                                        ], PlotFolder(
                                                      _DigiHits_TOA_HE_Scintillator,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_TOA"
                                                      ))

hgcalHitPlotter.append("DigiHits_TOT", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalEESensitive"
                                        ], PlotFolder(
                                                      _DigiHits_TOT_EE,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_TOT"
                                                      ))

hgcalHitPlotter.append("DigiHits_TOT", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHESiliconSensitive"
                                        ], PlotFolder(
                                                      _DigiHits_TOT_HE_Silicon,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_TOT"
                                                      ))

hgcalHitPlotter.append("DigiHits_TOT", [
                                        "DQMData/Run 1/HGCAL/Run summary/HGCalDigisV/HGCalHEScintillatorSensitive"
                                        ], PlotFolder(
                                                      _DigiHits_TOT_HE_Scintillator,
                                                      loopSubFolders=False,
                                                      purpose=PlotPurpose.Timing, page="DigiHits_TOT"
                                                      ))
#=================================================================================================
# hitCalibration
hgcalHitCalibPlotter = Plotter()

hgcalHitCalibPlotter.append("Layer_Occupancy", [
                                                "DQMData/Run 1/HGCalHitCalibration/Run summary",
                                                ], PlotFolder(
                                                              _LayerOccupancy,
                                                              loopSubFolders=False,
                                                              purpose=PlotPurpose.Timing, page="Layer_Occupancy"
                                                              ))
hgcalHitCalibPlotter.append("ReconstructableEnergyOverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _ReconstructableEnergyOverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="ReconstructableEnergyOverCPenergy"
        ))

hgcalHitCalibPlotter.append("ParticleFlowClusterHGCalFromMultiCl_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _ParticleFlowClusterHGCalFromMultiCl_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="ParticleFlowClusterHGCalFromMultiCl_Closest_EoverCPenergy"
        ))

hgcalHitCalibPlotter.append("PhotonsFromMultiCl_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _PhotonsFromMultiCl_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="PhotonsFromMultiCl_Closest_EoverCPenergy"
        ))

hgcalHitCalibPlotter.append("EcalDrivenGsfElectronsFromMultiCl_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _EcalDrivenGsfElectronsFromMultiCl_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="EcalDrivenGsfElectronsFromMultiCl_Closest_EoverCPenergy"
        ))
