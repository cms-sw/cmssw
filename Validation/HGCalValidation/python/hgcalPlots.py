from __future__ import print_function
import os
import sys
import copy
import collections

import six
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
_totclusternum_layer_EE_zminus = PlotGroup("totclusternum_layer_EE", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_totclusternum_layer_FH_zminus = PlotGroup("totclusternum_layer_FH", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_totclusternum_layer_BH_zminus = PlotGroup("totclusternum_layer_BH", [
  Plot("totclusternum_layer_{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

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


_bin_count = 50
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
_score_caloparticle_to_multiclusters = PlotGroup("ScoreCaloParticlesToMultiClusters", [
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
_score_multicluster_to_caloparticles = PlotGroup("ScoreMultiClustersToCaloParticles", [
        Plot("Score_multicl2caloparticle", **_common_score)
        ], ncols=1)

_common_shared= {"title": "Shared Energy CaloParticle To Multi Cluster ",
                 "stat": False,
                 "legend": True,
                 "xmin": 0,
                 "xmax": 1.0,
               }
_common_shared.update(_legend_common)
_shared_plots = [ Plot("SharedEnergy_caloparticle2multicl", **_common_shared) ]
_common_shared["xmin"] = -4.0
_common_shared["xmax"] = 4.0
_shared_plots.extend([Plot("SharedEnergy_caloparticle2multicl_vs_eta", xtitle="CaloParticle #eta", **_common_shared)])
_shared_plots.extend([Plot("SharedEnergy_caloparticle2multicl_vs_phi", xtitle="CaloParticle #phi", **_common_shared)])
_sharedEnergy_caloparticle_to_multicluster = PlotGroup("SharedEnergy_CaloParticleToMultiCluster", _shared_plots, ncols=3)

_common_shared= {"title": "Shared Energy Multi Cluster To CaloParticle ",
                 "stat": False,
                 "legend": True,
                 "xmin": 0,
                 "xmax": 1.0,
                }
_common_shared.update(_legend_common)
_shared_plots2 = [Plot("SharedEnergy_multicluster2caloparticle", **_common_shared)]
_common_shared["xmin"] = -4.0
_common_shared["xmax"] = 4.0
_shared_plots2.extend([Plot("SharedEnergy_multicl2caloparticle_vs_eta", xtitle="MultiCluster #eta", **_common_shared)])
_shared_plots2.extend([Plot("SharedEnergy_multicl2caloparticle_vs_phi", xtitle="MultiCluster #phi", **_common_shared)])
_sharedEnergy_multicluster_to_caloparticle = PlotGroup("SharedEnergy_MultiClusterToCaloParticle", _shared_plots2, ncols=3)


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

_common_eff = {"stat": False, "legend": False, "xbinlabelsize": 14, "xbinlabeloption": "d", "ymin": 0.0, "ymax": 1.1}
_effplots = [Plot("effic_eta", xtitle="", **_common_eff)]
_effplots.extend([Plot("effic_phi", xtitle="", **_common_eff)])
_effplots.extend([Plot("globalEfficiencies", xtitle="", **_common_eff)])
_efficiencies = PlotGroup("Efficiencies", _effplots, ncols=3)


_common_dup = {"stat": False, "legend": False, "xbinlabelsize": 14, "xbinlabeloption": "d", "ymin": 0.0, "ymax": 1.1}
_dupplots = [Plot("duplicate_eta", xtitle="", **_common_dup)]
_dupplots.extend([Plot("duplicate_phi", xtitle="", **_common_dup)])
_dupplots.extend([Plot("globalEfficiencies", xtitle="", **_common_dup)])
_duplicates = PlotGroup("Duplicates", _dupplots, ncols=3)

_common_fake = {"stat": False, "legend": False, "xbinlabelsize": 14, "xbinlabeloption": "d", "ymin": 0.0, "ymax": 1.1}
_fakeplots = [Plot("fake_eta", xtitle="", **_common_fake)]
_fakeplots.extend([Plot("fake_phi", xtitle="", **_common_fake)])
_fakeplots.extend([Plot("globalEfficiencies", xtitle="", **_common_fake)])
_fakes = PlotGroup("FakeRate", _fakeplots, ncols=3)

_common_merge = {"stat": False, "legend": False, "xbinlabelsize": 14, "xbinlabeloption": "d", "ymin": 0.0, "ymax": 1.1}
_mergeplots = [Plot("merge_eta", xtitle="", **_common_merge)]
_mergeplots.extend([Plot("merge_phi", xtitle="", **_common_merge)])
_mergeplots.extend([Plot("globalEfficiencies", xtitle="", **_common_merge)])
_merges = PlotGroup("MergeRate", _mergeplots, ncols=3)

_common_energy_score = dict(removeEmptyBins=True, xbinlabelsize=10, xbinlabeloption="d")
_common_energy_score["ymax"] = 1.
_common_energy_score["xmax"] = 1.0
_energyscore_cp2mcl = PlotOnSideGroup("Energy_vs_Score_CaloParticlesToMultiClusters", Plot("Energy_vs_Score_caloparticle2multi", drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1)
_common_energy_score["ymax"] = 1.
_common_energy_score["xmax"] = 1.0
_energyscore_mcl2cp = PlotOnSideGroup("Energy_vs_Score_MultiClustersToCaloParticles", Plot("Energy_vs_Score_multi2caloparticle", drawStyle="COLZ", adjustMarginRight=0.1, **_common_energy_score), ncols=1)

#Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

_totmulticlusternum = PlotGroup("TotalNumberofMultiClusters", [
  Plot("totmulticlusternum", xtitle="", **_common)
],ncols=1)

_multicluster_layernum_plots = [Plot("multicluster_firstlayer", xtitle="MultiCluster First Layer", **_common)]
_multicluster_layernum_plots.extend([Plot("multicluster_lastlayer", xtitle="MultiCluster Last Layer", **_common)])
_multicluster_layernum_plots.extend([Plot("multicluster_layersnum", xtitle="MultiCluster Number of Layers", **_common)])
_multicluster_layernum = PlotGroup("LayerNumbersOfMultiCluster", _multicluster_layernum_plots, ncols=3)

_common["xmax"] = 50
_clusternum_in_multicluster = PlotGroup("NumberofLayerClustersinMultiCluster",[
  Plot("clusternum_in_multicluster", xtitle="", **_common)
],ncols=1)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}
_common = {"stat": True, "drawStyle": "pcolz", "staty": 0.65}

_clusternum_in_multicluster_vs_layer = PlotGroup("NumberofLayerClustersinMultiClusterPerLayer",[
  Plot("clusternum_in_multicluster_vs_layer", xtitle="Layer number", ytitle = "<2d Layer Clusters in Multicluster>",  **_common)
],ncols=1)

_common["scale"] = 100.
#, ztitle = "% of clusters" normalizeToUnitArea=True
_multiplicity_numberOfEventsHistogram = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/ticlMultiClustersFromTrackstersEM/multiplicity_numberOfEventsHistogram"
_multiplicity_zminus_numberOfEventsHistogram = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/ticlMultiClustersFromTrackstersEM/multiplicity_zminus_numberOfEventsHistogram"
_multiplicity_zplus_numberOfEventsHistogram = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/ticlMultiClustersFromTrackstersEM/multiplicity_zplus_numberOfEventsHistogram"

_multiplicityOfLCinMCL_plots = [Plot("multiplicityOfLCinMCL", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Cluster size (n_{hit})", 
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)]
_multiplicityOfLCinMCL_plots.extend([Plot("multiplicityOfLCinMCL_vs_layerclusterenergy", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Cluster Energy (GeV)", 
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)]) 
_multiplicityOfLCinMCL_plots.extend([Plot("multiplicityOfLCinMCL_vs_layercluster_zplus", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Layer Number", 
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)])
_multiplicityOfLCinMCL_plots.extend([Plot("multiplicityOfLCinMCL_vs_layercluster_zminus", xtitle="Layer Cluster multiplicity in Multiclusters", ytitle = "Layer Number", 
                                drawCommand = "colz text45", normalizeToNumberOfEvents = True, **_common)])
_multiplicityOfLCinMCL = PlotGroup("MultiplcityofLCinMLC", _multiplicityOfLCinMCL_plots, ncols=2)

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65}
#--------------------------------------------------------------------------------------------
# z-
#--------------------------------------------------------------------------------------------
_clusternum_in_multicluster_perlayer_zminus_EE = PlotGroup("NumberofLayerClustersinMultiClusterPerLayer_zminus_EE", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm)
], ncols=7)

_clusternum_in_multicluster_perlayer_zminus_FH = PlotGroup("NumberofLayerClustersinMultiClusterPerLayer_zminus_FH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm)
], ncols=7)

_clusternum_in_multicluster_perlayer_zminus_BH = PlotGroup("NumberofLayerClustersinMultiClusterPerLayer_zminus_BH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm)
], ncols=7)

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_clusternum_in_multicluster_perlayer_zplus_EE = PlotGroup("NumberofLayerClustersinMultiClusterPerLayer_zplus_EE", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp)
], ncols=7)

_clusternum_in_multicluster_perlayer_zplus_FH = PlotGroup("NumberofLayerClustersinMultiClusterPerLayer_zplus_FH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp)
], ncols=7)

_clusternum_in_multicluster_perlayer_zplus_BH = PlotGroup("NumberofLayerClustersinMultiClusterPerLayer_zplus_BH", [
  Plot("clusternum_in_multicluster_perlayer{:02d}".format(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp)
], ncols=7)

#Coming back to the usual box definition
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#Some multiclusters quantities
_multicluster_eppe_plots = [Plot("multicluster_eta", xtitle="MultiCluster #eta", **_common)]
_multicluster_eppe_plots.extend([Plot("multicluster_phi", xtitle="MultiCluster #phi", **_common)])
_multicluster_eppe_plots.extend([Plot("multicluster_pt", xtitle="MultiCluster p_{T}", **_common)])
_multicluster_eppe_plots.extend([Plot("multicluster_energy", xtitle="MultiCluster Energy", **_common)])
_multicluster_eppe = PlotGroup("EtaPhiPtEnergy", _multicluster_eppe_plots, ncols=2)

_multicluster_xyz_plots = [Plot("multicluster_x", xtitle="MultiCluster x", **_common)]
_multicluster_xyz_plots.extend([Plot("multicluster_y", xtitle="MultiCluster y", **_common)])
_multicluster_xyz_plots.extend([Plot("multicluster_z", xtitle="MultiCluster z", **_common)])
_multicluster_xyz = PlotGroup("XYZ", _multicluster_xyz_plots, ncols=3)

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

_ParticleFlowClusterHGCalFromMultiCl_Closest_EoverCPenergy = PlotGroup("ParticleFlowClusterHGCalFromMultiCl", [
  Plot("hgcal_EoP_CPene_100_calib_fraction", title="hgcal_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_200_calib_fraction", title="hgcal_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_300_calib_fraction", title="hgcal_EoP_CPene_300_calib_fraction", **_common),
  Plot("hgcal_EoP_CPene_scint_calib_fraction", title="hgcal_EoP_CPene_scint_calib_fraction", **_common),
])

_EcalDrivenGsfElectronsFromMultiCl_Closest_EoverCPenergy = PlotGroup("EcalDrivenGsfElectronsFromMultiCl", [
  Plot("hgcal_ele_EoP_CPene_100_calib_fraction", title="hgcal_ele_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_200_calib_fraction", title="hgcal_ele_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_300_calib_fraction", title="hgcal_ele_EoP_CPene_300_calib_fraction", **_common),
  Plot("hgcal_ele_EoP_CPene_scint_calib_fraction", title="hgcal_ele_EoP_CPene_scint_calib_fraction", **_common),
])

_PhotonsFromMultiCl_Closest_EoverCPenergy = PlotGroup("PhotonsFromMultiCl", [
  Plot("hgcal_photon_EoP_CPene_100_calib_fraction", title="hgcal_photon_EoP_CPene_100_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_200_calib_fraction", title="hgcal_photon_EoP_CPene_200_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_300_calib_fraction", title="hgcal_photon_EoP_CPene_300_calib_fraction", **_common),
  Plot("hgcal_photon_EoP_CPene_scint_calib_fraction", title="hgcal_photon_EoP_CPene_scint_calib_fraction", **_common),
])

#=================================================================================================
hgcalLayerClustersPlotter = Plotter()
layerClustersLabel = 'Layer Clusters'

lc_general = [
  # [A] calculated "energy density" for cells in a) 120um, b) 200um, c) 300um, d) scint
  # (one entry per rechit, in the appropriate histo)
  _cellsenedens_thick,
  # [B] number of layer clusters per event in a) 120um, b) 200um, c) 300um, d) scint
  # (one entry per event in each of the four histos)
  _totclusternum_thick,
  # [G] Miscellaneous plots:
  # longdepthbarycentre: The longitudinal depth barycentre. One entry per event.
  # mixedhitscluster: Number of clusters per event with hits in different thicknesses.
  # num_reco_cluster_eta: Number of reco clusters vs eta
  _num_reco_cluster_eta,
  _energyclustered,
  _mixedhitsclusters,
  _longdepthbarycentre,
  # [H] SelectedCaloParticles plots
  _SelectedCaloParticles,
]
lc_zminus = [
  # [C] number of layer clusters per layer (one entry per event in each histo)
  _totclusternum_layer_EE_zminus,
  _totclusternum_layer_FH_zminus,
  _totclusternum_layer_BH_zminus,
  # [D] For each layer cluster:
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
  # [E] For each layer cluster:
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
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zminus,
  # [F] Looking at the fraction of true energy that has been clustered; by layer and overall
  _energyclustered_perlayer_EE_zminus,
  _energyclustered_perlayer_FH_zminus,
  _energyclustered_perlayer_BH_zminus,
  # [I] Score of CaloParticles wrt Layer Clusters
  _score_caloparticle_to_layerclusters_zminus,
  # [J] Score of LayerClusters wrt CaloParticles
  _score_layercluster_to_caloparticles_zminus,
  # [K] Shared Energy between CaloParticle and LayerClusters
  _sharedEnergy_caloparticle_to_layercluster_zminus,
  # [K2] Shared Energy between LayerClusters and CaloParticle
  _sharedEnergy_layercluster_to_caloparticle_zminus,
  # [L] Cell Association per Layer
  _cell_association_table_zminus,
  # [M] Efficiency Plots
  _efficiencies_zminus,
  _efficiencies_zminus_eta,
  _efficiencies_zminus_phi,
  # [L] Duplicate Plots
  _duplicates_zminus,
  _duplicates_zminus_eta,
  _duplicates_zminus_phi,
  # [M] Fake Rate Plots
  _fakes_zminus,
  _fakes_zminus_eta,
  _fakes_zminus_phi,
  # [N] Merge Rate Plots
  _merges_zminus,
  _merges_zminus_eta,
  _merges_zminus_phi,
  # [O] Energy vs Score 2D plots CP to LC
  _energyscore_cp2lc_zminus,
  # [P] Energy vs Score 2D plots LC to CP
  _energyscore_lc2cp_zminus
]
lc_zplus = [
  # number of layer clusters per layer (one entry per event in each histo)
  _totclusternum_layer_EE_zplus,
  _totclusternum_layer_FH_zplus,
  _totclusternum_layer_BH_zplus,
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
  _distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zplus,
  # Looking at the fraction of true energy that has been clustered; by layer and overall
  _energyclustered_perlayer_EE_zplus,
  _energyclustered_perlayer_FH_zplus,
  _energyclustered_perlayer_BH_zplus,
  # Score of CaloParticles wrt Layer Clusters
  _score_caloparticle_to_layerclusters_zplus,
  # Score of LayerClusters wrt CaloParticles
  _score_layercluster_to_caloparticles_zplus,
  # Shared Energy between CaloParticle and LayerClusters
  _sharedEnergy_caloparticle_to_layercluster_zplus,
  # Shared Energy between LayerClusters and CaloParticle
  _sharedEnergy_layercluster_to_caloparticle_zplus,
  # Cell Association per Layer
  _cell_association_table_zplus,
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
  _energyscore_cp2lc_zplus,
  _energyscore_lc2cp_zplus
]

def append_hgcalLayerClustersPlots(collection = "hgcalLayerClusters", name_collection = layerClustersLabel):
  regions = ["General", "zminus", "zplus"]
  setPlots = [lc_general, lc_zminus, lc_zplus]
  for reg, setPlot in zip(regions, setPlots):
    print(_hgcalFolders(collection))
    hgcalLayerClustersPlotter.append(collection+"_"+reg, [
                _hgcalFolders(collection)
                ], PlotFolder(
                *setPlot,
                loopSubFolders=False,
                purpose=PlotPurpose.Timing, page=layerClustersLabel, section=reg))

#=================================================================================================
def _hgcalFolders(lastDirName="hgcalLayerClusters"):
    return "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/"+lastDirName

_multiclustersAllPlots = [
  _efficiencies,
  _duplicates,
  _fakes,
  _merges,
  _multicluster_eppe,
  _multicluster_xyz,
  _totmulticlusternum,
  _score_caloparticle_to_multiclusters,
  _score_multicluster_to_caloparticles,
  _sharedEnergy_caloparticle_to_multicluster,
  _sharedEnergy_multicluster_to_caloparticle,
  #_energyscore_cp2mcl_mcl2cp,
  _energyscore_cp2mcl,
  _energyscore_mcl2cp,
  _clusternum_in_multicluster,
  _clusternum_in_multicluster_vs_layer,
  _clusternum_in_multicluster_perlayer_zminus_EE,
  _clusternum_in_multicluster_perlayer_zminus_FH,
  _clusternum_in_multicluster_perlayer_zminus_BH,
  _clusternum_in_multicluster_perlayer_zplus_EE,
  _clusternum_in_multicluster_perlayer_zplus_FH,
  _clusternum_in_multicluster_perlayer_zplus_BH,
  _multicluster_layernum,
  _multiplicityOfLCinMCL,
]

hgcalMultiClustersPlotter = Plotter()
def append_hgcalMultiClustersPlots(collection = 'ticlMultiClustersFromTrackstersMerge', name_collection = "MultiClustersMerge"):
  # Appending all plots for MCs
  hgcalMultiClustersPlotter.append(collection, [
              _hgcalFolders(collection)
              ], PlotFolder(
              *_multiclustersAllPlots,
              loopSubFolders=False,
              purpose=PlotPurpose.Timing, page="MultiClusters", section=name_collection))

  #We append here two PlotFolder because we want the text to be in percent
  #and the number of events are different in zplus and zminus
  #hgcalMultiClustersPlotter.append("Multiplicity", [
  #            dqmfolder
  #            ], PlotFolder(
  #            _multiplicityOfLCinMCL_vs_layercluster_zminus,
  #            loopSubFolders=False,
  #            purpose=PlotPurpose.Timing, page=collection,
  #            numberOfEventsHistogram=_multiplicity_zminus_numberOfEventsHistogram
  #            ))
  #
  #hgcalMultiClustersPlotter.append("Multiplicity", [
  #            dqmfolder
  #            ], PlotFolder(
  #            _multiplicityOfLCinMCL_vs_layercluster_zplus,
  #            loopSubFolders=False,
  #            purpose=PlotPurpose.Timing, page=collection,
  #            numberOfEventsHistogram=_multiplicity_zplus_numberOfEventsHistogram
  #            ))

#=================================================================================================
_common_Calo = {"stat": False, "drawStyle": "hist", "staty": 0.65, "ymin": 0.0, "ylog": False}

hgcalCaloParticlesPlotter = Plotter()
def append_hgcalCaloParticlesPlots(files, collection = '-211', name_collection = "pion-"):

  list_2D_histos = ["Energy of Rec-matched Hits vs layer",
                    "Energy of Rec-matched Hits vs layer (1SC)",
                    "Rec-matched Hits Sum Energy vs layer"]

  dqmfolder = "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/SelectedCaloParticles/" + collection
  print(dqmfolder)
  templateFile = ROOT.TFile.Open(files[0]) # assuming all files have same structure
  keys = gDirectory.GetDirectory(dqmfolder,True).GetListOfKeys()
  key = keys[0]
  while key:
    obj = key.ReadObj()
    name = obj.GetName()
    fileName = TString(name)
    fileName.ReplaceAll(" ","_")
    pg= PlotGroup(fileName.Data(),[
                  Plot(name,
                       xtitle=obj.GetXaxis().GetTitle(), ytitle=obj.GetYaxis().GetTitle(),
                       drawCommand = "",
                       normalizeToNumberOfEvents = True, **_common_Calo)
                  ],
                  ncols=1)

    if name in list_2D_histos :
        pg= PlotOnSideGroup(fileName.Data(),
                      Plot(name,
                           xtitle=obj.GetXaxis().GetTitle(), ytitle=obj.GetYaxis().GetTitle(),
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

hgcalHitCalibPlotter.append("ParticleFlowClusterHGCalFromMultiCl_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _ParticleFlowClusterHGCalFromMultiCl_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
        ))

hgcalHitCalibPlotter.append("PhotonsFromMultiCl_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _PhotonsFromMultiCl_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
        ))

hgcalHitCalibPlotter.append("EcalDrivenGsfElectronsFromMultiCl_Closest_EoverCPenergy", [
        "DQMData/Run 1/HGCalHitCalibration/Run summary",
        ], PlotFolder(
        _EcalDrivenGsfElectronsFromMultiCl_Closest_EoverCPenergy,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page=hitCalibrationLabel, section=hitCalibrationLabel
        ))
