import os
import copy
import collections

import six
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from Validation.RecoTrack.plotting.plotting import Plot, PlotGroup, PlotFolder, Plotter
from Validation.RecoTrack.plotting.html import PlotPurpose
import Validation.RecoTrack.plotting.plotting as plotting
import Validation.RecoTrack.plotting.validation as validation
import Validation.RecoTrack.plotting.html as html

#To be able to spot any issues both in -z and +z a layer id was introduced 
#that spans from 0 to 103. The mapping is : 
#-z: 0->51
#+z: 52->103
lastLayerEEzm = 28  # last layer of EE -z
lastLayerFHzm = 40  # last layer of FH -z
maxlayerzm = 52 # last layer of BH -z
lastLayerEEzp = 80  # last layer of EE +z
lastLayerFHzp = 92  # last layer of FH +z
maxlayerzp = 104 # last layer of BH +z

_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

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

_totclusternum_thick = PlotGroup("totclusternum_thick", [
    Plot("totclusternum_thick_120", xtitle="", **_common),
    Plot("totclusternum_thick_200", xtitle="", **_common),
    Plot("totclusternum_thick_300", xtitle="", **_common),
    Plot("totclusternum_thick_-1", xtitle="", **_common),
    ])

#We will plot the density in logy scale. 
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65, "ylog": True }

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
        Plot("totclusternum_layer_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_totclusternum_layer_FH_zminus = PlotGroup("totclusternum_layer_FH_zminus", [ 
        Plot("totclusternum_layer_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_totclusternum_layer_BH_zminus = PlotGroup("totclusternum_layer_BH_zminus", [ 
        Plot("totclusternum_layer_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_EE_zminus = PlotGroup("energyclustered_perlayer_EE_zminus", [ 
        Plot("energyclustered_perlayer%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_FH_zminus = PlotGroup("energyclustered_perlayer_FH_zminus", [ 
        Plot("energyclustered_perlayer%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_BH_zminus = PlotGroup("energyclustered_perlayer_BH_zminus", [ 
        Plot("energyclustered_perlayer%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_cellsnum_perthick_perlayer_120_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_120_EE_zminus", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_120_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_120_FH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_120_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_120_BH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#200 um 
_cellsnum_perthick_perlayer_200_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_200_EE_zminus", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_200_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_200_FH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_200_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_200_BH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#300 um 
_cellsnum_perthick_perlayer_300_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_300_EE_zminus", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_300_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_300_FH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_300_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_300_BH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#scint um 
_cellsnum_perthick_perlayer_scint_EE_zminus = PlotGroup("cellsnum_perthick_perlayer_-1_EE_zminus", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_scint_FH_zminus = PlotGroup("cellsnum_perthick_perlayer_-1_FH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_scint_BH_zminus = PlotGroup("cellsnum_perthick_perlayer_-1_BH_zminus", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetomaxcell_perthickperlayer_120_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_120_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_120_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_120_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetomaxcell_perthickperlayer_200_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_200_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_200_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_200_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetomaxcell_perthickperlayer_300_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_300_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_300_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_300_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetomaxcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_-1_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_-1_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_-1_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancebetseedandmaxcell_perthickperlayer_120_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_EE_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_120_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_FH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_120_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_BH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancebetseedandmaxcell_perthickperlayer_200_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_EE_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_200_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_FH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_200_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_BH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancebetseedandmaxcell_perthickperlayer_300_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_EE_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_300_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_FH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_300_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_BH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancebetseedandmaxcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_EE_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_FH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_BH_zminus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_EE_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_FH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zminus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_BH_zminus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetoseedcell_perthickperlayer_120_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_120_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_120_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_120_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetoseedcell_perthickperlayer_200_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_200_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_200_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_200_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetoseedcell_perthickperlayer_300_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_300_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_300_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_300_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetoseedcell_perthickperlayer_scint_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_-1_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_scint_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_-1_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_scint_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_-1_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#=====================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#We need points for the weighted plots
_common = {"stat": True, "drawStyle": "EP", "staty": 0.65 }

#120 um 
_distancetomaxcell_perthickperlayer_eneweighted_120_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_120_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_120_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetomaxcell_perthickperlayer_eneweighted_200_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_200_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_200_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetomaxcell_perthickperlayer_eneweighted_300_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_300_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_300_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_EE_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_FH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zminus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_BH_zminus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )


#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetoseedcell_perthickperlayer_eneweighted_120_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_120_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_120_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetoseedcell_perthickperlayer_eneweighted_200_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_200_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_200_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetoseedcell_perthickperlayer_eneweighted_300_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_300_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_300_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_EE_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_FH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzm,lastLayerFHzm) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zminus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_BH_zminus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzm,maxlayerzm) 
        ],
                                    ncols=4
                                    )

#Coming back to the usual definition 
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#--------------------------------------------------------------------------------------------
# z+
#--------------------------------------------------------------------------------------------
_totclusternum_layer_EE_zplus = PlotGroup("totclusternum_layer_EE_zplus", [ 
        Plot("totclusternum_layer_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_totclusternum_layer_FH_zplus = PlotGroup("totclusternum_layer_FH_zplus", [ 
        Plot("totclusternum_layer_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_totclusternum_layer_BH_zplus = PlotGroup("totclusternum_layer_BH_zplus", [ 
        Plot("totclusternum_layer_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_EE_zplus = PlotGroup("energyclustered_perlayer_EE_zplus", [ 
        Plot("energyclustered_perlayer%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_FH_zplus = PlotGroup("energyclustered_perlayer_FH_zplus", [ 
        Plot("energyclustered_perlayer%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_BH_zplus = PlotGroup("energyclustered_perlayer_BH_zplus", [ 
        Plot("energyclustered_perlayer%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_cellsnum_perthick_perlayer_120_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_120_EE_zplus", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_120_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_120_FH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_120_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_120_BH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#200 um 
_cellsnum_perthick_perlayer_200_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_200_EE_zplus", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_200_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_200_FH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_200_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_200_BH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#300 um 
_cellsnum_perthick_perlayer_300_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_300_EE_zplus", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_300_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_300_FH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_300_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_300_BH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#scint um 
_cellsnum_perthick_perlayer_scint_EE_zplus = PlotGroup("cellsnum_perthick_perlayer_-1_EE_zplus", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_scint_FH_zplus = PlotGroup("cellsnum_perthick_perlayer_-1_FH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_scint_BH_zplus = PlotGroup("cellsnum_perthick_perlayer_-1_BH_zplus", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetomaxcell_perthickperlayer_120_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_120_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_120_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_120_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetomaxcell_perthickperlayer_200_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_200_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_200_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_200_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetomaxcell_perthickperlayer_300_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_300_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_300_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_300_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetomaxcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_-1_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_-1_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_-1_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancebetseedandmaxcell_perthickperlayer_120_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_EE_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_120_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_FH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_120_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_120_BH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancebetseedandmaxcell_perthickperlayer_200_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_EE_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_200_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_FH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_200_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_200_BH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancebetseedandmaxcell_perthickperlayer_300_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_EE_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_300_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_FH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_300_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_300_BH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancebetseedandmaxcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_EE_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_FH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancebetseedandmaxcell_perthickperlayer_-1_BH_zplus", [ 
        Plot("distancebetseedandmaxcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_EE_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_FH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_BH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_EE_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_FH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_BH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_EE_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_FH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_BH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_EE_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_EE_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_FH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_FH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancebetseedandmaxcellvsclusterenergy_perthickperlayer_scint_BH_zplus = PlotGroup("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_BH_zplus", [ 
        Plot("distancebetseedandmaxcellvsclusterenergy_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )


#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetoseedcell_perthickperlayer_120_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_120_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_120_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_120_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetoseedcell_perthickperlayer_200_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_200_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_200_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_200_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetoseedcell_perthickperlayer_300_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_300_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_300_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_300_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetoseedcell_perthickperlayer_scint_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_-1_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_scint_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_-1_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_scint_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_-1_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#=====================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#We need points for the weighted plots
_common = {"stat": True, "drawStyle": "EP", "staty": 0.65 }

#120 um 
_distancetomaxcell_perthickperlayer_eneweighted_120_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_120_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_120_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetomaxcell_perthickperlayer_eneweighted_200_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_200_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_200_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetomaxcell_perthickperlayer_eneweighted_300_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_300_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_300_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetomaxcell_perthickperlayer_eneweighted_scint_EE_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_EE_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_scint_FH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_FH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_scint_BH_zplus = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_BH_zplus", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )


#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetoseedcell_perthickperlayer_eneweighted_120_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_120_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_120_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetoseedcell_perthickperlayer_eneweighted_200_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_200_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_200_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetoseedcell_perthickperlayer_eneweighted_300_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_300_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_300_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetoseedcell_perthickperlayer_eneweighted_scint_EE_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_EE_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(maxlayerzm,lastLayerEEzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_scint_FH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_FH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerEEzp,lastLayerFHzp) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_scint_BH_zplus = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_BH_zplus", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i), xtitle="", **_common) for i in range(lastLayerFHzp,maxlayerzp) 
        ],
                                    ncols=4
                                    )

#Just in case we add some plots below to be on the safe side. 
_common = {"stat": True, "drawStyle": "hist", "staty": 0.65 }

#_energyclustered = 


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

