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

#Could i get this from somewhere? 
lastLayerEE = 28  # last layer of EE
lastLayerFH = 40  # last layer of FH
maxlayer = 52 # last layer of BH

_common = {"stat": True, "drawStyle": "hist"}

_num_reco_cluster_eta = PlotGroup("num_reco_cluster_eta", [
        Plot("num_reco_cluster_eta", xtitle="", **_common),
        ],ncols=1)

_mixedhitscluster = PlotGroup("mixedhitscluster", [
        Plot("mixedhitscluster", xtitle="", **_common),
        ],ncols=1)

_energyclustered = PlotGroup("energyclustered", [
        Plot("energyclustered", xtitle="", **_common),
        ],ncols=1)

_longdepthbarycentre = PlotGroup("longdepthbarycentre", [
        Plot("longdepthbarycentre", xtitle="", **_common),
        ],ncols=1)

_totclusternum_thick = PlotGroup("totclusternum_thick", [
    Plot("totclusternum_thick_120", xtitle="", **_common),
    Plot("totclusternum_thick_200", xtitle="", **_common),
    Plot("totclusternum_thick_300", xtitle="", **_common),
    Plot("totclusternum_thick_-1", xtitle="", **_common),
    ])

_totclusternum_layer_EE = PlotGroup("totclusternum_layer_EE", [ 
        Plot("totclusternum_layer_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_totclusternum_layer_FH = PlotGroup("totclusternum_layer_FH", [ 
        Plot("totclusternum_layer_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_totclusternum_layer_BH = PlotGroup("totclusternum_layer_BH", [ 
        Plot("totclusternum_layer_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_EE = PlotGroup("energyclustered_perlayer_EE", [ 
        Plot("energyclustered_perlayer%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_FH = PlotGroup("energyclustered_perlayer_FH", [ 
        Plot("energyclustered_perlayer%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_energyclustered_perlayer_BH = PlotGroup("energyclustered_perlayer_BH", [ 
        Plot("energyclustered_perlayer%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )
_cellsenedens_thick =  PlotGroup("cellsenedens_thick", [
    Plot("cellsenedens_thick_120", xtitle="", **_common),
    Plot("cellsenedens_thick_200", xtitle="", **_common),
    Plot("cellsenedens_thick_300", xtitle="", **_common),
    Plot("cellsenedens_thick_-1", xtitle="", **_common),
    ])

#----------------------------------------------------------------------------------------------------------------
#120 um 
_cellsnum_perthick_perlayer_120_EE = PlotGroup("cellsnum_perthick_perlayer_120_EE", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_120_FH = PlotGroup("cellsnum_perthick_perlayer_120_FH", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_120_BH = PlotGroup("cellsnum_perthick_perlayer_120_BH", [ 
        Plot("cellsnum_perthick_perlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#200 um 
_cellsnum_perthick_perlayer_200_EE = PlotGroup("cellsnum_perthick_perlayer_200_EE", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_200_FH = PlotGroup("cellsnum_perthick_perlayer_200_FH", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_200_BH = PlotGroup("cellsnum_perthick_perlayer_200_BH", [ 
        Plot("cellsnum_perthick_perlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#300 um 
_cellsnum_perthick_perlayer_300_EE = PlotGroup("cellsnum_perthick_perlayer_300_EE", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_300_FH = PlotGroup("cellsnum_perthick_perlayer_300_FH", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_300_BH = PlotGroup("cellsnum_perthick_perlayer_300_BH", [ 
        Plot("cellsnum_perthick_perlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#scint um 
_cellsnum_perthick_perlayer_scint_EE = PlotGroup("cellsnum_perthick_perlayer_-1_EE", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_scint_FH = PlotGroup("cellsnum_perthick_perlayer_-1_FH", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_cellsnum_perthick_perlayer_scint_BH = PlotGroup("cellsnum_perthick_perlayer_-1_BH", [ 
        Plot("cellsnum_perthick_perlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetomaxcell_perthickperlayer_120_EE = PlotGroup("distancetomaxcell_perthickperlayer_120_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_120_FH = PlotGroup("distancetomaxcell_perthickperlayer_120_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_120_BH = PlotGroup("distancetomaxcell_perthickperlayer_120_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetomaxcell_perthickperlayer_200_EE = PlotGroup("distancetomaxcell_perthickperlayer_200_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_200_FH = PlotGroup("distancetomaxcell_perthickperlayer_200_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_200_BH = PlotGroup("distancetomaxcell_perthickperlayer_200_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetomaxcell_perthickperlayer_300_EE = PlotGroup("distancetomaxcell_perthickperlayer_300_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_300_FH = PlotGroup("distancetomaxcell_perthickperlayer_300_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_300_BH = PlotGroup("distancetomaxcell_perthickperlayer_300_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetomaxcell_perthickperlayer_scint_EE = PlotGroup("distancetomaxcell_perthickperlayer_-1_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_scint_FH = PlotGroup("distancetomaxcell_perthickperlayer_-1_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_scint_BH = PlotGroup("distancetomaxcell_perthickperlayer_-1_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )


#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetoseedcell_perthickperlayer_120_EE = PlotGroup("distancetoseedcell_perthickperlayer_120_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_120_FH = PlotGroup("distancetoseedcell_perthickperlayer_120_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_120_BH = PlotGroup("distancetoseedcell_perthickperlayer_120_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetoseedcell_perthickperlayer_200_EE = PlotGroup("distancetoseedcell_perthickperlayer_200_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_200_FH = PlotGroup("distancetoseedcell_perthickperlayer_200_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_200_BH = PlotGroup("distancetoseedcell_perthickperlayer_200_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetoseedcell_perthickperlayer_300_EE = PlotGroup("distancetoseedcell_perthickperlayer_300_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_300_FH = PlotGroup("distancetoseedcell_perthickperlayer_300_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_300_BH = PlotGroup("distancetoseedcell_perthickperlayer_300_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetoseedcell_perthickperlayer_scint_EE = PlotGroup("distancetoseedcell_perthickperlayer_-1_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_scint_FH = PlotGroup("distancetoseedcell_perthickperlayer_-1_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_scint_BH = PlotGroup("distancetoseedcell_perthickperlayer_-1_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#=====================================================================================================================
#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetomaxcell_perthickperlayer_eneweighted_120_EE = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_120_FH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_120_BH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_120_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetomaxcell_perthickperlayer_eneweighted_200_EE = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_200_FH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_200_BH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_200_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetomaxcell_perthickperlayer_eneweighted_300_EE = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_300_FH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_300_BH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_300_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetomaxcell_perthickperlayer_eneweighted_scint_EE = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_EE", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_scint_FH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_FH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetomaxcell_perthickperlayer_eneweighted_scint_BH = PlotGroup("distancetomaxcell_perthickperlayer_eneweighted_-1_BH", [ 
        Plot("distancetomaxcell_perthickperlayer_eneweighted_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )


#----------------------------------------------------------------------------------------------------------------
#120 um 
_distancetoseedcell_perthickperlayer_eneweighted_120_EE = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_120_FH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_120_BH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_120_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_120_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#200 um 
_distancetoseedcell_perthickperlayer_eneweighted_200_EE = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_200_FH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_200_BH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_200_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_200_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#300 um 
_distancetoseedcell_perthickperlayer_eneweighted_300_EE = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_300_FH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_300_BH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_300_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_300_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )

#scint um 
_distancetoseedcell_perthickperlayer_eneweighted_scint_EE = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_EE", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_scint_FH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_FH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerEE,lastLayerFH) 
        ],
                                    ncols=4
                                    )

_distancetoseedcell_perthickperlayer_eneweighted_scint_BH = PlotGroup("distancetoseedcell_perthickperlayer_eneweighted_-1_BH", [ 
        Plot("distancetoseedcell_perthickperlayer_eneweighted_-1_%s"%(i+1), xtitle="", **_common) for i in range(lastLayerFH,maxlayer) 
        ],
                                    ncols=4
                                    )



#_energyclustered = 


hgcalLayerClustersPlotter = Plotter()
hgcalLayerClustersPlotter.append("hgcalLayerClusters", [
        "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/hgcalLayerClusters",
        ], PlotFolder(
        _num_reco_cluster_eta,
        _mixedhitscluster,
        _energyclustered,
        _longdepthbarycentre,
        _totclusternum_thick, 
        _totclusternum_layer_EE,
        _totclusternum_layer_FH,
        _totclusternum_layer_BH,
        _energyclustered_perlayer_EE,
        _energyclustered_perlayer_FH,
        _energyclustered_perlayer_BH,
        _cellsenedens_thick, 
        _cellsnum_perthick_perlayer_120_EE,
        _cellsnum_perthick_perlayer_120_FH,
        _cellsnum_perthick_perlayer_120_BH,
        _cellsnum_perthick_perlayer_200_EE,
        _cellsnum_perthick_perlayer_200_FH,
        _cellsnum_perthick_perlayer_200_BH,
        _cellsnum_perthick_perlayer_300_EE,
        _cellsnum_perthick_perlayer_300_FH,
        _cellsnum_perthick_perlayer_300_BH,
        _cellsnum_perthick_perlayer_scint_EE,
        _cellsnum_perthick_perlayer_scint_FH,
        _cellsnum_perthick_perlayer_scint_BH,
        _distancetomaxcell_perthickperlayer_120_EE,
        _distancetomaxcell_perthickperlayer_120_FH,
        _distancetomaxcell_perthickperlayer_120_BH,
        _distancetomaxcell_perthickperlayer_200_EE,
        _distancetomaxcell_perthickperlayer_200_FH,
        _distancetomaxcell_perthickperlayer_200_BH,
        _distancetomaxcell_perthickperlayer_300_EE,
        _distancetomaxcell_perthickperlayer_300_FH,
        _distancetomaxcell_perthickperlayer_300_BH,
        _distancetomaxcell_perthickperlayer_scint_EE,
        _distancetomaxcell_perthickperlayer_scint_FH,
        _distancetomaxcell_perthickperlayer_scint_BH,
        _distancetoseedcell_perthickperlayer_120_EE,
        _distancetoseedcell_perthickperlayer_120_FH,
        _distancetoseedcell_perthickperlayer_120_BH,
        _distancetoseedcell_perthickperlayer_200_EE,
        _distancetoseedcell_perthickperlayer_200_FH,
        _distancetoseedcell_perthickperlayer_200_BH,
        _distancetoseedcell_perthickperlayer_300_EE,
        _distancetoseedcell_perthickperlayer_300_FH,
        _distancetoseedcell_perthickperlayer_300_BH,
        _distancetoseedcell_perthickperlayer_scint_EE,
        _distancetoseedcell_perthickperlayer_scint_FH,
        _distancetoseedcell_perthickperlayer_scint_BH,
        _distancetomaxcell_perthickperlayer_eneweighted_120_EE,
        _distancetomaxcell_perthickperlayer_eneweighted_120_FH,
        _distancetomaxcell_perthickperlayer_eneweighted_120_BH,
        _distancetomaxcell_perthickperlayer_eneweighted_200_EE,
        _distancetomaxcell_perthickperlayer_eneweighted_200_FH,
        _distancetomaxcell_perthickperlayer_eneweighted_200_BH,
        _distancetomaxcell_perthickperlayer_eneweighted_300_EE,
        _distancetomaxcell_perthickperlayer_eneweighted_300_FH,
        _distancetomaxcell_perthickperlayer_eneweighted_300_BH,
        _distancetomaxcell_perthickperlayer_eneweighted_scint_EE,
        _distancetomaxcell_perthickperlayer_eneweighted_scint_FH,
        _distancetomaxcell_perthickperlayer_eneweighted_scint_BH,
        _distancetoseedcell_perthickperlayer_eneweighted_120_EE,
        _distancetoseedcell_perthickperlayer_eneweighted_120_FH,
        _distancetoseedcell_perthickperlayer_eneweighted_120_BH,
        _distancetoseedcell_perthickperlayer_eneweighted_200_EE,
        _distancetoseedcell_perthickperlayer_eneweighted_200_FH,
        _distancetoseedcell_perthickperlayer_eneweighted_200_BH,
        _distancetoseedcell_perthickperlayer_eneweighted_300_EE,
        _distancetoseedcell_perthickperlayer_eneweighted_300_FH,
        _distancetoseedcell_perthickperlayer_eneweighted_300_BH,
        _distancetoseedcell_perthickperlayer_eneweighted_scint_EE,
        _distancetoseedcell_perthickperlayer_eneweighted_scint_FH,
        _distancetoseedcell_perthickperlayer_eneweighted_scint_BH,
        loopSubFolders=False,
        purpose=PlotPurpose.Timing, page="hgcalLayerClusters"   
        ))

