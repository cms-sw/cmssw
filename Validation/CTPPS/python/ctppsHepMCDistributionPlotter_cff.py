from Validation.CTPPS.CTPPSHepMCDistributionPlotter_cfi import *


from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(CTPPSHepMCDistributionPlotter, useNewLHCInfo = True)

from Configuration.Eras.Modifier_ctpps_directSim_cff import ctpps_directSim
ctpps_directSim.toModify(CTPPSHepMCDistributionPlotter, useNewLHCInfo = False)