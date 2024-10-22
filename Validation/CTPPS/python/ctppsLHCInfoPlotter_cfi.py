from Validation.CTPPS.ctppsLHCInfoPlotterDefault_cfi import ctppsLHCInfoPlotterDefault as _ctppsLHCInfoPlotterDefault
ctppsLHCInfoPlotter = _ctppsLHCInfoPlotterDefault.clone()
    
from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(ctppsLHCInfoPlotter, useNewLHCInfo = True)

from Configuration.Eras.Modifier_ctpps_directSim_cff import ctpps_directSim
ctpps_directSim.toModify(ctppsLHCInfoPlotter, useNewLHCInfo = False)
