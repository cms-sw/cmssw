#!/usr/bin/env python

# This is an example of plotting the standard tracking validation
# plots from an explicit set of DQM root files.

from Validation.RecoTrack.plotting.validation import SimpleValidation, SimpleSample
import Validation.RecoTrack.plotting.trackingPlots as trackingPlots
import Validation.RecoVertex.plotting.vertexPlots as vertexPlots


# Example of file - label pairs
filesLabels = [
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_1.root", "Option 1"),
    ("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO_2.root", "Option 2"),
]

outputDir = "plots"
subdirprefix = "sample"


# To auto-generate HTML pages, uncomment the commented lines below
val = SimpleValidation([x[0] for x in filesLabels], [x[1] for x in filesLabels], outputDir)
#report = val.createHtmlReport("INSERT_YOUR_BASE_URL_HERE", validationName="Short description of your comparison")
#report.beginSample(SimpleSample("prefix", "Sample name"))
val.doPlots(trackingPlots.plotter, subdirprefix=subdirprefix, plotterDrawArgs={"ratio": True},
#            htmlReport=report
)
## Uncomment this to include also vertex plots
##val.doPlots(vertexPlots.plotter, subdirprefix=subdirprefix, plotterDrawArgs={"ratio": True},
##            htmlReport=report
##)
#report.write()

