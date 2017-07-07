import collections
import itertools

import ROOT

import Validation.RecoTrack.plotting.plotting as plotting

def saveHistograms(tdirectory, histos):
    for h in histos:
        h.SetDirectory(tdirectory)

def applyStyle(h, color, markerStyle):
    h.SetMarkerStyle(markerStyle)
    h.SetMarkerColor(color)
    h.SetMarkerSize(1.2)
    h.SetLineColor(color)
    h.SetLineWidth(2)

# https://stackoverflow.com/questions/6076270/python-lambda-function-in-list-comprehensions
_defaultStyles = [(lambda c, m: (lambda h: applyStyle(h, c, m)))(color, ms) for color, ms in itertools.izip(plotting._plotStylesColor, plotting._plotStylesMarker)]

_ratioFactor = 1.25

def draw(name, histos, styles=_defaultStyles, legendLabels=[], **kwargs):
    width = 600
    height = 600
    ratioFactor = 1.25

    args = {}
    args.update(kwargs)
    if not "ratioFactor" in args:
        args["ratioFactor"] = _ratioFactor
    ratio = args.get("ratio", False)

    if ratio:
        height = int(height*ratioFactor)
    c = plotting._createCanvas(name, width, height)
    if ratio:
        plotting._modifyPadForRatio(c, ratioFactor)

    frame = drawSingle(c, histos, styles, **args)

    if len(legendLabels) > 0:
        if len(legendLabels) != len(histos):
            raise Exception("Got %d histos but %d legend labels" % (len(histos), len(legendLabels)))
        lx1 = 0.6
        lx2 = 0.9
        ly1 = 0.7
        ly2 = 0.85
        
        lx1 += legendDx
        lx2 += legendDx
        ly1 += legendDy
        ly2 += legendDy
        
        lx2 += legendDw
        ly1 -= legendDh
            
        legend = ROOT.TLegend(lx1, ly1, lx2, ly2)
        legend.SetLineColor(1)
        legend.SetLineWidth(1)
        legend.SetLineStyle(1)
        legend.SetFillColor(0)
        legend.SetMargin(0.07)
        legend.SetBorderSize(0)

        for h, l in zip(histos, legendLabels):
            legend.AddEntry(h, l, "L")

        legend.Draw()

    frame._pad.cd()

    c.Update()
    c.RedrawAxis()
    c.SaveAs(name+".png")
    c.SaveAs(name+".pdf")


def drawSingle(pad, histos, styles=_defaultStyles,
               nrows=1,
               xtitle=None, ytitle=None,
               drawOpt="HIST",
               legendDx=0, legendDy=0, legendDw=0, legendDh=0,
               xmin=None, ymin=0, xmax=None, ymax=None, xlog=False, ylog=False,
               xgrid=True, ygrid=True,
               ratio=False, ratioYmin=0.5, ratioYmax=1.5, ratioYTitle=plotting._ratioYTitle, ratioFactor=1.25):

    bounds = plotting._findBounds(histos, ylog, xmin, xmax, ymin, ymax)
    if ratio:
        ratioBounds = (bounds[0], ratioYmin, bounds[2], ratioYmax)
        frame = plotting.FrameRatio(pad, bounds, ratioBounds, ratioFactor, ratioYTitle=ratioYTitle, nrows=nrows)
        #frame._frameRatio.GetYaxis().SetLabelSize(0.12)
    else:
        frame = plotting.Frame(pad, bounds, nrows=nrows)

    if xtitle is not None:
        frame.setXTitle(xtitle)
    if ytitle is not None:
        frame.setYTitle(ytitle)

    frame.setLogx(xlog)
    frame.setLogy(ylog)
    frame.setGridx(xgrid)
    frame.setGridy(ygrid)

    if ratio:
        frame._pad.cd()
    for i, h in enumerate(histos):
        st = styles[i%len(styles)]
        st(h)
        h.Draw(drawOpt+" same")

    ratios = None
    if ratio:
        frame._padRatio.cd()
        ratios = plotting._calculateRatios(histos)
        for r in ratios[1:]:
            r.draw()
        frame._pad.cd()

    return frame


def drawMany(name, histoDicts, styles=_defaultStyles, opts={}, ncolumns=4):
    if len(histoDicts) == 0:
        return

    histoNames = histoDicts[0].keys()
    ratio = False
    ratioFactor = _ratioFactor
    for opt in opts.itervalues():
        if "ratio" in opt:
            ratio = True
        if "ratioFactor" in opt:
            ratioFactor = max(ratioFactor, opt["ratioFactor"])

    nhistos = len(histoNames)
    nrows = int((nhistos+ncolumns-1)/ncolumns)

    width = 500*ncolumns
    height = 500*nrows
    if ratio:
        height = int(_ratioFactor*height)

    canvas = plotting._createCanvas(name, width, height)
    canvas.Divide(ncolumns, nrows)

    histos = collections.defaultdict(list)

    for d in histoDicts:
        for n, h in d.iteritems():
            histos[n].append(h)

    for i, histoName in enumerate(histoNames):
        pad = canvas.cd(i+1)

        args = {}
        args.update(opts.get(histoName, {}))
        if "ratio" in args:
            if not "ratioFactor" in args:
                args["ratioFactor"] = _ratioFactor # use the default, not the max
            plotting._modifyPadForRatio(pad, args["ratioFactor"])

        frame = drawSingle(pad, histos[histoName], styles, nrows, **args)
        frame._pad.cd()
        frame._pad.Update()
        frame._pad.RedrawAxis()

    canvas.SaveAs(name+".png")
    canvas.SaveAs(name+".pdf")
