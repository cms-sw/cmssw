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

def draw(name, histos, styles, legendLabels=[],
         xtitle=None, ytitle=None,
         drawOpt="HIST",
         legendDx=0, legendDy=0, legendDw=0, legendDh=0,
         xmin=None, ymin=0, xmax=None, ymax=None, xlog=False, ylog=False,
         xgrid=True, ygrid=True,
         ratio=False, ratioYmin=0.5, ratioYmax=1.5, ratioYTitle=plotting._ratioYTitle
        ):

    width = 600
    height = 600
    ratioFactor = 1.25

    if ratio:
        height = int(height*ratioFactor)
    c = plotting._createCanvas(name, width, height)
    if ratio:
        plotting._modifyPadForRatio(c, ratioFactor)

    bounds = plotting._findBounds(histos, ylog, xmin, xmax, ymin, ymax)
    args = {"nrows": 1}
    if ratio:
        ratioBounds = (bounds[0], ratioYmin, bounds[2], ratioYmax)
        frame = plotting.FrameRatio(c, bounds, ratioBounds, ratioFactor, ratioYTitle=ratioYTitle, **args)
        #frame._frameRatio.GetYaxis().SetLabelSize(0.12)
    else:
        frame = plotting.Frame(c, bounds, **args)

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
    for h, st in zip(histos, styles):
        st(h)
        h.Draw(drawOpt+" same")

    ratios = None
    if ratio:
        frame._padRatio.cd()
        ratios = plotting._calculateRatios(histos)
        for r in ratios[1:]:
            r.draw()
        frame._pad.cd()

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
