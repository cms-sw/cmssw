import os
import numpy as np
import ROOT

def angleDiff(x1, x2):
    return np.arctan2(np.sin(x1 - x2), np.cos(x1 - x2))

def checkRootDir(afile, adir):
    if not afile.Get(adir):
        raise RuntimeError(f"Directory '{adir}' not found in {afile}")

def createDir(adir):
    if not os.path.exists(adir):
        os.makedirs(adir)
    return adir

def createIndexPHP(src, dest):
    """
    Copy index php file used for visualization in the browser.
    """
    php_file = os.path.join(src, 'index.php')
    if os.path.exists(php_file) and not os.path.exists(os.path.join(src, 'index.php')):
        os.system(f'cp {php_file} {dest}')

def debug(mes):
    print('### INFO: ' + mes)

def findBestGaussianCoreFit(histo, quiet=True, meanForRange=1., rmsForRange=0.1):
    """
    Implementation of a gaussian core fit.
    """
    if histo.Integral() < 50:
        debug(' [findBestGaussianCoreFit] There is not enough data for running the fit.')
        return None
    
    Xmax = histo.GetXaxis().GetBinCenter(histo.GetMaximumBin())

    gausTF1 = ROOT.TF1()

    Pvalue = 0.
    RangeLow = histo.GetBinLowEdge(2)
    RangeUp = histo.GetBinLowEdge(histo.GetNbinsX())

    PvalueBest = 0.    
    rms_step_minus = 2.2
    # the meanForRange and rmsForRange are initial estimates.
    RangeLowBest = meanForRange - rms_step_minus*rmsForRange
    RangeUpBest = meanForRange + rms_step_minus*rmsForRange
    
    range_max_dist = 0.7
    sigma_step = 0.1
    StepMinusBest, StepPlusBest, ChiSquareBest, ndfBest = (None for _ in range(4))

    while rms_step_minus>range_max_dist:
        RangeLow = meanForRange - rms_step_minus*rmsForRange
        rms_step_plus = rms_step_minus

        while rms_step_plus>range_max_dist:
            RangeUp = meanForRange + rms_step_plus*rmsForRange 
            histo.Fit("gaus", "0Q" if quiet else "0", "0", RangeLow, RangeUp)

            gausTF1 = histo.GetListOfFunctions().FindObject("gaus")
            ChiSquare = gausTF1.GetChisquare()
            ndf       = gausTF1.GetNDF()
            Pvalue = ROOT.TMath.Prob(ChiSquare, ndf)

            if Pvalue > PvalueBest:
                PvalueBest = Pvalue
                RangeLowBest = RangeLow
                RangeUpBest = RangeUp
                ndfBest = ndf
                ChiSquareBest = ChiSquare
                StepMinusBest = rms_step_minus
                StepPlusBest = rms_step_plus
                meanForRange = gausTF1.GetParameter(1)

            if not quiet:
                debug(f"[findBestGaussianCoreFit] \nFitting range used: [{meanForRange} - {rms_step_minus} sigma, {meanForRange} + {rms_step_plus} sigma ] ")
                debug(f"ChiSquare = {ChiSquare}, NDF = {ndf}, Prob = {Pvalue},  Best Prob so far = {PvalueBest}")
                debug(f"Sigma limit = {range_max_dist}")

            rms_step_plus -= sigma_step
            
        rms_step_minus -= sigma_step

    if quiet:
        histo.Fit("gaus", "0Q", "0", RangeLowBest, RangeUpBest)
    else:
        histo.Fit("gaus","0","0", RangeLowBest, RangeUpBest)
        debug("[findBestGaussianCoreFit] Fit found!")
        debug(f"Final fitting range used: [{meanForRange} - {StepMinusBest} rms(WHF), {meanForRange} + {StepPlusBest} rms(WHF) ]")
        debug(f"ChiSquare = {ChiSquareBest}, NDF = {ndfBest}, Prob = {PvalueBest}\n\n")
        
    return histo.GetListOfFunctions().FindObject("gaus")

def getParentDir(adir):
    pardir = os.path.dirname(adir)
    if adir[-1] == '/':
        pardir = os.path.dirname(pardir)
    return pardir
