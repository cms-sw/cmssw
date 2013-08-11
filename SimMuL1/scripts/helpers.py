def GetStat(h):
  return h.FindObject("stats")

def SetOptStat(h, op):
  stat = GetStat(h)
  stat.SetOptStat(op)
  return stat
    
def GetH(f, dir, name):
  return fi.Get("%s/%s;1"%(dir,name))

def Print(c, name):
    c.Print("%s/%s"%(pdir,name))

def myRebin(h, n):
    nb = h.GetNbinsX()
    entr = h.GetEntries()
    bin0 = h.GetBinContent(0)
    binN1 = h.GetBinContent(nb+1)
    if (nb % n):
        binN1 += h.Integral(nb - nb%n + 1, nb)
    h.Rebin(n)
    nb = h.GetNbinsX()
    h.SetBinContent(0, bin0)
    h.SetBinContent(nb+1, binN1)
    h.SetEntries(entr)

def scale(h):
    rate = 40000.
    nevents = 238000
    bx_window = 3
    bx_filling = 0.795
    h.Scale(rate*bx_filling/(bx_window*nevents))

if __name__ == "__main__":
    print "It's Working!"
