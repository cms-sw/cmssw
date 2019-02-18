ptbins = [
    10,24,32,43,56,74,97,133,174,245,300,362,430,507,592,686,846,1032,1248,1588,
    2000,2500,3000,4000,6000
]
etabins = [0.0, 0.5, 1.3, 2.1, 2.5, 3.0]

def response_distribution_name(iptbin, ietabin):
    #convert 0.5 -> "05"
    eta_string = "{0:.1f}".format(etabins[ietabin+1]).replace(".", "")
    return "reso_dist_{0:.0f}_{1:.0f}_eta{2}".format(ptbins[iptbin], ptbins[iptbin+1], eta_string)
