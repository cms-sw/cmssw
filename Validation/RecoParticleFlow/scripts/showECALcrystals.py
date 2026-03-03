import os
import argparse
import uproot
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as coll
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplhep as hep
import pandas as pd
hep.style.use("CMS")
from dataclasses import dataclass
    
def plotGeom(df):
    fig, ax = plt.subplots()
    params = dict(ax=ax, fontsize=15)
    hep.cms.text(' Phase-2 Simulation Preliminary', **params)
    hep.cms.lumitext("ECAL Geometry", **params)

    ax.scatter(df.crystalCenterEta, df.crystalCenterPhi, c='red', s=1)

    pat = []
    for row in df.itertuples(index=False):
        width = utils.angleDiff(row.crystalCorner2Eta, row.crystalCorner0Eta)
        height = utils.angleDiff(row.crystalCorner2Phi, row.crystalCorner0Phi)
        sq = patches.Rectangle((row.crystalCorner2Eta, row.crystalCorner2Phi),
                               width, height,
                               edgecolor='black', fill=False)
        pat.append(sq)
    ax.add_collection(coll.PatchCollection(pat, facecolor='none', edgecolor='black'))
    
    ax.scatter(df.crystalCenterEta, df.crystalCenterPhi, c='red', s=1)

    # ax.xaxis.set_major_locator(MultipleLocator(0.01))
    # ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    # ax.yaxis.set_major_locator(MultipleLocator(0.02))
    # ax.yaxis.set_minor_locator(MultipleLocator(0.005))
    
    plt.xlabel(f'$\eta$')
    plt.ylabel(f'$\phi$')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    fig.savefig('test.pdf')
    plt.close()

def add_black(cmap):
    """Add black at the end of a cmap."""
    colors = cmap(np.linspace(0, 1, cmap.N))
    new_colors = np.vstack((colors, [0, 0, 0, 1])) # add black
    return mcolors.ListedColormap(new_colors)

def plotEvent(geom, hits, clusters, out, zoom,
              era='Phase2', label='', var='energy', zlabel='', categorical=False):
    """
    Plot single event on top of the geometry.
    """
    pardir = utils.getParentDir(out)
    utils.createDir(pardir)
    utils.createIndexPHP(src=utils.getParentDir(pardir), dest=pardir)

    df = pd.merge(hits, geom, how='inner', left_on='detid', right_on='crystalDetId')
    # df = df[df[var]>0]
    # df[df.duplicated(subset='detid', keep=False) == True]
    
    fig, ax = plt.subplots()
    params = dict(ax=ax, fontsize=15)
    if era == 'Phase2': era='Phase-2'; en='14'
    elif era == 'Run3': era='Run-3'; en='13.6'
    hep.cms.text(f' {era} Simulation Preliminary', **params)
    hep.cms.lumitext(label + f' | {en} TeV', **params)

    if categorical:
        cmap = add_black(plt.get_cmap('tab10'))
        norm  = mcolors.Normalize(vmin=0, vmax=len(cmap.colors))
        colors = [cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(x-1) for x in df.clid]
        colors = [(r, g, b, 0.6) for (r, g, b, a) in colors] # change opacity
    else:
        cmap = cm.viridis
        # norm = mcolors.Normalize(vmin=df[var].min(), vmax=df[var].max())
        norm = mcolors.LogNorm(vmin=df[var].min(), vmax=df[var].max())
        colors = cmap(norm(df[var].values))

    # Add rectangles
    pat = []
    for row in df.itertuples(index=False):
        width = utils.angleDiff(row.crystalCorner2Eta, row.crystalCorner0Eta)
        height = utils.angleDiff(row.crystalCorner2Phi, row.crystalCorner0Phi)
        sq = patches.Rectangle((row.crystalCorner2Eta, row.crystalCorner2Phi),
                               width, height)
        pat.append(sq)
        
    ax.add_collection(coll.PatchCollection(pat, facecolor=colors, edgecolor='black'))

    # Add colorbar
    if not categorical:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=zlabel)

    plt.scatter(x=clusters.clusterEta, y=clusters.clusterPhi,
                color='red', marker='x', s=100, linewidth=4,
                label='Clusters')

    plt.xlabel(f'$\eta$')
    plt.ylabel(f'$\phi$')
    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-3.15, 3.15)
        
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    extensions = ('.pdf', '.png')
    for ext in extensions:
        fig.savefig(out + ext)
    print(f"INFO: Figure saved under {out}{'/'.join(extensions)}")
    plt.close()
        
def showECAL(infile, outfile, props):
    varsGeom = ['crystalDetId', 'crystalCenterEta', 'crystalCenterPhi',
                'crystalCorner0Eta', 'crystalCorner1Eta', 'crystalCorner2Eta', 'crystalCorner3Eta',
                'crystalCorner0Phi', 'crystalCorner1Phi', 'crystalCorner2Phi', 'crystalCorner3Phi']

    varsEventCommon = ['eventId']
    varsEvent = {"Reco": [], "Sim": []}
    for prefix in ("Reco", "Sim"):
        varsEvent[prefix].extend([x+prefix for x in
                                  ('energies', 'detids', 'nHits',
                                   'clusterEnergies', 'clusterEtas', 'clusterPhis',
                                   'clusterHitEnergies', 'clusterHitFractions',
                                   'clusterHitClids', 'clusterHitDetids')])
    varsEventAll = varsEventCommon + varsEvent['Reco'] + varsEvent['Sim']
        
    with uproot.open(infile) as file:
        dfGeom = file['ecalGeometryAnalyzer/Geometry'].arrays(varsGeom, library="pandas")
        dfEvent = file['ecalGeometryAnalyzer/Event'].arrays(varsEventAll, library="awkward")

    # filters
    # if zoom:
    #     filt = lambda x : ((x.crystalCenterEta > 0.18) & (x.crystalCenterEta < 0.3)
    #                        & (x.crystalCenterPhi > 0.1) & (x.crystalCenterPhi < 0.3))
    # else:
    #     filt = lambda x : x

    # dfGeom = dfGeom[filt(dfGeom)]
    plotGeom(dfGeom)

    for t in dfEvent.eventId[:props.nevents]:

        for prefix in ("Reco", "Sim"):
            print(f'INFO: Processing event {t} in mode {prefix}...')
            
            outname = "event" + prefix + str(t)
            if props.zoom:
                outname += "_zoom"

            dfEventTmp = dfEvent[dfEvent.eventId==t]

            # if prefix == "Sim": # energy cut to reduce hit multiplicity in plot
            #     en_mask = dfEventTmp.energiesSim > 0.0
            #     dfEventTmp['energies'+prefix] = dfEventTmp['energies'+prefix][en_mask]
            #     dfEventTmp['detids'+prefix] = dfEventTmp['detids'+prefix][en_mask]

            dfHits = pd.DataFrame({'energy': dfEventTmp['energies'+prefix][0],
                                   'detid': dfEventTmp['detids'+prefix][0]})

            dfClusters = pd.DataFrame({'clusterEnergy': dfEventTmp['clusterEnergies'+prefix][0],
                                       'clusterEta': dfEventTmp['clusterEtas'+prefix][0],
                                       'clusterPhi': dfEventTmp['clusterPhis'+prefix][0],
                                       })

            dfHitsInClusters = pd.DataFrame({'energy': dfEventTmp['clusterHitEnergies'+prefix][0],
                                             'frac': dfEventTmp['clusterHitFractions'+prefix][0],
                                             'detid': dfEventTmp['clusterHitDetids'+prefix][0],
                                             'clid': dfEventTmp['clusterHitClids'+prefix][0],
                                             })

            plotEvent(dfGeom, dfHits, dfClusters,
                      out=os.path.join(outfile, outname + '_allhits'),
                      era=props.era, label=props.label, var='energy', zoom=props.zoom,
                      zlabel=('PF' if prefix=='Reco' else 'Sim') + ' RecHit Energy [GeV]')
            plotEvent(dfGeom, dfHitsInClusters, dfClusters,
                      out=os.path.join(outfile, outname + '_clhits'),
                      var='frac', zoom=props.zoom, categorical=False,
                      zlabel=('PF' if prefix=='Reco' else 'Sim') + ' RecHit Energy [GeV]')
        
    print('INFO: Done.')

    
@dataclass
class InputArgs:
    zoom: bool
    nevents: int
    era: str
    label: str
    
if __name__ == '__main__':

    full_command = 'python3 Validation/RecoParticleFlow/scripts/showECALcrystals.py --file <input_root_file>'
    parser = argparse.ArgumentParser(description='Show position of crystals. \nExample command:\n' + full_command)

    parser.add_argument('-i', '--file',
                        help='Path to the input ROOT file.')
    parser.add_argument('-o', '--outdir',
                        help='Path to the output folder where the events will be stored.')
    parser.add_argument('-z', '--zoom',
                        help='Zoom over a hard-coded eta/phi region: (min_eta, max_eta, min_phi, max_phi)', nargs=4, type=float, default=None)
    parser.add_argument('-n', '--nevents',
                        help='Number of events to plot. If the input file has less events, then it plots all events of the input file.', default=10)
    parser.add_argument('-e', '--era', required=True,
                        help="Chose between ['Phase2', 'Run3'].")
    parser.add_argument('-l', '--sample_label', required=True,
                        help='Sample label for plotting.')
    
    args = parser.parse_args()
    props = InputArgs(zoom=args.zoom, nevents=args.nevents,
                      era=args.era, label=args.sample_label)
    showECAL(args.file, args.outdir, props)
