"""
If not available by default, to get bokeh on lxplus:
 $ cmsenv
 $ scram-venv
 $ python3 -m pip install bokeh    
"""

import os
import argparse
import uproot
import awkward as ak
import utils
import numpy as np
import pandas as pd
import utils
from dataclasses import dataclass

from bokeh.plotting import figure, output_file, save, ColumnDataSource
from bokeh.models import (HoverTool, Rect, ColumnDataSource, LinearColorMapper, LogColorMapper,
                          Div, Button, ColorBar, NumericInput, Dropdown, CDSView, GroupFilter,
                          BooleanFilter, CustomJS, Slider)
from bokeh.palettes import Viridis256, Category20
from bokeh.transform import linear_cmap, log_cmap
from bokeh.layouts import layout

def writeIntructions():
    text = """
    <b> Interactive event display </b>

    <br>
    
    <p>
    This event display was developed to enable a flexible visualization of PF clusters in ECAL.
    Its original goal was to facilitate PF validation at HLT.
    </p>

    <p>    
    This page can be created by running:

    <br>
    
    <pre>
    python3 Validation/RecoParticleFlow/scripts/showECALcrystals_interactive.py -i data.root --outdir . --nevents 10
    </pre>    
    within a CMSSW release.
    Check all options by adding "--help".

    <p>
    The input "data.root" file holds the geometry and event information, and can be in turn produced with:

    <br>
    
    <pre>
    cmsRun Validation/RecoParticleFlow/test/ecalGeometryAnalyzer_cfg.py input="your step2 file".root output=data.root maxEvents=100
    </pre>
    where the input file refers to a CMS step2 file, where cluster information and ECAL geometry is available.
    </p>

    <hr />
    
    <b> Capabilites </b>

    <p>
    This tool enables the comparison of simulated (left) and reconstructed (right) hits and clusters. Specifically, you can:

    <br>
    
    <ul>
    <li>Visualize different events by changing the event number in "Event ID selection"</li>
    <li>Filter crystals based on the deposited energy</li>
    <li>Select specific clusters by their ID</li>
    <li>Use selection tools on all figures simultaneously, available at the right of each plot (zoom, undo, ...)</li>
    </ul>
    </p>
    
    <p>
    The energy displayed corresponds to the total energy deposited in a given crystal.
    Hover the crystals with your mouse to inspect the contributions of individual clusters.
    </p>
    <p>
    Energy fractions in a given crystal are available mostly as a debugging tool: we expect "FracSum" to be one for all crystals.
    </p>
    
    <br>
    
    <p>
    <i>Note:</i> You might need to click twice on the "Show all clusters" buttons for them to work correctly.
    </p>

    <hr />
    """
    return Div(text=text)

def writeContacts():
    text = """
    <hr />
    <p>
    Tool developed under the <a href="https://nextgentriggers.web.cern.ch/">Next Generation Trigger project</a> (task 3.1.1).
    </p>
    <p>
    <i>Contact:</i> For bug reports or feature requests please write a message to <code>bruno.alves@cern.ch</code>.
    </p>
    <br>
    """
    return Div(text=text)

def createFigure(title):
    fig = figure(
        title=title,
        x_range=(-1.69,1.69),
        y_range=(-1.05*np.pi,1.05*np.pi),
        x_axis_label=r"$$\eta$$",
        y_axis_label=r"$$\phi$$",
        width=1100,
        height=700,
        tools="pan,wheel_zoom,box_zoom,undo,redo,reset,save",
        active_drag="box_zoom",
        active_scroll=None,
    )
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = None
    fig.toolbar.logo = None
    return fig

def plotGeom(df, output_path):
    """Plot ECAL geometry as interactive Bokeh plot."""
    output_file(output_path)

    df['width'] = utils.angleDiff(df.crystalCorner2Eta, df.crystalCorner0Eta)
    df['height'] = utils.angleDiff(df.crystalCorner2Phi, df.crystalCorner0Phi)
    # df = df.iloc[0:20000]
    source = ColumnDataSource(df)

    # Create figure
    p = createFigure(title='ECAL-Barrel Geometry')
    
    # Add rectangles for each crystal    
    p.add_tools(HoverTool(
        tooltips=[
            ("DetID", "@crystalDetId"),
            ("Center (η, φ)", "(@crystalCenterEta, @crystalCenterPhi)"),
        ],
        mode="mouse",
    ))

    # Use rect glyph for each crystal
    p.rect(
        x="crystalCenterEta",
        y="crystalCenterPhi",
        width="width",
        height="height",
        source=source,
        fill_color="lightgray",
        line_color="black",
        alpha=0.1,
    )

    # Add scatter for centers
    p.scatter(
        x="crystalCenterEta",
        y="crystalCenterPhi",
        source=source,
        color="red",
        size=6,
    )
    p.scatter(
        x="crystalCorner2Eta",
        y="crystalCorner2Phi",
        source=source,
        color="green",
        size=6,
    )
    p.scatter(
        x="crystalCorner0Eta",
        y="crystalCorner0Phi",
        source=source,
        color="blue",
        size=6,
    )

    save(p)
    print(f"INFO: Geometry plot saved to {output_path}")

def shift_phi_corners(phi0, phi1, phi2, phi3):
    corners = [phi0, phi1, phi2, phi3]
    # Check each pair of adjacent corners
    for i in range(4):
        j = (i + 1) % 4
        diff = abs(corners[i] - corners[j])
        if diff > np.pi:
            # Shift the larger value by -2π
            if corners[i] > corners[j]:
                corners[i] -= 2 * np.pi
            else:
                corners[j] -= 2 * np.pi
    return tuple(corners) + (corners[0],)  # Close the patch

def plotEvent(geom, hits, clusters, output_path):
    """Plot single event on top of the geometry, with interactive hover."""
    output_file(output_path)

    modes = ('Sim', 'Reco')
    eventDefault = 1
    p, df, hover, mapper_log, mapper_lin, color_bar, threshFilter, threshFilterId = ({} for _ in range(8))
    src, srcCluster, view, viewId, viewCluster, = ({} for _ in range (5))
    hit_renderer, cluster_renderer = ({} for _ in range(2))
    for mode in modes:
        df[mode] = pd.merge(hits[mode], geom, how="inner", left_on="detids", right_on="crystalDetId")

        fracs_in_df = 'fracs' in df[mode].columns
        clids_in_df = 'clids' in df[mode].columns

        df[mode]["eventId"] = df[mode]["eventId"].astype(str)
        if clids_in_df:
            df[mode]["clids"] = df[mode]["clids"].astype(str)

        view[mode] = CDSView(filters=[
            GroupFilter(column_name="eventId", group=str(eventDefault)),
            BooleanFilter([True] * len(df[mode]))]
        )
        threshFilter[mode] = view[mode].filters[1]

        if clids_in_df:
            viewId[mode] = CDSView(filters=[
                GroupFilter(column_name="eventId", group=str(eventDefault)),
                BooleanFilter([True] * len(df[mode])),
                BooleanFilter([True] * len(df[mode]))]
                                )
            threshFilterId[mode] = viewId[mode].filters[2]

        clusters[mode]["eventId"] = clusters[mode]["eventId"].astype(str)
        srcCluster[mode] = ColumnDataSource(clusters[mode])
        
        viewCluster[mode] = CDSView(filters=[
            GroupFilter(column_name="eventId", group=str(eventDefault)),]
        )
        
        # Create lists of lists for xs and ys
        df[mode]['xs'] = [
            (eta1, eta2, eta3, eta4, eta1)  # Close the patch by repeating the first point
            for eta1, eta2, eta3, eta4 in zip(
                    df[mode]["crystalCorner0Eta"], df[mode]["crystalCorner1Eta"],
                    df[mode]["crystalCorner2Eta"], df[mode]["crystalCorner3Eta"]
            )
        ]
        df[mode]['ys'] = [
            shift_phi_corners(phi0, phi1, phi2, phi3)  # Close the patch by repeating the first point
            for phi0, phi1, phi2, phi3 in zip(
                    df[mode]["crystalCorner0Phi"], df[mode]["crystalCorner1Phi"],
                    df[mode]["crystalCorner2Phi"], df[mode]["crystalCorner3Phi"]
            )
        ]
        
        # Aggregate data for unique patches
        patch_dict = {
            'eventId': df[mode]['eventId'],
            'xs': df[mode]['xs'],
            'ys': df[mode]['ys'],
            'energies': df[mode]['energies'],
        }
        aggr_dict = {'energies': 'sum'}
        if fracs_in_df:
            patch_dict.update({'fracs': df[mode]['fracs']})
            aggr_dict.update({'fracs': 'sum'})
        patch_data = pd.DataFrame(patch_dict)        
        aggr = patch_data.groupby(['eventId', 'xs', 'ys'], as_index=False).agg(aggr_dict)

        # Add summed variables to the original DataFrame
        energy_sum_map = aggr.set_index(['eventId', 'xs', 'ys'])['energies'].to_dict()
        df[mode]['energies_sum'] = df[mode].apply(
            lambda row: energy_sum_map.get((row['eventId'], row['xs'], row['ys']), None),
            axis=1
        )
        if fracs_in_df:
            frac_sum_map = aggr.set_index(['eventId', 'xs', 'ys'])['fracs'].to_dict()
            df[mode]['fracs_sum'] = df[mode].apply(
                lambda row: frac_sum_map.get((row['eventId'], row['xs'], row['ys']), None),
                axis=1
            )

        src[mode] = ColumnDataSource(df[mode])

        # Create figure
        p[mode] = [createFigure(title=mode + " Hits")]
        if clids_in_df:
            p[mode].append(createFigure(title=mode + " Cluster IDs"))

        hit_renderer[mode] = []
        if clids_in_df:
            # categorical figures
            colors = [Category20[20][int(i) % 20] for i in df[mode].clids]
            src[mode].add(colors, "colors")
            hit_renderer[mode].append(
                p[mode][1].patches(
                    xs="xs", ys="ys",
                    source=src[mode],
                    view=viewId[mode],
                    fill_color="colors",
                    line_color="black",
                    fill_alpha=0.5,
                )
            )
        
        # continuous figures
        mapper_kwargs = dict(palette=Viridis256, low=df[mode]['energies_sum'].min(), high=df[mode]['energies_sum'].max())
        mapper_log[mode] = LogColorMapper(**mapper_kwargs)
        mapper_lin[mode] = LinearColorMapper(**mapper_kwargs)
        color_bar[mode] = ColorBar(color_mapper=mapper_log[mode], label_standoff=12)

        hit_renderer[mode].append(
            p[mode][0].patches(
                xs="xs", ys="ys",
                source=src[mode],
                view=view[mode],
                fill_color=log_cmap('energies_sum', Viridis256, df[mode]['energies_sum'].min(), df[mode]['energies_sum'].max()),
                line_color="black"
            )
        )
        
        p[mode][0].add_layout(color_bar[mode], "right")

        # Add hover tool
        hover_string = ''
        if clids_in_df:
            hover_string += 'ClusterID: @clids, '
        if fracs_in_df:
            hover_string += "Frac: @fracs{0.000}, FracSum: @fracs_sum{0.000}, En: @energies, EnSum: @energies_sum"
        else:
            hover_string += "En: @energies, EnSum: @energies_sum"
        hover[mode] = HoverTool( # first string is the text
            renderers=hit_renderer[mode],
            tooltips=[ ("", hover_string), ],
            mode="mouse",
        )

        for idx in range(len(p[mode])):
            p[mode][idx].add_tools(hover[mode])

        # Add clusters to the energy/fraction plot
        if props.clusters:
            cluster_renderer[mode] = p[mode][0].scatter(
                x='clusterEtas' + mode,
                y='clusterPhis' + mode,
                source=srcCluster[mode],
                view=viewCluster[mode],
                color="red",
                marker="x",
                size=20,
                line_width=4,
                legend_label="Clusters",
            )
            p[mode][0].legend.label_text_font_size = '16pt'                                      

            cluster_hover_string = "Cluster Energy: @clusterEnergies" + mode
            p[mode][0].add_tools(
                HoverTool(renderers=[cluster_renderer[mode]],
                          tooltips=[("", cluster_hover_string), ]))
            
    p['Sim'][0].x_range, p['Sim'][0].y_range = p['Reco'][0].x_range, p['Reco'][0].y_range
    if clids_in_df:
        p['Sim'][1].x_range, p['Sim'][1].y_range = p['Reco'][0].x_range, p['Reco'][0].y_range
        p['Reco'][1].x_range, p['Reco'][1].y_range = p['Reco'][0].x_range, p['Reco'][0].y_range

    enSumMax = 2.
    slider = Slider(start=0, end=enSumMax, value=0.1, step=0.01, title="Min threshold for energies_sum", width=800)

    varNameHolder = ColumnDataSource(data=dict(value=["energies_sum"]))
    
    dfMin = min(df[mode].eventId.min() for mode in modes)
    dfMax = max(df[mode].eventId.max() for mode in modes)

    numInput = NumericInput(value=eventDefault, low=int(dfMin), high=int(dfMax),
                            title=f"Event ID selection (enter a number between {dfMin} and {dfMax}):")
    numInput_args = dict(
        srcSim=src["Sim"], srcReco=src["Reco"],
        srcClSim=srcCluster["Sim"], srcClReco=srcCluster["Reco"],
        viewEvSim=view["Sim"], viewEvReco=view["Reco"],
        viewClSim=viewCluster["Sim"], viewClReco=viewCluster["Reco"],
        select=numInput, slider=slider,
        threshSim=threshFilter["Sim"], threshReco=threshFilter["Reco"],
        varNameHolder=varNameHolder,
    )
    if clids_in_df:
        numInput_args.update({'viewIdSim': viewId["Sim"], 'viewIdReco': viewId["Reco"],})
        numInput_code = """
        const eid = select.value.toString();
        viewEvSim.filters[0].group = eid;
        viewEvReco.filters[0].group = eid;
        viewIdSim.filters[0].group = eid;
        viewIdReco.filters[0].group = eid;
        viewClSim.filters[0].group = eid;
        viewClReco.filters[0].group = eid;

        const minVal = slider.value;
        const v = varNameHolder.data['value'][0];
        
        const sim = srcSim.data;
        const rec = srcReco.data;

        let maskSim = [];
        let maskRec = [];
        
        for (let i = 0; i < sim[v].length; i++) {
        maskSim.push(sim[v][i] >= minVal && sim["eventId"][i] === eid);
        }
        for (let i = 0; i < rec[v].length; i++) {
        maskRec.push(rec[v][i] >= minVal && rec["eventId"][i] === eid);
        }
        
        threshSim.booleans = maskSim;
        threshReco.booleans = maskRec;

        srcSim.change.emit();
        srcReco.change.emit();
        srcClSim.change.emit();
        srcClReco.change.emit();
        viewEvSim.change.emit();
        viewEvReco.change.emit();
        viewIdSim.change.emit();
        viewIdReco.change.emit();
        viewClSim.change.emit();
        viewClReco.change.emit();
        """
    else:
        numInput_code = """
        const eid = select.value.toString();
        viewEvSim.filters[0].group = eid;
        viewEvReco.filters[0].group = eid;
        viewClSim.filters[0].group = eid;
        viewClReco.filters[0].group = eid;

        srcSim.change.emit();
        srcReco.change.emit();
        srcClSim.change.emit();
        srcClReco.change.emit();
        viewEvSim.change.emit();
        viewEvReco.change.emit();
        viewClSim.change.emit();
        viewClReco.change.emit();
        """
    
    numInput_callb = CustomJS(args=numInput_args, code=numInput_code)
    numInput.js_on_change("value", numInput_callb)

    title_template_one = f"Only one cluster available."
    title_template_more = "Cluster ID selection (enter a number between {} and {}):"
    if clids_in_df:
        dfClIdSimMin, dfClIdSimMax = df['Sim'].clids.min(), df['Sim'].clids.max()
        if dfClIdSimMin == dfClIdSimMax:
            title = title_template_one
        else:
            title = title_template_more.format(dfClIdSimMin, dfClIdSimMax)
        clIdInputSim = NumericInput(value=None, low=int(dfClIdSimMin), high=int(dfClIdSimMax),
                                    title=title)
        dfClIdRecoMin, dfClIdRecoMax = df['Reco'].clids.min(), df['Reco'].clids.max()
        if dfClIdRecoMin == dfClIdRecoMax:
            title = title_template_one
        else:
            title = title_template_more.format(dfClIdRecoMin, dfClIdRecoMax)
        clIdInputReco = NumericInput(value=None, low=int(dfClIdRecoMin), high=int(dfClIdRecoMax),
                                     title=title)

        clIdInput_code = """
        const eid = selectEvent.value.toString();
        const clid = selectId.value.toString();
        view.filters[0].group = eid;

        const clids = src.data.clids;
        const eventIds = src.data.eventId;
        const mask = clids.map((cid, i) => eventIds[i] === eid && cid === clid);
        view.filters[1].booleans = mask;
        
        src.change.emit();
        view.change.emit();
        """
        clIdInputSim_callb = CustomJS(args=dict(
            src=src['Sim'], view=viewId['Sim'],
            selectEvent=numInput, selectId=clIdInputSim,
        ), code=clIdInput_code)
        clIdInputSim.js_on_change("value", clIdInputSim_callb)

        clIdInputReco_callb = CustomJS(args=dict(
            src=src['Reco'], view=viewId['Reco'],
            selectEvent=numInput, selectId=clIdInputReco,
        ), code=clIdInput_code)
        clIdInputReco.js_on_change("value", clIdInputReco_callb)

        showAllButtonSim = Button(label="Show all clusters", button_type="success", width=150)
        showAllButtonReco = Button(label="Show all clusters", button_type="success", width=150)

        showAll_code = """
        selectId.value = NaN;
        const eid = selectEvent.value.toString();
        const eventIds = src.data.eventId;
        const mask = eventIds.map((evid) => evid === eid);
        view.filters[1].booleans = mask;
        src.change.emit();
        view.change.emit();
        """    
        showAllButtonSim.js_on_click(CustomJS(
            args=dict(view=viewId['Sim'], src=src['Sim'], selectEvent=numInput, selectId=clIdInputSim),
            code=showAll_code
        ))
        showAllButtonReco.js_on_click(CustomJS(
            args=dict(view=viewId['Reco'], src=src['Reco'], selectEvent=numInput, selectId=clIdInputReco),
            code=showAll_code
        ))

    menuVar_tuple = (('Energy [GeV]', 'energies_sum'),)
    if fracs_in_df:
        menuVar_tuple += (('Fraction', 'fracs_sum'),)
    menuVar = [*menuVar_tuple]
    dropVar = Dropdown(label="Z axis", button_type="warning", menu=menuVar, width=100)
    
    slider_calb = CustomJS(
        args=dict(
            srcSim=src["Sim"], srcReco=src["Reco"],
            viewSim=view["Sim"], viewReco=view["Reco"],
            threshSim=threshFilter["Sim"], threshReco=threshFilter["Reco"],
            slider=slider, select=numInput,
            varNameHolder=varNameHolder,
        ),
        code="""
        const minVal = slider.value;
        const eid = select.value.toString();
        const v = varNameHolder.data['value'][0];
        
        const sim = srcSim.data;
        const rec = srcReco.data;
        
        let maskSim = [];
        let maskRec = [];
        
        for (let i = 0; i < sim[v].length; i++) {
        maskSim.push(sim[v][i] >= minVal && sim["eventId"][i] === eid);
        }
        for (let i = 0; i < rec[v].length; i++) {
        maskRec.push(rec[v][i] >= minVal && rec["eventId"][i] === eid);
        }
        
        threshSim.booleans = maskSim;
        threshReco.booleans = maskRec;
        
        srcSim.change.emit();
        srcReco.change.emit();
        """
    )
    slider.js_on_change("value", slider_calb)

    mapperSim = {'energies': mapper_log['Sim'],'energies_sum': mapper_log['Sim']}
    mapperReco = {'energies': mapper_log['Reco'],'energies_sum': mapper_log['Reco']}
    maxVals = {'energies': 1.5, 'energies_sum': enSumMax}
    if fracs_in_df:
        mapperSim.update({'fracs': mapper_lin['Sim'],'fracs_sum': mapper_lin['Sim']})
        mapperReco.update({'fracs': mapper_lin['Reco'],'fracs_sum': mapper_lin['Reco']})
        maxVals.update({'fracs': 1.,'fracs_sum': 1.})
        
    dropVar_calb = CustomJS(
        args=dict(
            srcSim=src['Sim'], srcReco=src['Reco'],
            patchSim=p['Sim'][0].renderers[0], patchReco=p['Reco'][0].renderers[0],
            mapperSim=mapperSim, mapperReco=mapperReco,
            maxVals=maxVals,
            colorBarSim=color_bar['Sim'], colorBarReco=color_bar['Reco'],
            slider=slider,
            slider_callback=slider_calb,
            varNameHolder=varNameHolder,
        ),
        code="""
        const varName = this.item;
        // Update color mapper range
        colorBarSim.color_mapper = mapperSim[varName];
        colorBarSim.color_mapper.low = Math.min(...srcSim.data[varName]);
        colorBarSim.color_mapper.high = Math.max(...srcSim.data[varName]);
        colorBarReco.color_mapper = mapperReco[varName];
        colorBarReco.color_mapper.low = Math.min(...srcReco.data[varName]);
        colorBarReco.color_mapper.high = Math.max(...srcReco.data[varName]);
        // Update patch fill_color field
        patchSim.glyph.fill_color.field = varName;
        patchReco.glyph.fill_color.field = varName;
        // Update slider range and value
        slider.start = 0.;
        slider.end = maxVals[varName];
        slider.value = slider.start;
        slider.step = (slider.end - slider.start) / 500.;
        slider.title = "Min threshold for " + varName;
        // Update the varName in the slider callback
        varNameHolder.data['value'][0] = varName;
        // Update the sources
        srcSim.change.emit();
        srcReco.change.emit();
        varNameHolder.change.emit();
        """
    )
    dropVar.js_on_event("menu_item_click", dropVar_calb)

    lay = [[writeIntructions()],
           [numInput, Div(text='', width=40, height=1), slider],
           [dropVar,],
           [p['Sim'][0], p['Reco'][0]]]

    if clids_in_df:
        lay.append([Div(text='', width=30, height=1), clIdInputSim, showAllButtonSim,
                    Div(text='', width=650, height=1), clIdInputReco, showAllButtonReco])
        lay.append([p['Sim'][1], p['Reco'][1]])

    lay.append([writeContacts()])
    
    save(layout(lay))
    print(f"INFO: Event plot saved to {output_path}")

def showECAL(infile, outdir, props, outname='EventDisplay'):
    utils.createDir(outdir)
    parentDir = os.path.dirname(outdir)
    if outdir[-1] == '/':
        parentDir = os.path.dirname(parentDir)
    utils.createIndexPHP(src=parentDir, dest=outdir)

    varsGeom = [
        "crystalDetId",
        "crystalCenterEta",
        "crystalCenterPhi",
        "crystalCorner0Eta",
        "crystalCorner1Eta",
        "crystalCorner2Eta",
        "crystalCorner3Eta",
        "crystalCorner0Phi",
        "crystalCorner1Phi",
        "crystalCorner2Phi",
        "crystalCorner3Phi",
    ]
    varsEventCommon = ["eventId"]
    varsEvent = {"Reco": [], "Sim": []}
    for pfix in ("Reco", "Sim"):
        varsEvent[pfix].extend(
            [x + pfix for x in (
                "energies",
                "detids",
                "nHits",
                "clusterEnergies",
                "clusterEtas",
                "clusterPhis",
                "clusterHitEnergies",
                "clusterHitFractions",
                "clusterHitClids",
                "clusterHitDetids",
            )]
        )
    varsEventAll = varsEventCommon + varsEvent["Reco"] + varsEvent["Sim"]

    with uproot.open(infile) as file:
        dfGeom = file["ecalGeometryAnalyzer/Geometry"].arrays(varsGeom, library="pandas")
        if not props.geom:
            dfEvent = file["ecalGeometryAnalyzer/Event"].arrays(varsEventAll, entry_stop=props.nevents, library="awkward")

    if props.geom:
        plotGeom(dfGeom, output_path=os.path.join(outdir, "geom.html"))
        return

    dfHits, dfClusters, dfHitsInClusters = ({} for _ in range(3))
    for pfix in ("Reco", "Sim"):
        dfHits[pfix] = ak.to_dataframe(
            dfEvent[["eventId", "energies"+pfix, "detids"+pfix]]
        ).rename(columns={'energies'+pfix: 'energies', 'detids'+pfix: 'detids'})
        dfClusters[pfix] = ak.to_dataframe(dfEvent[["eventId", "clusterEnergies"+pfix, "clusterEtas"+pfix, "clusterPhis"+pfix]])
        dfHitsInClusters[pfix] = ak.to_dataframe(
            dfEvent[["eventId", "clusterHitEnergies"+pfix, "clusterHitDetids"+pfix, "clusterHitFractions"+pfix, "clusterHitClids"+pfix]]
        ).rename(columns={"clusterHitEnergies"+pfix: 'energies', "clusterHitDetids"+pfix: 'detids',
                          "clusterHitFractions"+pfix: 'fracs', "clusterHitClids"+pfix: 'clids'})

    plotEvent(
        dfGeom,
        dfHitsInClusters,
        dfClusters,
        output_path=os.path.join(outdir, outname + "_clusterHits.html"),
    )

    if props.allhits:
        plotEvent(
            dfGeom,
            dfHits, 
            dfClusters,
            output_path=os.path.join(outdir, outname + "_allHits.html"),
        )

    print("INFO: Done.")

@dataclass
class InputArgs:
    nevents: int
    geom: bool = False
    clusters: bool = False
    allhits: bool = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show position of crystals.")
    parser.add_argument("-i", "--file", help="Path to the input ROOT file.")
    parser.add_argument("-o", "--outdir", help="Path to the output folder where the script outputs will be stored.")
    parser.add_argument("--outname", default='EventDisplay', help="Name of the output html file with the event display.")
    parser.add_argument("-n", "--nevents", help="Number of events to plot. If the input file has less events, then it plots all events of the input file.", default=10, type=int)
    parser.add_argument("-c", "--clusters", help="Add cluster information.", default=False, action='store_true')
    geom_help_str = "Plot only the geometry. It highlights the position of the center and corners of each ECAL crystal."
    parser.add_argument("-g", "--geom", help=geom_help_str, default=False, action='store_true')
    all_hits_str = "On top of the clustered hits, add an identical visualization with all hits."
    parser.add_argument("-a", "--allhits", help=all_hits_str, default=False, action='store_true')

    args = parser.parse_args()
    props = InputArgs(nevents=args.nevents, geom=args.geom, clusters=args.clusters, allhits=args.allhits)
    showECAL(args.file, args.outdir, props, args.outname)
