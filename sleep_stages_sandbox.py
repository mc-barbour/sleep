#%%
import importlib
# Replace with the actual module name

from edf_sleep_data import *
from resp_waveforms import *
import loop_gain

# importlib.reload(edf_sleep_data)

#%%
filename = 'Y:/powell_w/PSG EDF Exports/RCResp_2024_07_19.EDF'
study = EDFSleepStudy.from_edf_file(filename)
print(study.summary())
# 
# # Access all respiratory signals
respiratory_data = study.get_respiratory_signals()
# 
# # Access only the required respiratory signals
required_signals = study.get_required_respiratory_signals()
# 
# # Get apnea events
apnea_events = study.get_apnea_events()
print(apnea_events)

fig = study.plot_signals(
    signal_names=['flow', 'pflow', 'resp_chest', 'resp_abdomen'],
    start_index=int(10 * 60 * study.sample_rate),  # 10 minutes
    length=int(600 * study.sample_rate),            # 30 seconds
    title="Respiratory Signals - 30 second window",
    show_annotations=True
)
fig.show()
# %%
# %%
study.get_unique_annotation_types()
study.get_annotation_summary()
# %%
R = study.get_annotations_by_type("Sleep stage R")
N1 = study.get_annotations_by_type("Sleep stage N1")
N2 = study.get_annotations_by_type("Sleep stage N2")
N3 = study.get_annotations_by_type("Sleep stage N3")
W = study.get_annotations_by_type("Sleep stage W")
# %%

sleep_stages = R + N1 + N2 + N3 + W
sorted_sleep_epochs = sorted(sleep_stages, key=lambda x: x['onset'])
# %%

stage_indexed = []
for epoch in sorted_sleep_epochs:
    if epoch["description"] == "Sleep stage W":
        epoch = 5
    elif epoch["description"] == "Sleep stage R":
        epoch = 4
    elif epoch["description"] == "Sleep stage N1":
        epoch = 3
    elif epoch["description"] == "Sleep stage N2":
        epoch = 2
    elif epoch["description"] == "Sleep stage N3":
        epoch = 1
    else:
        print("Warning - no sleep clasification")
    stage_indexed.append(epoch)

# %%

fig = go.Figure()
fig.add_trace(go.Scatter(y=stage_indexed, line_width=3,marker_color='firebrick'))
fig.update_xaxes(title="Epochs", showline=True, mirror=True)
fig.update_yaxes(title="Sleep Stage",  showline=True, mirror=True)
fig.update_layout(font_size=16, font_family='serif', template='plotly_white', yaxis=dict(
        tickmode="array",
        tickvals=[1, 2, 3, 4, 5],
        ticktext=["N3", "N2", "N1", "R", "W"]
        
    )
)
fig.show()

# %%
