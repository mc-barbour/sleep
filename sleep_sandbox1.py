#%%
import importlib
# Replace with the actual module name

from edf_sleep_data import *

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

annotations = study.annotations

events = annotations[2]
unique_events = np.unique(events)
# %%
study.get_unique_annotation_types()
study.get_annotation_summary()
# %%
R_sleep = study.get_annotations_by_type("Sleep stage R")
nonREM = study.get_annotations_by_type('Sleep stage N2')
EEGwake = study.get_annotations_by_type("EEG arousal")
hypopneas = study.get_annotations_by_type("Hypopnea")
cent_apneas = study.get_annotations_by_type("Central Apnea")
# %%
nonREM_sleep_windows = []
start = nonREM[0]['onset']
window_start = start  
sleep_window_length = 0
for count in range(len(nonREM) - 1):

    next_start = start + nonREM[count]['duration']
    if nonREM[count+1]['onset'] == next_start:
        sleep_window_length += 30

    else:
        print('break', sleep_window_length)
        window = {"onset": window_start,
                  "duration": sleep_window_length}
        nonREM_sleep_windows.append(window)
        sleep_window_length = 0
        window_start = nonREM[count+1]['onset']
    start = nonREM[count+1]['onset']


# %% find windows with hyponea and central app in the sleep windows
found_hypopneas = []
found_centapneas = []
valid_sleep_windows = []
for window in nonREM_sleep_windows:
    start = window["onset"]
    end = window["duration"] + start

    for hyp in hypopneas:
        if (hyp['onset'] >= start) and  (hyp['onset'] <= end):
            print('found', hyp, window)
            found_hypopneas.append(hyp)
            valid_sleep_windows.append(window)
    
    for cent in cent_apneas:
        if (cent['onset'] >= start) and  (cent['onset'] <= end):
            print('found', cent, window)
            found_centapneas.append(hyp)
            valid_sleep_windows.append(window)




# %% Let's plot the windows and make sure hypopnea are in there
for window in valid_sleep_windows:
    fig = study.plot_signals(
        signal_names=['flow', 'pflow', 'resp_chest', 'resp_abdomen'],
        start_index=int(window['onset'] * study.sample_rate),  # 10 minutes
        length=int(window['duration'] * study.sample_rate),            # 30 seconds
        title="REM window w/ apnea",
        height=1200,
        annotation_types=['Hypopnea', 'Central Apnea']
    )
    fig.show()

# %% let's zoom in on a good sample
from scipy.integrate import trapz
from scipy.signal import find_peaks
start = int(valid_sleep_windows[1]['onset'] * study.sample_rate)
end = start + int(valid_sleep_windows[1]['duration'] * study.sample_rate)

estimated_RR = 3 #sec
min_samples = 0.9*estimated_RR * study.sample_rate

flow = study.pflow[start:end]
flow_grad = np.gradient(flow)

time = np.arange(len(flow)) / study.sample_rate

peaks = find_peaks(flow, distance=min_samples)

breath_minute_volume_array = []
breath_plot_index = []
breath_time = []
for count in range(len(peaks[0])-1):
    s1 = peaks[0][count]
    s2 = peaks[0][count+1]
    breath_plot_index.append(s1 + (s2-s1) / 2)
    breath_time.append((s1 + (s2-s1) / 2 ) / study.sample_rate)
    breath = flow[s1:s2]
    insp = breath[np.where(breath <= 0)[0]]
    exp = breath[np.where(breath >= 0)[0]]

    insp_vol = np.trapz(insp, dx = 1/study.sample_rate)
    exp_vol = np.trapz(exp, dx = 1/study.sample_rate)
    breath_duration = (s2 - s1) / study.sample_rate
    breath_minute_volume = -1 * insp_vol / breath_duration # arbitray volume / s
    breath_minute_volume_array.append(breath_minute_volume)


def func_vchem(time, minute_vent, tau, LG, sigma):
    return -LG /(1 + time * tau) * np.exp(-time * sigma) * minute_vent


vchem = func_vchem(np.array(breath_time), np.array(breath_minute_volume_array), tau=1, LG=1, sigma=0.2 )


fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=time, y=flow), row=1, col=1)
fig.add_trace(go.Scatter(x=time[peaks[0]],y=flow[peaks[0]], mode='markers', marker_color='red'))
fig.add_trace(go.Scatter(x=breath_time, y=breath_minute_volume_array), row=2, col=1)
fig.add_trace(go.Scatter(x=breath_time, y=vchem), row=2, col=1)

fig.update_xaxes(range = (0,400))
fig.show()




# %%
