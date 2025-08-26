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
        annotation_types=['Hypopnea', 'Central Apnea', 'EEG arousal']
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
breath_durations = []
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
    breath_durations.append(breath_duration)

minute_ventilation_norm = np.array(breath_minute_volume_array) - np.mean(breath_minute_volume_array)



def compute_vchem(minute_vent, breath_duration, tau, LG, sigma, vchem0):
    # still need to add delay function

    vchem = np.zeros_like(minute_vent)
    vchem[0] = vchem0                                                                                                                                                                                                                                                                                                                 
    
    for i in range(len(minute_vent) - 1):
        alpha = ((tau / breath_duration[i]) / (1 + tau / breath_duration[i]) + 1 - (breath_duration[i] / tau ) )
        beta = -LG * (1 / (1 + tau / breath_duration[i]) + 1 / (tau / breath_duration[i]))
        print(alpha, beta)
        vchem[i+1] = alpha / 2 * vchem[i] + beta / 2 * minute_vent[i+1]
    return vchem
        



def func_vchem(time, minute_vent, tau, LG, sigma):
    return -LG /(1 + time * tau) * np.exp(-time * sigma) * minute_vent


# vchem = func_vchem(np.array(breath_time), np.array(breath_minute_volume_array), tau=1, LG=1, sigma=1.0 )
vchem = compute_vchem(minute_ventilation_norm, breath_durations, tau=4, LG=0.5, sigma=1, vchem0=breath_minute_volume_array[0]*0.9)

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=time, y=flow), row=1, col=1)
fig.add_trace(go.Scatter(x=time[peaks[0]],y=flow[peaks[0]], mode='markers', marker_color='red'))
fig.add_trace(go.Scatter(x=breath_time, y=minute_ventilation_norm), row=2, col=1)
fig.add_trace(go.Scatter(x=breath_time, y=vchem), row=2, col=1)

fig.update_xaxes(range = (0,400))
fig.show()




# %% create idealized waveform

apnea_events = [
        (15, 12),   # 12-second apnea at 15s
        (40, 18),   # 18-second apnea at 40s
        (75, 15),   # 15-second apnea at 75s
        (105, 10),  # 10-second apnea at 105s
    ]
    
time, signal = generate_apnea_waveform(
        duration=120,
        sample_rate=50,
        breathing_frequency=0.7,  \
        apnea_events=apnea_events,
        pre_apnea_decline_time=4.0,
        post_apnea_recovery_time=6.0,
        recovery_overshoot=1.8,
        noise_level=0.08
    )

fig = go.Figure()
fig.add_trace(go.Scatter(x=time, y=signal))
fig.show()

# %%
importlib.reload(loop_gain)



        

minute_ventilation_norm, breath_durations, breath_time = loop_gain.compute_minute_ventilation(signal, time, estimated_RR=1.4, sample_rate=50 )
# delayed_minute_vent = minute_vent_delay(minute_ventilation_norm, breath_time, sigma=6)

vchem, start_idx = loop_gain.compute_vchem(minute_ventilation_norm, breath_durations, breath_time, tau=6, LG=1.2, sigma=10, vchem0=minute_ventilation_norm[0]*0.9)
fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=time, y=signal), row=1, col=1)
fig.add_trace(go.Scatter(x=breath_time, y=minute_ventilation_norm), row=2, col=1)
fig.add_trace(go.Scatter(x=breath_time[start_idx::], y=vchem), row=2, col=1)
# fig.add_trace(go.Scatter(x=breath_time, y=delayed_minute_vent), row=2, col=1)




fig.show()

# %%
