#%%
from edf_sleep_data import *


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
hypopneas = study.get_annotations_by_type("Hypopnea")
cent_apneas = study.get_annotations_by_type("Central Apnea")
# %%
R_sleep_windows = []
start = R_sleep[0]['onset']
window_start = start  
sleep_window_length = 0
for count in range(len(R_sleep) - 1):

    next_start = start + R_sleep[count]['duration']
    if R_sleep[count+1]['onset'] == next_start:
        sleep_window_length += 30

    else:
        print('break', sleep_window_length)
        window = {"onset": window_start,
                  "duration": sleep_window_length}
        R_sleep_windows.append(window)
        sleep_window_length = 0
        window_start = R_sleep[count+1]['onset']
    start = R_sleep[count+1]['onset']


# %%
found_hypopneas = []
found_centapneas = []
valid_sleep_windows = []
for window in R_sleep_windows:
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




# %%
