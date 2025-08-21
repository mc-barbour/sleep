#%%
import pandas as pd
import numpy as np
import pyedflib
from pathlib import Path
import datetime


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


#%%
def edf_to_df(f, start=0, n_samples=1000):
    """
    Read edf and return edf object and df
"""
    
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    # sigbufs = np.zeros((n, f.getNSamples()[0]))
    sigbufs = np.zeros((n, n_samples-start))
    freq = f.getSampleFrequency(1)
    	
    for i in np.arange(n):
        sigbufs[i,:] = f.readSignal(i,start,n_samples)
        print("Loading: ", signal_labels[i])
        
    
    df = pd.DataFrame(data = sigbufs.T, columns=signal_labels)
    
    return df
    
#%%
f = pyedflib.EdfReader(filename)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))

for i in range(n):
    print(f.getNSamples()[i])




#%% Load the edf file _ will need to close if reloading f.close()


#filename = '/Volumes/files/RadResearch/Projects/SLEEP_STUDY_DATA/AIRWAY_MAIDA_20JUN23/RESEARCH~ ROOM_a7173d7d-93ff-4f23-aaa9-3573bb41eeb4.edf'
filename = '/Volumes/files/RadResearch/Projects/SLEEP_STUDY_DATA/Weston_sleepEDF_test.edf'
filename = 'Z:/Projects/SLEEP_STUDY_DATA/Weston_sleepEDF_test.edf'
# filename = '/Volumes/files/RadResearch/Projects/SLEEP_STUDY_DATA/EDF1.EDF'
#filename = '/Volumes/Active/powell_w/PSG EDF Exports/RCResp_2024_07_19.EDF'
#filename = '/Users/mbarb1/Desktop/PRS_SJ_2024_11_05.EDF'
filename = 'Y:/powell_w/PSG EDF Exports/RCResp_2024_07_19.EDF'

f = pyedflib.EdfReader(filename)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
freq = int(f.getSampleFrequency(1))
#%%
total_samples = f.samples_in_file(1)
#total_samples = int(2e5)

df = edf_to_df(f, start=0, n_samples=total_samples)
print(df.columns)
#%%
fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Scatter(y = df['Resp Chest'], name = 'Resp Chest'), row=1, col=1)
fig.add_trace(go.Scatter(y = df['Resp Abdomen'], name = 'Resp Abd'), row=2, col=1)
fig.add_trace(go.Scatter(y = df['PFLOW'], name = 'PFlow'), row=3, col=1)
fig.add_trace(go.Scatter(y = df['Resp Thermocan+'], name = 'thermocan'), row=4, col=1)

fig.show()


#%%

time = np.linspace(0,len(df)/int(freq), len(df))

minutes = 2
plot_index = int(minutes*freq*60)

fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Scatter(y = df['Resp Chest'][0:plot_index],x =time[0:plot_index], name = 'Resp Chest'), row=1, col=1)
fig.add_trace(go.Scatter(y = df['Resp Abdomen'][0:plot_index], x=time[0:plot_index], name = 'Resp Abd'), row=2, col=1)
fig.add_trace(go.Scatter(y = df['Resp Thermocan+'][0:plot_index],x=time[0:plot_index], name = 'Pressure'), row=3, col=1)

fig.show()

#%%

fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Scatter(y = df['Resp Chest'][0:plot_index],x =time[0:plot_index], name = 'Resp Chest'), row=1, col=1)
fig.add_trace(go.Scatter(y = df['Resp Abdomen'][0:plot_index], x=time[0:plot_index], name = 'Resp Abd'), row=2, col=1)

fig.add_trace(go.Scatter(y = np.gradient(df['Resp Abdomen'][0:plot_index]), x=time[0:plot_index], name = 'Resp Abd'), row=3, col=1)
fig.add_trace(go.Scatter(y = df['Resp Thermocan+'][0:plot_index],x=time[0:plot_index], name = 'Pressure'), row=4, col=1)

fig.show()

#%% let's find the right section of the exam


start = f.getStartdatetime()

target_time = datetime.datetime(start.year, start.month, start.day, 23, 32, 17 , 0)

delta = target_time - start

start_int = int(freq * delta.seconds) - 20000
minutes = 5
plot_index = start_int + int(minutes*freq*60)
# plot_index=int(8720*freq)

fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Scatter(y = -1*df['Resp Thermocan+'][start_int:plot_index],x=time[start_int:plot_index], name = 'Thermocan'), row=1, col=1)
fig.add_trace(go.Scatter(y = df['XSum'][start_int:plot_index] * 10000,x=time[start_int:plot_index], name = 'SUM'), row=2, col=1)
fig.add_trace(go.Scatter(y = df['Resp Chest'][start_int:plot_index] + df['Resp Abdomen'][start_int:plot_index],x=time[start_int:plot_index], name = 'Actual Sum'), row=2, col=1)

fig.add_trace(go.Scatter(y = df['Resp Chest'][start_int:plot_index],x =time[start_int:plot_index], name = 'Resp Chest'), row=3, col=1)
fig.add_trace(go.Scatter(y = df['Resp Abdomen'][start_int:plot_index], x=time[start_int:plot_index], name = 'Resp Abd'), row=4, col=1)

# fig.add_trace(go.Scatter(y = np.gradient(df['Resp Abdomen'][start_int:plot_index]), x=time[start_int:plot_index], name = 'Resp Abd'), row=4, col=1)
# fig.add_trace(go.Scatter(y = df['Resp Thermocan+'][start_int:plot_index],x=time[start_int:plot_index], name = 'Pressure'), row=4, col=1)

fig.show()

#%% try and find hypopnia events


annotations = f.readAnnotations()

events = annotations[2]
unique_events = np.unique(events)
event_time = annotations[0]
event_dur = annotations[1]
hypop_int = np.where(events == 'Hypopnea')
normal_breathing = np.where(events == 'Normal Breathing')
periodic_breathing = np.where(events == 'Periodic Breathing')
central_apnea = np.where(events == 'Central Apnea')

sleep_stageN1 = np.where(events == 'Sleep stage N1')
sleep_stageN2 = np.where(events == 'Sleep stage N2')
sleep_stageN3 = np.where(events == 'Sleep stage N3')

sleep_stageR = np.where(events == 'Sleep stage R')
sleep_stageW = np.where(events == 'Sleep stage W')

for ind in hypop_int:
    print(event_time[hypop_int] * freq - start_int)

#%% print sleep stage durations

for count, index in enumerate(sleep_stageN3[0]):
    print(event_dur[index])
    


#%%
fig = make_subplots(rows=3, cols=1)

fig.add_trace(go.Scatter(y = df['Resp Thermocan+'][start_int:plot_index], name = 'Thermocan'), row=1, col=1)

fig.add_trace(go.Scatter(y = df['PFLOW'][start_int:plot_index], name = 'PFLOW'), row=2, col=1)
fig.add_trace(go.Scatter(y = df['Resp Abdomen'][start_int:plot_index], name = 'Resp Abd'), row=3, col=1)

# fig.add_trace(go.Scatter(y = np.gradient(df['Resp Abdomen'][start_int:plot_index]), x=time[start_int:plot_index], name = 'Resp Abd'), row=4, col=1)
# fig.add_trace(go.Scatter(y = df['Resp Thermocan+'][start_int:plot_index],x=time[start_int:plot_index], name = 'Pressure'), row=4, col=1)

fig.show()

#%%

fig = go.Figure()
for label in signal_labels:
    fig.add_trace(go.Scatter(y=df[label][start_int:plot_index],x=time[start_int:plot_index], name=label))

fig.show()




#%% create plots around specific annotation events


event_index = hypop_int[0][5]
event_index = central_apnea[0][7]
event_time_start = event_time[event_index]

event_duration = event_dur[event_index]

start_int = int(freq * event_time_start) - 5000
minutes = 3
plot_index = start_int + int(minutes*freq*60)

fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Scatter(y = -1*df['Resp Thermocan+'][start_int:plot_index],x=time[start_int:plot_index], name = 'Thermocan'), row=1, col=1)
fig.add_trace(go.Scatter(y = df['PFLOW'][start_int:plot_index] * 10000,x=time[start_int:plot_index], name = 'PFlow'), row=2, col=1)
# fig.add_trace(go.Scatter(y = df['Resp Chest'][start_int:plot_index] + df['Resp Abdomen'][start_int:plot_index],x=time[start_int:plot_index], name = 'Actual Sum'), row=2, col=1)

fig.add_trace(go.Scatter(y = df['Resp Chest'][start_int:plot_index],x =time[start_int:plot_index], name = 'Resp Chest'), row=3, col=1)
fig.add_trace(go.Scatter(y = df['Resp Abdomen'][start_int:plot_index], x=time[start_int:plot_index], name = 'Resp Abd'), row=4, col=1)


if event_duration == '-1':
    # fig.add_vline(x=time[start_int], row='all')
    fig.add_vline(x=event_time_start, row='all')

else:
    # fig.add_vline(x=time[start_int], row='all')
    # fig.add_vline(x=time[start_int + int(event_duration * freq)], row='all')
    fig.add_vline(x=event_time_start, row='all')
    fig.add_vline(x=event_time_start + event_duration, row='all')
#     fig.add_shape(
#     type="rect",
#     x0=event_time_start,
#     x1=event_time_start + event_duration,
#     y0=-5000,
#     y1=5000,
#     fillcolor="rgba(255, 0, 0, 0.2)", # Use RGBA for transparency
#     line=dict(width=0) # Remove border
# )

fig.update_xaxes(title='Time (s)')
# fig.add_trace(go.Scatter(y = np.gradient(df['Resp Abdomen'][start_int:plot_index]), x=time[start_int:plot_index], name = 'Resp Abd'), row=4, col=1)
# fig.add_trace(go.Scatter(y = df['Resp Thermocan+'][start_int:plot_index],x=time[start_int:plot_index], name = 'Pressure'), row=4, col=1)

fig.show()








#%% Pulling in Robin patient - want to find sleep states

annotations = f.readAnnotations()
event_time = annotations[0]
event_dur = annotations[1]
events = annotations[2]
unique_events = np.unique(events)
print(unique_events)


hypop_int = np.where(events == 'Hypopnea')
normal_breathing = np.where(events == 'Normal Breathing')
periodic_breathing = np.where(events == 'Periodic Breathing')
central_apnea = np.where(events == 'Central Apnea')

sleep_stageN1 = np.where(events == 'Sleep stage N1')
sleep_stageN2 = np.where(events == 'Sleep stage N2')
sleep_stageN3 = np.where(events == 'Sleep stage N3')

sleep_stageR = np.where(events == 'Sleep stage R')
sleep_stageW = np.where(events == 'Sleep stage W')

supine = np.where(events == 'Body Position: Supine')
left = np.where(events == 'Body Position: Left')
right= np.where(events == 'Body Position: Right')

#%%
time = np.linspace(0,len(df)/int(freq), len(df))
i = 274
start_ind = 50000 * i
end_ind = 50000 * (i+1)
fig = make_subplots(rows=4, cols=1)

fig.add_trace(go.Scatter(x=time[start_ind:end_ind], y = df['Resp DyMedix+'][start_ind:end_ind], name = 'PLeth'), row=1, col=1)
fig.add_trace(go.Scatter(x=time[start_ind:end_ind], y = df['PFLOW'][start_ind:end_ind], name = 'PFLOW'), row=2, col=1)
fig.add_trace(go.Scatter(x=time[start_ind:end_ind], y = df['Resp Abd'][start_ind:end_ind], name = 'Resp Abd'), row=3, col=1)
fig.add_trace(go.Scatter(x=time[start_ind:end_ind], y = df['Resp Chest'][start_ind:end_ind], name = 'Resp Chest'), row=4, col=1)


time_start = time[start_ind]
time_end = time[end_ind]
events_time_band = np.where((event_time < time_end) & (event_time > time_start))[0]
annotations_time_band = events[events_time_band[0]:events_time_band[-1]+1]
print(annotations_time_band, events_time_band)

fig.show()


