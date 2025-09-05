from scipy.integrate import trapz
from scipy.signal import find_peaks
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def linear_interpolate(x1, y1, x2, y2, x_interp):
    y_interp = y1 + (x_interp - x1) * (y2 - y1) / (x2 - x1)
    return y_interp




def compute_minute_ventilation(flow_signal, time, estimated_RR, sample_rate, plot_peaks=True):
    
    min_samples = 0.9*estimated_RR * sample_rate

    peaks = find_peaks(flow_signal, distance=min_samples)

    breath_minute_volume_array = []
    breath_plot_index = []
    breath_time = []
    breath_durations = []
    for count in range(len(peaks[0])-1):
        s1 = peaks[0][count]
        s2 = peaks[0][count+1]
        breath_plot_index.append(s1 + (s2-s1) / 2)
        breath_time.append((s1 + (s2-s1) / 2 ) / sample_rate)
        breath = flow_signal[s1:s2]
        insp = breath[np.where(breath <= 0)[0]]
        exp = breath[np.where(breath >= 0)[0]]

        insp_vol = np.trapz(insp, dx = 1 / sample_rate)
        exp_vol = np.trapz(exp, dx = 1 / sample_rate)
        breath_duration = (s2 - s1) / sample_rate
        breath_minute_volume = -1 * insp_vol / breath_duration # arbitray volume / s
        breath_minute_volume_array.append(breath_minute_volume)
        breath_durations.append(breath_duration)

    minute_ventilation_norm = np.array(breath_minute_volume_array) / np.mean(breath_minute_volume_array)

    if plot_peaks:
        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scatter(x=time, y=flow_signal), row=1, col=1)
        fig.add_trace(go.Scatter(x=time[peaks[0]],y=flow_signal[peaks[0]], mode='markers', marker_color='red'))
        fig.add_trace(go.Scatter(x=breath_time, y=minute_ventilation_norm), row=2, col=1)
        # fig.add_trace(go.Scatter(x=breath_time, y=vchem), row=2, col=1)

        fig.show()

    return np.array(minute_ventilation_norm), np.array(breath_durations), np.array(breath_time)

def compute_vent_delay(minute_vent, breath_time, sigma):
    # breath_time = np.array(breath_time)
    first_breath_idx = np.where(breath_time-sigma > 0)[0][1]
    
    # delayed_minute_volume = np.zeros_like(minute_vent[first_breath_idx::])
    delayed_minute_vent = []
    for n in range(first_breath_idx,len(minute_vent)):
        delay_time = breath_time[n] - sigma
        if delay_time < 0:
            print("AHHHH")

        diffs = np.abs(breath_time - delay_time)

        # Get the indices that would sort the differences
        sorted_indices = np.argsort(diffs)

        # The first two indices in sorted_indices correspond to the two closest values
        closest_indices = sorted_indices[:2]

        delayed_minute_vent.append(linear_interpolate(breath_time[closest_indices[0]], minute_vent[closest_indices[0]], breath_time[closest_indices[1]], minute_vent[closest_indices[1]], delay_time))

    return delayed_minute_vent, first_breath_idx

def compute_vchem(minute_vent, breath_duration, breath_time, tau, LG, sigma, vchem0):
    
    delayed_minute_vent, first_breath_idx = compute_vent_delay(minute_vent, breath_time, sigma)

    vchem = np.zeros_like(minute_vent[first_breath_idx::])
    vchem[0] = vchem0                                                                                                                                                                                                                                                                                                                 
    
    for i in range(len(minute_vent[first_breath_idx::]) - 1):
        alpha = ((tau / breath_duration[i]) / (1 + tau / breath_duration[i]) + 1 - (breath_duration[i] / tau ) )
        beta = -LG * (1 / (1 + tau / breath_duration[i]) + 1 / (tau / breath_duration[i]))
        print(alpha, beta)
        vchem[i+1] = alpha / 2 * vchem[i] + beta / 2 * delayed_minute_vent[i+1]
    return vchem, first_breath_idx
        