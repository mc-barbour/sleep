import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def generate_apnea_waveform(
    duration: float = 120,  # Total duration in seconds
    sample_rate: float = 100,  # Samples per second
    breathing_frequency: float = 0.25,  # Breaths per second (15 breaths/min)
    normal_amplitude: float = 1.0,  # Normal breathing amplitude
    apnea_events: List[Tuple[float, float]] = None,  # List of (start_time, duration) for apnea events
    pre_apnea_decline_time: float = 3.0,  # Time for amplitude to decline before apnea
    post_apnea_recovery_time: float = 5.0,  # Time for amplitude to recover after apnea
    recovery_overshoot: float = 1.5,  # Amplitude overshoot during recovery (as multiplier)
    noise_level: float = 0.05  # Amount of noise to add for realism
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an idealized breathing waveform with apnea events.
    
    Args:
        duration: Total signal duration in seconds
        sample_rate: Sampling rate in Hz
        breathing_frequency: Breathing rate in Hz (e.g., 0.25 = 15 breaths/min)
        normal_amplitude: Normal breathing amplitude
        apnea_events: List of tuples (start_time, apnea_duration) in seconds
        pre_apnea_decline_time: Time for breathing to decline before apnea
        post_apnea_recovery_time: Time for breathing to recover after apnea
        recovery_overshoot: Amplitude multiplier during recovery phase
        noise_level: Standard deviation of gaussian noise to add
    
    Returns:
        Tuple of (time_array, breathing_signal)
    """
    
    # Default apnea events if none provided
    if apnea_events is None:
        apnea_events = [
            (20, 15),   # 15-second apnea starting at 20s
            (50, 20),   # 20-second apnea starting at 50s  
            (90, 12),   # 12-second apnea starting at 90s
        ]
    
    # Generate time array
    n_samples = int(duration * sample_rate)
    time_array = np.linspace(0, duration, n_samples)
    
    # Generate base sine wave (breathing)
    base_breathing = np.sin(2 * np.pi * breathing_frequency * time_array)
    
    # Initialize amplitude envelope
    amplitude_envelope = np.ones_like(time_array) * normal_amplitude
    
    # Process each apnea event
    for start_time, apnea_duration in apnea_events:
        # Calculate time indices
        decline_start_idx = int((start_time - pre_apnea_decline_time) * sample_rate)
        apnea_start_idx = int(start_time * sample_rate)
        apnea_end_idx = int((start_time + apnea_duration) * sample_rate)
        recovery_end_idx = int((start_time + apnea_duration + post_apnea_recovery_time) * sample_rate)
        
        # Ensure indices are within bounds
        decline_start_idx = max(0, decline_start_idx)
        recovery_end_idx = min(len(time_array), recovery_end_idx)
        
        # Pre-apnea decline phase
        if decline_start_idx < apnea_start_idx:
            decline_length = apnea_start_idx - decline_start_idx
            decline_curve = np.linspace(normal_amplitude, 0, decline_length)
            # Use exponential decay for more realistic decline
            decline_curve = normal_amplitude * np.exp(-3 * np.linspace(0, 1, decline_length))
            amplitude_envelope[decline_start_idx:apnea_start_idx] = decline_curve
        
        # Apnea phase (no breathing)
        if apnea_start_idx < apnea_end_idx:
            amplitude_envelope[apnea_start_idx:apnea_end_idx] = 0
        
        # Post-apnea recovery phase
        if apnea_end_idx < recovery_end_idx:
            recovery_length = recovery_end_idx - apnea_end_idx
            # Create recovery curve with overshoot
            recovery_curve = np.zeros(recovery_length)
            
            # Rapid initial increase with overshoot
            overshoot_duration = int(recovery_length * 0.3)  # 30% of recovery time for overshoot
            if overshoot_duration > 0:
                overshoot_curve = normal_amplitude * recovery_overshoot * (
                    1 - np.exp(-4 * np.linspace(0, 1, overshoot_duration))
                )
                recovery_curve[:overshoot_duration] = overshoot_curve
            
            # Gradual return to normal
            if overshoot_duration < recovery_length:
                normalization_curve = np.linspace(
                    normal_amplitude * recovery_overshoot,
                    normal_amplitude,
                    recovery_length - overshoot_duration
                )
                recovery_curve[overshoot_duration:] = normalization_curve
            
            amplitude_envelope[apnea_end_idx:recovery_end_idx] = recovery_curve
    
    # Apply amplitude envelope to base breathing
    breathing_signal = base_breathing * amplitude_envelope
    
    # Add realistic noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(breathing_signal))
        breathing_signal += noise
    
    return time_array, breathing_signal
