from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pyedflib
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


pio.renderers.default = "browser"



@dataclass
class EDFSleepStudy:
    """
    Dataclass for EDF sleep study data with focus on respiratory traces and annotations.
    """
    # File metadata
    file_path: Path
    patient_id: str
    start_datetime: str
    duration_seconds: float
    sample_rate: float
    n_samples: int

    
    # Respiratory signals - these are the main signals of interest
    pflow: Optional[np.ndarray] = None  # PFLOW
    flow: Optional[np.ndarray] = None   # Flow
    resp_therm: Optional[np.ndarray] = None  # Resp Therm
    resp_thermocan: Optional[np.ndarray] = None  # Resp Thermocan+
    resp_dymedix: Optional[np.ndarray] = None   # Resp DyMedix+
    xsum: Optional[np.ndarray] = None   # XSum
    resp_chest: Optional[np.ndarray] = None     # Resp Chest
    resp_abdomen: Optional[np.ndarray] = None   # Resp Abdomen
    spo2: Optional[np.ndarray] = None   # SpO2
    
    # Additional physiological signals
    ecg_la_ra: Optional[np.ndarray] = None      # ECG LA-RA
    position: Optional[np.ndarray] = None       # Position
    pleth: Optional[np.ndarray] = None          # Pleth
    pr: Optional[np.ndarray] = None             # PR (Pulse Rate)
    etco2: Optional[np.ndarray] = None          # EtCO2
    etwave: Optional[np.ndarray] = None         # EtWave
    tcco2: Optional[np.ndarray] = None          # TcCO2
    
    # Other signals
    patient_event: Optional[np.ndarray] = None  # Patient Event
    snore_mic: Optional[np.ndarray] = None      # Snore mic+-Ref
    snore_dr: Optional[np.ndarray] = None       # Snore_DR
    
    # Signal metadata
    signal_labels: List[str] = field(default_factory=list)
    signal_sample_rates: Dict[str, float] = field(default_factory=dict)
    signal_units: Dict[str, str] = field(default_factory=dict)
    signal_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Annotations
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Raw EDF header information
    header: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_edf_file(cls, file_path: str, validate_required_signals: bool = True):
        """
        Create EDFSleepStudy instance from EDF file.
        
        Args:
            file_path: Path to the EDF file
            validate_required_signals: If True, validates that required respiratory signals are present
        
        Returns:
            EDFSleepStudy instance
            
        Raises:
            ValueError: If required signals are missing and validation is enabled
        """
        file_path = Path(file_path)
        
        # Expected signal labels for this specific setup
        expected_signals = [
            'Patient Event', 'Snore mic+-Ref', 'Snore_DR', 'ECG LA-RA', 'Flow', 
            'PFLOW', 'Resp Therm', 'Resp Thermocan+', 'Resp DyMedix+', 'XSum', 
            'Resp Chest', 'Resp Abdomen', 'Position', 'SpO2', 'Pleth', 'PR', 
            'EtCO2', 'EtWave', 'TcCO2'
        ]
        
        # Critical respiratory signals that must be present
        required_signals = ['PFLOW', 'Flow', 'Resp Therm']
        
        with pyedflib.EdfReader(str(file_path)) as edf:
            # Get header information
            header = {
                'technician': edf.getTechnician(),
                'recording_additional': edf.getRecordingAdditional(),
                'patient_name': edf.getPatientName(),
                'patient_additional': edf.getPatientAdditional(),
                'patient_code': edf.getPatientCode(),
                'equipment': edf.getEquipment(),
                'admin_code': edf.getAdmincode(),
                'gender': edf.getSex(),
                'birthdate': edf.getBirthdate(),
                'startdate': edf.getStartdatetime(),
                'n_signals': edf.signals_in_file,
                'n_samples': edf.getNSamples()[0],
                'file_duration': edf.file_duration,
                'datarecords_in_file': edf.datarecords_in_file
            }
            
            # Get signal information
            signal_labels = edf.getSignalLabels()
            signal_sample_rates = {}
            signal_units = {}
            signal_ranges = {}
            
            for i in range(edf.signals_in_file):
                label = signal_labels[i]
                signal_sample_rates[label] = edf.getSampleFrequency(i)
                signal_units[label] = edf.getPhysicalDimension(i)
                signal_ranges[label] = (edf.getPhysicalMinimum(i), edf.getPhysicalMaximum(i))
            
            # Initialize the dataclass
            study = cls(
                file_path=file_path,
                patient_id=header.get('patient_code', 'Unknown'),
                n_samples=header.get('n_samples'),
                start_datetime=str(header.get('startdate', 'Unknown')),
                duration_seconds=header['file_duration'],
                sample_rate=max(signal_sample_rates.values()) if signal_sample_rates else 0,
                signal_labels=signal_labels,
                signal_sample_rates=signal_sample_rates,
                signal_units=signal_units,
                signal_ranges=signal_ranges,
                header=header
            )
            
            # Validate required signals are present
            if validate_required_signals:
                missing_signals = [sig for sig in required_signals if sig not in signal_labels]
                if missing_signals:
                    raise ValueError(f"Required signals missing from EDF file: {missing_signals}")
            
            # Load all signals based on expected labels
            study._load_signals_by_exact_match(edf)
            
            # Load annotations
            study._load_annotations(edf)
            
        return study
    
    def get_unique_annotation_types(self) -> List[str]:
        """Get list of unique annotation descriptions/types in the dataset."""
        if not self.annotations:
            return []
        
        unique_descriptions = list(set(ann['description'] for ann in self.annotations))
        return sorted(unique_descriptions)
    
    def get_annotation_summary(self) -> Dict[str, int]:
        """Get count of each annotation type."""
        if not self.annotations:
            return {}
        
        annotation_counts = {}
        for ann in self.annotations:
            description = ann['description']
            annotation_counts[description] = annotation_counts.get(description, 0) + 1
        
        # Sort by count (descending)
        return dict(sorted(annotation_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _load_signals_by_exact_match(self, edf: pyedflib.EdfReader):
        """Load signals based on exact label matching."""
        
        # Create mapping from signal labels to indices
        label_to_index = {label: i for i, label in enumerate(self.signal_labels)}
        
        # Load each signal if present
        if 'Patient Event' in label_to_index:
            self.patient_event = edf.readSignal(label_to_index['Patient Event'])
            
        if 'Snore mic+-Ref' in label_to_index:
            self.snore_mic = edf.readSignal(label_to_index['Snore mic+-Ref'])
            
        if 'Snore_DR' in label_to_index:
            self.snore_dr = edf.readSignal(label_to_index['Snore_DR'])
            
        if 'ECG LA-RA' in label_to_index:
            self.ecg_la_ra = edf.readSignal(label_to_index['ECG LA-RA'])
            
        # Respiratory signals
        if 'Flow' in label_to_index:
            self.flow = edf.readSignal(label_to_index['Flow'])
            
        if 'PFLOW' in label_to_index:
            self.pflow = edf.readSignal(label_to_index['PFLOW'])
            
        if 'Resp Therm' in label_to_index:
            self.resp_therm = edf.readSignal(label_to_index['Resp Therm'])
            
        if 'Resp Thermocan+' in label_to_index:
            self.resp_thermocan = edf.readSignal(label_to_index['Resp Thermocan+'])
            
        if 'Resp DyMedix+' in label_to_index:
            self.resp_dymedix = edf.readSignal(label_to_index['Resp DyMedix+'])
            
        if 'XSum' in label_to_index:
            self.xsum = edf.readSignal(label_to_index['XSum'])
            
        if 'Resp Chest' in label_to_index:
            self.resp_chest = edf.readSignal(label_to_index['Resp Chest'])
            
        if 'Resp Abdomen' in label_to_index:
            self.resp_abdomen = edf.readSignal(label_to_index['Resp Abdomen'])
            
        if 'Position' in label_to_index:
            self.position = edf.readSignal(label_to_index['Position'])
            
        if 'SpO2' in label_to_index:
            self.spo2 = edf.readSignal(label_to_index['SpO2'])
            
        if 'Pleth' in label_to_index:
            self.pleth = edf.readSignal(label_to_index['Pleth'])
            
        if 'PR' in label_to_index:
            self.pr = edf.readSignal(label_to_index['PR'])
            
        if 'EtCO2' in label_to_index:
            self.etco2 = edf.readSignal(label_to_index['EtCO2'])
            
        if 'EtWave' in label_to_index:
            self.etwave = edf.readSignal(label_to_index['EtWave'])
            
        if 'TcCO2' in label_to_index:
            self.tcco2 = edf.readSignal(label_to_index['TcCO2'])
    
    def _load_annotations(self, edf: pyedflib.EdfReader):
        """Load annotations from EDF file."""
        try:
            annotations_data = edf.readAnnotations()
            if annotations_data:
                onset_times, durations, descriptions = annotations_data
                self.annotations = [
                    {
                        'onset': onset,
                        'duration': duration,
                        'description': desc
                    }
                    for onset, duration, desc in zip(onset_times, durations, descriptions)
                ]
        except Exception as e:
            print(f"Warning: Could not load annotations: {e}")
            self.annotations = []
    
    def get_respiratory_signals(self) -> Dict[str, np.ndarray]:
        """Get all loaded respiratory signals as a dictionary."""
        signals = {}
        
        if self.pflow is not None:
            signals['pflow'] = self.pflow
        if self.flow is not None:
            signals['flow'] = self.flow
        if self.resp_therm is not None:
            signals['resp_therm'] = self.resp_therm
        if self.resp_thermocan is not None:
            signals['resp_thermocan'] = self.resp_thermocan
        if self.resp_dymedix is not None:
            signals['resp_dymedix'] = self.resp_dymedix
        if self.xsum is not None:
            signals['xsum'] = self.xsum
        if self.resp_chest is not None:
            signals['resp_chest'] = self.resp_chest
        if self.resp_abdomen is not None:
            signals['resp_abdomen'] = self.resp_abdomen
        if self.spo2 is not None:
            signals['spo2'] = self.spo2
            
        return signals
    
    def get_required_respiratory_signals(self) -> Dict[str, np.ndarray]:
        """Get the three required respiratory signals."""
        signals = {}
        
        if self.pflow is not None:
            signals['pflow'] = self.pflow
        if self.flow is not None:
            signals['flow'] = self.flow
        if self.resp_therm is not None:
            signals['resp_therm'] = self.resp_therm
            
        return signals
    
    def get_annotations_by_type(self, annotation_pattern: str) -> List[Dict[str, Any]]:
        """
        Filter annotations by description pattern.
        
        Args:
            annotation_pattern: Pattern to search for in annotation descriptions
            
        Returns:
            List of matching annotations
        """
        return [
            ann for ann in self.annotations 
            if annotation_pattern.lower() in ann['description'].lower()
        ]
    

    
    def get_apnea_events(self) -> List[Dict[str, Any]]:
        """Get apnea-related annotations."""
        apnea_patterns = ['apnea', 'hypopnea', 'central', 'obstructive', 'mixed']
        apnea_events = []
        
        for pattern in apnea_patterns:
            apnea_events.extend(self.get_annotations_by_type(pattern))
        
        return apnea_events
    
    def summary(self) -> str:
        """Generate a summary of the sleep study data."""
        respiratory_signals = self.get_respiratory_signals()
        
        
        summary_lines = [
            f"EDF Sleep Study Summary",
            f"========================",
            f"Patient ID: {self.patient_id}",
            f"Start Time: {self.start_datetime}",
            f"Duration: {self.duration_seconds/3600:.2f} hours",
            f"Number of Samples: {self.n_samples}",
            f"",
            f"Loaded Respiratory Signals ({len(respiratory_signals)}):",
        ]
        
        for signal_name in respiratory_signals:
            summary_lines.append(f"  - {signal_name}")
        
        summary_lines.extend([
            f"",
            f"Total Annotations: {len(self.annotations)}",
            f"Apnea Events: {len(self.get_apnea_events())}",
            f"",
            f"Available Signal Labels:",
        ])
        
        for label in self.signal_labels:
            freq = self.signal_sample_rates.get(label, 'Unknown')
            unit = self.signal_units.get(label, 'Unknown')
            summary_lines.append(f"  - {label} ({freq} Hz, {unit})")
        
        return "\n".join(summary_lines)
    
    def plot_signals(self, 
                    signal_names: List[str], 
                    start_index: int = 0, 
                    length: Optional[int] = None,
                    title: str = "Sleep Study Signals",
                    height: int = 800,
                    annotation_types: Optional[List[str]] = None) -> go.Figure:
        """
        Plot 4 signal traces using Plotly with specified start index and length.
        
        Args:
            signal_names: List of 4 signal names to plot (must be exact attribute names)
            start_index: Starting sample index
            length: Number of samples to plot (if None, plots remaining signal)
            title: Plot title
            height: Plot height in pixels
            show_annotations: Whether to overlay annotations on the plot
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If signal_names doesn't contain exactly 4 signals or signals don't exist
        """
        if len(signal_names) != 4:
            raise ValueError("Must specify exactly 4 signal names")
        
        # Get all available signals
        all_signals = self._get_all_signals_dict()
        
        # Validate that all requested signals exist and are loaded
        missing_signals = []
        for signal_name in signal_names:
            if signal_name not in all_signals or all_signals[signal_name] is None:
                missing_signals.append(signal_name)
        
        if missing_signals:
            available_signals = [name for name, data in all_signals.items() if data is not None]
            raise ValueError(f"Signals not found or not loaded: {missing_signals}. "
                           f"Available signals: {available_signals}")
        
        # Create subplot figure
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=signal_names
        )
        
        # Color palette for the traces
        colors = ['#1f77b4', "#c47e15", "#053a05", "#8d1616"]
        
        # Calculate time arrays and plot each signal
        for i, signal_name in enumerate(signal_names):
            signal_data = all_signals[signal_name]
            
            # Handle start_index and length
            if start_index >= len(signal_data):
                raise ValueError(f"Start index {start_index} exceeds signal length {len(signal_data)}")
            
            end_index = start_index + (length if length is not None else len(signal_data) - start_index)
            end_index = min(end_index, len(signal_data))
            
            # Extract signal segment
            signal_segment = signal_data[start_index:end_index]
            
            # Create time array based on sample rate
            sample_rate = self.signal_sample_rates.get(
                self._get_original_signal_label(signal_name), 
                self.sample_rate
            )
            time_array = np.arange(len(signal_segment)) / sample_rate + (start_index / sample_rate)
            
            # Get signal units for y-axis label
            original_label = self._get_original_signal_label(signal_name)
            units = self.signal_units.get(original_label, '')
            y_label = f"{signal_name} ({units})" if units else signal_name
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=time_array,
                    y=signal_segment,
                    mode='lines',
                    name=signal_name,
                    line=dict(color=colors[i], width=1),
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            # Update y-axis title
            fig.update_yaxes(title_text=y_label, row=i+1, col=1)
        
        # Add annotations if requested
        if self.annotations:
            self._add_annotations_to_plot(fig, start_index, end_index, self.sample_rate, annotation_types)
        
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Update x-axis title (only on bottom subplot)
        fig.update_xaxes(title_text="Time (seconds)", row=4, col=1)
        
        return fig
    
    def _get_all_signals_dict(self) -> Dict[str, Optional[np.ndarray]]:
        """Get dictionary of all signal attribute names and their data."""
        return {
            'patient_event': self.patient_event,
            'snore_mic': self.snore_mic,
            'snore_dr': self.snore_dr,
            'ecg_la_ra': self.ecg_la_ra,
            'flow': self.flow,
            'pflow': self.pflow,
            'resp_therm': self.resp_therm,
            'resp_thermocan': self.resp_thermocan,
            'resp_dymedix': self.resp_dymedix,
            'xsum': self.xsum,
            'resp_chest': self.resp_chest,
            'resp_abdomen': self.resp_abdomen,
            'position': self.position,
            'spo2': self.spo2,
            'pleth': self.pleth,
            'pr': self.pr,
            'etco2': self.etco2,
            'etwave': self.etwave,
            'tcco2': self.tcco2
        }
    
    def _get_original_signal_label(self, attribute_name: str) -> str:
        """Map attribute name back to original signal label for metadata lookup."""
        mapping = {
            'patient_event': 'Patient Event',
            'snore_mic': 'Snore mic+-Ref',
            'snore_dr': 'Snore_DR',
            'ecg_la_ra': 'ECG LA-RA',
            'flow': 'Flow',
            'pflow': 'PFLOW',
            'resp_therm': 'Resp Therm',
            'resp_thermocan': 'Resp Thermocan+',
            'resp_dymedix': 'Resp DyMedix+',
            'xsum': 'XSum',
            'resp_chest': 'Resp Chest',
            'resp_abdomen': 'Resp Abdomen',
            'position': 'Position',
            'spo2': 'SpO2',
            'pleth': 'Pleth',
            'pr': 'PR',
            'etco2': 'EtCO2',
            'etwave': 'EtWave',
            'tcco2': 'TcCO2'
        }
        return mapping.get(attribute_name, attribute_name)
    
    # def _add_annotations_to_plot(self, fig: go.Figure, start_index: int, end_index: int, sample_rate: float):
    #     """Add annotation markers to the plot."""
    #     start_time = start_index / sample_rate
    #     end_time = end_index / sample_rate
        
    #     # Filter annotations that fall within the time window
    #     relevant_annotations = [
    #         ann for ann in self.annotations
    #         if start_time <= ann['onset'] <= end_time
    #     ]
        
    #     # Add vertical lines for annotations
    #     for ann in relevant_annotations:
    #         fig.add_vline(
    #             x=ann['onset'],
    #             line=dict(color="red", width=1, dash="dot"),
    #             annotation_text=ann['description'][:20] + "..." if len(ann['description']) > 20 else ann['description'],
    #             annotation_position="top"
    #         )
    
    def _add_annotations_to_plot(self, fig: go.Figure, start_index: int, end_index: int, sample_rate: float, annotation_types: Optional[List[str]] = None):
        """Add annotation markers to the plot."""
        start_time = start_index / sample_rate
        end_time = end_index / sample_rate
        
        # Filter annotations that fall within the time window
        relevant_annotations = [
            ann for ann in self.annotations
            if start_time <= ann['onset'] <= end_time
        ]
        
        # Further filter by annotation types if specified
        if annotation_types is not None:
            filtered_annotations = []
            for ann in relevant_annotations:
                # Check if any of the requested types are in the annotation description
                if any(ann_type.lower() in ann['description'].lower() for ann_type in annotation_types):
                    filtered_annotations.append(ann)
            relevant_annotations = filtered_annotations
        
        # Color mapping for different annotation types
        annotation_colors = {
            'apnea': 'red',
            'hypopnea': 'orange', 
            'arousal': 'purple',
            'desaturation': 'blue',
            'stage': 'green',
            'movement': 'brown',
            'default': 'gray'
        }
        
        # Add vertical lines for annotations
        for ann in relevant_annotations:
            # Determine color based on annotation type
            color = annotation_colors['default']
            for ann_type, ann_color in annotation_colors.items():
                if ann_type in ann['description'].lower():
                    color = ann_color
                    break
            
            # Truncate long annotation descriptions
            display_text = ann['description'][:25] + "..." if len(ann['description']) > 25 else ann['description']
            
            fig.add_vline(
                x=ann['onset'],
                line=dict(color=color, width=2, dash="dot"),
                annotation_text=display_text,
                annotation_position="top",
                annotation_textangle=90,
                annotation_font_size=8
            )
    
    def get_available_signals(self) -> List[str]:
        """Get list of available (loaded) signal names for plotting."""
        all_signals = self._get_all_signals_dict()
        return [name for name, data in all_signals.items() if data is not None]


# Example usage:
if __name__ == "__main__":
    # Load an EDF file with validation (default)
    
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
    
    # Plot different signals with annotations
    fig2 = study.plot_signals(
        signal_names=['spo2', 'pr', 'position', 'flow'],
        start_index=0,
        length=int(5 * 60 * study.sample_rate),  # First 5 minutes
        title="Mixed Signals with Annotations",
        show_annotations=True
    )
    fig2.show()
    # 
    # # Load without validation (if some required signals might be missing)
    # study_no_validation = EDFSleepStudy.from_edf_file("path/to/file.edf", validate_required_signals=False)
    pass