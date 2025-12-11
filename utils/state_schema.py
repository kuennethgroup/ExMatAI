"""
State schema for LangGraph workflow
"""

from typing import TypedDict, List, Dict, Optional, Annotated
import operator

class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    page: int

class DetectedStructure(TypedDict):
    bbox: BoundingBox
    image_crop: str  # Path to cropped image
    nearby_text: List[str]
    matched_label: Optional[str]
    confidence: float

class MaterialInfo(TypedDict):
    full_name: str
    abbreviation: Optional[str]
    role: str  # cathode, anode, electrolyte_salt, electrolyte_solvent, binder, conductive_additive
    monomer_name: Optional[str]
    wt_percent: Optional[float]
    binder: Optional[str]
    binder_wt_percent: Optional[float]
    conductive_material: Optional[str]
    conductive_wt_percent: Optional[float]
    mentions: List[str]

class ElectrolyteComponent(TypedDict):
    type: str  # salt or solvent
    name: str
    smiles: Optional[str]
    concentration: Optional[str]
    confidence: float
    source_page: Optional[int]

class ProcessingParameters(TypedDict):
    current_rate_mAg: Optional[float]
    mass_ratio: Optional[str]
    areal_ratio: Optional[float]
    temperature_C: Optional[float]
    loading_rate_mg_cm2: Optional[float]
    cell_setup: Optional[str]
    reference_electrode: Optional[str]

class PlotDataPoint(TypedDict):
    cycle: int
    capacity_mAh_g: float

class VoltageProfilePoint(TypedDict):
    capacity_mAh_g: float
    voltage_V: float

class VoltageProfile(TypedDict):
    cycle: int
    profile: List[VoltageProfilePoint]

class FigureReference(TypedDict):
    figure: Optional[int]
    subplot: Optional[str]
    page: int

class PerformanceData(TypedDict):
    specific_capacity_vs_cycles: List[Dict]
    voltage_profiles: List[Dict]

class MaterialComposition(TypedDict):
    material_name: str
    full_name: Optional[str]
    abbreviation: Optional[str]
    smiles: Optional[str]
    wt_percent: Optional[float]
    binder: Optional[str]
    binder_wt_percent: Optional[float]
    conductive_material: Optional[str]
    conductive_wt_percent: Optional[float]
    confidence: float
    source_page: Optional[int]
    role: Optional[str]

class Experiment(TypedDict):
    experiment_id: int
    type: str  # Half-cell or Full-cell
    subtype: Optional[str]  # Positive, Negative, or None
    cathode: MaterialComposition
    anode: MaterialComposition
    electrolyte: Dict
    processing_parameters: ProcessingParameters
    reference_electrode: Optional[str]
    cell_setup: str
    performance_data: PerformanceData

class AgentState(TypedDict):
    # Input
    pdf_path: str
    
    # OCR outputs
    extracted_text: str
    text_by_page: Dict[int, str]
    structure_images: List[Dict]
    plot_images: List[Dict]
    
    # Text Analysis outputs
    identified_materials: List[MaterialInfo]
    battery_stacks: List[Dict]
    processing_params: ProcessingParameters
    paper_metadata: Dict
    validation_passed: bool
    
    # Structure Detection outputs
    detected_structures: List[DetectedStructure]
    
    # SMILES outputs
    material_smiles: Dict[str, Dict]
    
    # Plots outputs
    cycling_data: List[Dict]
    voltage_data: List[Dict]
    
    # Final outputs
    experiments: List[Experiment]
    output_json: Dict
    
    # Control
    messages: Annotated[List[str], operator.add]
    current_agent: str
    errors: Annotated[List[str], operator.add]
