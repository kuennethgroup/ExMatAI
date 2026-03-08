"""
Text Analysis Agent - Text LLM Extraction + Figure Regex Extraction
Absorbs logic from modules/text_extract.py.

Produces:
  - experiments_data  (list of structured experiment dicts via qwen3.5:35b)
  - figures_data      (list of {Figure_ID, Image_Path, Caption} via regex)
"""

import json
import os
import re
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from utils.state_schema import WorkflowState

# =======================================================================
# Pydantic Schema - one record per battery experiment setup
# =======================================================================


class ExtractedExperiment(BaseModel):
    Subtype: Optional[str] = Field(
        default=None, description="e.g., Positive, Negative, or Full Cell"
    )
    Type_of_battery: Optional[str] = Field(
        default=None, description="e.g., Half-cell, Full-cell"
    )
    Battery_type: Optional[str] = Field(
        default=None, description="e.g., Dual-ion, Lithium-ion, Potassium-ion"
    )

    # Material Identifiers & Structure References
    Material_Name_Negative: Optional[str] = Field(
        default=None, description="Text name of the negative electrode material"
    )
    Structure_Figure_Negative: Optional[str] = Field(
        default=None,
        description="Figure reference for the chemical structure of the negative electrode (e.g., 'Fig. 1a')",
    )
    Material_Name_Positive: Optional[str] = Field(
        default=None, description="Text name of the positive electrode material"
    )
    Structure_Figure_Positive: Optional[str] = Field(
        default=None,
        description="Figure reference for the chemical structure of the positive electrode (e.g., 'Fig. 1b')",
    )

    # Compositions
    wt_percent_active_material: Optional[str] = Field(
        default=None, description="Weight percentage of the active material"
    )
    conductive_material: Optional[str] = Field(
        default=None, description="Name of conductive additive, e.g., Carbon Black, Super P"
    )
    wt_percent_conductive_mat: Optional[str] = Field(
        default=None, description="Weight percentage of conductive material"
    )
    binder: Optional[str] = Field(default=None, description="Name of the binder, e.g., PVDF")
    wt_percent_binder: Optional[str] = Field(
        default=None, description="Weight percentage of the binder"
    )
    Weight_ratio_neg_pos: Optional[str] = Field(
        default=None, description="Weight ratio of negative to positive electrode"
    )

    # Electrolyte & Cell Details
    Electrolyte: Optional[str] = Field(
        default=None, description="Full description of the electrolyte used"
    )
    Salt_amount: Optional[str] = Field(
        default=None, description="Amount or concentration of the salt"
    )
    Solvent_amount: Optional[str] = Field(
        default=None, description="Amount or ratio of the solvent"
    )
    Cell_setup: Optional[str] = Field(
        default=None, description="e.g., coin cell, 3-electrode"
    )
    Reference_electrode: Optional[str] = Field(
        default=None,
        description="e.g., Ag/AgCl. Leave null if it is a coin cell or 2-electrode setup.",
    )
    Loading_rate_mg_cm2: Optional[str] = Field(
        default=None, description="Mass loading rate in mg/cm2"
    )
    Temperature: Optional[str] = Field(
        default=None, description="Testing temperature in Celsius"
    )

    # Text-Based Summary Metrics
    Reported_C_rate: Optional[str] = Field(
        default=None, description="C-rate or current density mentioned in the text setup"
    )
    Reported_Specific_Capacity: Optional[str] = Field(
        default=None, description="Specific capacity value reported in the text"
    )
    Max_Reported_Cycles: Optional[int] = Field(
        default=None, description="Maximum number of cycles mentioned in the text"
    )

    # Plot References
    Cycle_Data_Figure: Optional[List[str]] = Field(
        default_factory=list, description="List of figure references showing cycle data/stability plots (e.g., ['Fig. 3b', 'Fig. 3c'])"
    )
    Voltage_Profile_Figure: Optional[List[str]] = Field(
        default_factory=list, description="List of figure references showing Voltage vs. Specific Capacity plots (e.g., ['Fig. 3a'])"
    )


class ExperimentExtraction(BaseModel):
    experiments: List[ExtractedExperiment] = Field(
        description="List of extracted battery experiment records from the paper."
    )


# =======================================================================
# Helper: extract figure metadata from .mmd via regex
# =======================================================================

def _extract_figures_from_mmd(paper_content: str) -> list:
    """Parse figure references from DeepSeek-OCR .mmd output.

    Handles two patterns:
      A) Same-line:  ![](path) <center>Fig. N ...</center>
      B) Multi-line: ![](path)\\n\\n<center>Figure N | ...</center>
    """
    figures: list = []

    # Pattern A: image and caption on same line (original modules format)
    pattern_a = re.compile(r'\!\[\]\((.*?)\)\s*<center>(Fig\..*?)</center>', re.DOTALL)

    # Pattern B: image on one line, caption on a nearby line (DeepSeek-OCR format)
    # Allows blank lines / page-split markers between image and caption
    pattern_b = re.compile(
        r'\!\[\]\((.*?)\)'           # ![](path)
        r'[\s\S]*?'                  # any whitespace / blank lines (non-greedy)
        r'<center>(Figure\s+\d+.*?)</center>',  # <center>Figure N | ...</center>
        re.DOTALL,
    )

    # Try pattern A first
    matches = pattern_a.findall(paper_content)
    if matches:
        for raw_path, caption_raw in matches:
            raw_path = raw_path.strip()
            image_path = raw_path if raw_path.startswith("./") else f"./{raw_path}"
            caption = " ".join(caption_raw.split())
            fig_id_match = re.search(r'(Fig\.\s*\d+)', caption)
            fig_id = fig_id_match.group(1) if fig_id_match else "Unknown"
            figures.append({
                "Figure_ID": fig_id,
                "Image_Path": image_path,
                "Caption": caption,
            })
    else:
        # Use pattern B (DeepSeek-OCR format)
        matches = pattern_b.findall(paper_content)
        for raw_path, caption_raw in matches:
            raw_path = raw_path.strip()
            image_path = raw_path if raw_path.startswith("./") else f"./{raw_path}"
            caption = " ".join(caption_raw.split())
            # Extract Figure ID: "Figure 1" or "Figure 2" etc.
            fig_id_match = re.search(r'(Figure\s+\d+)', caption)
            fig_id = fig_id_match.group(1) if fig_id_match else "Unknown"
            figures.append({
                "Figure_ID": fig_id,
                "Image_Path": image_path,
                "Caption": caption,
            })

    return figures


# =======================================================================
# Helper: split paper into sections for prioritized LLM input
# =======================================================================

# Section heading patterns ranked by information priority
_PRIORITY_HEADINGS = [
    # Tier 1 – electrode recipes, electrolyte, cell setup, testing conditions
    ["method", "experimental", "electrode preparation", "cell assembly",
     "electrochemical measurement", "battery fabrication", "materials and methods"],
    # Tier 2 – results discussion with performance metrics & figure references
    ["result", "electrochemical performance", "electrochemical characterization",
     "discussion", "performance"],
    # Tier 3 – overview / abstract (material names, key highlights)
    ["abstract", "introduction", "conclusion", "summary"],
]

# Headings to SKIP (won't contribute to extraction)
_SKIP_HEADINGS = [
    "reference", "acknowledgement", "author contribution",
    "competing interest", "additional information", "supplementary",
    "data availability",
]


def _split_paper_sections(paper_content: str) -> dict:
    """Split .mmd paper into named sections using ## headings.

    Returns dict: {heading_lower: section_text, ...}
    '_preamble' key holds everything before the first heading (title + abstract).
    """
    sections = {}
    # Split on ## headings (markdown H2)
    parts = re.split(r'^(##\s+.+)$', paper_content, flags=re.MULTILINE)

    # parts = [preamble, heading1, body1, heading2, body2, ...]
    if parts:
        preamble = parts[0].strip()
        if preamble:
            sections["_preamble"] = preamble

    i = 1
    while i < len(parts) - 1:
        heading = parts[i].lstrip("#").strip()
        body = parts[i + 1].strip()
        sections[heading.lower()] = f"## {heading}\n\n{body}"
        i += 2

    return sections


def _clean_for_llm(text: str) -> str:
    """Strip noise that wastes tokens without helping extraction."""
    # Remove image references
    text = re.sub(r'!\[\]\(.*?\)', '', text)
    # Remove <center>...</center> figure captions (already extracted by regex)
    text = re.sub(r'<center>.*?</center>', '', text, flags=re.DOTALL)
    # Remove page split markers
    text = re.sub(r'<--- Page Split --->', '', text)
    # Remove HTML tags
    text = re.sub(r'</?sup>', '', text)
    # Simplify some LaTeX noise but keep numbers and units readable
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\\(([^)]*?)\\\)', r'\1', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Keywords that indicate a Methods sub-paragraph is relevant to extraction
_RELEVANT_METHOD_KW = [
    "electrode", "electrolyte", "cell", "battery", "coin",
    "active material", "binder", "pvdf", "graphene", "carbon",
    "slurry", "loading", "anode", "cathode", "separator",
    "charge", "discharge", "galvanostatic", "cycling", "current",
    "voltage", "capacity", "electrochem", "assembly", "fabricat",
    "preparation", "counter electrode", "working electrode",
]

# Keywords that indicate a sub-paragraph should be SKIPPED
_IRRELEVANT_METHOD_KW = [
    "nmr spectr", "solid-state nmr", "solid- state nmr",
    "dft", "density functional", "gaussian",
    "xrd", "x-ray diffraction", "transmission electron",
    "synchrotron", "srpes", "photoemission",
    "morphology characterization",
    "single crystal", "crystallograph",
]


def _filter_methods_section(methods_text: str) -> str:
    """Sub-split Methods section into paragraphs and keep only the ones
    relevant to battery fabrication & testing (drop NMR, DFT, morphology).

    Relevant keywords OVERRIDE irrelevant ones - if a paragraph mentions
    both NMR and electrode preparation, it is KEPT.
    """
    paragraphs = re.split(r'\n\n+', methods_text)

    relevant = []
    for para in paragraphs:
        para_lower = para.lower()

        has_relevant = any(kw in para_lower for kw in _RELEVANT_METHOD_KW)
        has_irrelevant = any(kw in para_lower for kw in _IRRELEVANT_METHOD_KW)

        # Keep if it has any relevant keyword (even if also irrelevant)
        if has_relevant:
            relevant.append(para)
        # Skip if only irrelevant
        elif has_irrelevant:
            continue
        # Also skip paragraphs with no match at all (generic text)

    if not relevant:
        return methods_text

    return "\n\n".join(relevant)


def _build_prioritized_text(sections: dict, max_chars: int = 120000) -> str:
    """Build LLM input text maintaining high information density,
    staying under max_chars to fit context window.
    """
    blocks: list[str] = []
    used_keys: set[str] = set()

    # Always include preamble (title + abstract) – short, high info density
    if "_preamble" in sections:
        blocks.append(_clean_for_llm(sections["_preamble"]))
        used_keys.add("_preamble")

    # Add sections by priority tier
    for tier in _PRIORITY_HEADINGS:
        for heading_key, section_text in sections.items():
            if heading_key in used_keys:
                continue
            if any(skip in heading_key for skip in _SKIP_HEADINGS):
                continue
            if any(kw in heading_key for kw in tier):
                cleaned = _clean_for_llm(section_text)
                if sum(len(b) for b in blocks) + len(cleaned) < max_chars:
                    blocks.append(cleaned)
                    used_keys.add(heading_key)
                    
    # Ensure no data is lost by appending any unassigned valid sections
    for heading_key, section_text in sections.items():
        if heading_key in used_keys:
            continue
        if any(skip in heading_key for skip in _SKIP_HEADINGS):
            continue
        cleaned = _clean_for_llm(section_text)
        if sum(len(b) for b in blocks) + len(cleaned) < max_chars:
            blocks.append(cleaned)
            used_keys.add(heading_key)

    return "\n\n---\n\n".join(blocks)


# =======================================================================
# LangGraph Node
# =======================================================================

def extract_text_data(state: WorkflowState) -> WorkflowState:
    """LangGraph node: extract experiments + figure metadata from the .mmd file."""
    print("\n" + "=" * 70)
    print("Step 2: Starting text extraction using qwen3.5:35b and regex")
    print("=" * 70)

    mmd_path = state["mmd_path"]
    output_dir = state["output_dir"]

    with open(mmd_path, "r", encoding="utf-8") as f:
        paper_content = f.read()

    # -- 1. Regex figure extraction --------------------------------------
    print("  Extracting figure metadata using regex...")
    figures_data = _extract_figures_from_mmd(paper_content)
    print(f"  Found {len(figures_data)} figures in the document.")

    # Save intermediate
    figures_json_path = os.path.join(output_dir, "figures.json")
    with open(figures_json_path, "w", encoding="utf-8") as f:
        json.dump({"figures": figures_data}, f, indent=4)
    print(f"  Saved figure metadata to: {figures_json_path}")

    # -- 2. Split paper into sections ------------------------------------
    print("  Splitting the paper into logical sections for analysis...")
    sections = _split_paper_sections(paper_content)
    print(f"  Identified {len(sections)} sections: {list(sections.keys())}")

    prioritized_text = _build_prioritized_text(sections)
    print(f"  Compiled prioritized text block of {len(prioritized_text)} characters.")

    # Save prioritized text for debugging
    debug_path = os.path.join(output_dir, "llm_input_text.txt")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(prioritized_text)

    # -- 3. LLM experiment extraction ------------------------------------
    print("  Setting up qwen3.5:35b for extracting experiment details...")
    llm = ChatOllama(
        model="qwen3.5:35b",
        base_url="http://localhost:11434",
        temperature=0.0,
        num_ctx=32768,
    )

    # Use raw JSON output instead of structured output (more reliable with Qwen)
    system_prompt = """You are an expert Materials Science AI assistant part of the ExMat AI pipeline.
Your task is to extract battery material properties and experiment conditions from the provided scientific paper text.

The text below is organized with the most important sections FIRST:
- The Methods/Experimental section comes first (contains electrode compositions, electrolyte, cell setup)
- Then Results/Performance sections (contains performance data and figure references)
- Then Abstract/Introduction (contains material names and key highlights)

Read ALL sections carefully before extracting, paying special attention to the Methods section.

You MUST output valid JSON with this EXACT structure:
```json
{
  "experiments": [
    {
      "Subtype": "Positive or Negative or Full Cell",
      "Type_of_battery": "Half-cell or Full-cell",
      "Battery_type": "Lithium-ion, Dual-ion, etc.",
      "Material_Name_Negative": "name of negative/anode material",
      "Structure_Figure_Negative": "Fig. reference or null",
      "Material_Name_Positive": "name of positive/cathode material",
      "Structure_Figure_Positive": "Fig. reference or null",
      "wt_percent_active_material": "percentage as string",
      "conductive_material": "name",
      "wt_percent_conductive_mat": "percentage as string",
      "binder": "name",
      "wt_percent_binder": "percentage as string",
      "Weight_ratio_neg_pos": "ratio or null",
      "Electrolyte": "full electrolyte description",
      "Salt_amount": "concentration",
      "Solvent_amount": "ratio or null",
      "Cell_setup": "e.g. 2016-type coin cell",
      "Reference_electrode": "e.g. Li metal or null",
      "Loading_rate_mg_cm2": "mg/cm2 or null",
      "Temperature": "temperature or null",
      "Reported_C_rate": "e.g. 400 mA g-1 (1C)",
      "Reported_Specific_Capacity": "e.g. 395 mAh g-1",
      "Max_Reported_Cycles": 10000,
      "Cycle_Data_Figure": ["Fig. 3b", "Fig. 3c"],
      "Voltage_Profile_Figure": ["Fig. 3a"]
    }
  ]
}
```

CRITICAL INSTRUCTIONS:

1. Create a SEPARATE record for EACH distinct material tested.

2. ELECTRODE COMPOSITIONS - Look in Methods for weight percentages:
   - "30 wt% active material, 60 wt% graphene, 10 wt% PVDF" ->
     wt_percent_active_material="30", conductive_material="graphene",
     wt_percent_conductive_mat="60", binder="PVDF", wt_percent_binder="10"
   - COPY these values to ALL experiment records using the same electrode prep.

3. ELECTROLYTE - Extract full description, salt concentration, solvent ratio.
   COPY to ALL experiment records using the same electrolyte.

4. CELL SETUP - e.g. "2016-type coin cell", "lithium chip as counter electrode"
   COPY to ALL experiment records using the same setup.

5. FIGURE REFERENCES - extract EXACT figure references from text.
   - For Cycle_Data_Figure and Voltage_Profile_Figure, extract ALL applicable sub-figures as a list of strings.
   - For Structure_Figure, prefer simple 2D chemical structure drawings (e.g., 'Fig. 2a') over complex 3D arrays or synthesis diagrams (e.g., 'Fig. 1b'). Look for captions describing "chemical structures of...".

6. PERFORMANCE - C-rates, specific capacity, max cycles from text.

7. Set null ONLY if the value is truly not mentioned anywhere in the text.

CRITICAL: DO NOT output any markdown tables. DO NOT output conversational text. Output ONLY the raw JSON object starting with `{` and ending with `}`."""

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract ALL battery experiment records from the following paper text:\n\n{prioritized_text}"),
    ]

    print("  Sending text to the LLM. This may take a moment...")
    try:
        result = llm.invoke(messages)
        raw_text = result.content.strip()

        # Handle <think>...</think> blocks from Qwen
        if "<think>" in raw_text:
            raw_text = raw_text.split("</think>")[-1].strip()

        # Parse JSON from response (handle ```json ... ``` blocks)
        if "```json" in raw_text:
            raw_text = raw_text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```", 1)[1].rsplit("```", 1)[0].strip()
        else:
            # Fallback for conversational output: find first { and last }
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                raw_text = raw_text[start:end+1]

        parsed = json.loads(raw_text)
        experiments_data = parsed.get("experiments", [])
        print(f"  Successfully extracted {len(experiments_data)} experiment records.")

        # Log non-null field counts for debugging
        if experiments_data:
            non_null = sum(1 for v in experiments_data[0].values() if v is not None)
            total = len(experiments_data[0])
            print(f"  Coverage: populated {non_null} out of {total} fields in the first experiment.")

    except json.JSONDecodeError as e:
        print(f"  Failed to parse JSON from the model response: {e}")
        print(f"  Raw LLM output preview: {raw_text[:500]}")
        experiments_data = []
    except Exception as e:
        print(f"  LLM extraction encountered an error: {e}")
        experiments_data = []

    # Save intermediate
    experiments_json_path = os.path.join(output_dir, "experiments.json")
    with open(experiments_json_path, "w", encoding="utf-8") as f:
        json.dump({"experiments": experiments_data}, f, indent=4)
    print(f"  Saved extracted experiments to: {experiments_json_path}")

    return {
        "experiments_data": experiments_data,
        "figures_data": figures_data,
    }