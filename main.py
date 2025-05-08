import re
import math
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, conlist

MODEL_ASR_CORRECTION = "qwen3" # Low Reasoning, Text Output - Capable 7B
MODEL_SEGMENTATION = "qwen3" # Moderate Reasoning, Pydantic List - Test Pydantic adherence
MODEL_ACTION_EXTRACTION = "qwen3" # High Reasoning, Complex Pydantic List - POTENTIAL RELIABILITY RISK with 7B
MODEL_SUMMARY = "qwen3" # Moderate Reasoning, Text Output - Capable 7B
MODEL_CLARIFICATION = "qwen3" # High Reasoning, Complex Pydantic Object - POTENTIAL RELIABILITY RISK with 7B
MODEL_MEETING_MINUTES = "qwen3" # Moderate Reasoning, Structured Text Output - Test formatting quality

ASR_CHUNK_SIZE = 20 
ASR_CHUNK_OVERLAP = 5

class PydanticSpeakerUtterance(BaseModel):
    """Represents a single utterance with its speaker and timestamp. """
    speaker: str = Field(description="The identified speaker of the utterance.")
    utterance: str = Field(description="The text content of the utterance.")
    timestamp: str = Field(description="The timestamp of the utterance.")

class PydanticSegment(BaseModel):
    """Represents a topical segment of the meeting"""
    topic_name: str = Field(description="A concise name for the main topic of this segment.")
    utterance_indices: conlist(int, min_length=1) = Field(description="List of 0-based indices of utterances from the original transcript belonging to this topic.")

class PydanticActionItem(BaseModel):
    """Represents an action item identified in the meeting."""
    action_description: str = Field(description="A clear and concise description of the task or action to be performed.")
    assigned_to: Optional[str] = Field(None, description="The person(s) responsible. Should be from a provided speaker list or 'Unassigned'.")
    due_date: Optional[str] = Field(None, description="Any mentioned due date(e.g., 'next Friday', 'by EOD'). Null if not mentioned.")
    is_ambiguous: bool = Field(default=False, description="True if the action, assignee, or due date is vague or requires clarification.")
    ambiguity_reason: Optional[str] = Field(None, description="If ambiguous, a brief explaination of the ambiguity.")
    source_utterance_indices: List[int] = Field(default_factory=list, description="0-based indices of the utterances from which this action item was derived.")

# Renamed model for consistency
class PydanticSegmentWithUtterances(BaseModel):
    """Internal representation of a segment with full utterance details."""
    topic_name: str
    utterances_info: List[PydanticSpeakerUtterance]


# --- Define the State for the Graph (using Pydantic models) ---
class WorkflowState(TypedDict):
    raw_transcription: str
    parsed_transcription: Optional[List[PydanticSpeakerUtterance]]
    corrected_transcription: Optional[List[PydanticSpeakerUtterance]]
    quality_assessment: Optional[Literal["good", "poor"]]
    segmentation_input_text: Optional[str] # Used by Summary node (raw text)
    segmented_transcription: Optional[List[PydanticSegmentWithUtterances]] # Stores segments with utterance objects
    extracted_action_items: Optional[List[PydanticActionItem]]
    summary: Optional[str]
    action_validation_status: Optional[Literal["all_valid", "needs_clarification"]]
    finalized_action_items: Optional[List[PydanticActionItem]]
    meeting_minutes: Optional[str]
    formatted_outputs: Optional[Dict[str, Any]]
    error_message: Optional[str]

def get_ollama_llm(model_name: str, temperature: float=0.1) -> ChatOllama:
    """Helper to initialize ChatOllama"""
    return ChatOllama(
        model=model_name,
        temperature=temperature
    )

# --- Helper function to parse corrected chunk text ---
def parse_corrected_chunk_output(
    chunk_text: str, original_chunk_start_index: int, original_chunk: List[PydanticSpeakerUtterance]
) -> List[PydanticSpeakerUtterance]:
    """
    Uses the original chunk info for robustness.
    """
    corrected_utterances: List[PydanticSpeakerUtterance] = []
    utterance_line_pattern = re.compile(r"^\[(\d+)\]\s*(Speaker \w+)\s*(\(T\d{3}\)):\s*(.*)$", re.MULTILINE)

    # Create a dictionary mapping expected key (like "[idx]Speaker (Timestamp)") to the parsed corrected utterance text
    parsed_lines: Dict[str, str] = {}
    for line in chunk_text.strip().split("\n"):
        match = utterance_line_pattern.match(line)
        if match:
            try:
                # Use the captured global index, speaker, and timestamp to create a key
                global_idx = int(match.group(1))
                speaker = match.group(2).strip()
                timestamp = match.group(3).strip()
                corrected_text = match.group(4).strip()
                # Key format: "[global_idx]Speaker (Timestamp)" - ensure it matches the expected format
                key = f"[{global_idx}]{speaker} {timestamp}"
                parsed_lines[key] = corrected_text
            except (ValueError, IndexError) as e:
                print(f"Warning: Regex matched but failed to parse groups for line: '{line[:100]}...' Error: {e}")
                continue
        elif line.strip():
            pass # Suppress non-matching lines by default

    for i, original_item in enumerate(original_chunk):
        global_idx = original_chunk_start_index + i
        # Construct the key we expect from the LLM output for this original item
        expected_key = f"[{global_idx}]{original_item.speaker} ({original_item.timestamp})"

        corrected_utterance_text = parsed_lines.get(expected_key)

        if corrected_utterance_text is None:
            final_utterance_text = original_item.utterance
        else:
            final_utterance_text = corrected_utterance_text

        # Create the Pydantic object using the original speaker/timestamp and the final text
        corrected_utterances.append(PydanticSpeakerUtterance(
            speaker=original_item.speaker,
            utterance=final_utterance_text,
            timestamp=original_item.timestamp 
        ))

    return corrected_utterances

# --- Node Functions ---

def ingest_transcription_node(state: WorkflowState) -> WorkflowState:
    print("--- Ingesting Transcription ---")
    if not state.get("raw_transcription"):
        # Corrected: Added return statement
        return {**state, "error_message": "No raw transcription provided."}

    print(f"Raw transcription received: {state['raw_transcription'][:100]}...")
    return state

def preprocess_transcription_parsing_node(state: WorkflowState) -> WorkflowState:
    print("--- Preprocessing parsing Transcription ---")
    raw_text = state.get("raw_transcription")

    if not raw_text:
        return {**state, "error_message": "Raw transcription missing for parsing."}

    parsed: List[PydanticSpeakerUtterance] = []
    # Regex for initial parsing (assuming raw text is Speaker: Utterance format)
    simple_utterance_pattern = re.compile(r"^(Speaker \w+):\s*(.*)", re.MULTILINE)
    matches_simple = simple_utterance_pattern.findall(raw_text)

    for i, (speaker, utterance) in enumerate(matches_simple):
        parsed.append(PydanticSpeakerUtterance(
            speaker=speaker.strip(),
            utterance=utterance.strip(),
            timestamp=f"T{i:03d}" 
        ))

    if not parsed:
        print("Warning: Could not parse any utterances. Check transcription format and regex.")
        return {**state, "parsed_transcription": [], "error_message": "Failed to parse transcription."}

    print(f"Parsed {len(parsed)} utterances.")
    return {**state, "parsed_transcription": parsed}


# ASR Correction Node with Simplified Chunking
def llm_asr_correction_node(state: WorkflowState) -> WorkflowState:
    print(f"--- ASR Correction using Ollama model: {MODEL_ASR_CORRECTION} (Chunking enabled) ---")
    parsed_transcription = state.get("parsed_transcription")
    if not parsed_transcription:
        return {**state, "error_message": "Parsed transcription missing for ASR correction."}

    llm = get_ollama_llm(MODEL_ASR_CORRECTION)

    chunk_correction_prompt_template = PromptTemplate.from_template(
        "You are an expert ASR (Automatic Speech Recognition) correction assistant. "
        "Review the following chunk of meeting transcript. "
        "Correct any ASR errors (grammar, spelling, punctuation, misheard words) using the context within this chunk. "
        "Preserve the original meaning and speaker's intent. "
        "If an utterance seems correct, output it as is. Avoid making unnecessary changes or adding extra commentary.\n\n"
        "Output format: For each utterance, output its global index, speaker, timestamp, and corrected text on a new line, exactly like this:\n"
        "[Index] Speaker (Timestamp): Corrected Utterance Text\n\n"
        "Example: [0] Speaker Alpha (T000): Hello team, let's start.\n\n"
        "Transcript Chunk:\n{transcript_chunk}\n\n"
        "Corrected Transcript Chunk (Maintain order and format):"
    )

    chunk_asr_chain = chunk_correction_prompt_template | llm | StrOutputParser()

    all_corrected_utterances: List[PydanticSpeakerUtterance] = []
    total_utterances = len(parsed_transcription)

    # Simplified non-overlapping chunking logic
    chunk_size = ASR_CHUNK_SIZE

    print(f"Processing {total_utterances} utterances in chunks of size {chunk_size}.")

    # Loop through the transcript in non-overlapping chunks
    for i in range(0, total_utterances, chunk_size):
        start_index = i
        end_index = min(start_index + chunk_size, total_utterances)
        current_chunk_original = parsed_transcription[start_index:end_index]

        if not current_chunk_original:
             # This should not be reached if total_utterances > 0 and chunk_size > 0
             print(f"Warning: Skipping empty chunk slice starting at index {i}")
             continue

        print(f"Processing chunk indices {start_index} to {end_index-1}...")

        # Format the current chunk for the prompt
        current_chunk_text = ""
        for j, item in enumerate(current_chunk_original):
            global_idx = start_index + j # Use the GLOBAL index relative to the full transcript
            current_chunk_text += f"[{global_idx}] {item.speaker} ({item.timestamp}): {item.utterance}\n"

        try:
            # Invoke the LLM chain for the current chunk
            corrected_chunk_text_raw = chunk_asr_chain.invoke({
                "transcript_chunk": current_chunk_text
            })

            parsed_chunk_utterances = parse_corrected_chunk_output(
                corrected_chunk_text_raw, start_index, current_chunk_original
            )

            # Extend the main list with the corrected utterances from this chunk

            all_corrected_utterances.extend(parsed_chunk_utterances)

        except Exception as e:
            print(f"Error processing chunk (indices {start_index} to {end_index-1}): {e}. Falling back to original text for this chunk.")
            # Fallback: If LLM or parsing fails for a chunk, use the original utterances for that chunk
            all_corrected_utterances.extend(current_chunk_original)

    if len(all_corrected_utterances) != total_utterances:
         print(f"Warning: Final corrected utterance count ({len(all_corrected_utterances)}) does not match original ({total_utterances}). This may impact downstream steps relying on indices.")
    corrected_transcription_pydantic = all_corrected_utterances 

    segmentation_input_chunks = []
    if corrected_transcription_pydantic:
        for item in corrected_transcription_pydantic:
             segmentation_input_chunks.append(f"{item.speaker} ({item.timestamp}): {item.utterance}")
    else:
         print("Warning: Corrected transcription list is empty.")

    print(f"ASR chunking correction completed. Final corrected utterance count: {len(corrected_transcription_pydantic)}")

    return {
        **state,
        "corrected_transcription": corrected_transcription_pydantic,
        "segmentation_input_text": "\n".join(segmentation_input_chunks)
    }

# --- Keep check_transcription_quality_node and handle_poor_quality_node ---
def check_transcription_quality_node(state: WorkflowState) -> WorkflowState:
    print("--- Checking Transcription Quality ---")
    corrected_transcription = state.get("corrected_transcription")
    if not corrected_transcription or len(corrected_transcription)==0:
        print("Quality Check: Poor (No Content)")
        return {**state, "quality_assessment": "poor", "error_message": "No content after ASR correction."}

    total_words = 0
    short_utterances = 0
    for item in corrected_transcription:
        words = len(item.utterance.split())
        total_words+=words
        if words < 3:
            short_utterances += 1
    avg_words = total_words / len(corrected_transcription) if corrected_transcription else 0

    if avg_words < 2.0 or (short_utterances / len(corrected_transcription) / len(corrected_transcription) > 0.6): # Corrected calculation for ratio
        print(f"Quality Check: Poor (Avg words: {avg_words:.2f}, High short utterances: {short_utterances})")
        return {**state, "quality_assessment": "poor", "error_message": "Transcription quality deemed poor."}

    print("Quality Check: Good")
    return {**state, "quality_assessment": "good"}

def handle_poor_quality_node(state: WorkflowState) -> WorkflowState:
    print("--- Handling Poor Quality Transcription ---")
    error_msg = state.get("error_message", "Processing halted due to poor quality.")
    print(f"Error: {error_msg}")
    return {**state, "formatted_outputs": {"status": "Failed: Poor Transcription Quality", "details": error_msg}}

# --- Keep segment_transcription_node ---
def segment_transcription_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Segmenting Transcription using Ollama model: {MODEL_SEGMENTATION} ---")
    corrected_transcription_list = state.get("corrected_transcription")
    if not corrected_transcription_list:
        return {**state, "error_message": "Corrected transcription missing for segmentation."}

    parser = PydanticOutputParser(pydantic_object=List[PydanticSegment])
    llm = get_ollama_llm(MODEL_SEGMENTATION, temperature=0.2)

    prompt_template = PromptTemplate.from_template(
        """You are a topic segmentation expert. Given the following meeting transcript, identify the main topics discussed. Group consecutive utterances under each topic. The transcript has {num_utterances} utterances, presented below with their 0-based indices. Focus on identifying distinct topical segments.

        {format_instructions}

        Numbered Transcript:
        {numbered_transcript}

        Respond with the list of segments as per the format instructions.
        """
    )
    segmentation_chain = prompt_template | llm | parser

    numbered_transcript = ""
    for i, item in enumerate(corrected_transcription_list):
        numbered_transcript += f"[{i}] {item.speaker}: {item.utterance}\n"

    segments_with_utterances: List[PydanticSegmentWithUtterances] = []
    try:
        llm_segments_output: List[PydanticSegment] = segmentation_chain.invoke({
            "numbered_transcript": numbered_transcript[:24000], # Increased truncation slightly, adjust based on model
            "num_utterances": len(corrected_transcription_list),
            "format_instructions": parser.get_format_instructions(),
        })

        for seg_info in llm_segments_output:
            segment_utterances_pydantic: List[PydanticSpeakerUtterance] = [] # Corrected: Use PydanticSpeakerUtterance
            for idx in seg_info.utterance_indices:
                if 0 <= idx < len(corrected_transcription_list):
                    segment_utterances_pydantic.append(corrected_transcription_list[idx])
                else:
                     print(f"Warning: LLM hallucinated index {idx} in segmentation.") # Warn about hallucinated indices
            if segment_utterances_pydantic:
                segments_with_utterances.append(PydanticSegmentWithUtterances(
                    topic_name=seg_info.topic_name,
                    utterances_info=segment_utterances_pydantic
                ))
            else:
                 print(f"Warning: Segment '{seg_info.topic_name}' has no valid utterances.") # Warn about segments with no valid indices

        if not segments_with_utterances and corrected_transcription_list: # Fallback only if there is content
            raise ValueError("LLM returned no valid segments or mapping failed.")


    except Exception as e:
        print(f"Error during segmentation or parsing: {e}. Using full transcript as one segment.")
        if corrected_transcription_list: # Fallback only if there is content
             segments_with_utterances = [PydanticSegmentWithUtterances(
                topic_name="General Discussion",
                utterances_info=corrected_transcription_list
             )]
        else:
             segments_with_utterances = [] # Empty if no content was available


    print(f"Segmentation completed into {len(segments_with_utterances)} segments.")
    return {**state, "segmented_transcription": segments_with_utterances}

# --- Keep extract_action_items_node ---
def extract_action_items_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Extracting Action Items using Ollama model: {MODEL_ACTION_EXTRACTION} ---")
    corrected_transcript_list = state.get("corrected_transcription")
    if not corrected_transcript_list:
        return {**state, "error_message": "Corrected transcription missing for action item extraction."}

    parser = PydanticOutputParser(pydantic_object=List[PydanticActionItem])
    llm = get_ollama_llm(MODEL_ACTION_EXTRACTION, temperature=0.1)
    speaker_list = sorted(list(set(item.speaker for item in corrected_transcript_list)))

    prompt_template = PromptTemplate.from_template(
        """
        You are an expert in identifying action items from meeting transcripts. Analyze the provided meeting transcript carefully. Identify all tasks, commitments, or actions. Use the speaker list: {speaker_list} for assignments. For each action item, provide details as specified in the format instructions. Pay close attention to 'is_ambiguous' and 'ambiguity_reason' fields. Also, correctly list the 'source_utterance_indices' (0-based from the numbered transcript below).

        {format_instructions}

        Numbered Transcript:
        {numbered_transcript}

        Respond with the list of action items.
        """
    )
    action_item_chain = prompt_template | llm | parser

    numbered_transcript_for_actions = ""
    for i, item in enumerate(corrected_transcript_list):
        numbered_transcript_for_actions += f"[{i}] {item.speaker}: {item.utterance}\n"

    extracted_pydantic_items: List[PydanticActionItem] = []
    try:
        extracted_pydantic_items = action_item_chain.invoke({
            "numbered_transcript": numbered_transcript_for_actions,
            "speaker_list": ", ".join(speaker_list),
            "format_instructions": parser.get_format_instructions(),
        })
    except Exception as e:
        print(f"Error during action item extraction or parsing: {e}")
        extracted_pydantic_items = []


    print(f"Extracted {len(extracted_pydantic_items)} potential action items.")
    return {**state, "extracted_action_items": extracted_pydantic_items}

# --- Keep generate_summary_node ---
def generate_summary_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Generating Summary using Ollama model: {MODEL_SUMMARY} ---")
    # segmentation_input_text should be the raw text from corrected transcription
    full_corrected_text = state.get("segmentation_input_text")
    if not full_corrected_text:
           return {**state, "error_message": "Corrected transcription text missing for summary."}

    llm = get_ollama_llm(MODEL_SUMMARY, temperature=0.3)
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert meeting summarizer. Based on the following meeting transcript, provide a concise summary of key discussion points, decisions, and main outcomes. Aim for 3-5 bullet points or a short paragraph.

        Transcript:
        {transcript}

        Summary:
        """
    )
    summary_chain = prompt_template | llm | StrOutputParser()
    summary = "Summary generation failed."
    try:
        # Truncate for safety if needed, adjust based on model's context window
        summary = summary_chain.invoke({"transcript": full_corrected_text[:24000]})
    except Exception as e:
        print(f"Error during summary generation: {e}")
    print(f"Summary generated: {summary[:100]}...")
    return {**state, "summary": summary}


# --- Keep validate_action_items_node ---
def validate_action_items_node(state: WorkflowState) -> WorkflowState:
    print("--- Validating Action Items ---")
    action_items = state.get("extracted_action_items")
    if action_items is None:
        # If extraction failed or returned None, treat as all valid but empty list
        return {**state, "action_validation_status": "all_valid", "finalized_action_items": []}

    needs_clarification = any(item.is_ambiguous for item in action_items)
    if needs_clarification:
        print("Validation: Some action items need clarification.")
        return {**state, "action_validation_status": "needs_clarification"}
    else:
        print("Validation: All action items seem valid.")
        return {**state, "action_validation_status": "all_valid", "finalized_action_items": action_items}

# --- Keep attempt_llm_clarification_node ---
def attempt_llm_clarification_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Attempting LLM Clarification using Ollama: {MODEL_CLARIFICATION} ---")
    action_items = state.get("extracted_action_items", [])
    corrected_transcript_list = state.get("corrected_transcription")

    if not corrected_transcript_list:
        print("Cannot attempt clarification: missing corrected transcript list.")
        # If clarification step is reached but transcript is missing, return items as is
        return {**state, "finalized_action_items": action_items}

    # Corrected: Added pydantic_object=
    parser = PydanticOutputParser(pydantic_object=PydanticActionItem)
    llm = get_ollama_llm(MODEL_CLARIFICATION, temperature=0.2)

    prompt_template = PromptTemplate.from_template(
        """
        Resolve ambiguities in the following action item based on the transcript context.

        Original Action Item:
        Description: {action_description}
        Assigned To: {assigned_to}
        Due Date: {due_date}
        Ambiguity Reason: {ambiguity_reason}

        Relevant Transcript Context (utterances around source, with their 0-based indices):
        {context_utterances}

        If you can clarify, update the fields and set 'is_ambiguous' to false.
        If it remains ambiguous, keep 'is_ambiguous' true and refine 'ambiguity_reason'.
        Preserve the original list of 'source_utterance_indices': {source_utterance_indices_str}
        {format_instructions}


        Clarified Action Item (JSON object):
        """
    )
    clarification_chain = prompt_template | llm | parser

    clarified_items_list: List[PydanticActionItem] = []
    for item in action_items:
        if item.is_ambiguous:
            print(f"Attempting to clarify: {item.action_description[:50]}...")
            context_utterances_text = "No specific source utterances found or could not retrieve context."
            context_window_size = 5
            if item.source_utterance_indices:
                try:
                    # Determine min and max index to center context window
                    min_source_idx = min(item.source_utterance_indices)
                    max_source_idx = max(item.source_utterance_indices)

                    context_start_idx = max(0, min_source_idx - context_window_size)
                    context_end_idx = min(len(corrected_transcript_list) -1, max_source_idx + context_window_size)

                    context_lines = []
                    # Iterate through indices to build context text
                    for k in range(context_start_idx, context_end_idx + 1):
                        if 0 <= k < len(corrected_transcript_list): # Double check index bounds
                            context_lines.append(f"[{k}] {corrected_transcript_list[k].speaker}: {corrected_transcript_list[k].utterance}")
                    context_utterances_text = "\n".join(context_lines)

                    # Optional: Truncate context if it gets too long for the LLM's context window
                    if len(context_utterances_text) > 4000: # Adjust based on model capability
                         context_utterances_text = context_utterances_text[:4000] + "\n... (context truncated)"

                except Exception as context_e:
                     print(f"Error retrieving context for clarification (indices {item.source_utterance_indices}): {context_e}")
                     context_utterances_text = "Error retrieving context."
            else:
                 context_utterances_text = "No source indices available to retrieve context."


            try:
                clarified_action = clarification_chain.invoke({
                    "action_description": item.action_description,
                    "assigned_to": item.assigned_to or "N/A",
                    "due_date": item.due_date or "N/A",
                    "ambiguity_reason": item.ambiguity_reason or "No reason given",
                    "context_utterances": context_utterances_text,
                    # Ensure source_utterance_indices_str is passed as string
                    "source_utterance_indices_str": str(item.source_utterance_indices),
                    "format_instructions": parser.get_format_instructions(),
                })
                # Ensure source_utterance_indices is preserved from original, not hallucinated by LLM
                # LLMs can sometimes drop or hallucinate fields in structured output.
                clarified_action.source_utterance_indices = item.source_utterance_indices
                clarified_items_list.append(clarified_action)
            except Exception as e:
                print(f"Error during LLM clarification for '{item.action_description[:50]}...': {e}. Keeping original.")
                clarified_items_list.append(item) # Keep original on error if clarification fails
        else:
            clarified_items_list.append(item) # Keep non-ambiguous items as is

    return {**state, "finalized_action_items": clarified_items_list}

# --- Keep generate_meeting_minutes_node ---
def generate_meeting_minutes_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Generating Meeting Minutes using Ollama model: {MODEL_MEETING_MINUTES} ---")
    summary = state.get("summary", "No summary available.")
    action_items = state.get("finalized_action_items", [])
    # full_corrected_text is used for 'Key Discussion Points'
    full_corrected_text = state.get("segmentation_input_text", "Full transcript not available.")
    # Use a bit more text for title, split on newline and take first line, strip whitespace
    meeting_title = state.get("raw_transcription", "Meeting")[:100].split('\n')[0].strip()
    meeting_date = "Today" # Or derive from actual data if available

    attendees_list = []
    if state.get("corrected_transcription"):
        # Corrected: Use PydanticSpeakerUtterance
        attendees_list = sorted(list(set(item.speaker for item in state["corrected_transcription"]))) # type: ignore

    action_items_str = ""
    if action_items:
        for i, ai in enumerate(action_items):
            assignee = ai.assigned_to or "Unassigned"
            due = ai.due_date or "N/A"
            flag = " (AMBIGUOUS)" if ai.is_ambiguous else ""
            ambiguity_detail = f" - Reason: {ai.ambiguity_reason}" if ai.is_ambiguous and ai.ambiguity_reason else ""
            # You could optionally add source indices here for reference, e.g., f" (Source: {ai.source_utterance_indices})"
            action_items_str += f"- [ ] {ai.action_description}{flag}\n  Assigned To: {assignee}\n  Due Date: {due}{ambiguity_detail}\n"
    else: action_items_str = "No action items identified."

    llm = get_ollama_llm(MODEL_MEETING_MINUTES, temperature=0.3)
    prompt_template = PromptTemplate.from_template(
        """
        You are a professional scribe creating Minutes of Meeting (MoM).
        Generate comprehensive minutes based on the provided information.\n\n"
        Meeting Title: {title}
        Date: {date}
        Attendees: {attendees}

        1. Meeting Summary:\n{summary}

        2. Key Discussion Points / Decisions:
        Analyze the provided full transcript and list 3-7 key topics, discussion points, or decisions that were made, different from the summary and action items unless they are major points. Use bullet points.

        3. Action Items:\n{action_items}

        Full Transcript (for context and deriving key discussion points):\n{full_transcript}

        Compile the MoM in a clean, professional, and well-formatted markdown format.
        """
    )
    mom_chain = prompt_template | llm | StrOutputParser()
    mom = "Meeting minutes generation failed."
    try:
        # Truncate full transcript for LLM context if necessary
        transcript_for_mom = full_corrected_text[:16000] if full_corrected_text else "Transcript not available." # Adjust truncation based on model

        mom = mom_chain.invoke({
            "title": meeting_title,
            "date": meeting_date,
            "attendees": ", ".join(attendees_list) if attendees_list else "N/A",
            "summary": summary,
            "action_items": action_items_str,
            "full_transcript": transcript_for_mom
        })
    except Exception as e:
        print(f"Error during MoM generation: {e}")
    print("Meeting minutes generated.")
    return {**state, "meeting_minutes": mom}


# --- Keep format_outputs_node ---
def format_outputs_node(state: WorkflowState) -> WorkflowState:
    print("--- Formatting Outputs ---")
    final_action_items = state.get("finalized_action_items", [])
    # Convert Pydantic action items to dicts for JSON serializable output
    action_items_dict = [item.model_dump() for item in final_action_items]

    final_outputs = {
        "summary": state.get("summary", "N/A"),
        "action_items": action_items_dict,
        "meeting_minutes": state.get("meeting_minutes", "N/A"),
        # Check quality assessment status to determine overall success/failure
        "status": "Success" if state.get("quality_assessment") == "good" and not state.get("error_message") else "Failed",
        "error_details": state.get("error_message", None)
    }

    # Ensure the poor quality failure status overrides success if it occurred earlier
    if state.get("quality_assessment") == "poor":
         final_outputs["status"] = "Failed: Poor Transcription Quality"
         final_outputs["details"] = state.get("error_message", "Processing halted early due to poor quality.")
         # Clear other outputs if quality was poor and processing stopped early
         if not state.get("meeting_minutes"): # Check if minutes were generated (implying it passed quality)
             final_outputs["summary"] = "N/A"
             final_outputs["action_items"] = []
             final_outputs["meeting_minutes"] = "N/A"


    print("Outputs formatted.")
    return {**state, "formatted_outputs": final_outputs}

# --- Conditional Edge Functions ---
def decide_quality_branch(state: WorkflowState) -> Literal["segment_transcription", "handle_poor_quality"]:
    # Corrected check for quality_assessment being set
    if state.get("quality_assessment") == "good":
        return "segment_transcription"
    else:
        # Assume poor if not 'good', including if it's None (error occurred before quality check)
        return "handle_poor_quality"

def decide_action_item_validation_branch(state: WorkflowState) -> Literal["attempt_clarification", "generate_mom"]:
    return "attempt_clarification" if state.get("action_validation_status") == "needs_clarification" else "generate_mom"


# --- Build the Graph ---
def build_graph() -> StateGraph:
    workflow = StateGraph(WorkflowState)
    workflow.add_node("ingest_transcription", ingest_transcription_node)
    workflow.add_node("preprocess_parsing", preprocess_transcription_parsing_node)
    workflow.add_node("llm_asr_correction", llm_asr_correction_node)
    workflow.add_node("check_quality", check_transcription_quality_node)
    workflow.add_node("handle_poor_quality", handle_poor_quality_node)
    workflow.add_node("segment_transcription", segment_transcription_node)
    workflow.add_node("extract_actions", extract_action_items_node)
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("validate_actions", validate_action_items_node)
    workflow.add_node("attempt_clarification", attempt_llm_clarification_node)
    workflow.add_node("generate_mom", generate_meeting_minutes_node)
    workflow.add_node("format_outputs", format_outputs_node)

    workflow.set_entry_point("ingest_transcription")
    workflow.add_edge("ingest_transcription", "preprocess_parsing")
    workflow.add_edge("preprocess_parsing", "llm_asr_correction")
    workflow.add_edge("llm_asr_correction", "check_quality")
    workflow.add_conditional_edges("check_quality", decide_quality_branch, {
        "segment_transcription": "segment_transcription",
        "handle_poor_quality": "handle_poor_quality",
    })
    workflow.add_edge("handle_poor_quality", END)
    workflow.add_edge("segment_transcription", "extract_actions")
    workflow.add_edge("extract_actions", "generate_summary")
    workflow.add_edge("generate_summary", "validate_actions")
    workflow.add_conditional_edges("validate_actions", decide_action_item_validation_branch, {
        "attempt_clarification": "attempt_clarification",
        "generate_mom": "generate_mom"
    })
    workflow.add_edge("attempt_clarification", "generate_mom")
    workflow.add_edge("generate_mom", "format_outputs")
    workflow.add_edge("format_outputs", END)
    return workflow.compile()

# --- Main Execution ---
if __name__ == "__main__":

    app = build_graph()
    # Moved sample_transcription definition below build_graph to avoid potential issues

    # --- Sample Transcript ---
    # Your existing sample transcript (now used below)

    # --- Test Cases ---
    # You can swap this out with different sample transcripts to test the workflow
    sample_transcription_basic = """
Speaker Alpha: Hello team, let's start. We need to discuss the Q2 project plan.
Speaker Beta: Agreed. I think the main deliverable should be finalized by next Wednesday.
Speaker Alpha: Okay, Beta, can you take the lead on drafting that?
Speaker Gamma: I can help Beta with the market research part.
Speaker Alpha: Good idea, Gamma. Let's wrap this up.
""" # Basic, clear, no errors, simple action

    sample_transcription_asr_errors = """
Speaker Alpha: Hello team, let's start. We need to discuz the Q2 project plan.
Speaker Beta: Agreed. I think the main deliverable should be finalized by next Wensday.
Speaker Alpha: Okay, Beta, can you take the lead on drafting that?
Speaker Gamma: I can help Beta with the market resurch part. Maybe sumone can look into the budget?
Speaker Alpha: Good idea, Gamma. Let's assign the budget review to... well, we need to figure that out. It is very importnt.
Speaker Beta: I also think we should shedule a follow-up for early next week.
Speaker Alpha: Yes, a follow-up meeting is essential. I'll send out an invite. End of meeting.
Speaker Delta: Just to add, regarding the marketing campain, we need to ensure the cretives are approved by EOD Fryday. Is that clear for evryone?
Speaker Alpha: Yes, Delta, that's kritical. Mark, can you handle the creative aprovals?
Speaker Mark: I can take that on. Aprovals by Fryday EOD. Got it.
Speaker Gamma: And I'll get that market resurch over to Beta by Tusday morning.
Speaker Beta: Pefect, thanks Gamma. That will help finalize the deliverable draft by Wensday.
Speaker Alpha: Excellent. So Beta drafts deliverable by Wensday, Gamma provides resurch by Tusday, Mark approves cretives by Fryday. We also need to confirm that budget review owner. I will follow up offline.
Speaker Delta: Sounds good.
Speaker Alpha: Alright, anything else? No? Meeting ajourned.
""" # Includes multiple ASR errors, clear and ambiguous action items, specific and vague due dates

    sample_transcription_segmentation = """
Speaker Alpha: Okay team, let's dive into project X updates. Sarah, how's the development coming along?
Speaker Sarah: Development is on track. We finished module A yesterday and are starting B today. Should be done by Friday.
Speaker Beta: Great. Any blockers?
Speaker Sarah: None so far, just standard integration tests remaining.
Speaker Alpha: Good. Let's switch gears to the Q3 budget proposal. David, did you get that draft put together?
Speaker David: Yes, I sent it out for review this morning. We're proposing an increase in the marketing spend.
Speaker Gamma: I saw that. I have some questions about the ROI projections on slide 5. Can we discuss those?
Speaker David: Absolutely. Let's go over those figures now.
Speaker Alpha: Alright, thanks for the budget overview. Finally, let's talk about next steps for the onboarding process improvements.
Speaker Chris: I put together a quick doc outlining potential changes. We need to pilot this with the next batch of hires.
Speaker Alpha: Okay, Chris, can you schedule a meeting next week to walk us through the pilot plan?
Speaker Chris: Will do.
Speaker Alpha: Excellent. Meeting adjourned.
""" # Includes distinct topic shifts for segmentation testing

    sample_transcription_ambiguity = """
Speaker Alpha: We need someone to handle the follow-up emails after this call. It's important we get those out quickly.
Speaker Beta: I can probably take that.
Speaker Alpha: Great, Beta. Please do. And don't forget the report needs to be updated soon.
Speaker Gamma: Who is updating the report?
Speaker Alpha: Uh, someone from the team. We talked about it last week. Let's get that sorted out by the end of the week, shall we?
Speaker Beta: The proposal also needs a final review before sending it out.
Speaker Alpha: Right. Could someone give that a quick look? Ideally tomorrow.
""" # Focuses on ambiguous assignments and vague due dates

    sample_transcription_long_test_chunking = sample_transcription_asr_errors * 3 # Repeat ASR error transcript to make it longer than ASR_CHUNK_SIZE

    # --- Select a test case ---
    # Choose one of the sample_transcription variables above
    current_test_transcript = sample_transcription_long_test_chunking # Example: using the long test case

    print("\n--- Using Sample Transcript ---")
    print(current_test_transcript[:200] + "...\n") # Print first 200 chars of the selected transcript

    initial_state: WorkflowState = {"raw_transcription": current_test_transcript}

    print("\n--- Starting Transcription Processing Workflow ---")
    # Added error handling around invocation in case of graph execution errors
    try:
        final_state = app.invoke(initial_state)
        print("\n--- Workflow Execution Complete ---")
    except Exception as e:
        print(f"\n--- Workflow Execution Failed ---")
        print(f"An error occurred during workflow execution: {e}")
        # Capture the error in a simplified final state for reporting
        final_state = {"formatted_outputs": {"status": "Failed: Workflow Execution Error", "details": str(e)}}


    print("\n--- Final Formatted Outputs ---")
    if final_state.get("formatted_outputs"):
        output_data = final_state["formatted_outputs"]
        print(f"Status: {output_data.get('status')}")
        if output_data.get('status') == "Success":
            print("\nSummary:")
            print(output_data.get("summary", "N/A"))
            print("\nAction Items:")
            # Corrected loop to iterate through list of dicts
            for i, ai_dict in enumerate(output_data.get("action_items", [])):
                print(f"  {i+1}. Description: {ai_dict.get('action_description')}")
                print(f"      Assigned To: {ai_dict.get('assigned_to', 'N/A')}")
                print(f"      Due Date: {ai_dict.get('due_date', 'N/A')}")
                if ai_dict.get('is_ambiguous'):
                    print(f"      AMBIGUOUS: {ai_dict.get('ambiguity_reason', 'No reason given')}")
                print(f"      Source Indices: {ai_dict.get('source_utterance_indices', [])}")
            print("\nMeeting Minutes:")
            print(output_data.get("meeting_minutes", "N/A"))
        else:
            print(f"Error Details: {output_data.get('details')}")
    else:
        print("No formatted outputs found in the final state.")
        print("\n--- Full Final State (selected fields) ---")
        # Corrected loop to handle potential None values safely
        for key, value in final_state.items():
            if key not in ['raw_transcription', 'segmentation_input_text']: # Exclude very verbose fields
                 # Safely print value, truncating if string and long
                 value_repr = str(value)
                 print(f"{key}: {value_repr[:500] + '...' if len(value_repr) > 500 else value_repr}")
