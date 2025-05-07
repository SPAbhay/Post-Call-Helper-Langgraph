import re
import math
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, conlist

MODEL_ASR_CORRECTION = "qwen3" # without reasoning
MODEL_SEGMENTATION = "qwen3" # with reasoning
MODEL_ACTION_EXTRACTION = "qwen3" # with reasoning
MODEL_SUMMARY = "qwen3" # with reasoning
MODEL_CLARIFICATION = "qwen3" # with reasoning 
MODEL_MEETING_MINUTES = "qwen3" # with reasoning

ASR_CHUNK_SIZE = 20 # Number of utterances per chunk
ASR_CHUNK_OVERLAP = 5 # Number of utterances to overlap between chunks

class SpeakerUtterance(BaseModel):
    """Represents a single utterance with its speaker and timestamp. """
    speaker: str = Field(description="The identified speaker of the utterance.")
    utterance: str = Field(description="The text content of the utterance.")
    timestamp: str = Field(description="The timestamp of the utterance.")
    
class Segment(BaseModel):
    """Represents a topical segment of the meeting"""
    topic_name: str = Field(description="A concise name for the main topic of this segment.")
    utterance_indices: conlist(int, min_length=1) = Field(description="List of 0-based indices of utterances from the original transcript belonging to this topic.")
    
class ActionItem(BaseModel):
    """Represents an action item identified in the meeting."""
    action_description: str = Field(description="A clear and concise description of the task or action to be performed.")
    assigned_to: Optional[str] = Field(None, description="The person(s) responsible. Should be from a provided speaker list or 'Unassigned'.")
    due_date: Optional[str] = Field(None, description="Any mentioned due date(e.g., 'next Friday', 'by EOD'). Null if not mentioned.")
    is_ambiguous: bool = Field(default=False, description="True if the action, assignee, or due date is vague or requires clarification.")
    ambiguity_reason: Optional[str] = Field(None, description="If ambiguous, a brief explaination of the ambiguity.")
    source_utterance_indices: List[int] = Field(default_factory=list, description="0-based indices of the utterances from which this action item was derived.")
    
class SegmentWithUtterances(BaseModel):
    """Internal representation of a segment with full utterance details."""
    topic_name: str
    utterances_info: List[SpeakerUtterance]

    
class WorkflowState(TypedDict):
    raw_transcription: str
    parsed_transcription: Optional[List[SpeakerUtterance]]    
    corrected_transcription: Optional[List[SpeakerUtterance]]
    quality_assessment: Optional[Literal["good", "poor"]]
    segmentation_input_text: Optional[str]
    segmented_transcription: Optional[List[SegmentWithUtterances]]
    extracted_action_items: Optional[List[ActionItem]]
    summary: Optional[str]
    action_validation_status: Optional[Literal["all_valid", "needs_clarification"]]
    finalized_action_items: Optional[List[ActionItem]]
    meeting_minutes: Optional[str]
    formatted_outputs: Optional[Dict[str, Any]]
    error_message: Optional[str]
    
def get_ollama_llm(model_name: str, temperature: float=0.1) -> ChatOllama:
    """Helper to initialize ChatOllama"""
    return ChatOllama(
        model=model_name, 
        temperature=temperature
    )
    
def parse_corrected_chunk_output(chunk_text: str, original_chunk_start_index: int, original_chunk: List[SpeakerUtterance]) -> List[SpeakerUtterance]:
    """Parses the LLM's corrected text output for a chunk back into Utterance objects."""
    corrected_utterances: List[SpeakerUtterance] = []
    
    utterance_line_pattern = re.compile(r"^\[(\d+)\]\s*(Speaker \w+)\s*(\(T\d{3}\)):\s*(.*)$", re.MULTILINE)

    parsed_lines: Dict[str, str] = {}
    for line in chunk_text.strip().split("\n"):
        match = utterance_line_pattern.match(line)
        if match:
            try:
                global_idx = int(match.group(1))
                speaker = match.group(2).strip()
                timestamp = match.group(3).strip()
                corrected_text = match.group(4).strip()
                key = f"[{global_idx}]{speaker} {timestamp}"
                parsed_lines[key] = corrected_text
            except (ValueError, IndexError):
                # Handle cases where regex matched but groups were not as expected (shouldn't happen with this regex)
                print(f"Warning: Regex matched but failed to parse groups for line: {line[:100]}...")
                continue
        elif line.strip():
            # Optional: Print lines that didn't match for debugging LLM formatting issues
            print(f"Debug: Line did not match pattern: {line[:100]}...")

    for i, original_item in enumerate(original_chunk):
        global_idx = original_chunk_start_index + i
        expected_key = f"[{global_idx}]{original_item.speaker} ({original_item.timestamp})"
        
        corrected_utterance_text = parsed_lines.get(expected_key)
        
        if corrected_utterance_text is None:
            # If we couldn't find a corrected version for this specific utterance key,
            # fall back to the original utterance text.
            print(f"Warning: No corrected version found for utterance [{global_idx}] - falling back to original.")
            final_utterance_text = original_item.utterance
        else:
            final_utterance_text = corrected_utterance_text

        corrected_utterances.append(SpeakerUtterance(
            speaker=original_item.speaker,
            utterance=final_utterance_text,
            timestamp=original_item.timestamp # Preserve original timestamp
        ))

    return corrected_utterances
    
def ingest_transcription_node(state: WorkflowState) -> WorkflowState:
    print("--- Ingesting Transcription ---")
    if not state.get("raw_transcription"):
        {**state, "error_message": "No raw transcription provided"}

    print(f"Raw transcription received: {state['raw_transcription'][:100]}...")
    return state

def preprocess_transcription_parsing_node(state: WorkflowState) -> WorkflowState:
    print("--- Preprocessing parsing Transcription ---")
    raw_text = state.get("raw_transcription")
    
    if not raw_text:
        return {**state, "error_message": "Raw transcription missing for parsing."}
    
    parsed: List[SpeakerUtterance] = []
    simple_utterance_pattern = re.compile(r"^(Speaker \w+):\s*(.*)", re.MULTILINE)
    matches_simple = simple_utterance_pattern.findall(raw_text)
    
    for i, (speaker, utterance) in enumerate(matches_simple):
        parsed.append(SpeakerUtterance(
            speaker=speaker.strip(),
            utterance=utterance.strip(),
            timestamp=f"T{i:03d}"
        ))
        
    if not parsed:
        print("Warning: Could not parse any utterances. Check transcription format and regex.")
        return {**state, "parsed_transcription": [], "error_message": "Failed to parse transcription."}

    print(f"Parsed {len(parsed)} utterances.")
    return {**state, "parsed_transcription": parsed}

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
        "Output format: For each utterance, output its global index, speaker, timestamp, and corrected text on a new line, like this:\n"
        "[Index] Speaker (Timestamp): Corrected Utterance Text\n\n"
        "Example: [0] Speaker Alpha (T000): Hello team, let's start.\n\n"
        "Transcript Chunk:\n{transcript_chunk}\n\n"
        "Corrected Transcript Chunk:"
    )

    chunk_asr_chain = chunk_correction_prompt_template | llm | StrOutputParser()

    corrected_transcription_pydantic: List[SpeakerUtterance] = []
    total_utterances = len(parsed_transcription)

    # Calculate number of steps (chunks), accounting for overlap
    step_size = ASR_CHUNK_SIZE - ASR_CHUNK_OVERLAP
    num_steps = math.ceil(total_utterances / step_size) if total_utterances > 0 else 0

    print(f"Processing {total_utterances} utterances in {num_steps} chunks (size={ASR_CHUNK_SIZE}, overlap={ASR_CHUNK_OVERLAP}).")

    for i in range(num_steps):
        start_index = i * step_size
        end_index = min(start_index + ASR_CHUNK_SIZE, total_utterances)

        if i == num_steps - 1 and end_index < start_index + 1:
            start_index = max(0, end_index - ASR_CHUNK_SIZE) # Adjust start for last small chunk if needed
            current_chunk = parsed_transcription[start_index:end_index]
        else:
            current_chunk = parsed_transcription[start_index:end_index]


        if not current_chunk:
             print(f"Warning: Skipping empty chunk at step {i}")
             continue

        print(f"Processing chunk {i+1}/{num_steps} (indices {start_index} to {end_index-1})...")

        # Format the current chunk for the prompt
        current_chunk_text = ""
        for j, item in enumerate(current_chunk):
            # Use the GLOBAL index for the LLM's reference
            global_idx = start_index + j
            current_chunk_text += f"[{global_idx}] {item.speaker} ({item.timestamp}): {item.utterance}\n"

        try:
            corrected_chunk_text = chunk_asr_chain.invoke({
                "transcript_chunk": current_chunk_text
            })

            # Parse the corrected chunk text output from the LLM
            parsed_chunk_utterances = parse_corrected_chunk_output(
                corrected_chunk_text, start_index, current_chunk
            )

            if i == num_steps - 1:
                 corrected_transcription_pydantic.extend(parsed_chunk_utterances)
            else:
                 corrected_transcription_pydantic.extend(parsed_chunk_utterances[:step_size])

        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}. Falling back to original text for this chunk.")
            if i == num_steps - 1:
                 corrected_transcription_pydantic.extend(current_chunk)
            else:
                 corrected_transcription_pydantic.extend(current_chunk[:step_size])

    segmentation_input_chunks = []
    if corrected_transcription_pydantic:
        for item in corrected_transcription_pydantic:
            # Using the original timestamp (which encodes the original order)
            segmentation_input_chunks.append(f"{item.speaker} ({item.timestamp}): {item.utterance}")
    else:
         print("Warning: Corrected transcription list is empty.")


    print(f"ASR chunking correction completed. Total corrected utterances: {len(corrected_transcription_pydantic)}")

    return {
        **state,
        "corrected_transcription": corrected_transcription_pydantic,
        "segmentation_input_text": "\n".join(segmentation_input_chunks)
    }
    
def check_transcription_quality_node(state: WorkflowState) -> WorkflowState:
    print("--- Checking Transcription Quality ---")
    corrected_transcription = state.get("corrected_trnascription")
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
    
    if avg_words < 2.0 or (short_utterances / len(corrected_transcription) > 0.6):
        print(f"Quality Check: Poor (Avg words: {avg_words:.2f}, High short utterances: {short_utterances})")
        return {**state, "quality_assessment": "poor", "error_message": "Transcription quality deemed poor."}

    print("Quality Check: Good")
    return {**state, "quality_assessment": "good"}

def handle_poor_quality_node(state: WorkflowState) -> WorkflowState:
    print("--- Handling Poor Quality Transcription ---")
    error_msg = state.get("error_message", "Processing halted due to poor quality.")
    print(f"Error: {error_msg}")
    return {**state, "formatted_outputs": {"status": "Failed: Poor Transcription Quality", "details": error_msg}}

def segment_transcription_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Segmenting Transcription using Ollama model: {MODEL_SEGMENTATION} ---")
    corrected_transcription_list = state.get("corrected_transcription")
    if not corrected_transcription_list:
        return {**state, "error_message": "Corrected transcription missing for segmentation."}
    
    parser = PydanticOutputParser(List[Segment])
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
    
    segments_with_utterances: List[SegmentWithUtterances] = []
    try:
        llm_segments_output: List[PydanticSegment] = segmentation_chain.invoke({
            "numbered_transcript": numbered_transcript[:16000], # Truncate for safety
            "num_utterances": len(corrected_transcription_list),
            "format_instructions": parser.get_format_instructions(),
        })

        for seg_info in llm_segments_output:
            segment_utterances_pydantic: List[SpeakerUtterance] = []
            for idx in seg_info.utterance_indices:
                if 0 <= idx < len(corrected_transcription_list):
                    segment_utterances_pydantic.append(corrected_transcription_list[idx])
            if segment_utterances_pydantic:
                segments_with_utterances.append(SegmentWithUtterances(
                    topic_name=seg_info.topic_name,
                    utterances_info=segment_utterances_pydantic
                ))
        if not segments_with_utterances:
            raise ValueError("LLM returned no valid segments or mapping failed.")

    except Exception as e: 
        print(f"Error during segmentation or parsing: {e}. Using full transcript as one segment.")
        segments_with_utterances = [SegmentWithUtterances(
            topic_name="General Discussion",
            utterances_info=corrected_transcription_list
        )]

    print(f"Segmentation completed into {len(segments_with_utterances)} segments.")
    return {**state, "segmented_transcription": segments_with_utterances}

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

def generate_summary_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Generating Summary using Ollama model: {MODEL_SUMMARY} ---")
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
        summary = summary_chain.invoke({"transcript": full_corrected_text}) 
    except Exception as e:
        print(f"Error during summary generation: {e}")
    print(f"Summary generated: {summary[:100]}...")
    return {**state, "summary": summary}

def validate_action_items_node(state: WorkflowState) -> WorkflowState:
    print("--- Validating Action Items ---")
    action_items = state.get("extracted_action_items")
    if action_items is None:
        return {**state, "action_validation_status": "all_valid", "finalized_action_items": []}

    needs_clarification = any(item.is_ambiguous for item in action_items)
    if needs_clarification:
        print("Validation: Some action items need clarification.")
        return {**state, "action_validation_status": "needs_clarification"}
    else:
        print("Validation: All action items seem valid.")
        return {**state, "action_validation_status": "all_valid", "finalized_action_items": action_items}

def attempt_llm_clarification_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Attempting LLM Clarification using Ollama: {MODEL_CLARIFICATION} ---")
    action_items = state.get("extracted_action_items", [])
    corrected_transcript_list = state.get("corrected_transcription")
    full_transcript_text = state.get("segmentation_input_text")

    if not corrected_transcript_list:
        print("Cannot attempt clarification: missing corrected transcript list.")
        return {**state, "finalized_action_items": action_items}

    parser = PydanticOutputParser(pydantic_object=PydanticActionItem)
    llm = get_ollama_llm(MODEL_CLARIFICATION, temperature=0.2)

    prompt_template = PromptTemplate.from_template(
        """
        Resolve ambiguities in the following action item based on the transcript context. 
        
        Original Action Item:
        Description: {action_description}
        Assignee: {assigned_to}
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
                    min_idx = max(0, min(item.source_utterance_indices) - context_window_size)
                    max_idx = min(len(corrected_transcript_list) -1, max(item.source_utterance_indices) + context_window_size)
                    context_lines = []
                    for k in range(min_idx, max_idx + 1):
                        if 0 <= k < len(corrected_transcript_list): # Double check index bounds
                            context_lines.append(f"[{k}] {corrected_transcript_list[k].speaker}: {corrected_transcript_list[k].utterance}")
                    context_utterances_text = "\n".join(context_lines)
                    if len(context_utterances_text) > 4000:
                         context_utterances_text = context_utterances_text[:4000] + "\n... (context truncated)"

                except Exception as context_e:
                     print(f"Error retrieving context for clarification: {context_e}")
                     context_utterances_text = "Error retrieving context."


            try:
                clarified_action = clarification_chain.invoke({
                    "action_description": item.action_description,
                    "assigned_to": item.assigned_to or "N/A",
                    "due_date": item.due_date or "N/A",
                    "ambiguity_reason": item.ambiguity_reason or "No reason given",
                    "context_utterances": context_utterances_text,
                    "source_utterance_indices_str": str(item.source_utterance_indices),
                    "format_instructions": parser.get_format_instructions(),
                })
                clarified_action.source_utterance_indices = item.source_utterance_indices
                clarified_items_list.append(clarified_action)
            except Exception as e:
                print(f"Error during LLM clarification for '{item.action_description[:50]}...': {e}. Keeping original.")
                clarified_items_list.append(item) 
        else:
            clarified_items_list.append(item) 

    return {**state, "finalized_action_items": clarified_items_list}

def generate_meeting_minutes_node(state: WorkflowState) -> WorkflowState:
    print(f"--- Generating Meeting Minutes using Ollama model: {MODEL_MEETING_MINUTES} ---")
    summary = state.get("summary", "No summary available.")
    action_items = state.get("finalized_action_items", [])
    full_corrected_text = state.get("segmentation_input_text", "Full transcript not available.")
    meeting_title = state.get("raw_transcription", "Meeting")[:50].split('\n')[0].strip() # Use a bit more text for title, split on newline
    meeting_date = "Today" 

    attendees_list = []
    if state.get("corrected_transcription"):
        attendees_list = sorted(list(set(item.speaker for item in state["corrected_transcription"]))) # type: ignore

    action_items_str = ""
    if action_items:
        for i, ai in enumerate(action_items):
            assignee = ai.assigned_to or "Unassigned"
            due = ai.due_date or "N/A"
            flag = " (AMBIGUOUS)" if ai.is_ambiguous else ""
            ambiguity_detail = f" - Reason: {ai.ambiguity_reason}" if ai.is_ambiguous and ai.ambiguity_reason else ""
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
        Analyze the provided full transcript and list 3-7 key topics, discussion points, or decisions that were made, different from the summary and action items unless they are major points.
        
        3. Action Items:\n{action_items}
        
        Full Transcript (for context and deriving key discussion points):\n{full_transcript}
        
        Compile the MoM in a clean, professional, and well-formatted markdown format.
        """
    )
    mom_chain = prompt_template | llm | StrOutputParser()
    mom = "Meeting minutes generation failed."
    try:
        transcript_for_mom = full_corrected_text if full_corrected_text else "Transcript not available."

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

def format_outputs_node(state: WorkflowState) -> WorkflowState:
    print("--- Formatting Outputs ---")
    final_action_items = state.get("finalized_action_items", [])
    action_items_dict = [item.model_dump() for item in final_action_items]

    final_outputs = {
        "summary": state.get("summary", "N/A"),
        "action_items": action_items_dict,
        "meeting_minutes": state.get("meeting_minutes", "N/A"),
        "status": "Success" if state.get("formatted_outputs", {}).get("status") != "Failed: Poor Transcription Quality" else "Failed: Poor Transcription Quality",
        "error_details": state.get("error_message", None)
    }

    if state.get("quality_assessment") == "poor":
         final_outputs["status"] = "Failed: Poor Transcription Quality"
         final_outputs["details"] = state.get("error_message", "Processing halted early due to poor quality.")
         if not state.get("meeting_minutes"):
             final_outputs["summary"] = "N/A"
             final_outputs["action_items"] = []
             final_outputs["meeting_minutes"] = "N/A"


    print("Outputs formatted.")
    return {**state, "formatted_outputs": final_outputs}

def decide_quality_branch(state: WorkflowState) -> Literal["segment_transcription", "handle_poor_quality"]:
    return "segment_transcription" if state.get("quality_assessment") == "good" else "handle_poor_quality"

def decide_action_item_validation_branch(state: WorkflowState) -> Literal["attempt_clarification", "generate_mom"]:
    return "attempt_clarification" if state.get("action_validation_status") == "needs_clarification" else "generate_mom"

graph = StateGraph(WorkflowState)
graph.add_node("ingest_transcription", ingest_transcription_node)
graph.add_node("preprocess_parsing", preprocess_transcription_parsing_node)
graph.add_node("llm_asr_correction", llm_asr_correction_node)
graph.add_node("check_quality", check_transcription_quality_node)
graph.add_node("handle_poor_quality", handle_poor_quality_node)
graph.add_node("segment_transcription", segment_transcription_node)
graph.add_node("extract_actions", extract_action_items_node)
graph.add_node("generate_summary", generate_summary_node)
graph.add_node("validate_actions", validate_action_items_node)
graph.add_node("attempt_clarification", attempt_llm_clarification_node)
graph.add_node("generate_mom", generate_meeting_minutes_node)
graph.add_node("format_outputs", format_outputs_node)

graph.set_entry_point("ingest_transcription")
graph.add_edge("ingest_transcription", "preprocess_parsing")
graph.add_edge("preprocess_parsing", "llm_asr_correction")
graph.add_edge("llm_asr_correction", "check_quality")
graph.add_conditional_edges("check_quality", decide_quality_branch, {
    "segment_transcription": "segment_transcription",
    "handle_poor_quality": "handle_poor_quality",
})
graph.add_edge("handle_poor_quality", END)
graph.add_edge("segment_transcription", "extract_actions")
graph.add_edge("extract_actions", "generate_summary")
graph.add_edge("generate_summary", "validate_actions")
graph.add_conditional_edges("validate_actions", decide_action_item_validation_branch, {
    "attempt_clarification": "attempt_clarification",
    "generate_mom": "generate_mom"
})
graph.add_edge("attempt_clarification", "generate_mom")
graph.add_edge("generate_mom", "format_outputs")
graph.add_edge("format_outputs", END)

app = graph.compile()

sample_transcription = """
Speaker Alpha: Hello team, let's start. We need to discuz the Q2 project plan.
Speaker Beta: Agreed. I think the main deliverable should be finalized by next Wensday.
Speaker Alpha: Okay, Beta, can you take the lead on drafting that?
Speaker Gamma: I can help Beta with the market research part. Maybe someone can look into the budget?
Speaker Alpha: Good idea, Gamma. Let's assign the budget review to... well, we need to figure that out. It is very importnt.
Speaker Beta: I also think we should shedule a follow-up for early next week.
Speaker Alpha: Yes, a follow-up meeting is essential. I'll send out an invite. End of meeting.
Speaker Delta: Just to add, regarding the marketing campaign, we need to ensure the creatives are approved by EOD Friday. Is that clear for everyone?
Speaker Alpha: Yes, Delta, that's critical. Mark, can you handle the creative approvals?
Speaker Mark: I can take that on. Approvals by Friday EOD. Got it.
Speaker Gamma: And I'll get that market research over to Beta by Tuesday morning.
Speaker Beta: Perfect, thanks Gamma. That will help finalize the deliverable draft by Wednesday.
Speaker Alpha: Excellent. So Beta drafts deliverable by Wednesday, Gamma provides research by Tuesday, Mark approves creatives by Friday. We also need to confirm that budget review owner. I will follow up offline.
Speaker Delta: Sounds good.
Speaker Alpha: Alright, anything else? No? Meeting adjourned.
"""

initial_state = {"raw_transcription": sample_transcription}
print("\n--- Starting Transcription Processing Workflow ---")
final_state = app.invoke(initial_state)
print("\n--- Workflow Execution Complete ---")
print("\n--- Final Formatted Outputs ---")
if final_state.get("formatted_outputs"):
    output_data = final_state["formatted_outputs"]
    print(f"Status: {output_data.get('status')}")
    if output_data.get('status') == "Success":
        print("\nSummary:")
        print(output_data.get("summary", "N/A"))
        print("\nAction Items:")
        for i, ai_dict in enumerate(output_data.get("action_items", [])):
            # ai_dict is now a dictionary from PydanticActionItem.model_dump()
            print(f"  {i+1}. Description: {ai_dict.get('action_description')}")
            print(f"      Assigned To: {ai_dict.get('assigned_to', 'N/A')}")
            print(f"      Due Date: {ai_dict.get('due_date', 'N/A')}")
            if ai_dict.get('is_ambiguous'):
                print(f"      AMBIGUOUS: {ai_dict.get('ambiguity_reason', 'No reason given')}")
            print(f"      Source Indices: {ai_dict.get('source_utterance_indices', [])}") # Print source indices
        print("\nMeeting Minutes:")
        print(output_data.get("meeting_minutes", "N/A"))
    else:
        print(f"Error Details: {output_data.get('details')}")
else:
    print("No formatted outputs found in the final state.")
    print("\n--- Full Final State (selected fields) ---")
    for key, value in final_state.items():
        if key not in ['raw_transcription', 'segmentation_input_text']: # Exclude very verbose fields
                print(f"{key}: {str(value)[:500] + '...' if isinstance(value, str) and len(value) > 500 else value}")
