# Meeting Transcript Processor

An automated pipeline for processing meeting transcripts using LangGraph and LLMs. This system extracts action items, generates summaries, and creates professional meeting minutes from raw speaker transcript data.

## Workflow Overview

<img width="242" alt="image" src="https://github.com/user-attachments/assets/92219435-0111-4be2-9896-d237e92dbf49" />

## Features

- **ASR Error Correction**: Fixes spelling, grammar, and punctuation errors in transcripts
- **Topic Segmentation**: Identifies distinct discussion topics within the meeting
- **Action Item Extraction**: Extracts tasks, owners, and due dates 
- **Ambiguity Resolution**: Attempts to clarify ambiguous action items through context analysis
- **Meeting Summary**: Generates concise meeting summaries
- **Meeting Minutes**: Creates professional meeting minutes in Markdown format

## Architecture

This project uses:
- **LangGraph**: For orchestrating the multi-step workflow
- **LangChain**: For creating LLM chains and prompts
- **Ollama**: For running LLMs locally
- **Pydantic**: For model validation and data structures

## Processing Pipeline

1. **Ingestion**: Accepts raw transcript text
2. **Parsing**: Extracts speaker information and utterances
3. **ASR Correction**: Fixes transcription errors (chunked for long transcripts)
4. **Quality Assessment**: Evaluates transcript quality
5. **Segmentation**: Groups transcript by discussion topics
6. **Action Item Extraction**: Identifies tasks, assignees, and due dates
7. **Summary Generation**: Creates a concise meeting summary
8. **Action Item Validation**: Checks for ambiguities
9. **Clarification**: Attempts to resolve ambiguous action items
10. **Meeting Minutes Generation**: Creates formatted meeting minutes
11. **Output Formatting**: Prepares final structured output

## Requirements

- Python 3.9+
- Ollama with a compatible model (default: qwen3)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/meeting-transcript-processor.git
cd meeting-transcript-processor

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running with the required model
ollama pull qwen3
```

## Usage

```python
from transcript_processor import build_graph

# Initialize the workflow
app = build_graph()

# Process a transcript
transcript = """
Speaker Alpha: Hello team, let's start. We need to discuss the Q2 project plan.
Speaker Beta: Agreed. I think the main deliverable should be finalized by next Wednesday.
Speaker Alpha: Okay, Beta, can you take the lead on drafting that?
Speaker Gamma: I can help Beta with the market research part.
Speaker Alpha: Good idea, Gamma. Let's wrap this up.
"""

# Run the workflow
result = app.invoke({"raw_transcription": transcript})

# Access the outputs
print(result["formatted_outputs"]["status"])
print(result["formatted_outputs"]["summary"])
print(result["formatted_outputs"]["meeting_minutes"])
```

## Configuration

You can configure the LLM models used for each component by modifying the following constants at the top of the file:

```python
MODEL_ASR_CORRECTION = "qwen3"
MODEL_SEGMENTATION = "qwen3"
MODEL_ACTION_EXTRACTION = "qwen3"
MODEL_SUMMARY = "qwen3"
MODEL_CLARIFICATION = "qwen3"
MODEL_MEETING_MINUTES = "qwen3"
```

For processing long transcripts, you can also adjust the chunking parameters:

```python
ASR_CHUNK_SIZE = 20
ASR_CHUNK_OVERLAP = 5
```

## Example Output

The final output contains:

- A concise meeting summary
- Extracted action items with assignees and due dates
- Formatted meeting minutes in Markdown
- Processing status information
