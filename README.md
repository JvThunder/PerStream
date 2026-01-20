# PerStream: Personal Memory Assistant for Egocentric Video Streams

A multimodal AI system that processes egocentric video streams to build and query a personalized memory graph, supporting both **passive (query-driven)** and **proactive (autonomous)** response modes.

## Key Features

- **Dual-Mode Operation**: Query-driven (passive) and autonomous (proactive) response generation
- **Personalized Memory Graph (PMG)**: Two-layer architecture with 30 semantic memory categories
- **Real-Time Processing**: Continuous video stream understanding with memory augmentation
- **Remember Gate Mechanism**: Intelligent frame filtering using category similarity
- **Memory Efficiency**: Granular reduction strategies (NSBG/GSBN) for VRAM management
- **Multi-GPU Support**: Distributed model loading for large-scale inference
- **Spatial Preservation**: Pooled embeddings maintain patch grid structure for precise retrieval

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO STREAM INPUT                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Frame Extraction      │
          │  (OpenCV + Resize)     │
          └────────┬───────────────┘
                   │
                   ▼
          ┌────────────────────────┐
          │  Optical Flow Sampling │
          │  (Keyframe Selection)  │
          └────────┬───────────────┘
                   │
        ┌──────────┴───────────────┐
        │                          │
        ▼                          ▼
    [Remember Gate]          [Caption Gen]
    [Category Match]         [Triplet Ext]
        │                          │
        └──────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │ Personalized Memory Graph    │
    │  (30 categories, triplets)   │
    └──────────┬───────────────────┘
               │
        ┌──────┴─────┐
        │            │
        ▼            ▼
    [Passive]   [Proactive]
    [Query]     [Response]
        │            │
        └──────┬─────┘
               │
               ▼
        [User Response]
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+
- Conda

### Setup

```bash
# Create conda environment
conda create -n perstream python=3.10
conda activate perstream

# Install PyTorch with CUDA support
pip install torch==2.3.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Environment Variables

Set up your OpenAI API key for triplet extraction in .env:
```bash
OPENAI_API_KEY=sk-proj-your-api-key-here
```

## Project Structure

```
PerStream/
├── src/                                  # Core source code
│   ├── core/                             # Core algorithms and memory management
│   │   ├── perstream.py                  # Main orchestration system
│   │   ├── personalized_memory_graph.py  # Two-layer memory architecture
│   │   ├── memory_subcategories.py       # 30 semantic memory categories
│   │   ├── passive_user_query.py         # Query-driven response generation
│   │   ├── proactive_user_query.py       # Autonomous response generation
│   │   └── memory_dataset_class.py       # PyTorch Dataset for evaluation
│   │
│   ├── utils/                            # Utility functions
│   │   ├── perstream_utils.py            # Core utilities (embeddings, triplets, gates)
│   │   ├── video_utils.py                # Video processing and optical flow
│   │   └── model_utils.py                # Model loading and projection MLP
│   │
│   ├── train/                            # Training scripts
│   │   ├── train_perstream.py            # LoRA fine-tuning for Qwen2.5-Omni
│   │   └── train_projection.py           # Visual-semantic alignment training
│   │
│   └── eval/                             # Evaluation scripts
│       ├── eval_passive_dataset.py       # Passive mode evaluation
│       ├── eval_proactive_dataset.py     # Proactive mode evaluation
│       ├── eval_passive_reduction.py     # Memory reduction evaluation
│       ├── eval_proactive_reduction.py   # Memory reduction evaluation
│       ├── score_passive_judge.py        # LLM-based passive scoring
│       └── score_proactive_judge.py      # LLM-based proactive scoring
│
├── dataset/                              # Dataset files
│   ├── drivenact.json                    # DrivenAct dataset annotations
│   └── ego4d.json                        # Ego4D dataset annotations
│
├── finetuned_models/                     # Pre-trained model checkpoints
├── ckpts/                                # Projection model checkpoints
├── scripts/                              # Bash scripts for execution
├── evaluation/                           # Evaluation results and logs
├── sample_videos/                        # Test video samples
├── requirements.txt                      # Python dependencies
└── setup.py                              # Package setup configuration
```

## Usage

### Running Inference

Run the main inference pipeline on a video:

```bash
bash scripts/perstream.sh
```

Or run directly with Python:

```python
from src.core.perstream import video_stream_with_memory

# Process video with memory augmentation
video_stream_with_memory(
    video_path="path/to/video.mp4",
    buffer_size=4,
    gamma_threshold=0.3,
    enable_proactive=True
)
```

### Training

#### LoRA Fine-tuning

Fine-tune Qwen2.5-Omni on memory-augmented video QA:

```bash
bash scripts/train.sh
```

Or run directly:

```bash
python -m src.train.train_perstream \
    --model_name "Qwen/Qwen2.5-Omni-7B" \
    --dataset_path "dataset/drivenact.json" \
    --output_dir "finetuned_models/custom_model" \
    --num_epochs 3 \
    --learning_rate 2e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 8
```

**Training Configuration:**
- LoRA: r=8, alpha=16, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Mixed precision: bf16/fp16

#### Projection Model Training

Train visual-semantic alignment:

```bash
bash scripts/proj_align.sh
```

### Evaluation

Run evaluation on datasets:

```bash
# Passive mode evaluation
python -m src.eval.eval_passive_dataset \
    --dataset_path "dataset/drivenact.json" \
    --model_path "finetuned_models/qwen_lora_finetune_drivenact_passive"

# Proactive mode evaluation
python -m src.eval.eval_proactive_dataset \
    --dataset_path "dataset/drivenact.json" \
    --model_path "finetuned_models/qwen_lora_finetune_drivenact_proactive"

# LLM-based scoring
python -m src.eval.score_passive_judge --results_path "evaluation/passive_results.json"
python -m src.eval.score_proactive_judge --results_path "evaluation/proactive_results.json"
```

## Using Your Own Dataset

### Dataset Format

Create a JSON file with the following structure:

```json
[
  {
    "participant_id": 1,
    "video_id": "path/to/video_file",
    "timestamp": "120.5",
    "caption": "Description of the scene",
    "split": "train",
    "type_1_memories": [
      "Short factual memory 1 (habits, preferences)",
      "Short factual memory 2"
    ],
    "type_2_memories": [
      "Longer narrative memory describing an episodic event...",
      "Another episodic memory..."
    ],
    "type": "passive",
    "question": "What did I do yesterday at the store?",
    "answer": "You bought groceries and talked to the cashier about..."
  }
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | int | Unique identifier for the participant |
| `video_id` | string | Path to video file (relative or absolute) |
| `timestamp` | string | Timestamp in seconds where the event occurs |
| `caption` | string | Brief description of the current scene |
| `split` | string | Dataset split: "train", "val", or "test" |
| `type_1_memories` | list[str] | Short factual memories (habits, preferences, facts) |
| `type_2_memories` | list[str] | Long narrative memories (episodic events) |
| `type` | string | Query type: "passive" (user-initiated) or "proactive" (system-initiated) |
| `question` | string | User query (for passive type) or context (for proactive) |
| `answer` | string | Ground truth answer |

### Memory Types

- **Type 1 (Factual)**: Short, declarative statements about habits/preferences
  - Example: "I keep snacks in my car.", "I prefer coffee over tea."

- **Type 2 (Episodic)**: Longer narratives describing past events
  - Example: "Last Tuesday, I accidentally left my umbrella at the coffee shop and had to go back the next day to retrieve it."

### Video Requirements

- **Format**: MP4, AVI, or other OpenCV-compatible formats
- **Resolution**: 224p+ recommended (auto-resized during processing)
- **Frame Rate**: Any (keyframes selected via optical flow)

### Example: Creating a Custom Dataset

```python
import json

dataset = [
    {
        "participant_id": 1,
        "video_id": "videos/cooking_session_01.mp4",
        "timestamp": "45.0",
        "caption": "chopping vegetables on cutting board",
        "split": "train",
        "type_1_memories": [
            "I am vegetarian.",
            "I prefer organic vegetables.",
            "I usually cook dinner at 7 PM."
        ],
        "type_2_memories": [
            "Last week I tried a new recipe for vegetable stir-fry and it turned out great. I used extra garlic.",
            "Two months ago I cut my finger while chopping onions and had to bandage it."
        ],
        "type": "passive",
        "question": "What vegetables do I usually cook with?",
        "answer": "Based on your memories, you typically cook with organic vegetables. You've recently made vegetable stir-fry with extra garlic."
    }
]

with open("dataset/my_custom_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
```

### Running with Custom Dataset

```bash
# Training
python -m src.train.train_perstream \
    --dataset_path "dataset/my_custom_dataset.json" \
    --output_dir "finetuned_models/my_model"

# Evaluation
python -m src.eval.eval_passive_dataset \
    --dataset_path "dataset/my_custom_dataset.json"
```

## Configuration Parameters

### Core Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `gamma_threshold` | 0.3 | 0.0-1.0 | Remember gate similarity threshold |
| `delta_dfs_threshold` | 0.3 | 0.0-1.0 | PMG retrieval activation threshold |
| `buffer_size` | 4 | 1-32 | Frames per processing batch |
| `pool_size` | (8, 8) | (2,2)-(16,16) | Adaptive pooling patches |
| `top_k` | 5 | 1-20 | Number of top memories to retrieve |
| `queue_size` | 4 | 1-64 | Short-term memory FIFO size |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 200 | Maximum generation length |
| `temperature` | 1.0 (passive), 0.0 (proactive) | Sampling temperature |
| `do_sample` | True/False | Enable/disable sampling |

### Memory Reduction

| Strategy | Description |
|----------|-------------|
| NSBG | Node Scan by Granularity: Process cold→hot nodes, remove fine→coarse storage |
| GSBN | Granularity Scan by Node: Process fine→coarse granularities, remove cold→hot nodes |

**Granularity Hierarchy** (finest to coarsest):
1. `v_nearest`: Nearest visual vector (~1-10 MB/node)
2. `v_mean`: Mean visual vector (~1-10 MB/node)
3. `v_M`: Caption embedding (384-dim, ~1.5 KB/node)
4. `M`: Raw text caption (~100-500 bytes/node)

## Memory Categories

The system organizes memories into 30 semantic categories:

| Category Group | Categories |
|----------------|------------|
| Personal Identity | name, identification, vehicle, contact, education |
| Basic Profile | nickname, age, physical, birth, employment |
| Health/Medical | medical, health |
| Transactions | transactions, financial, cards |
| Daily Behavior | apps, schedule, browsing, mentioned, phrases |
| Location | travel, location, places |
| Social | relationships, birthdays, interests, addresses |
| Personal Data | personal_interests, notes, media, app_data, contacts_list, chat, messages |

## Model Information

### Base Model

- **Model**: Qwen2.5-Omni-7B
- **Vision Encoder**: Spatial visual features (3584-dim)
- **Language Decoder**: Text generation with memory context
- **Projection**: 3584 → 384 dimensional alignment

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `buffer_size` to 2-4
- Enable gradient checkpointing during training
- Use memory reduction strategies (NSBG/GSBN)
- Enable `low_memory_mode` in dataset class

**OpenAI API Errors**
- Verify `OPENAI_API_KEY` is set correctly
- Check API rate limits
- Triplet extraction has retry logic (1 retry)

**Video Loading Issues**
- Ensure video format is OpenCV-compatible
- Check file path permissions
- Verify ffmpeg is installed

### Performance Tips

1. Use optical flow sampling to reduce frame redundancy
2. Adjust `gamma_threshold` based on video content density
3. For long videos, enable memory reduction periodically
4. Use multi-GPU setup for large models

## Citation

If you use PerStream in your research, please cite:

```bibtex
@article{perstream2024,
  title={PerStream: Personal Memory Assistant for Egocentric Video Streams},
  author={...},
  journal={...},
  year={2024}
}
```

## License

[Add your license information here]

## Acknowledgments

- Built on [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)
- Uses [Sentence Transformers](https://www.sbert.net/) for text embeddings
- OpenAI GPT-3.5-turbo for triplet extraction
