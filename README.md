# 🎙️ VoiceRAG

### Real-Time RAG Voice Agent for Customer Support

VoiceRAG is an end-to-end real-time voice AI system built for customer-support
conversations.

It combines streaming speech recognition, Voice Activity Detection, Retrieval-
Augmented Generation, semantic reranking, Qwen3-30B, streaming Text-to-Speech,
WebSockets, conversational memory, and real-time interruption handling into a
low-latency voice interaction pipeline.

## 🧠 What is VoiceRAG?

VoiceRAG is designed to make a RAG-based customer-support system feel like a
real conversation rather than a traditional chatbot.

The user can speak naturally, ask follow-up questions, and interrupt the agent
while it is speaking.

Instead of waiting for every stage to finish sequentially, VoiceRAG streams
information through the pipeline so that the agent can begin responding as
early as possible.

```text
User Speech
     ↓
Speech Recognition
     ↓
Retrieval
     ↓
Semantic Reranking
     ↓
Qwen3-30B
     ↓
Streaming TTS
     ↓
Audio Response
`````
Immediately after the introduction, put your **actual before/after results**.


## 🚀 Performance: Before vs After

A major focus of VoiceRAG has been reducing the latency of real-time voice
interaction.

The original implementation relied on a much more sequential pipeline. Through
streaming architecture and LLM inference optimization, the system achieved
significant reductions in response latency.

| Metric | Before | After | Improvement |
|---|---:|---:|---:|
| 🎧 Time-to-First-Audio | ~9.5s | **1.13s** | **88% reduction** |
| 🤖 Qwen3-30B Generation | ~64s | **2.09s** | **97% reduction** |
| ⚡ Qwen TTFT | — | **0.44s** | — |
| 📚 Retrieval + Reranking | — | **~0.2–1.1s** | — |

### 🎧 Time-to-First-Audio

**~9.5s → 1.13s**

An approximately **88% reduction** in the time between the user finishing
their request and the first AI-generated audio being played.

The major improvement came from streaming the response instead of waiting for
the complete LLM response before starting TTS.

### 🤖 Qwen3-30B Generation

**~64s → 2.09s**

An approximately **97% reduction** in total Qwen3-30B generation time.

The optimization included:

- Q4_K_M quantization
- Flash Attention
- Reasoning-budget tuning
- GPU-based inference
- Streaming generation

The optimized setup currently achieves approximately **0.44s Time-to-First-
Token (TTFT)**.

> Performance can vary depending on GPU environment, query length, context
> size, response length, and runtime conditions.
```

## 🏗️ How It Works

VoiceRAG uses a streaming pipeline where the major stages work together rather
than waiting for the entire previous stage to finish.

```text
                         🎤 USER
                            │
                            ▼
                     ┌─────────────┐
                     │  Silero VAD │
                     └──────┬──────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Nemotron   │
                    │ Streaming ASR│
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Retrieval  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Reranking   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Qwen3-30B   │
                    └──────┬───────┘
                           │
                    Streaming Output
                           │
                           ▼
                    ┌──────────────┐
                    │     TTS      │
                    └──────┬───────┘
                           │
                           ▼
                     🔊 SPEAKER
```

Current Status / What's Next

**This is important because you told me the project is NOT finished.**

Put this near the end:

```markdown
## 🚧 Current Status

VoiceRAG is an **active work-in-progress**.

The core real-time voice pipeline is currently functional, including:

- Streaming speech recognition
- VAD-based utterance detection
- RAG retrieval
- Semantic reranking
- Qwen3-30B inference
- Streaming LLM output
- Sentence-level TTS
- Conversational memory
- WebSocket communication
- Real-time TTS interruption

### 🔬 What's Next

The next stage is focused on making the agent more robust in real-world
conversational environments.

Planned work includes:

- 🧠 Semantic interruption detection
- 🔊 Improved background-noise robustness
- 🗣️ Handling overlapping speech and background speakers
- 🎙️ Testing different voice-agent configurations
- 🤖 Testing additional voice/LLM models
- ⚡ Further latency optimization
- 🔄 More robust conversation-state handling
- 🧪 Broader real-world voice-agent evaluation

```
The long-term goal is to move from a fast RAG voice pipeline toward a more
natural, context-aware voice agent capable of handling real-world
conversations.

## 🛠️ Tech Stack

### AI / ML

- Python
- PyTorch
- Silero VAD
- Nemotron
- Qwen3-30B
- Retrieval-Augmented Generation
- Semantic Reranking
- Text-to-Speech

### Systems

- WebSockets
- NumPy
- sounddevice
- Google Colab
- Kaggle GPU
- macOS

---

## 🎯 Project Goal

Build a voice agent that can **listen, reason, retrieve, speak, and respond
naturally in real time**.

VoiceRAG focuses on combining:

**Grounded RAG + optimized LLM inference + streaming + conversational memory +
real-time interruption**

into a practical customer-support voice agent.
